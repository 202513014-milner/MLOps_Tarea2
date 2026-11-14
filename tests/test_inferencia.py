"""Pruebas de la inferencia mensual (P3, parte de Kim): TLV, grupos de ejecución y flujo del mes.

No necesitan Docker, Airflow ni MLflow: el champion es un Pipeline falso con la misma forma que el real
(`preprocessing` + `model`) y el registro en MLflow se reemplaza por una función que cuenta llamadas.
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import inference
from src.posprocessing import grupos_mes, puntuar

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'config/config.yaml').read_text(encoding='utf8'))
FRESCURA = CFG['postprocessing']['freshness_prob']


def _mes(rng, n):
    return pd.DataFrame({'codunicocli': [str(i) for i in range(n)], 'x1': rng.normal(0, 1, n), 'x2': rng.normal(0, 1, n),
                         'monto': rng.gamma(2, 500, n), 'prob_value_contact': np.where(rng.random(n) < .2, np.nan, rng.random(n)),
                         'grp_campecs06m': rng.choice(['G1', 'G2', 'G3', 'G4', 'G5'], n), 'p_fecinformacion': 202607,
                         'target': rng.integers(0, 2, n)})


# ------------------------------- posprocesamiento -------------------------------
def test_tlv_sigue_la_formula_de_negocio():
    df = pd.DataFrame({'monto': [1000.0, 0.0, np.nan], 'prob_value_contact': [0.5, np.nan, 0.2], 'grp_campecs06m': ['G1', 'G9', 'G4']})
    out = puntuar(df, [0.8, 0.6, 0.4], CFG)
    esperado = [0.8 * 0.5 * np.log(1001) * FRESCURA['G1'],
                0.6 * CFG['postprocessing']['contact_fill'] * np.log(1) * FRESCURA['default'],  # grupo desconocido -> default, monto 0
                0.4 * 0.2 * np.log(1) * FRESCURA['G4']]                                          # monto nulo -> 0
    assert out['puntuacion_tlv'].tolist() == pytest.approx(esperado)


def test_grupos_tlv_proporciones_y_orden():
    rng = np.random.default_rng(1)
    out = puntuar(_mes(rng, 20000), rng.random(20000), CFG)
    g = grupos_mes(out['puntuacion_tlv'], CFG)
    assert (g == 1).mean() == pytest.approx(0.05, abs=1e-3) and (g == 10).mean() == pytest.approx(0.035, abs=1e-3)
    assert set(g) == set(range(1, 11))
    assert out.loc[g == 1, 'puntuacion_tlv'].min() >= out.loc[g == 10, 'puntuacion_tlv'].max()


# ------------------------------- flujo mensual -------------------------------
@pytest.fixture
def entorno(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    (tmp_path / 'in').mkdir()
    for mes in CFG['inference']['months']:
        _mes(rng, 3000).to_csv(tmp_path / 'in' / f'{mes}_extrac.csv', index=False)
    tr = _mes(rng, 3000)
    prep = ColumnTransformer([('n', StandardScaler(), ['x1', 'x2', 'monto'])], remainder='drop')
    pipe = Pipeline([('preprocessing', prep), ('model', LogisticRegression())]).fit(tr[['x1', 'x2', 'monto']], tr['target'])
    cfg = {**CFG, 'input_dir': str(tmp_path / 'in'), 'output_dir': str(tmp_path / 'out'),
           'postprocessing': dict(CFG['postprocessing']), 'inference': {**CFG['inference'], 'expected_algorithm': None}}
    monkeypatch.setattr(inference, 'info_champion', lambda c: {'model_name': 'm', 'model_version': '1'})
    monkeypatch.setattr(inference, 'cargar_champion', lambda c: pipe)
    runs = []
    monkeypatch.setattr(inference, 'registrar_run', lambda cfg_, suf, key, *a, **k: runs.append((suf, key)) or f'run{len(runs)}')
    return cfg, runs


def correr(cfg, mes):
    ctx = inference.cargar_datos(cfg, mes)
    for paso in (inference.preprocesar_datos, inference.generar_scores, inference.posprocesar, inference.registrar_inferencia):
        ctx = paso(cfg, ctx)
    return ctx


def test_flujo_mensual_genera_salidas_completas(entorno):
    cfg, runs = entorno
    ctx = correr(cfg, 'p7')
    d = Path(ctx['dir'])
    s = pd.read_csv(d / 'scores_clientes.csv', dtype={'codunicocli': 'string'})
    assert list(s.columns) == ['codunicocli', 'prob', 'puntuacion_tlv', 'grupo_ejec_tlv']
    assert len(s) == ctx['n_clientes'] == 3000 and s['codunicocli'].is_unique
    assert s['prob'].between(0, 1).all() and (s['puntuacion_tlv'] >= 0).all()
    assert set(s['grupo_ejec_tlv']) == set(range(1, 11))
    assert pd.read_csv(d / 'resumen_grupos.csv')['clientes'].sum() == 3000
    assert 'target' not in pd.read_pickle(d / 'datos.pkl') and ctx['target_descartado'] is True   # el target nunca llega al scoring
    assert set(json.loads((d / 'compute_logs.json').read_text())) == {'cargar_datos', 'preprocesar_datos', 'generar_scores', 'posprocesar'}
    assert json.loads((d / 'inference_context.json').read_text())['inference_run_id'] == ctx['inference_run_id']
    assert runs == [('inference', ctx['key'])]


def test_p7_a_p10_cada_mes_en_su_carpeta(entorno):
    cfg, _ = entorno
    dirs = {mes: correr(cfg, mes)['dir'] for mes in cfg['inference']['months']}
    assert len(set(dirs.values())) == 4 and all(Path(d).parent.name == m for m, d in dirs.items())


def test_idempotencia_misma_entrada_misma_carpeta_y_mismos_scores(entorno):
    cfg, runs = entorno
    a = correr(cfg, 'p8')
    antes = pd.read_csv(Path(a['dir']) / 'scores_clientes.csv')
    b = correr(cfg, 'p8')
    assert a['dir'] == b['dir'] and a['key'] == b['key']
    pd.testing.assert_frame_equal(antes, pd.read_csv(Path(b['dir']) / 'scores_clientes.csv'))
    assert len({k for _, k in runs}) == 1                       # el registro usa la misma idempotency_key (no crea otro run)
    assert len(list((Path(cfg['output_dir']) / 'inference' / 'p8').iterdir())) == 1


def test_cambiar_datos_o_posprocesamiento_crea_otra_corrida(entorno):
    cfg, _ = entorno
    base = correr(cfg, 'p9')['key']
    cfg['postprocessing']['freshness_prob'] = {**FRESCURA, 'G1': 0.07}
    assert correr(cfg, 'p9')['key'] != base
    cfg['postprocessing'] = dict(CFG['postprocessing'])
    assert correr(cfg, 'p9')['key'] == base                     # volver a la config original reutiliza la corrida
    (Path(cfg['input_dir']) / 'p9_extrac.csv').write_text((Path(cfg['input_dir']) / 'p10_extrac.csv').read_text())
    assert correr(cfg, 'p9')['key'] != base


def test_entrada_incompleta_falla_antes_de_puntuar(entorno):
    cfg, _ = entorno
    f = Path(cfg['input_dir']) / 'p10_extrac.csv'
    pd.read_csv(f).drop(columns=['monto']).to_csv(f, index=False)
    with pytest.raises(ValueError, match='faltan columnas'):
        inference.cargar_datos(cfg, 'p10')


def test_champion_con_otro_algoritmo_se_rechaza(monkeypatch):
    modelo = types.SimpleNamespace(named_steps={'model': LogisticRegression()})
    mlflow_sklearn = types.SimpleNamespace(load_model=lambda uri: modelo)
    monkeypatch.setitem(sys.modules, 'mlflow', types.SimpleNamespace(sklearn=mlflow_sklearn))
    monkeypatch.setitem(sys.modules, 'mlflow.sklearn', mlflow_sklearn)
    monkeypatch.setattr(inference, '_mlflow', lambda cfg: None)
    with pytest.raises(RuntimeError, match='se esperaba CatBoost'):
        inference.cargar_champion({**CFG, 'inference': {**CFG['inference'], 'expected_algorithm': 'CatBoost'}})


def test_posprocesamiento_rechaza_tlv_incorrecto(entorno, monkeypatch):
    """Comprueba que un TLV incorrecto detiene la inferencia."""

    # Guardar la función original de Kim
    puntuar_original = inference.puntuar

    # Simular un error en el cálculo del TLV
    def puntuar_incorrecto(df, scores, cfg):
        resultado = puntuar_original(df, scores, cfg)
        resultado.loc[resultado.index[0], "puntuacion_tlv"] += 1
        return resultado

    # Sustituir temporalmente la función durante esta prueba
    monkeypatch.setattr(inference, "puntuar", puntuar_incorrecto)

    # El pipeline debe detectar el TLV incorrecto y detenerse
    with pytest.raises(ValueError, match="TLV incorrecto"):
        correr(entorno[0], "p7")
