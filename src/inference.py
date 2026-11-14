"""Pipeline de inferencia mensual (P3): datos -> preprocesamiento -> scores -> posprocesamiento -> registro.

El champion NO se reentrena: se carga del Model Registry por alias. El preprocesamiento es el paso
`preprocessing` ya ajustado de su Pipeline (el mismo del entrenamiento). Cada tarea recibe y devuelve un
dict liviano (rutas y metadatos) por XCom; los datos pesados viajan por archivos en la carpeta del mes.
"""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from src.monitoring import ComputeMonitor
from src.posprocessing import grupos_mes, puntuar
from src.provenance import airflow_info, docker_info, git_info, sha256_file
from src.inference_validation import validar_probabilidades, validar_tlv


def cargar_cfg():
    """Config del proyecto (config/config.yaml con las rutas de Docker)."""
    from src.stages import cargar_config
    return cargar_config('config/config.yaml')


def _mlflow(cfg):
    """Apunta MLflow al servidor de la config y devuelve el módulo."""
    import mlflow
    from src.tracking import tracking_uri
    mlflow.set_tracking_uri(tracking_uri(cfg))
    return mlflow


def info_champion(cfg):
    """Nombre y versión del modelo con el alias champion (sin descargar el modelo)."""
    from mlflow.tracking import MlflowClient
    _mlflow(cfg)
    r = cfg['registry']
    return {'model_name': r['model_name'], 'model_version': str(MlflowClient().get_model_version_by_alias(r['model_name'], r['alias']).version)}


def cargar_champion(cfg):
    """Pipeline del champion; falla si el alias no apunta al algoritmo esperado (no cambia ningún alias)."""
    import mlflow.sklearn
    _mlflow(cfg)
    r = cfg['registry']
    modelo = mlflow.sklearn.load_model(f"models:/{r['model_name']}@{r['alias']}")
    algo, esperado = type(modelo.named_steps['model']).__name__, cfg['inference'].get('expected_algorithm')
    if esperado and esperado.lower() not in algo.lower():
        raise RuntimeError(f"El alias {r['alias']} apunta a {algo}, se esperaba {esperado}. Revisar el Model Registry.")
    return modelo


def transformar(modelo, X):
    """Aplica el preprocesamiento ajustado del champion y devuelve un DataFrame denso."""
    prep = modelo.named_steps['preprocessing']
    a = prep.transform(X)
    a = a.toarray() if hasattr(a, 'toarray') else np.asarray(a)
    try:
        cols = list(prep.get_feature_names_out())
    except Exception:
        cols = [f'x{i}' for i in range(a.shape[1])]
    return pd.DataFrame(a, columns=cols)


@contextmanager
def medir(carpeta, tarea):
    """Mide tiempo, CPU y RAM de una tarea y lo guarda en <carpeta>/tareas/<tarea>.json."""
    with ComputeMonitor(tarea) as m:
        yield
    (Path(carpeta) / 'tareas').mkdir(exist_ok=True)
    (Path(carpeta) / 'tareas' / f'{tarea}.json').write_text(json.dumps(m.summary, indent=1))


def compute_logs(carpeta):
    """Une los logs de cómputo de las tareas en compute_logs.json."""
    logs = {f.stem: json.loads(f.read_text()) for f in sorted((Path(carpeta) / 'tareas').glob('*.json'))}
    (Path(carpeta) / 'compute_logs.json').write_text(json.dumps(logs, indent=1))
    return logs


def registrar_run(cfg, sufijo, key, nombre, tags, metricas, archivos):
    """Crea (o reutiliza, por idempotency_key) un run de MLflow en el experimento <exp>-<sufijo>."""
    mlflow = _mlflow(cfg)
    from mlflow.tracking import MlflowClient
    exp = f"{cfg['mlflow']['experiment']}-{sufijo}"
    mlflow.set_experiment(exp)
    c = MlflowClient()
    prev = c.search_runs([c.get_experiment_by_name(exp).experiment_id], f"tags.idempotency_key = '{key}'", max_results=1)
    with mlflow.start_run(run_id=prev[0].info.run_id) if prev else mlflow.start_run(run_name=nombre) as run:
        mlflow.set_tags({**{k: str(v) for k, v in tags.items()}, 'idempotency_key': key, **git_info(), **docker_info(),
                         'airflow_dag_run_id': airflow_info()['airflow_dag_run_id']})
        mlflow.log_metrics({k: float(v) for k, v in metricas.items() if v == v}, step=0)
        for f in archivos:
            mlflow.log_artifact(str(f))
        return run.info.run_id


# ----------------------------- tareas del DAG -----------------------------
def cargar_datos(cfg, mes):
    """Lee el mes, DESCARTA el target, valida y fija la carpeta idempotente de la corrida."""
    path = Path(cfg['input_dir']) / cfg['inference']['input_pattern'].format(mes=mes)
    champ, data_hash = info_champion(cfg), sha256_file(path)
    cfg_hash = json.dumps(cfg['postprocessing'], sort_keys=True)  # solo lo que cambia los scores/grupos
    key = hashlib.sha256(f"{mes}|{data_hash}|{champ['model_version']}|{cfg_hash}".encode()).hexdigest()[:12]
    d = Path(cfg['output_dir']) / cfg['inference']['output_subdir'] / mes / key  # misma entrada + modelo + config => misma carpeta (idempotente)
    d.mkdir(parents=True, exist_ok=True)
    with medir(d, 'cargar_datos'):
        df = pd.read_csv(path, sep=cfg.get('csv_sep', ','), encoding=cfg.get('csv_encoding', 'utf-8'),
                         dtype={cfg['inference']['id_col']: 'string'}, low_memory=False)
        tenia_target = cfg['target_col'] in df
        df = df.drop(columns=[cfg['target_col']], errors='ignore')  # el target nunca llega al scoring
        p = cfg['postprocessing']
        faltan = [c for c in (p['amount_col'], p['contact_col'], p['freshness_col'], cfg['inference']['id_col']) if c not in df]
        if df.empty or faltan:
            raise ValueError(f'{mes}: archivo vacío o faltan columnas {faltan}')
        df.to_pickle(d / 'datos.pkl')
    return {'mes': mes, 'dir': str(d), 'key': key, 'data_hash': data_hash, 'n_clientes': len(df), 'target_descartado': tenia_target, **champ}


def preprocesar_datos(cfg, ctx):
    """Aplica el preprocesamiento del champion a las columnas de entrada que él espera (prep.pkl)."""
    d = Path(ctx['dir'])
    with medir(d, 'preprocesar_datos'):
        modelo, df = cargar_champion(cfg), pd.read_pickle(d / 'datos.pkl')
        cols = list(modelo.feature_names_in_)
        faltan = [c for c in cols if c not in df]
        if faltan:
            raise ValueError(f'Faltan columnas que usa el champion: {faltan[:10]}')
        transformar(modelo, df[cols]).to_pickle(d / 'prep.pkl')
    return ctx


def generar_scores(cfg, ctx):
    """Probabilidad de compra con el estimador del champion (predict_proba, sin fit)."""
    d = Path(ctx['dir'])
    with medir(d, 'generar_scores'):
        s = cargar_champion(cfg).named_steps['model'].predict_proba(pd.read_pickle(d / 'prep.pkl').to_numpy())[:, 1]
        if not np.isfinite(s).all() or s.min() < 0 or s.max() > 1:
            raise ValueError('Scores fuera de [0, 1]')
        np.save(d / 'scores.npy', s)
    return ctx


def posprocesar(cfg, ctx):
    """Puntuación TLV, validación y grupos de ejecución 1-10."""
    d, p, idc = Path(ctx['dir']), cfg['postprocessing'], cfg['inference']['id_col']

    with medir(d, 'posprocesar'):
        datos = pd.read_pickle(d / 'datos.pkl')[
            [idc, p['amount_col'], p['contact_col'], p['freshness_col']]
        ]

        # Calcular puntuación TLV
        out = puntuar(datos, np.load(d / 'scores.npy'), cfg)

        # Validar las probabilidades generadas
        validar_probabilidades(out, columna="prob")

        # Adaptar las columnas para la validación del TLV
        datos_validacion = out.rename(
            columns={
                "prob": "probabilidad",
                "puntuacion_tlv": "tlv",
            }
        ).copy()

        # Aplicar el mismo tratamiento utilizado en el posprocesamiento
        datos_validacion["prob_value_contact"] = (
            pd.to_numeric(
                datos_validacion["prob_value_contact"],
                errors="coerce"
            ).fillna(p["contact_fill"])
        )

        datos_validacion["monto"] = (
            pd.to_numeric(
                datos_validacion["monto"],
                errors="coerce"
            ).fillna(0).clip(lower=0)
        )

        # Validar la fórmula del TLV
        validar_tlv(
            datos_validacion,
            factores_freshness=p["freshness_prob"]
        )

        # Asignar grupos de ejecución
        out["grupo_ejec_tlv"] = grupos_mes(
            out["puntuacion_tlv"], cfg
        )

        # Guardar resultados individuales
        out[
            [idc, "prob", "puntuacion_tlv", "grupo_ejec_tlv"]
        ].to_csv(d / "scores_clientes.csv", index=False)

        # Guardar resumen por grupo
        g = out["grupo_ejec_tlv"].value_counts().reindex(
            range(1, 11), fill_value=0
        )

        pd.DataFrame({
            "grupo_ejec_tlv": g.index,
            "clientes": g.values,
            "pct": (g / g.sum()).round(4).values,
        }).to_csv(d / "resumen_grupos.csv", index=False)

    return ctx


def registrar_inferencia(cfg, ctx):
    """Run de inferencia en MLflow (idempotente). Los scores ya están en disco: lo que falle después no los invalida."""
    d = Path(ctx['dir'])
    logs = compute_logs(d)
    tags = {k: ctx[k] for k in ('mes', 'model_name', 'model_version', 'data_hash', 'n_clientes')}
    metricas = {'n_clientes': ctx['n_clientes'], 'score_medio': np.load(d / 'scores.npy').mean(), **{f'{t}_seg': v['wall_seconds'] for t, v in logs.items()}}
    rid = registrar_run(cfg, 'inference', ctx['key'], f"inferencia_{ctx['mes']}", tags, metricas, [d / 'resumen_grupos.csv', d / 'compute_logs.json'])
    ctx = {**ctx, 'inference_run_id': rid}
    (d / 'inference_context.json').write_text(json.dumps(ctx))  # punto de entrega: lo lee quien monitorea o valida el mes
    return ctx
