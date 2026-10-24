"""Evaluación prospectiva del campeón fijo en cuatro archivos mensuales reservados."""
import json
import pickle
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.preprocessing import muestreo_estratificado
from src.evaluation import obtener_scores
from src.registry import crear_directorio_corrida, leer_json_compatible
from src.ops_metrics import business_metrics, calcular_decay, decay_kind, top_fraction


def periodo_mensual(value):
    text = str(value).strip().removesuffix('.0')
    if len(text) == 6 and text.isdigit(): text = text[:4] + '-' + text[4:]
    return pd.Period(text, freq='M')


def resolver_campeon(config, source_dir=None):
    explicit = source_dir or config['backtesting'].get('source_run_dir')
    if explicit:
        paths = [Path(explicit) / 'results_summary.json']
    else:
        # Preferir campeón de paralelismo; si no existe, aceptar HPO como fuente.
        parallel = sorted(Path(config['output_dir']).glob('parallel/*/results_summary.json'), key=lambda p:p.stat().st_mtime, reverse=True)
        hpo = sorted(Path(config['output_dir']).glob('*/results_summary.json'), key=lambda p:p.stat().st_mtime, reverse=True)
        paths = parallel + hpo
    for path in paths:
        if not path.is_file(): continue
        summary = leer_json_compatible(path)
        name = summary.get('campeon')
        if not name:
            if explicit: raise ValueError('La corrida no seleccionó un campeón válido.')
            continue
        model_path = path.parent / f'{name}_model.pkl'
        metadata_path = path.parent / f'{name}_metadata.json'
        if not model_path.is_file() or not metadata_path.is_file():
            if explicit: raise FileNotFoundError('Falta el .pkl o metadata del campeón en la carpeta indicada.')
            continue
        metadata = leer_json_compatible(metadata_path)
        manifest = metadata.get('data_split') or summary.get('data_split')
        if not manifest or not manifest.get('feature_columns'):
            raise ValueError('La corrida debe incluir el manifiesto temporal P2 y las columnas del pipeline.')
        return path.parent, name, model_path, metadata, manifest
    raise FileNotFoundError('Indicar backtesting.source_run_dir: carpeta de una corrida exitosa de paralelismo o HPO con campeón y artefactos.')


def graficar_auc(results, output_path=None):
    import matplotlib.pyplot as plt
    rows = results['waves']
    x = [row['wave'] for row in rows]
    auc = [row['auc'] if row['auc'] is not None else np.nan for row in rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(x, auc, marker='o', linewidth=2, label='AUC mensual del campeón')
    ax.axhline(results['reference_auc'], color='tab:orange', linestyle='--', label='AUC de validación')
    ax.axhline(.5, color='gray', linestyle=':', label='Referencia AUC = 0.5')
    ax.set_xticks(x, [f'T+{r["wave"]}\n{r["period"] or r["file"]}' for r in rows])
    ax.set_ylim(0, 1.03); ax.set_xlabel('Ola / mes futuro'); ax.set_ylabel('AUC ROC')
    ax.set_title(f'Backtesting temporal — {results["champion"]}')
    for r in rows:
        if r['auc'] is not None: ax.annotate(f'{r["auc"]:.4f}', (r['wave'], r['auc']), xytext=(0, 8), textcoords='offset points', ha='center')
    ax.grid(alpha=.25); ax.legend(loc='best'); fig.tight_layout()
    if output_path: fig.savefig(output_path, dpi=160, bbox_inches='tight')
    return fig


def run_backtesting_champion_legacy(config, source_dir=None):
    source, name, model_path, metadata, manifest = resolver_campeon(config, source_dir)
    opts = config['backtesting']; temporal = config['temporal']
    development = [r['file'] for r in manifest['development']]
    future = list(temporal['backtesting_files'])
    if len(future) < 4 or len(set(future)) != len(future) or set(development) & set(future):
        raise ValueError('Se requieren >=4 archivos futuros únicos y separados del entrenamiento.')
    if development != temporal['development_files'] or future != manifest['reserved_backtesting_files']:
        raise ValueError('Los archivos temporales no coinciden con los de la corrida del campeón.')
    cutoff_file = opts.get('cutoff_file')
    if cutoff_file != development[-1]:
        raise ValueError('cutoff_file debe coincidir con el último CSV de desarrollo del campeón. Cambiar el corte requiere una nueva corrida de entrenamiento.')
    features = manifest['feature_columns']; target = config['target_col']
    reference = float(metadata['performance']['auc_test'])
    if not np.isfinite(reference) or not 0 < reference <= 1:
        raise ValueError('AUC de validación de referencia inválido.')
    fraction = opts.get('sample_fraction', config.get('sample_fraction', .1))
    if not 0 < fraction <= 1: raise ValueError('Fracción de backtesting debe estar en (0,1].')
    # No fit ni transformadores nuevos: el pickle contiene el pipeline aprendido.
    with model_path.open('rb') as f: model = pickle.load(f)
    if not hasattr(model, 'named_steps') or 'preprocessing' not in model.named_steps:
        raise ValueError('Se requiere el pipeline completo con preprocesamiento P2.')
    period_col = temporal.get('period_col')
    cutoff = periodo_mensual(opts['cutoff_period']) if opts.get('cutoff_period') else None
    if cutoff is None and manifest.get('development_periods'):
        cutoff = periodo_mensual(manifest['development_periods'][-1])
    if period_col and cutoff is None:
        raise ValueError('Configurar cutoff_period (YYYY-MM) para verificar meses futuros.')
    if cutoff and manifest.get('development_periods') and cutoff != periodo_mensual(manifest['development_periods'][-1]):
        raise ValueError('cutoff_period no coincide con el periodo de entrenamiento guardado.')
    labels = temporal.get('file_periods', {}) or {}
    if cutoff and not period_col:
        if labels.get(cutoff_file) is None or periodo_mensual(labels[cutoff_file]) != cutoff:
            raise ValueError('Sin columna mensual, cutoff_period requiere file_periods con los meses reales de desarrollo y futuros.')
    waves = []
    for index, filename in enumerate(future, 1):
        start = time.perf_counter()
        df = pd.read_csv(Path(config['input_dir']) / filename,
                         sep=config.get('csv_sep', ','), encoding=config.get('csv_encoding', 'utf-8'))
        if target not in df or df[target].isna().any() or not set(df[target].unique()).issubset({0,1}):
            raise ValueError(f'{filename}: target debe ser binario 0/1 y sin nulos.')
        missing = [col for col in features if col not in df]
        if missing: raise ValueError(f'{filename}: faltan predictores: {missing}')
        actual = None
        if period_col:
            if period_col not in df or df[period_col].isna().any() or df[period_col].nunique() != 1:
                raise ValueError(f'{filename}: se requiere exactamente un periodo mensual válido.')
            actual = periodo_mensual(df[period_col].iloc[0])
        elif filename in labels: actual = periodo_mensual(labels[filename])
        if cutoff and (actual is None or actual != cutoff + index):
            raise ValueError(f'{filename}: debe corresponder a {cutoff + index}, se encontró {actual}.')
        original_n = len(df)
        original_rate = float(df[target].mean()) if original_n else None
        # Muestreo por clase cuando hay ambas; una sola clase no permite calcular AUC.
        if df[target].nunique() == 2:
            sampled = muestreo_estratificado(df, target, fraction, config.get('random_seed',42) + 100 + index)
        else:
            sampled = df.sample(frac=fraction, random_state=config.get('random_seed',42) + 100 + index)
        auc = None; status = 'ok'; predict_seconds = 0.0; biz = {}
        if sampled[target].nunique() < 2:
            status = 'auc_undefined_single_class_or_empty'
        else:
            prediction_start = time.perf_counter()
            scores = obtener_scores(model, sampled[features])
            predict_seconds = time.perf_counter() - prediction_start
            auc = float(roc_auc_score(sampled[target], scores))
            biz = business_metrics(sampled[target], scores, top_fraction(config))
        decay = calcular_decay(reference, auc, decay_kind(config))
        waves.append({'wave': index, 'file': filename, 'period': str(actual) if actual else None,
                      'rows_original': original_n, 'rows_evaluated': len(sampled),
                      'target_rate_original': original_rate,
                      'target_rate_sample': float(sampled[target].mean()) if len(sampled) else None,
                      'auc': auc, 'decay_percent': decay, 'status': status,
                      'drift_alert': bool(decay is not None and decay > float(config.get('max_decay_percent', 10.0))),
                      'capture_rate_top_k': biz.get('capture_rate_top_k'), 'lift_top_k': biz.get('lift_top_k'),
                      'prediction_seconds': predict_seconds, 'total_seconds': time.perf_counter() - start})
        print(f'[Backtesting] T+{index} {filename} | AUC={auc} | decay={decay}% | filas={len(sampled)}', flush=True)
    valid = [r['auc'] for r in waves if r['auc'] is not None]
    results = {'champion': name, 'source_run_dir': str(source.resolve()),
               'cutoff_file': cutoff_file, 'cutoff_period': str(cutoff) if cutoff else None,
               'chronology_verified': bool(cutoff),
               'chronology_basis': 'CSV monthly column' if period_col else ('manual file_periods mapping' if cutoff else 'declared file order'),
               'chronology_note': 'Meses verificados' if cutoff else 'Orden mensual declarado por archivos; fechas reales no verificadas. Configurar period_col/cutoff_period o file_periods para verificarlas.',
               'evaluation_mode': 'fixed_origin_no_retraining', 'reference_auc': reference,
               'reference': 'AUC holdout de validación del campeón',
               'decay_formula': '100 * (reference_auc - monthly_auc) / reference_auc; negativo = mejora',
               'sample_fraction': fraction, 'waves': waves,
               'mean_auc_valid_waves': float(np.mean(valid)) if valid else None,
               'drift_alerts': sum(1 for r in waves if r['drift_alert']),
               'valid_waves': len(valid), 'status': 'complete' if len(valid)==len(waves) else 'partial_auc_undefined',
               'timestamp': datetime.now().isoformat(), 'model_refitted': False,
               'source_model_sha256': __import__('hashlib').sha256(model_path.read_bytes()).hexdigest()}
    run_dir = Path(crear_directorio_corrida(str(Path(config['output_dir']) / 'backtesting')))
    (run_dir / 'backtesting_results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    pd.DataFrame(waves).to_csv(run_dir / 'backtesting_results.csv', index=False, encoding='utf-8-sig')
    fig = graficar_auc(results, run_dir / 'auc_por_ola.png')
    import matplotlib.pyplot as plt
    plt.close(fig)
    lines = ['## Backtesting temporal', f'Campeón: {name}. AUC de referencia: {reference:.4f}.',
             '| Ola | Archivo | Mes | Filas muestra | AUC | Decay % |', '|---|---|---|---:|---:|---:|']
    for row in waves:
        auc_text = f'{row["auc"]:.4f}' if row['auc'] is not None else 'No definido'
        decay_text = f'{row["decay_percent"]:.2f}' if row['decay_percent'] is not None else 'No definido'
        lines.append(f'| T+{row["wave"]} | {row["file"]} | {row["period"] or "Sin fecha verificada"} | {row["rows_evaluated"]} | {auc_text} | {decay_text} |')
    lines += ['', results['chronology_note'], '', 'Campeón fijo, sin reentrenamiento. Decay relativo al AUC de validación; negativo indica mejora. El promedio se calcula solo sobre olas con AUC definido.', '', '![AUC por ola](auc_por_ola.png)']
    report = '\n'.join(lines)
    (run_dir / 'README.md').write_text(report, encoding='utf-8')
    if config.get('update_root_readme', True):
        root = Path(__file__).resolve().parents[1]
        readme = root / 'README.md'
        prior = readme.read_text(encoding='utf-8').split('<!-- BACKTESTING_REAL -->')[0]
        report_root = report.replace('(auc_por_ola.png)', '(' + __import__('os').path.relpath(run_dir / 'auc_por_ola.png', root).replace('\\','/') + ')')
        readme.write_text(prior + '\n<!-- BACKTESTING_REAL -->\n' + report_root, encoding='utf-8')
    print(f'[Backtesting] Resultados y gráfico: {run_dir}')
    return str(run_dir), results


def run_backtesting(config, source_dir=None):
    """API P2 compatible: evalúa todos los modelos disponibles, conserva campeón previo."""
    from src.backtesting_all import evaluar_artefactos
    source,name,_,_,manifest=resolver_campeon(config,source_dir)
    summary=leer_json_compatible(source/'results_summary.json')
    entries={n:{'model':n,'kind':'final','model_path':str(source/f'{n}_model.pkl'),
                'metadata_path':str(source/f'{n}_metadata.json')} for n in summary['models']}
    dest=Path(crear_directorio_corrida(str(Path(config['output_dir'])/'backtesting')))
    local={**config,'validation':{'date_column':'p_fecinformacion',**config.get('validation',{})}}
    reports=evaluar_artefactos(local,entries,manifest,dest)
    selected=reports[name]
    result={**selected,'champion':name,'models':reports}
    (dest/'backtesting_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return str(dest),result
