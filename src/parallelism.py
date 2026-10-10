"""Benchmark secuencial/paralelo de nueve modelos con HP procedentes de HPO."""
import json
import time
import os
import subprocess
from pathlib import Path
from contextlib import nullcontext
from tempfile import TemporaryDirectory
from filelock import FileLock
from importlib.metadata import version
import numpy as np
from joblib import Parallel, delayed, parallel_config
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from threadpoolctl import threadpool_limits
from src.hpo import definir_modelos_hpo, crear_pipeline, obtener_scores
from src.registry import crear_directorio_corrida, guardar_modelo, guardar_metadata, leer_json_compatible


def resolver_hpo(config, manifest):
    """Resolver hpo: operación reutilizable del pipeline P2."""
    explicit = config['parallel'].get('hpo_run_dir')
    candidates = [Path(explicit) / 'results_summary.json'] if explicit else sorted(
        Path(config['output_dir']).glob('*/results_summary.json'),
        key=lambda p: p.stat().st_mtime, reverse=True)
    expected = set(definir_modelos_hpo(42, 1))
    for path in candidates:
        if not path.exists():
            continue
        data = leer_json_compatible(path)
        models = data.get('models', {})
        if set(models) != expected or not all('hpo' in m for m in models.values()):
            continue
        old = data.get('config', {})
        keys = ['random_seed', 'sample_fraction', 'test_size', 'target_col',
                'cols_excluir', 'cols_categoricas_ohe', 'cols_categoricas_label']
        temporal_keys = ['development_files', 'backtesting_files', 'period_col']
        temporal_mismatch = any(old.get('temporal', {}).get(k) != config['temporal'].get(k) for k in temporal_keys)
        if any(old.get(k) != config.get(k) for k in keys) or temporal_mismatch:
            if explicit:
                raise ValueError('La corrida HPO usa otra configuración de datos; elegir una compatible.')
            continue
        if data.get('data_split', {}).get('feature_columns') != manifest['feature_columns']:
            raise ValueError('Las columnas actuales no coinciden con la corrida HPO.')
        return path.parent, {k: v['hpo'] for k, v in models.items()}
    raise FileNotFoundError('Se requiere una corrida HPO real de los nueve modelos. Configure parallel.hpo_run_dir con su carpeta data/output/FECHA; no se inventan HP ganadores.')


def detectar_gpu():
    """Detectar gpu: operación reutilizable del pipeline P2."""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return {'available': True, 'name': result.stdout.strip().splitlines()[0]}
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        from catboost.utils import get_gpu_device_count
        if get_gpu_device_count() > 0:
            return {'available': True, 'name': 'CUDA detectada por CatBoost'}
    except Exception:
        pass
    return {'available': False, 'name': None}


def verificar_dispositivo(name, estimator, requested):
    """Verificar dispositivo: operación reutilizable del pipeline P2."""
    if name == 'xgb' and requested == 'GPU':
        device = json.loads(estimator.get_booster().save_config())['learner']['generic_param']['device']
        if not device.startswith('cuda'):
            raise RuntimeError('XGBoost volvió a CPU: no se contabiliza como GPU.')


def planificar_dispositivos(estimators, gpu):
    """Planificar dispositivos: operación reutilizable del pipeline P2."""
    plan = {name: {'device': 'CPU', 'parameters': {}, 'reason': 'Implementación CPU'} for name in estimators}
    if not gpu['available']:
        for item in plan.values(): item['reason'] = 'No hay GPU disponible; ejecución CPU'
        return plan
    # Pruebas reales cortas: detectar hardware no basta para verificar el backend.
    X = np.random.default_rng(42).normal(size=(200, 4))
    y = (X[:, 0] > 0).astype(int)
    choices = {'xgb': [{'device': 'cuda:0', 'tree_method': 'hist'}],
               'catb': [{'task_type': 'GPU', 'devices': '0'}],
               'lgbm': [{'device_type': 'cuda', 'gpu_device_id': 0}, {'device_type': 'gpu', 'gpu_device_id': 0}]}
    for name, options in choices.items():
        errors = []
        for parameters in options:
            probe = clone(estimators[name]).set_params(**parameters)
            probe.set_params(**({'iterations': 2} if name == 'catb' else {'n_estimators': 2}))
            try:
                probe.fit(X, y)
                verificar_dispositivo(name, probe, 'GPU')
                plan[name] = {'device': 'GPU', 'parameters': parameters, 'reason': 'Backend GPU verificado mediante fit'}
                break
            except Exception as exc:
                errors.append(str(exc)[:300])
        else:
            plan[name]['reason'] = 'Backend GPU no utilizable: ' + ' | '.join(errors)
    if not any(item['device'] == 'GPU' for item in plan.values()):
        raise RuntimeError('Hay GPU detectada, pero ningún backend pudo usarla. Revisar CUDA/instalación; esta corrida no cumpliría la consigna.')
    return plan


def entrenar_proceso(name, pipeline, X, y, device, gpu_lock):
    # Un ajuste a la vez en GPU; CPU y GPU sí trabajan simultáneamente.
    """Entrenar proceso: operación reutilizable del pipeline P2."""
    start = time.perf_counter()
    with (FileLock(gpu_lock) if device == 'GPU' else nullcontext()):
        acquired = time.perf_counter()
        with threadpool_limits(limits=1):
            pipeline.fit(X, y)
        verificar_dispositivo(name, pipeline.named_steps['model'], device)
        end = time.perf_counter()
    timing = {'fit_seconds': end - acquired, 'gpu_wait_seconds': acquired - start if device == 'GPU' else 0.0,
              'process_seconds': end - start, 'pid': os.getpid(), 'device': device}
    print(f'[{name}] {device} | fit: {timing["fit_seconds"]:.2f}s | '
          f'espera GPU: {timing["gpu_wait_seconds"]:.2f}s | proceso: {timing["process_seconds"]:.2f}s', flush=True)
    return name, pipeline, timing


def metricas(model, X, y, X_val, y_val, seconds):
    """Metricas: operación reutilizable del pipeline P2."""
    auc_train = float(roc_auc_score(y, obtener_scores(model, X)))
    scores = obtener_scores(model, X_val)
    auc = float(roc_auc_score(y_val, scores))
    fpr, tpr, _ = roc_curve(y_val, scores)
    return {'auc_train': auc_train, 'auc_test': auc, 'auc': auc, 'gini': 2 * auc - 1,
            'ks': float(np.max(tpr - fpr)), 'f1': float(f1_score(y_val, model.predict(X_val), zero_division=0)),
            'decay_percent': 100 * (auc_train - auc) / auc_train,
            'training_time_segs': seconds, 'ronda': 1}


def run_parallel(X, y, X_val, y_val, config, manifest, source_dir=None):
    """Run parallel: operación reutilizable del pipeline P2."""
    from src.evaluation import GetBestModel
    if source_dir:
        config = {**config, 'parallel': {**config['parallel'], 'hpo_run_dir': str(source_dir)}}
    source, hpo = resolver_hpo(config, manifest)
    workers = config.get('n_jobs', config['parallel'].get('workers', 2))
    if workers == -1:
        workers = max(1, __import__('os').cpu_count() or 1)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError('n_jobs debe ser -1 o un entero >=1.')
    estimators = definir_modelos_hpo(config['random_seed'], float((y == 0).sum() / (y == 1).sum()))
    for name, model in estimators.items():
        model.set_params(**hpo[name]['best_parameters'])
    gpu = detectar_gpu()
    if not config.get('use_gpu', True):
        gpu = {**gpu, 'available': False, 'disabled_by_config': True}
    print(f'[Hardware] GPU disponible: {gpu["available"]}; {gpu["name"]}', flush=True)
    plan = planificar_dispositivos(estimators, gpu)
    pipelines = {name: crear_pipeline(X, model.set_params(**plan[name]['parameters']), config)
                 for name, model in estimators.items()}
    # Intercalar tareas GPU/CPU evita ocupar todos los workers esperando GPU.
    gpu_names = [n for n in pipelines if plan[n]['device'] == 'GPU']
    cpu_names = [n for n in pipelines if plan[n]['device'] == 'CPU']
    order = []
    while gpu_names or cpu_names:
        if gpu_names: order.append(gpu_names.pop(0))
        if cpu_names: order.append(cpu_names.pop(0))
    pipelines = {n: pipelines[n] for n in order}
    for name in pipelines:
        print(f'[Hardware] {name}: {plan[name]["device"]} — {plan[name]["reason"]}', flush=True)
    versions = {name: version(name) for name in ['scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'pandas', 'numpy', 'joblib']}
    # Cada modo usa exactamente los mismos datos, HP y dispositivos. Preflight y evaluación excluidos.
    results, times = {}, {}
    with TemporaryDirectory(prefix='automl_gpu_lock_') as lock_dir:
        lock = str(Path(lock_dir) / 'gpu0.lock')
        for mode in (['sequential', 'parallel'] if config.get('parallel_enabled', True) else ['sequential']):
            print(f'\n[BENCHMARK] {mode}; workers={1 if mode == "sequential" else workers}', flush=True)
            start = time.perf_counter()
            if mode == 'sequential':
                trained = [entrenar_proceso(n, clone(p), X, y, plan[n]['device'], lock) for n, p in pipelines.items()]
            else:
                with parallel_config(backend='loky', inner_max_num_threads=1):
                    trained = Parallel(n_jobs=workers, pre_dispatch=workers, batch_size=1)(
                        delayed(entrenar_proceso)(n, clone(p), X, y, plan[n]['device'], lock)
                        for n, p in pipelines.items())
            times[mode] = time.perf_counter() - start
            results[mode] = {n: {'model': p, 'timing': timing} for n, p, timing in trained}
    parallel_ran = 'parallel' in results
    selected_mode = 'parallel' if parallel_ran else 'sequential'
    speedup = times['sequential'] / times['parallel'] if parallel_ran else None
    if not parallel_ran:
        results['parallel'] = results['sequential']
        times['parallel'] = times['sequential']
    parallelism = {'backend': 'joblib' if parallel_ran else 'sequential',
        'n_workers': workers if parallel_ran else 1, 't_sequential_s': times['sequential'],
        't_parallel_s': times['parallel'] if parallel_ran else None, 'speed_up': speedup,
        'parallel_executed': parallel_ran, 'cpu_cores': __import__('os').cpu_count()} 
    run_dir = crear_directorio_corrida(str(Path(config['output_dir']) / 'parallel'))
    models, comparisons = {}, {}
    for name in pipelines:
        info = results['parallel'][name]
        perf = metricas(info['model'], X, y, X_val, y_val, info['timing']['fit_seconds'])
        comparisons[name] = {mode: results[mode][name]['timing'] for mode in results}
        models[name] = {'model': info['model'], 'performance': perf, 'library_versions': versions}
        guardar_modelo(info['model'], name, run_dir)
        path = guardar_metadata(name, perf, info['model'].named_steps['model'].get_params(), versions, run_dir)
        metadata = leer_json_compatible(Path(path))
        metadata.update(hpo=hpo[name], hpo_source=str(source), hardware=plan[name], execution_device=plan[name]['device'].lower(),
                        parallel=comparisons[name], parallelism=parallelism, data_split=manifest)
        Path(path).write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    champion, _, _ = GetBestModel(models, config)
    summary = {'stage': selected_mode, 'parallelism': parallelism, 'campeon': champion, 'workers': workers,
               'hpo_source': str(source), 'hardware': gpu, 'device_plan': plan,
               'T_sequential_seconds': times['sequential'], 'T_parallel_seconds': times['parallel'],
               'speedup': speedup, 'per_model': comparisons, 'data_split': manifest,
               'models': {n: {'performance': v['performance'], 'hpo': hpo[n]} for n, v in models.items()}}
    Path(run_dir, 'results_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    if speedup is None:
        Path(run_dir, 'README.md').write_text('Entrenamiento secuencial; benchmark paralelo no ejecutado. Speed-up no definido.', encoding='utf-8')
        return run_dir, summary
    print('\nModelo | dispositivo | fit secuencial (s) | fit paralelo (s) | espera GPU paralelo (s)')
    rows = []
    for name, data in comparisons.items():
        a, b = data['sequential'], data['parallel']
        row = f'{name} | {b["device"]} | {a["fit_seconds"]:.3f} | {b["fit_seconds"]:.3f} | {b["gpu_wait_seconds"]:.3f}'
        rows.append('| ' + row + ' |'); print(row)
    report = '\n'.join(['# Benchmark de entrenamiento',
        f'Fuente HPO: `{source}`. Workers: {workers}.',
        '| Modelo | Dispositivo | Fit secuencial s | Fit paralelo s | Espera GPU paralelo s |',
        '|---|---|---:|---:|---:|', *rows,
        '\n| T secuencial s | T paralelo s | Speed-up |', '|---:|---:|---:|',
        f'| {times["sequential"]:.3f} | {times["parallel"]:.3f} | {speedup:.4f} |',
        '\nIncluye ajuste del preprocesamiento, entrenamiento y overhead de despacho/transferencia de resultados. '
        'Excluye HPO, carga de CSV, preflight GPU, evaluación y escritura de artefactos. '
        'Una repetición, secuencial primero; no es comparación CPU contra GPU. '
        'Los tiempos por fit excluyen espera GPU. El tiempo global no es suma de tiempos paralelos. '
        'Un speed-up menor que 1 indica que el overhead/contención domina. '
        'GPU se serializa mediante lock para evitar agotar VRAM; puede solaparse con CPU. '
        'La CPU mantiene un hilo por estimador. CatBoost GPU puede variar numéricamente entre ejecuciones.'])
    Path(run_dir, 'README.md').write_text(report, encoding='utf-8')
    # README raíz incorpora el resultado real de la corrida ejecutada por el usuario.
    root_readme = Path(__file__).resolve().parents[1] / 'README.md'
    base = root_readme.read_text(encoding='utf-8').split('<!-- BENCHMARK_REAL -->')[0]
    root_readme.write_text(base + '\n<!-- BENCHMARK_REAL -->\n' + report, encoding='utf-8')
    print(f'T secuencial={times["sequential"]:.2f}s; T paralelo={times["parallel"]:.2f}s; speed-up={speedup:.3f}x')
    return run_dir, summary
