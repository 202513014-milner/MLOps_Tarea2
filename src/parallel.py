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
from src.ops_metrics import business_metrics, calcular_decay, decay_kind, top_fraction


def resolver_hpo(config, manifest):
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
    if name == 'xgb' and requested == 'GPU':
        device = json.loads(estimator.get_booster().save_config())['learner']['generic_param']['device']
        if not device.startswith('cuda'):
            raise RuntimeError('XGBoost volvió a CPU: no se contabiliza como GPU.')


def planificar_dispositivos(estimators, gpu):
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
        print('[Hardware] Ningún backend GPU utilizable: se continúa en CPU.')
    return plan


def entrenar_proceso(name, pipeline, X, y, device, gpu_lock, interval=2.0):
    """Entrena y mide el proceso y sus hijos; GPU es telemetría global del dispositivo."""
    from src.monitoring import ComputeMonitor
    # Un ajuste a la vez en GPU; CPU y GPU sí trabajan simultáneamente.
    start = time.perf_counter()
    with (FileLock(gpu_lock) if device == 'GPU' else nullcontext()):
        acquired = time.perf_counter()
        with ComputeMonitor(name, interval=interval) as monitor:
            with threadpool_limits(limits=1):
                pipeline.fit(X, y)
        verificar_dispositivo(name, pipeline.named_steps['model'], device)
        end = time.perf_counter()
    timing = {'fit_seconds': end - acquired, 'gpu_wait_seconds': acquired - start if device == 'GPU' else 0.0,
              'process_seconds': end - start, 'pid': os.getpid(), 'device': device,
              'compute': monitor.summary, 'estimator_threads': 1,
              'gpu_metrics_scope': 'device_global_not_exclusive_to_process'}
    print(f'[{name}] {device} | fit: {timing["fit_seconds"]:.2f}s | '
          f'espera GPU: {timing["gpu_wait_seconds"]:.2f}s | proceso: {timing["process_seconds"]:.2f}s', flush=True)
    return name, pipeline, timing


def metricas(model, X, y, X_val, y_val, seconds, config=None):
    config = config or {}
    auc_train = float(roc_auc_score(y, obtener_scores(model, X)))
    scores = obtener_scores(model, X_val)
    auc = float(roc_auc_score(y_val, scores))
    fpr, tpr, _ = roc_curve(y_val, scores)
    biz = business_metrics(y_val, scores, top_fraction(config))
    return {'auc_train': auc_train, 'auc_test': auc, 'auc': auc, 'gini': 2 * auc - 1,
            'ks': float(np.max(tpr - fpr)), 'f1': float(f1_score(y_val, model.predict(X_val), zero_division=0)),
            'decay_percent': calcular_decay(auc_train, auc, decay_kind(config)),
            'training_time_segs': seconds, 'ronda': 1,
            **{k: v for k, v in biz.items() if k != 'top_fraction'}}


def run_parallel(X, y, X_val, y_val, config, manifest, source_dir=None):
    from src.training import GetBestModel
    if source_dir:
        config = {**config, 'parallel': {**config['parallel'], 'hpo_run_dir': str(source_dir)}}
    source, hpo = resolver_hpo(config, manifest)
    opts = config.get('parallel', {})
    mode = opts.get('mode', 'parallel')
    if mode not in ('sequential', 'parallel', 'benchmark'):
        raise ValueError('parallel.mode: sequential | parallel | benchmark')
    requested = int(opts.get('workers', 2))
    available = getattr(os, 'process_cpu_count', os.cpu_count)() or 1
    workers = min(available, requested if requested > 0 else available)
    if requested == 0 or requested < -1:
        raise ValueError('workers debe ser -1 o >=1')
    estimators = definir_modelos_hpo(config['random_seed'], float((y == 0).sum() / (y == 1).sum()))
    for name, model in estimators.items():
        model.set_params(**hpo[name]['best_parameters'])
    gpu = detectar_gpu()
    permitted = {**gpu, 'available': gpu['available'] and config.get('use_gpu', False)}
    plan = planificar_dispositivos(estimators, permitted)
    pipelines = {n: crear_pipeline(X, m.set_params(**plan[n]['parameters']), config) for n, m in estimators.items()}
    results, times = {}, {}
    interval = config.get('compute_logging', {}).get('interval_seconds', 2.0)
    modes = ['sequential', 'parallel'] if mode == 'benchmark' else [mode]
    with TemporaryDirectory(prefix='automl_gpu_lock_') as locks:
        lock = str(Path(locks) / 'gpu.lock')
        for execution in modes:
            started = time.perf_counter()
            if execution == 'sequential':
                trained = [entrenar_proceso(n, clone(m), X, y, plan[n]['device'], lock, interval) for n,m in pipelines.items()]
            else:
                with parallel_config(backend='loky', inner_max_num_threads=1):
                    trained = Parallel(n_jobs=workers, pre_dispatch=workers)(
                        delayed(entrenar_proceso)(n, clone(m), X, y, plan[n]['device'], lock, interval)
                        for n,m in pipelines.items())
            times[execution] = time.perf_counter() - started
            results[execution] = {n: {'model': m, 'timing': t} for n,m,t in trained}
    selected = 'parallel' if 'parallel' in results else 'sequential'
    run_dir = Path(crear_directorio_corrida(str(Path(config['output_dir']) / 'parallel')))
    versions = {n: version(n) for n in ['scikit-learn','xgboost','lightgbm','catboost','numpy','pandas']}
    models, comparisons, executions = {}, {}, {}
    for name in pipelines:
        comparisons[name] = {}
        executions[name] = {}
        for execution in modes:
            info = results[execution][name]
            dest = run_dir if execution == selected else run_dir / execution
            dest.mkdir(exist_ok=True)
            perf = metricas(info['model'], X, y, X_val, y_val, info['timing']['fit_seconds'], config)
            model_path = guardar_modelo(info['model'], name, str(dest))
            meta_path = guardar_metadata(name, perf, info['model'].named_steps['model'].get_params(), versions, str(dest))
            meta = leer_json_compatible(meta_path)
            meta.update(hpo=hpo[name], data_split=manifest, hardware=plan[name], compute=info['timing']['compute'],
                        execution_mode=execution, estimator_threads=1)
            Path(meta_path).write_text(json.dumps(meta, indent=2, default=str), encoding='utf-8')
            comparisons[name][execution] = info['timing']
            executions[name][execution] = {'model_path': model_path, 'metadata_path': meta_path, 'performance': perf,
                                          'compute': info['timing']['compute'], 'device': plan[name]['device']}
            if execution == selected:
                models[name] = {'model': info['model'], 'performance': perf}
    champion, _, _ = GetBestModel(models, config)
    summary = {'stage': selected, 'selected_mode': selected, 'campeon': champion, 'workers': workers,
               'hpo_source': str(source), 'hardware': gpu, 'device_plan': plan,
               'T_sequential_seconds': times.get('sequential'), 'T_parallel_seconds': times.get('parallel'),
               'speedup': times['sequential']/times['parallel'] if len(times)==2 else None,
               'per_model': comparisons, 'executions': executions, 'data_split': manifest,
               'config': config, 'models': {n: {'performance': v['performance']} for n,v in models.items()}}
    (run_dir/'results_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    (run_dir/'README.md').write_text('Modo: '+mode+'; seleccionado: '+selected+'\n'+json.dumps(times,indent=2),encoding='utf-8')
    return str(run_dir), summary
