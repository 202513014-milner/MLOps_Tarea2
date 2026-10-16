"""Etapas del pipeline de entrenamiento: una función por tarea del DAG (D3).

Todas son **idempotentes** (D7):
- Cada corrida del DAG tiene un ``run_key`` (por defecto el run_id de Airflow, saneado).
- Las salidas de una etapa viven en ``<output_dir>/runs/<run_key>/<etapa>[/models/<m>]/``.
- Una etapa solo se salta si existe ``_SUCCESS.json`` con la **misma huella** (config +
  hash de datos + commit + huellas aguas arriba). Si no, se borra lo parcial y se recalcula.
- El run de MLflow lleva ``idempotency_key``: un reintento sustituye al run anterior.
Reintentar una tarea o relanzar el DAG con el mismo run_key nunca duplica ni mezcla resultados.

CLI:  python -m src.stages <etapa> --config config/config.yaml --run-key demo [--model xgb]
"""
import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import mlflow
import yaml

from src.monitoring import ComputeMonitor, info_hardware, tamano_modelo_mb
from src.provenance import git_info, hash_config, hash_datos, tags_trazabilidad, code_hash, docker_info, sha256_file
from src.tracking import log_compute, run_rastreado, serie_logger
from src.validation import validar_datos

ROOT = Path(__file__).resolve().parents[1]
MODELOS = ['xgb', 'lgbm', 'catb', 'logistic_regression', 'random_forest', 'ridge_classifier',
           'adaboost', 'gradient_boosting_sklearn', 'extra_trees']
ETAPAS = ['validate_data', 'prepare_data', 'hpo_model', 'select_hpo', 'benchmark',
          'get_best_model', 'backtesting', 'register_champion', 'generate_report']
VERSIONES = ['scikit-learn', 'xgboost', 'lightgbm', 'catboost', 'pandas', 'numpy', 'optuna', 'mlflow']


def versiones():
    """Versiones instaladas; en la imagen de Airflow solo hay mlflow-skinny, no mlflow."""
    out = {}
    for name in VERSIONES:
        for candidate in (name, name + '-skinny'):
            try:
                out[name] = version(candidate)
                break
            except PackageNotFoundError:
                continue
        else:
            out[name] = 'no-instalado'
    return out


# --------------------------------------------------------------------------
# Configuración y rutas
# --------------------------------------------------------------------------
def sanear(run_key):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(run_key)).strip('_') or 'manual'


def cargar_config(path='config/config.yaml', overrides=None):
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = ROOT / path
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    # Rutas inyectables por entorno: en Docker los datos y salidas son volúmenes.
    if os.environ.get('DATA_INPUT_DIR'):
        cfg['input_dir'] = os.environ['DATA_INPUT_DIR']
    if os.environ.get('DATA_OUTPUT_DIR'):
        cfg['output_dir'] = os.environ['DATA_OUTPUT_DIR']
    for key in ('input_dir', 'output_dir'):
        if not Path(cfg[key]).is_absolute():
            cfg[key] = str((ROOT / cfg[key]).resolve())
    cfg.setdefault('update_root_readme', False)
    return _fusionar(cfg, overrides or {})


def _fusionar(base, extra):
    """Mezcla profunda: los overrides del DAG (p. ej. hpo.optuna.n_trials) no pisan el resto."""
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _fusionar(base[key], value)
        else:
            base[key] = value
    return base


def run_key_por_defecto(cfg):
    """Clave de corrida determinista (D7): misma config + mismos datos + mismo commit => misma clave.

    Así, relanzar el DAG sin indicar ``run_key`` reutiliza la corrida existente (las etapas se saltan
    por huella) en lugar de crear un conjunto nuevo de runs. Si cambian datos, config o commit, la
    clave cambia y la corrida nueva no pisa a la anterior."""
    temporal = cfg['temporal']
    files = list(temporal['development_files']) + list(temporal['backtesting_files'])
    try:
        data = hash_datos(cfg['input_dir'], files, Path(cfg['output_dir']) / '.hash_cache.json')['data_hash']
    except OSError:
        data = 'datos-no-disponibles'  # validate_data fallará con un mensaje claro
    return 'auto-' + _huella(config=hash_config(cfg), data=data, git=git_info()['git_commit'], code=code_hash(), image=docker_info())[:20]


def raiz_corrida(cfg, run_key):
    return Path(cfg['output_dir']) / 'runs' / sanear(run_key)


def dir_etapa(cfg, run_key, etapa, modelo=None):
    base = raiz_corrida(cfg, run_key) / etapa
    return base / 'models' / modelo if modelo else base


def _json_seguro(obj):
    import numpy as np
    if isinstance(obj, (np.floating,)):
        return None if obj != obj else float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def escribir_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_json_seguro),
                          encoding='utf-8')


def leer_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _huella(**parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def _huella_etapa(cfg, run_key, etapa, modelo=None):
    """Huella guardada de una etapa ya completada (para encadenar la invalidación aguas abajo)."""
    marker = dir_etapa(cfg, run_key, etapa, modelo) / '_SUCCESS.json'
    return leer_json(marker)['fingerprint'] if marker.exists() else None


def hash_de_datos(cfg, run_key):
    """Hash de los CSV: lo calcula validate_data; el resto de etapas lo reutilizan."""
    fp = dir_etapa(cfg, run_key, 'validate_data') / 'data_fingerprint.json'
    if fp.exists():
        return leer_json(fp)['data_hash']
    temporal = cfg['temporal']
    files = list(temporal['development_files']) + list(temporal['backtesting_files'])
    return hash_datos(cfg['input_dir'], [f for f in files if (Path(cfg['input_dir'])/f).is_file()], Path(cfg['output_dir']) / '.hash_cache.json')['data_hash']


# --------------------------------------------------------------------------
# Corredor idempotente + MLflow + cómputo
# --------------------------------------------------------------------------
def _bind(cfg, run_key):
    """Rechaza reutilizar una etiqueta explícita para otra identidad de ejecución."""
    from filelock import FileLock
    root = raiz_corrida(cfg, run_key); root.mkdir(parents=True, exist_ok=True)
    identity = run_key_por_defecto(cfg)
    with FileLock(str(root/'.identity.lock')):
        path = root/'identity.json'
        if path.exists() and leer_json(path)['identity'] != identity:
            raise ValueError('Esta run_key pertenece a otros datos/config/código/imagen. Use otra clave o deje vacío.')
        if not path.exists():
            escribir_json(path, {'identity':identity, 'traceability':tags_trazabilidad(cfg,run_key=run_key)})
    return root


def _record_attempt(run_id):
    """Vincula una reejecución cacheada sin modificar la procedencia original."""
    from mlflow.tracking import MlflowClient
    attempt = os.environ.get('AIRFLOW_CTX_DAG_RUN_ID','local')
    MlflowClient().set_tag(run_id, 'airflow_attempt.'+_huella(attempt=attempt)[:12], attempt)


def ejecutar_etapa(cfg, run_key, etapa, cuerpo, modelo=None, aguas_arriba=(), force=False):
    """Ejecuta bajo bloqueo; reanuda el run existente y conserva resultados completos.

    force permite reparar una etapa incompleta; no sobreescribe una etapa completada.
    Para otro experimento, cambiar la configuración (p. ej. semilla) o usar otra run_key.
    """
    from filelock import FileLock
    from src.tracking import configurar
    run_key = sanear(run_key); root = _bind(cfg, run_key)
    configurar(cfg)
    destination = dir_etapa(cfg,run_key,etapa,modelo)
    lock = root/('.'+etapa+'_'+str(modelo or 'all')+'.lock')
    with FileLock(str(lock)):
        marker = destination/'_SUCCESS.json'
        if marker.exists() and (destination/'result.json').exists():
            res=leer_json(destination/'result.json')
            if all(Path(x).exists() for x in res.get('required_files', [])):
                _record_attempt(res['mlflow_run_id'])
                return {**res,'cached':True,'fingerprint':leer_json(marker)['fingerprint']}
        destination.mkdir(parents=True,exist_ok=True)
        fingerprint=_huella(identity=run_key_por_defecto(cfg), stage=etapa, model=modelo)
        interval=cfg.get('compute_logging',{}).get('interval_seconds',2.0)
        with run_rastreado(cfg,etapa,run_key,modelo=modelo,data_hash=hash_de_datos(cfg,run_key)) as run:
            monitor=ComputeMonitor(etapa,interval=interval,on_sample=serie_logger(run.info.run_id))
            try:
                with monitor:
                    result=cuerpo(destination,run) or {}
            finally:
                log_compute(monitor.summary)
                mlflow.log_dict({'scope':'process_and_children; GPU=device_global', 'summary':monitor.summary},'compute/compute_logs.json')
            result.update(compute=monitor.summary,mlflow_run_id=run.info.run_id,cached=False)
            mlflow.set_tag('compute_mode',result.get('execution_device','cpu_only'))
            result.setdefault('required_files',[])
            escribir_json(destination/'result.json',result)
            mlflow.log_artifact(str(destination/'result.json'),'stage')
            if etapa=='generate_report':
                logs=leer_json(root/'compute_logs.json')
                logs['tasks']['generate_report']=monitor.summary
                logs['report_measurement_scope']='Report assembly; final log upload/marker I/O excluded.'
                for folder in (root,destination):
                    escribir_json(folder/'compute_logs.json',logs)
                mlflow.log_artifact(str(root/'compute_logs.json'),'report')
        escribir_json(marker,{'fingerprint':fingerprint,'mlflow_run_id':run.info.run_id})
        return {**result,'fingerprint':fingerprint}


def _cargar_dataset(cfg,run_key):
    """Carga el dataset preparado; XCom solo contiene su ruta."""
    return joblib.load(dir_etapa(cfg,run_key,'prepare_data')/'dataset.joblib')


def validate_data(cfg,run_key,force=False):
    """Compuerta obligatoria: hashes y contrato de diez meses consecutivos."""
    def cuerpo(destino,run):
        files=cfg['temporal']['development_files']+cfg['temporal']['backtesting_files']
        try:
            report=validar_datos(cfg,destino/'validation_report.json')
        except Exception:
            mlflow.set_tag('validation_ok','false')
            if (destino/'validation_report.json').exists():
                mlflow.log_artifact(str(destino/'validation_report.json'),'validation')
            raise
        data=hash_datos(cfg['input_dir'],files)
        escribir_json(destino/'data_fingerprint.json',data)
        mlflow.set_tag('validation_ok','true')
        for file in ('validation_report.json','data_fingerprint.json'):
            mlflow.log_artifact(str(destino/file),'validation')
        return {'ok':True,**data,'required_files':[str(destino/'data_fingerprint.json')]}
    return ejecutar_etapa(cfg,run_key,'validate_data',cuerpo,force=force)


def prepare_data(cfg, run_key, force=False):
    from src.preprocessing import cargar_desarrollo_temporal

    def cuerpo(destino, run):
        X_train, X_val, y_train, y_val, manifest = cargar_desarrollo_temporal(cfg)
        joblib.dump((X_train, y_train, X_val, y_val, manifest), destino / 'dataset.joblib')
        escribir_json(destino / 'manifest.json', manifest)
        mlflow.log_artifact(str(destino / 'manifest.json'), artifact_path='prepare')
        mlflow.log_metrics({'rows_train': len(X_train), 'rows_validation': len(X_val),
                            'n_features': X_train.shape[1],
                            'positive_rate_train': float(y_train.mean()),
                            'positive_rate_validation': float(y_val.mean()),
                            'dataset_size_mb': tamano_modelo_mb(destino / 'dataset.joblib')})
        return {'rows_train': len(X_train), 'rows_validation': len(X_val), 'n_features': X_train.shape[1],
                'dataset_path': str(destino / 'dataset.joblib'), 'required_files':[str(destino/'dataset.joblib'),str(destino/'manifest.json')]}
    return ejecutar_etapa(cfg, run_key, 'prepare_data', cuerpo, force=force,
                          aguas_arriba=[_huella_etapa(cfg, run_key, 'validate_data')])


def _guardar_roc(model, X_val, y_val, nombre, path):
    """Curva ROC del modelo sobre la validación (meses 1–6) como PNG; devuelve la ruta."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    from src.evaluation import obtener_scores
    scores = obtener_scores(model, X_val)
    fpr, tpr, _ = roc_curve(y_val, scores)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_val, scores):.4f}')
    ax.plot([0, 1], [0, 1], 'k:', label='Azar')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title(f'ROC validación — {nombre}')
    ax.legend(loc='lower right'); fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------
# Etapa 3 — HPO de UN modelo (una tarea del DAG por modelo)
# --------------------------------------------------------------------------
def hpo_model(cfg, run_key, modelo, force=False):
    from src.hpo import definir_modelos_hpo, hpo_un_modelo
    from src.registry import guardar_metadata, guardar_modelo
    if modelo not in MODELOS:
        raise ValueError(f'Modelo desconocido: {modelo}. Opciones: {MODELOS}')

    def cuerpo(destino, run):
        X_train, y_train, X_val, y_val, manifest = _cargar_dataset(cfg, run_key)
        ratio = float((y_train == 0).sum() / (y_train == 1).sum())
        estimator = definir_modelos_hpo(cfg.get('random_seed', 42), ratio)[modelo]
        on_trial = lambda i, v: mlflow.log_metric('hpo.trial_auc_cv', v, step=i)
        model, perf, hpo, table = hpo_un_modelo(modelo, estimator, X_train, y_train, X_val, y_val, cfg, on_trial)
        pkl = guardar_modelo(model, modelo, str(destino))
        versions = versiones()
        meta_path = guardar_metadata(modelo, perf, model.named_steps['model'].get_params(), versions, str(destino))
        metadata = leer_json(meta_path)
        metadata.update(hpo=hpo, hardware='CPU', data_split=manifest)
        escribir_json(meta_path, metadata)
        csv = destino / f'{modelo}_hpo_results.csv'
        table.to_csv(csv, index=False)
        size = tamano_modelo_mb(pkl)
        roc_png = _guardar_roc(model, X_val, y_val, modelo, destino / f'{modelo}_roc.png')
        mlflow.log_artifact(pkl, artifact_path='model')       # modelo serializado
        mlflow.log_artifact(roc_png, artifact_path='hpo')     # curva ROC (validación)
        mlflow.log_params({f'best.{k}': v for k, v in hpo['best_parameters'].items()})
        mlflow.log_metrics({f'perf.{k}': float(v) for k, v in perf.items() if isinstance(v, (int, float)) and v == v})
        mlflow.log_metrics({'model_size_mb': size, 'hpo.n_trials': hpo['n_trials'],
                            'hpo.elapsed_seconds': hpo['elapsed_seconds']})
        mlflow.log_artifact(meta_path, artifact_path='hpo')
        mlflow.log_artifact(str(csv), artifact_path='hpo')
        return {'model': modelo, 'performance': perf, 'hpo': hpo, 'model_path': pkl,
                'metadata_path': meta_path, 'model_size_mb': size, 'required_files':[pkl,meta_path], 'execution_device':'cpu_only'}
    return ejecutar_etapa(cfg, run_key, 'hpo', cuerpo, modelo=modelo, force=force,
                          aguas_arriba=[_huella_etapa(cfg, run_key, 'prepare_data')])


# --------------------------------------------------------------------------
# Etapa 4 — consolidación HPO (formato results_summary.json compatible con paralelismo)
# --------------------------------------------------------------------------
def select_hpo(cfg, run_key, force=False):
    from src.training import GetBestModel

    def cuerpo(destino, run):
        manifest = leer_json(dir_etapa(cfg, run_key, 'prepare_data') / 'manifest.json')
        models = {}
        for m in MODELOS:
            res_path = dir_etapa(cfg, run_key, 'hpo', m) / 'result.json'
            if not res_path.exists():
                raise FileNotFoundError(f'Falta el HPO de {m}: {res_path}')
            res = leer_json(res_path)
            models[m] = {'model': None, 'performance': res['performance'], 'hpo': res['hpo']}
        champion, _, perf = GetBestModel(models, cfg)
        summary = {'campeon': champion, 'config': cfg, 'data_split': manifest, 'stage': 'HPO (DAG)',
                   'models': {k: {'performance': v['performance'], 'hpo': v['hpo']} for k, v in models.items()}}
        escribir_json(destino / 'results_summary.json', summary)
        mlflow.log_artifact(str(destino / 'results_summary.json'), artifact_path='hpo')
        mlflow.set_tag('hpo_champion', str(champion))
        primary = (cfg.get('business_metric') or {}).get('primary', 'capture_rate_top_k')
        mlflow.log_metrics({f'{k}.{primary}': float(v['performance'][primary]) for k, v in models.items()
                            if v['performance'].get(primary) is not None})
        return {'campeon_hpo': champion, 'summary_path': str(destino / 'results_summary.json')}
    ups = [_huella_etapa(cfg, run_key, 'hpo', m) for m in MODELOS]
    return ejecutar_etapa(cfg, run_key, 'select_hpo', cuerpo, force=force, aguas_arriba=ups)


# --------------------------------------------------------------------------
# Etapa 5 — benchmark secuencial vs paralelo (9 modelos, mismos HP)
# --------------------------------------------------------------------------
def _numeric(metrics):
    """Solo números finitos para MLflow; valores indefinidos permanecen null en JSON."""
    import math
    return {k:float(v) for k,v in metrics.items() if isinstance(v,(int,float)) and math.isfinite(v)}


def _log_fit(cfg,run_key,name,kind,item,X_val,y_val):
    """Run completo por artefacto final (también cada variante del benchmark)."""
    with run_rastreado(cfg,'training_'+kind,run_key,modelo=name,data_hash=hash_de_datos(cfg,run_key)) as run:
        meta=leer_json(item['metadata_path'])
        mlflow.log_params({str(k):str(v)[:500] for k,v in meta['hyperparameters'].items()})
        mlflow.log_metrics(_numeric(meta['performance']))
        log_compute(item['compute'])
        mlflow.set_tags({'execution_device':item['device'],'compute_mode':'gpu' if item['device']=='GPU' else 'cpu_only',
                         'gpu_metrics_scope':'device_global', 'estimator_threads':'1'})
        mlflow.log_metric('model_size_mb',tamano_modelo_mb(item['model_path']))
        with open(item['model_path'],'rb') as f:
            model=pickle.load(f)
        roc=_guardar_roc(model,X_val,y_val,name,Path(item['model_path']).with_suffix('.roc.png'))
        for file in (item['model_path'],item['metadata_path'],roc):
            mlflow.log_artifact(str(file),'model')
        mlflow.log_dict({'summary':item['compute'],'device':item['device'],'estimator_threads':1},'compute/compute_logs.json')
        return run.info.run_id


def benchmark(cfg,run_key,force=False):
    """Entrenamiento configurable; benchmark opcional y runs por modelo/modo."""
    from src.parallel import run_parallel
    def cuerpo(destino,run):
        X,y,Xv,yv,manifest=_cargar_dataset(cfg,run_key)
        local={**cfg,'output_dir':str(destino)}
        pointer=destino/'training_pointer.json'
        # Si falló solo el tracking, reusar los artefactos exactos ya entrenados.
        if pointer.exists():
            out=leer_json(pointer)['directory'];summary=leer_json(Path(out)/'results_summary.json')
        else:
            out,summary=run_parallel(X,y,Xv,yv,local,manifest,dir_etapa(cfg,run_key,'select_hpo'))
            escribir_json(pointer,{'directory':out})
        for name,variants in summary['executions'].items():
            for kind,item in variants.items():
                item['mlflow_run_id']=_log_fit(cfg,run_key,name,kind,item,Xv,yv)
        escribir_json(Path(out)/'results_summary.json',summary)
        mlflow.log_artifact(str(Path(out)/'results_summary.json'),'training')
        return {'parallel_dir':out,'campeon_paralelo':summary['campeon'],
                'speedup':summary['speedup'],'T_sequential_seconds':summary['T_sequential_seconds'],
                'T_parallel_seconds':summary['T_parallel_seconds'],
                'execution_device':'mixed_cpu_gpu' if any(d['device']=='GPU' for d in summary['device_plan'].values()) else 'cpu_only',
                'required_files':[str(Path(out)/'results_summary.json')]+[i['model_path'] for v in summary['executions'].values() for i in v.values()]}
    return ejecutar_etapa(cfg,run_key,'benchmark',cuerpo,force=force)


def get_best_model_stage(cfg,run_key,force=False):
    """Selecciona exclusivamente en holdout p1–p6; no utiliza los meses futuros."""
    from src.training import GetBestModel
    def cuerpo(destino,run):
        bench=leer_json(dir_etapa(cfg,run_key,'benchmark')/'result.json')
        folder=Path(bench['parallel_dir']);summary=leer_json(folder/'results_summary.json')
        models={n:{'model':None,'performance':v['performance']} for n,v in summary['models'].items()}
        name,_,perf=GetBestModel(models,cfg)
        primary=cfg['business_metric']['primary']
        champ={'champion':name,'performance':perf,'primary_metric':primary,
               'primary_value':None if perf is None else perf.get(primary), 'parallel_dir':str(folder),
               'selected_mode':summary['selected_mode'],'consistent_with_benchmark':name==summary['campeon'],
               'criteria':{'business_metric':cfg['business_metric'],'min_auc_test':cfg['min_auc_test'],
                           'max_decay':cfg['max_decay_percent'],'decay':cfg['decay'],
                           'none_policy':'sin campeón; evaluar y reportar candidatos; no promover'}}
        if name:
            item=summary['executions'][name][summary['selected_mode']]
            champ.update(model_path=item['model_path'],metadata_path=item['metadata_path'],model_run_id=item['mlflow_run_id'])
        escribir_json(destino/'champion.json',champ)
        mlflow.log_artifact(str(destino/'champion.json'),'selection')
        mlflow.set_tag('champion',name or 'none')
        return champ
    return ejecutar_etapa(cfg,run_key,'get_best_model',cuerpo,force=force)


def backtesting(cfg,run_key,force=False):
    """Backtesting de HPO y cada entrenamiento final; actualiza sus propios runs."""
    from src.backtesting_all import evaluar_artefactos
    from mlflow.tracking import MlflowClient
    def cuerpo(destino,run):
        champ=leer_json(dir_etapa(cfg,run_key,'get_best_model')/'champion.json')
        summary=leer_json(Path(champ['parallel_dir'])/'results_summary.json')
        manifest=leer_json(dir_etapa(cfg,run_key,'prepare_data')/'manifest.json')
        entries={}
        for name in MODELOS:
            h=leer_json(dir_etapa(cfg,run_key,'hpo',name)/'result.json')
            entries['hpo_'+name]={**h,'model':name,'kind':'hpo'}
            for kind,item in summary['executions'][name].items():
                entries[kind+'_'+name]={**item,'model':name,'kind':kind}
        reports=evaluar_artefactos(cfg,entries,manifest,destino)
        client=MlflowClient()
        for key,item in entries.items():
            r=reports[key];rid=item['mlflow_run_id']
            for w in r['waves']:
                for metric in ('auc','decay_percent','capture_rate_top_k','lift_top_k','precision_top_k'):
                    if w.get(metric) is not None:
                        client.log_metric(rid,f'backtest.T{w["wave"]}.{metric}',float(w[metric]))
            for file in (destino/key).iterdir():
                client.log_artifact(rid,str(file),'backtesting')
            meta=leer_json(item['metadata_path']);meta['backtesting']=r
            escribir_json(item['metadata_path'],meta)
            client.log_artifact(rid,item['metadata_path'],'model' if item['kind']!='hpo' else 'hpo')
        mlflow.log_artifact(str(destino/'backtesting_results.json'),'backtesting')
        return {'backtest_dir':str(destino),'models':reports,'valid_waves':min(r['valid_waves'] for r in reports.values()),
                'status':'complete' if all(r['valid_waves']==4 for r in reports.values()) else 'partial'}
    return ejecutar_etapa(cfg,run_key,'backtesting',cuerpo,force=force)


def register_champion(cfg,run_key,force=False):
    """Registro serializado, versión idempotente y comparación en el mismo conjunto evaluado."""
    from mlflow.tracking import MlflowClient
    from mlflow.exceptions import MlflowException
    from filelock import FileLock
    import mlflow.sklearn
    def cuerpo(destino,run):
        champ=leer_json(dir_etapa(cfg,run_key,'get_best_model')/'champion.json')
        result={'champion':champ['champion'],'promoted':False,'version':None,'alias':cfg['registry']['alias']}
        if not champ['champion']:
            result['reason']='Ningún candidato cumple filtros; no se registra campeón.'
        else:
            reg=cfg['registry'];name=reg['model_name'];client=MlflowClient()
            primary=champ['primary_metric'];value=champ['primary_value']
            version_key=_huella(run=run_key,model=sha256_file(champ['model_path']))
            eval_key=_huella(data=hash_de_datos(cfg,run_key),seed=cfg['random_seed'],test=cfg['test_size'],sample=cfg['sample_fraction'],business=cfg['business_metric'])
            with FileLock(str(Path(cfg['output_dir'])/'.registry.lock')):
                try:
                    client.get_registered_model(name)
                except MlflowException as exc:
                    if exc.error_code!='RESOURCE_DOES_NOT_EXIST':raise
                    client.create_registered_model(name)
                existing=[v for v in client.search_model_versions(f"name='{name}'") if v.tags.get('version_key')==version_key]
                if existing:
                    version=str(existing[0].version)
                else:
                    model_dir=destino/'mlflow_model'
                    if not model_dir.exists():
                        with open(champ['model_path'],'rb') as f:model=pickle.load(f)
                        mlflow.sklearn.save_model(model,str(model_dir),serialization_format='cloudpickle')
                    client.log_artifacts(champ['model_run_id'],str(model_dir),'registered_model')
                    uri=client.get_run(champ['model_run_id']).info.artifact_uri+'/registered_model'
                    v=client.create_model_version(name,source=uri,run_id=champ['model_run_id'],tags={
                        'version_key':version_key,'evaluation_key':eval_key,primary:str(value),'run_key':run_key})
                    version=str(v.version)
                # Consultar alias declarados evita confundir errores HTTP con alias ausente.
                registered=client.get_registered_model(name)
                current_version=registered.aliases.get(reg['alias'])
                current=client.get_model_version(name,current_version) if current_version else None
                promoted=False;reason=''
                if current is None:
                    promoted=True;reason='Primer campeón'
                elif str(current.version)==version:
                    promoted=True;reason='Versión ya promovida; no modificar su evaluación'
                elif current.tags.get('evaluation_key')!=eval_key:
                    reason='Evaluaciones no comparables: requiere reevaluar campeón vigente en el mismo holdout'
                else:
                    old=float(current.tags[primary]);margin=float(reg.get('min_margin',0))
                    promoted=value>=old+margin
                    reason=f'{primary}: candidato={value}; vigente={old}; margen={margin}'
                alias=reg['alias'] if promoted else reg.get('challenger_alias','challenger')
                client.set_registered_model_alias(name,alias,version)
                client.set_model_version_tag(name,version,'promotion_decision',reason)
                result.update(version=version,alias=alias,promoted=promoted,reason=reason,
                              registered_model=name,model_run_id=champ['model_run_id'])
        escribir_json(destino/'promotion.json',result)
        mlflow.log_artifact(str(destino/'promotion.json'),'registry')
        return result
    return ejecutar_etapa(cfg,run_key,'register_champion',cuerpo,force=force)


def generate_report(cfg,run_key,force=False):
    """Reporte consolidado con artefactos finales coherentes y backtesting de todos los modelos."""
    def cuerpo(destino,run):
        root=raiz_corrida(cfg,run_key)
        champ=leer_json(dir_etapa(cfg,run_key,'get_best_model')/'champion.json')
        summary=leer_json(Path(champ['parallel_dir'])/'results_summary.json')
        bt=leer_json(dir_etapa(cfg,run_key,'backtesting')/'backtesting_results.json')
        promo=leer_json(dir_etapa(cfg,run_key,'register_champion')/'promotion.json')
        tasks={}
        for stage in ETAPAS:
            if stage in ('hpo_model','generate_report'):continue
            tasks[stage]=leer_json(dir_etapa(cfg,run_key,stage)/'result.json').get('compute')
        models={};logs={}
        for name in MODELOS:
            h=leer_json(dir_etapa(cfg,run_key,'hpo',name)/'result.json')
            tasks['hpo/'+name]=h['compute']
            logs[name]={'hpo':h['compute'],'training':{k:i['compute'] for k,i in summary['executions'][name].items()}}
            item=summary['executions'][name][summary['selected_mode']]
            models[name]={'performance':item['performance'],'hpo':h['hpo'], 'device':item['device'],
                          'model_size_mb':tamano_modelo_mb(item['model_path']),'model_sha256':sha256_file(item['model_path']),
                          'mlflow_run_id':item['mlflow_run_id'],'backtesting':bt[summary['selected_mode']+'_'+name]}
        comp={'run_key':run_key,'tasks':tasks,'models':logs,'hardware':info_hardware(),
              'measurement_scope':'CPU/RAM: process and children; GPU: device-global, not exclusive; CPU logical count is capacity, estimator_threads=1 is configured limit.',
              'compute_mode':'mixed_cpu_gpu' if any(m['device']=='GPU' for m in models.values()) else 'cpu_only'}
        results={'run_key':run_key,'champion':champ['champion'],'selection':champ,'models':models,'registry':promo,
                 'traceability':tags_trazabilidad(cfg,data_hash=hash_de_datos(cfg,run_key),run_key=run_key),
                 'benchmark':{k:summary[k] for k in ('T_sequential_seconds','T_parallel_seconds','speedup')}}
        fmt=lambda x:'—' if x is None else f'{x:.4f}' if isinstance(x,(int,float)) else str(x)
        lines=['# Resultados de la corrida',f'Campeón: {champ["champion"] or "ninguno"}.',
               '| Modelo | AUC train | AUC test | Decay | Captura top-k | Dispositivo | MB |',
               '|---|---:|---:|---:|---:|---|---:|']
        for n,m in models.items():
            v=m['performance'];lines.append('| '+' | '.join(map(fmt,[n,v['auc_train'],v['auc_test'],v['decay_percent'],v['capture_rate_top_k'],m['device'],m['model_size_mb']]))+' |')
        lines+=['','## Backtesting de modelos finales (sin reentrenar ni reseleccionar)',
                '| Modelo | Ola | Mes | AUC | Captura | Lift | Alerta |','|---|---|---|---:|---:|---:|---|']
        for n,m in models.items():
            for w in m['backtesting']['waves']:
                lines.append('| '+' | '.join(map(fmt,[n,w['wave'],w['period'],w['auc'],w['capture_rate_top_k'],w['lift_top_k'],str(w['drift_alert'])]))+' |')
        lines+=['','## Recursos por entrenamiento final','| Modelo | Modo | CPU media % | CPU max % | RAM media MB | RAM max MB | Fit s | Hilos configurados |', '|---|---|---:|---:|---:|---:|---:|---:|']
        for n,variants in summary['executions'].items():
            for mode,item in variants.items():
                c=item['compute'];lines.append('| '+' | '.join(map(fmt,[n,mode,c['cpu_percent_avg'],c['cpu_percent_max'],c['ram_rss_mb_avg'],c['ram_rss_mb_max'],item['performance']['training_time_segs'],1]))+' |')
        lines+=['',comp['measurement_scope'], '', 'La selección usa validación p1–p6; p7–p10 se reportan sin cambiar al campeón.']
        for folder in (root,destino):
            escribir_json(folder/'results_summary.json',results);escribir_json(folder/'compute_logs.json',comp)
            (folder/'report.md').write_text('\n'.join(lines),encoding='utf-8')
        for f in ('results_summary.json','compute_logs.json','report.md'):
            mlflow.log_artifact(str(destino/f),'report')
        return {'report_dir':str(root),'champion':champ['champion'],'required_files':[str(root/f) for f in ('results_summary.json','compute_logs.json','report.md')]}
    return ejecutar_etapa(cfg,run_key,'generate_report',cuerpo,force=force)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def ejecutar(etapa, cfg, run_key, modelo=None, force=False):
    if etapa == 'validate_data':
        return validate_data(cfg, run_key, force)
    if etapa == 'prepare_data':
        return prepare_data(cfg, run_key, force)
    if etapa == 'hpo_model':
        return hpo_model(cfg, run_key, modelo, force)
    if etapa == 'select_hpo':
        return select_hpo(cfg, run_key, force)
    if etapa == 'benchmark':
        return benchmark(cfg, run_key, force)
    if etapa == 'get_best_model':
        return get_best_model_stage(cfg, run_key, force)
    if etapa == 'backtesting':
        return backtesting(cfg, run_key, force)
    if etapa == 'register_champion':
        return register_champion(cfg, run_key, force)
    if etapa == 'generate_report':
        return generate_report(cfg, run_key, force)
    raise ValueError(f'Etapa desconocida: {etapa}')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Ejecuta una etapa idempotente del pipeline.')
    parser.add_argument('etapa', choices=ETAPAS + ['all'])
    parser.add_argument('--config', default='config/config.yaml')
    parser.add_argument('--run-key', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)
    cfg = cargar_config(args.config)
    args.run_key = args.run_key or run_key_por_defecto(cfg)
    if args.etapa == 'all':  # ejecución local sin Airflow (útil para depurar y para los tests)
        for etapa in ETAPAS:
            if etapa == 'hpo_model':
                for m in MODELOS:
                    ejecutar(etapa, cfg, args.run_key, m, args.force)
            else:
                ejecutar(etapa, cfg, args.run_key, None, args.force)
        return
    print(json.dumps(ejecutar(args.etapa, cfg, args.run_key, args.model, args.force),
                     indent=2, ensure_ascii=False, default=_json_seguro))


if __name__ == '__main__':
    sys.exit(main())
