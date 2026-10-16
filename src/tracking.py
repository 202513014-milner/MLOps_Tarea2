"""Gestor de experimentos (MLflow) con trazabilidad completa e idempotencia (D6, D7).

Cada run registra como tags: commit de git, hash de los datos, imagen Docker, hash de la
configuración y la ejecución del DAG de Airflow (dag_id, run_id, task_id, intento).
La config efectiva se guarda como artefacto YAML y sus parámetros clave como params.

Idempotencia: cada run lleva el tag ``idempotency_key = <run_key>:<etapa>[:<modelo>]``. Si una
tarea se reintenta o se vuelve a ejecutar con el mismo ``run_key``, se reanuda el mismo run y se preservan sus intentos anteriores.
"""
import contextlib
import os
import tempfile
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from src.provenance import tags_trazabilidad

DEFAULT_EXPERIMENT = 'automl-venta-financiera'


def tracking_uri(config=None):
    uri = os.environ.get('MLFLOW_TRACKING_URI') or (config or {}).get('mlflow', {}).get('tracking_uri')
    if uri:
        return uri
    db = Path((config or {}).get('output_dir', 'data/output')).parent / 'mlflow' / 'mlflow.db'
    db.parent.mkdir(parents=True, exist_ok=True)
    return f'sqlite:///{db.resolve().as_posix()}'


def configurar(config):
    mlflow.set_tracking_uri(tracking_uri(config))
    name = os.environ.get('MLFLOW_EXPERIMENT_NAME') or config.get('mlflow', {}).get('experiment', DEFAULT_EXPERIMENT)
    exp = mlflow.set_experiment(name)
    return exp.experiment_id


def _aplanar(d, prefix=''):
    for key, value in d.items():
        name = f'{prefix}{key}'
        if isinstance(value, dict):
            yield from _aplanar(value, name + '.')
        elif isinstance(value, (list, tuple)):
            yield name, ','.join(map(str, value))[:500]
        else:
            yield name, value


PARAMS_CLAVE = ['random_seed', 'test_size', 'sample_fraction', 'max_decay_percent', 'min_auc_test',
                'decay', 'business_metric', 'hpo.strategy', 'hpo.cv_folds', 'hpo.optuna.n_trials',
                'hpo.optuna.timeout_seconds', 'parallel.workers', 'backtesting.sample_fraction']


def _params_clave(config):
    flat = dict(_aplanar(config))
    keep = {}
    for k, v in flat.items():
        if any(k == p or k.startswith(p + '.') for p in PARAMS_CLAVE):
            keep[k] = v
    return keep


@contextlib.contextmanager
def run_rastreado(config, etapa, run_key, modelo=None, data_hash=None, tags_extra=None):
    """Reanuda el mismo run por identidad; nunca elimina runs históricos.

    El llamador serializa cada etapa con FileLock. La identidad incluye la huella
    de la corrida; cambios de datos/config/código usan otra carpeta y otra clave.
    """
    exp_id = configurar(config)
    key = f'{run_key}:{etapa}' + (f':{modelo}' if modelo else '')
    client = MlflowClient()
    found = client.search_runs([exp_id], filter_string=f"tags.idempotency_key = '{key}'", max_results=2)
    if len(found) > 1:
        raise RuntimeError('Runs duplicados preexistentes: revisar idempotency_key '+key)
    tags = tags_trazabilidad(config, data_hash=data_hash, run_key=run_key,
                            extra={'etapa': etapa, 'idempotency_key': key,
                                   **({'modelo': modelo} if modelo else {}), **(tags_extra or {})})
    rid = found[0].info.run_id if found else None
    with mlflow.start_run(run_id=rid, experiment_id=exp_id,
                          run_name=f'{etapa}/{modelo or "all"} [{run_key}]', nested=mlflow.active_run() is not None) as run:
        if not rid:
            mlflow.set_tags({k: str(v) for k,v in tags.items()})
        else:
            # Conservar origen y añadir cada intento de Airflow sin pisarlo.
            dag_run = tags.get('airflow_dag_run_id', 'local')
            import hashlib
            mlflow.set_tag('airflow_attempt.'+hashlib.sha256(str(dag_run).encode()).hexdigest()[:12], str(dag_run))
        mlflow.log_params({k: str(v)[:500] for k,v in _params_clave(config).items()})
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp)/'config.yaml'
            cfg.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding='utf-8')
            mlflow.log_artifact(str(cfg), artifact_path='config')
        yield run


def log_compute(summary, prefix='compute'):
    """Resumen de ComputeMonitor como métricas MLflow (compute.cpu_percent_avg, ...)."""
    mlflow.log_metrics({f'{prefix}.{k}': float(v) for k, v in summary.items() if v is not None})


def serie_logger(run_id):
    """Callback para ComputeMonitor. Usa el cliente con run_id explícito porque el run activo
    de MLflow es local al hilo y el muestreador corre en otro hilo."""
    client = MlflowClient()

    def _log(step, sample):
        import time
        ts = int(time.time() * 1000)
        client.log_batch(run_id, metrics=[
            mlflow.entities.Metric(f'ts.{k}', float(v), ts, step) for k, v in sample.items() if k != 't'])
    return _log


def log_dict_artifact(data, filename, artifact_path=None):
    mlflow.log_dict(data, f'{artifact_path}/{filename}' if artifact_path else filename)
