"""DAG de entrenamiento AutoML (Práctica 3).

validate_data ─► prepare_data ─► hpo.<9 modelos en paralelo> ─► select_hpo ─► benchmark
                                                                               │
          generate_report ◄─ register_champion ◄─ backtesting ◄─ get_best_model ◄┘

Decisiones (detalle y justificación en docs/decisiones_diseno.md):
- schedule=None: ejecución manual (D9: el reentrenamiento lo dispara una persona o un
  disparador externo; no hay cron dentro del DAG).
- Reintentos, pool, timeout y demás parámetros del DAG se leen de ``config/config.yaml``
  (sección ``airflow``); las variables de entorno AUTOML_* permiten sobrescribirlos sin tocar código.
- D3 granularidad: una tarea por etapa y una tarea por modelo en HPO; el benchmark es una
  sola tarea porque mide secuencial vs paralelo dentro del mismo proceso.
- D2 cómputo: las 9 tareas de HPO corren en paralelo limitadas por el pool ``automl_compute``;
  el benchmark compara además secuencial vs joblib/loky con los mismos hiperparámetros.
- D7 idempotencia: cada tarea escribe en runs/<run_key>/<etapa>/ y se salta si ya terminó con
  la misma huella. Si no se indica ``run_key``, se deriva de forma determinista de
  (config, datos, commit): relanzar el DAG con lo mismo reutiliza la corrida en vez de duplicarla.
- La compuerta de validación falla SIN reintentos (AirflowFailException) y corta el DAG.
- Por XCom solo viajan rutas e identificadores livianos; los modelos y datasets viven en disco/MLflow.
"""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yaml
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup

# El código vive en el proyecto montado; PYTHONPATH lo cubre en Docker, esto cubre ejecución local.
PROJECT_ROOT = Path(os.environ.get('AUTOML_PROJECT_ROOT', Path(__file__).resolve().parents[1]))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = os.environ.get('AUTOML_CONFIG', 'config/config.yaml')


def _leer_config_airflow():
    """Sección ``airflow`` de config.yaml (el DAG se parsea sin importar el resto del proyecto)."""
    path = Path(CONFIG_PATH)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return (yaml.safe_load(path.read_text(encoding='utf-8')) or {}).get('airflow', {}) or {}
    except OSError:
        return {}


AF = _leer_config_airflow()


def _opt(env, key, default, cast=float):
    """Valor de la variable de entorno si existe; si no, config.yaml; si no, el valor por defecto."""
    return cast(os.environ.get(env, AF.get(key, default)))


MODELOS = ['xgb', 'lgbm', 'catb', 'logistic_regression', 'random_forest', 'ridge_classifier',
           'adaboost', 'gradient_boosting_sklearn', 'extra_trees']
POOL = os.environ.get('AUTOML_POOL', AF.get('pool', 'automl_compute'))

default_args = {
    'owner': 'mlops',
    'retries': _opt('AUTOML_RETRIES', 'retries', 2, int),
    'retry_delay': timedelta(seconds=_opt('AUTOML_RETRY_DELAY_SECONDS', 'retry_delay_seconds', 60)),
    'retry_exponential_backoff': bool(AF.get('retry_exponential_backoff', True)),
    'max_retry_delay': timedelta(minutes=float(AF.get('max_retry_delay_minutes', 15))),
    'execution_timeout': timedelta(hours=_opt('AUTOML_TASK_TIMEOUT_HOURS', 'task_timeout_hours', 4)),
}


def _overrides(params):
    over = {}
    if params.get('n_trials'):
        over.setdefault('hpo', {}).setdefault('optuna', {})['n_trials'] = int(params['n_trials'])
    if params.get('sample_fraction'):
        over['sample_fraction'] = float(params['sample_fraction'])
        over.setdefault('backtesting', {})['sample_fraction'] = float(params['sample_fraction'])
    if params.get('workers'):
        over.setdefault('parallel', {})['workers'] = int(params['workers'])
    return over


def ejecutar_etapa(etapa, modelo=None, **context):
    """Callable común: delega en src.stages, que ya es idempotente.

    Devuelve por XCom solo metadatos livianos (run_key, etapa, id del run de MLflow)."""
    from src import stages
    from src.validation import DataValidationError
    params = context['params']
    # Contexto del DAG para los tags de trazabilidad (airflow_dag_run_id), también en `airflow dags test`.
    os.environ['AIRFLOW_CTX_DAG_RUN_ID'] = str(context['run_id'])
    cfg = stages.cargar_config(params['config_path'], overrides=_overrides(params))
    if etapa=='validate_data':
        run_key=params.get('run_key') or stages.run_key_por_defecto(cfg)
    else:
        upstream=context['ti'].xcom_pull(task_ids='validate_data')
        if not upstream: raise RuntimeError('Ejecutar validate_data antes de esta etapa.')
        run_key=upstream['run_key']
    try:
        result = stages.ejecutar(etapa, cfg, run_key, modelo, force=bool(params.get('force')))
    except DataValidationError as exc:
        raise AirflowFailException(str(exc))  # datos malos: reintentar no sirve
    # XCom liviano: los resultados completos viven en runs/<run_key>/ y en MLflow.
    return {'run_key': stages.sanear(run_key), 'etapa': etapa, 'modelo': modelo,
            'cached': bool(result.get('cached')), 'mlflow_run_id': result.get('mlflow_run_id')}


with DAG(
    dag_id=AF.get('dag_id', 'automl_training_pipeline'),
    description='AutoML: validación, HPO (Optuna) por modelo, benchmark CPU/GPU, GetBestModel, backtesting, registro y reporte',
    schedule=AF.get('schedule', None),   # None = ejecución manual
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=int(AF.get('max_active_runs', 1)),
    default_args=default_args,
    params={
        'config_path': Param(CONFIG_PATH, type='string', description='YAML de configuración del proyecto'),
        'run_key': Param('', type=['string', 'null'], description='Clave de idempotencia. Vacío = derivada de '
                         '(config, datos, commit): relanzar con lo mismo reutiliza la corrida. '
                         'Reusar una clave retoma una corrida sin recalcular lo ya terminado.'),
        'force': Param(False, type='boolean', description='Reparar etapas incompletas; resultados completos se conservan'),
        'n_trials': Param(None, type=['integer', 'null'], description='Trials de Optuna por modelo (vacío = config)'),
        'sample_fraction': Param(None, type=['number', 'null'], description='Fracción de muestreo (vacío = config)'),
        'workers': Param(None, type=['integer', 'null'], description='Workers del benchmark paralelo (vacío = config)'),
    },
    tags=['mlops', 'automl', 'mcd402'],
    doc_md=__doc__,
) as dag:

    def tarea(etapa, task_id=None, modelo=None, **kwargs):
        return PythonOperator(task_id=task_id or etapa, python_callable=ejecutar_etapa,
                              op_kwargs={'etapa': etapa, 'modelo': modelo}, **kwargs)

    validate = tarea('validate_data')
    prepare = tarea('prepare_data')
    with TaskGroup('hpo', tooltip='Una tarea por modelo; paralelas, limitadas por el pool') as hpo_group:
        hpo_tasks = [tarea('hpo_model', task_id=m, modelo=m, pool=POOL) for m in MODELOS]
    select = tarea('select_hpo')
    bench = tarea('benchmark')
    best = tarea('get_best_model')
    backtest = tarea('backtesting')
    register = tarea('register_champion')
    report = tarea('generate_report')

    validate >> prepare >> hpo_group >> select >> bench >> best >> backtest >> register >> report
