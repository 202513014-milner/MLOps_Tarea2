"""DAG de inferencia mensual (P3). Un mes por ejecución, schedule=None.

Disparo:  airflow dags trigger automl_inference_pipeline --conf '{"mes": "p7"}'   (meses: p7, p8, p9, p10)
Cadena:   cargar_datos >> preprocesar_datos >> generar_scores >> posprocesar >> registrar_inferencia
El champion no se reentrena: se carga del Model Registry por alias. Al terminar quedan en la carpeta del mes los scores,
el TLV y el grupo de ejecución de cada cliente (ver docs/INFERENCIA_KIM.md). Monitoreo y validación de resultados no forman
parte de este DAG: leen lo que deja `registrar_inferencia` (inference_context.json).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

from src.inference import cargar_cfg

CFG = cargar_cfg()
IC = CFG['inference']
PASOS = ['preprocesar_datos', 'generar_scores', 'posprocesar', 'registrar_inferencia']


def _cargar(dag_run, **_):
    """Primera tarea: toma el mes de --conf (o el predeterminado de config.yaml) y lo carga."""
    mes = (dag_run.conf or {}).get('mes', IC['default_month'])
    if mes not in IC['months']:
        raise ValueError(f"mes debe ser uno de {IC['months']}")
    from src.inference import cargar_datos
    return cargar_datos(CFG, mes)


def _paso(nombre, anterior):
    """Tarea que ejecuta `src.inference.<nombre>` con el contexto liviano de la tarea anterior."""
    def run(ti, **_):
        """Recibe por XCom el contexto del paso previo (rutas y metadatos)."""
        from src import inference
        return getattr(inference, nombre)(CFG, ti.xcom_pull(task_ids=anterior))
    return run


with DAG(dag_id=IC['dag_id'], start_date=datetime(2026, 1, 1), schedule=None, catchup=False, max_active_runs=1,
         default_args={'owner': 'mlops', 'retries': IC['retries'], 'retry_delay': timedelta(seconds=IC['retry_delay_seconds'])},
         tags=['p3', 'inferencia']) as dag:
    tareas = [PythonOperator(task_id='cargar_datos', python_callable=_cargar)]
    for anterior, nombre in zip(['cargar_datos'] + PASOS, PASOS):
        tareas.append(PythonOperator(task_id=nombre, python_callable=_paso(nombre, anterior)))
    for a, b in zip(tareas, tareas[1:]):
        a >> b
