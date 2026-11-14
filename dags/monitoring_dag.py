"""DAG independiente para monitorear una inferencia mensual."""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from src.stages import cargar_config

CFG = cargar_config("config/config.yaml")
MON = CFG.get("monitoring", {})


def _monitorear(dag_run, **_):
    conf = dag_run.conf or {}
    mes = conf.get("mes")
    key = conf.get("key")

    if not mes or not key:
        raise ValueError("El DAG requiere 'mes' y 'key' en dag_run.conf.")

    from src.monitoring_p3 import monitorear_desde_contexto

    return monitorear_desde_contexto(
        CFG,
        mes=mes,
        key=key,
        inference_run_id=conf.get("inference_run_id"),
    )


with DAG(
    dag_id=MON.get("dag_id", "automl_monitoring_pipeline"),
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "mlops",
        "retries": MON.get("retries", 1),
        "retry_delay": timedelta(
            seconds=MON.get("retry_delay_seconds", 30)
        ),
    },
    tags=["p3", "monitoreo"],
) as dag:
    PythonOperator(
        task_id="monitorear",
        python_callable=_monitorear,
    )