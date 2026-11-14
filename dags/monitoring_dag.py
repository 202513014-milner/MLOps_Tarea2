"""DAG independiente para monitorear una inferencia mensual."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator
from src.stages import cargar_config

CFG = cargar_config("config/config.yaml")
MON = CFG.get("monitoring", {})


def _monitorear(dag_run, **_):
    conf = dag_run.conf or {}
    mes = conf.get("mes")

    if not mes:
        raise ValueError("El DAG requiere 'mes' en dag_run.conf.")

    base = Path(CFG["output_dir"]) / "inference" / mes
    key = conf.get("key")

    if key:
        contexto_path = base / key / "inference_context.json"
        if not contexto_path.is_file():
            raise FileNotFoundError(f"No existe el contexto de inferencia: {contexto_path}")
    else:
        candidatos = list(base.glob("*/inference_context.json"))
        if not candidatos:
            raise FileNotFoundError(f"No hay inferencias registradas para {mes}.")
        contexto_path = max(candidatos, key=lambda p: p.stat().st_mtime)

    contexto = json.loads(contexto_path.read_text(encoding="utf-8"))
    key = contexto.get("key") or contexto_path.parent.name
    inference_run_id = conf.get("inference_run_id") or contexto.get("inference_run_id")

    from src.monitoring_p3 import monitorear_desde_contexto

    return monitorear_desde_contexto(
        CFG,
        mes=mes,
        key=key,
        inference_run_id=inference_run_id,
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
        "retry_delay": timedelta(seconds=MON.get("retry_delay_seconds", 30)),
    },
    tags=["p3", "monitoreo"],
) as dag:
    PythonOperator(
        task_id="monitorear",
        python_callable=_monitorear,
    )