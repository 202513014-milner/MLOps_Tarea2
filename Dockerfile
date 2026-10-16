# Imagen del pipeline: Airflow 2.10 + las librerías de ML/MLOps del proyecto.
# El código (src/, dags/, config/) NO se copia: se monta como volumen para iterar sin reconstruir.
FROM apache/airflow:2.10.5-python3.11

USER root
# libgomp1: requerido por LightGBM/XGBoost (OpenMP). git: para registrar git_commit/git_branch de cada run.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 git \
    && rm -rf /var/lib/apt/lists/*

USER airflow
ARG AIRFLOW_VERSION=2.10.5
ARG PYTHON_VERSION=3.11
COPY requirements-docker.txt /tmp/requirements-docker.txt
# Mantener la versión de Airflow al resolver dependencias adicionales.
# MLflow servidor usa otra imagen; aquí solo va el cliente.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements-docker.txt

ENV PYTHONPATH=/opt/airflow/project \
    AUTOML_PROJECT_ROOT=/opt/airflow/project \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /opt/airflow/project
