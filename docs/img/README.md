Capturas reales de la corrida final (`auto-bde95c04df4aa4894f91`):

- `airflow_graph.png`: DAG con sus 17 tareas en éxito.
- `airflow_idempotencia.png`: ejecuciones del DAG (original y repetición, ambas en éxito).
- `mlflow_idempotencia.png`: 26 runs de la corrida tras la repetición, sin duplicados.
- `mlflow_tags.png`, `mlflow_run_tags.png`: runs de entrenamiento y tags de trazabilidad (`git_commit`, `git_branch`, `docker_image`, `config_hash`, `data_hash`, `airflow_dag_run_id`).
