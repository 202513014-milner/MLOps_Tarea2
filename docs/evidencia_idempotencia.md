# Evidencia de idempotencia

## Qué se probó

Relanzar el DAG `automl_training_pipeline` sobre la **corrida final** (`config/config.yaml`, 10 % de muestra,
15 trials, clave `auto-bde95c04df4aa4894f91`) con los mismos parámetros efectivos, datos, código e imagen, y
comprobar que las etapas se reutilizan y no se duplican runs ni carpetas.

## Ejecuciones del DAG (Airflow, 2026-10-24, UTC)

| # | Ejecución (`airflow_dag_run_id`) | Inicio | Fin | Duración | Clave de corrida | Resultado |
|---|---|---|---|---|---|---|
| 1 | `manual__2026-10-24T13:52:05+00:00` | 13:52:47 | 15:32:28 | 1:39:40 | `auto-bde95c04df4aa4894f91` | 17/17 tareas en éxito; crea la corrida |
| 2 | `manual__2026-10-24T17:13:59+00:00` | 17:14:03 | 17:17:17 | 3:14 | `auto-bde95c04df4aa4894f91` | 17/17 tareas en éxito; reutiliza la corrida |

**Configuración de cada disparo.** La ejecución 1 se lanzó con
`{"config_path":"config/config.yaml","run_key":"","force":false}`. La ejecución 2 se lanzó sin parámetros
adicionales: Airflow rellenó los valores por defecto del DAG (`run_key`, `n_trials`, `sample_fraction` y
`workers` nulos). `run_key` vacío o nulo produce lo mismo (el DAG deriva la clave a partir de configuración,
datos y commit) y los parámetros nulos no sobrescriben `config.yaml`; por eso ambas ejecuciones son el
mismo experimento.

## Qué se verificó

1. **La clave se reutilizó.** La repetición no creó ninguna carpeta nueva en `data/output/runs/`: los archivos de
   bloqueo de `auto-bde95c04df4aa4894f91` se actualizaron entre las 17:14 y las 17:17 UTC y no apareció otra clave.
2. **No se reentrenó.** La repetición duró 3:14 frente a 1:39:40 de la ejecución original.
3. **Sin duplicados en MLflow** (experimento `automl-venta-financiera`). El filtro
   `tags.idempotency_key LIKE 'auto-bde95c04df4aa4894f91%'` devuelve **26 runs** después de la repetición, exactamente los que creó la
   ejecución 1; si la repetición hubiera duplicado runs, serían 52. Los 26 runs son 17 de etapa (validate_data,
   prepare_data, 9 de HPO, select_hpo, benchmark, get_best_model, backtesting, register_champion,
   generate_report) y 9 de modelos finales del benchmark. Captura: `docs/img/mlflow_idempotencia.png`.
4. **Procedencia.** Por diseño, al reutilizar una etapa completa el pipeline añade a su run un tag
   `airflow_attempt.<hash>` con el Id de la ejecución 2 y no modifica los tags originales (`_record_attempt` en
   `src/stages.py`). Este tag no se verificó con captura en esta corrida.

## Limitaciones

- La repetición tardó 3 min aunque todas las etapas se reutilizan. Por lectura del código, una causa probable es que
  cada tarea recalcula el hash de los ~1.1 GB de CSV al verificar la identidad de la corrida; no se midió.
- No se probaron aquí la reanudación tras una tarea incompleta ni el rechazo de una etiqueta manual incompatible con
  datos o configuración distintos; esos casos los cubre `tests/test_integracion.py` con datos sintéticos
  (`VALIDACION.md`).

## Capturas

- Corrida final: `docs/img/airflow_graph.png`, `mlflow_tags.png`, `mlflow_run_tags.png`.
- Repetición: `docs/img/airflow_idempotencia.png` (ejecuciones del DAG en éxito) y `docs/img/mlflow_idempotencia.png` (26 runs).
