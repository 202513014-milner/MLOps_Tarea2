# Evidencia de idempotencia (versión rápida)

> **Alcance.** Por falta de tiempo, la idempotencia se verificó con una **corrida reducida** y no con la
> corrida completa. La corrida reducida usó `config/config.yaml` con dos parámetros cambiados desde
> Airflow (`sample_fraction = 0.01` y `n_trials = 2`); **no** usó el perfil `config/smoke.yaml`
> (que además es secuencial, con 1 worker y sin límite de tiempo). Sus métricas **no** son resultados
> finales. Lo que se demuestra aquí es el comportamiento del pipeline al relanzarse, no el rendimiento.

## Qué se probó

Disparar el DAG `automl_training_pipeline` una segunda vez con **los mismos parámetros, datos, código e imagen**
y comprobar que las etapas se reutilizan y no se duplican runs en MLflow.

## Ejecuciones del DAG (Airflow, 2026-10-24, UTC)

| # | Ejecución (`airflow_dag_run_id`) | Configuración | Clave de corrida | Resultado |
|---|---|---|---|---|
| 1 | `manual__2026-10-24T13:52:05` | 10 % de muestra, 15 trials (valores de `config.yaml`) | `auto-bde95c04df4aa4894f91` | 17/17 tareas en éxito (~1 h 40 min) |
| 2 | `manual__2026-10-24T15:53:09` | 1 % de muestra, 2 trials | `auto-7e3324ccc84a7112fbc4` | 17/17 tareas en éxito |
| 3 | `manual__2026-10-24T16:27:51` | **Igual que la 2** (relanzamiento) | `auto-7e3324ccc84a7112fbc4` | 17/17 tareas en éxito (~9 min, mínimo de las 3 ejecuciones) |

Las claves distintas de 1 y 2 son esperadas: la clave se deriva del hash de la configuración, y las dos
corridas solo difieren en `sample_fraction` (0.1 vs 0.01) y `hpo.optuna.n_trials` (15 vs 2).
El resto de la identidad (`code_hash`, `git_commit`, imagen Docker) es idéntico en ambas.

## Resultados en MLflow (experimento `automl-venta-financiera`)

Consulta a la API de MLflow después del relanzamiento:

| Clave de corrida | Runs totales | Runs de etapa con tag `airflow_attempt.*` |
|---|---|---|
| `auto-7e3324ccc84a7112fbc4` | **26** | **17** (valor: `manual__2026-10-24T16:27:51+00:00`) |
| `auto-bde95c04df4aa4894f91` | 26 | 0 (no se relanzó) |
| **Total** | **52** | |

Desglose de los 26 runs por corrida: 17 de etapa (validate_data, prepare_data, 9 de HPO, select_hpo,
benchmark, get_best_model, backtesting, register_champion, generate_report) + 9 de modelos finales del benchmark.

## Conclusiones verificadas

1. **La clave se reutilizó.** El relanzamiento (ejecución 3) operó sobre `auto-7e3324cc…`; no generó una clave nueva.
2. **No se duplicaron runs.** El número de runs por corrida sigue siendo 26 (52 en total) tras el relanzamiento.
3. **Se conservó la procedencia original y se añadió el intento.** Los 17 runs de etapa recibieron un tag
   `airflow_attempt.<hash>` con el ID de la ejecución 3, sin reemplazar sus tags originales.
4. **No se crearon carpetas nuevas** en `data/output/runs/`: solo existen las dos claves anteriores
   (`auto-bde95c04…` y `auto-7e3324cc…`).

## Limitaciones

- La corrida completa (10 % y 15 trials, `auto-bde95c04…`) **no** se relanzó, por lo que su idempotencia no
  está evidenciada directamente; se infiere del mismo mecanismo (`ejecutar_etapa` con `_SUCCESS.json` y huellas).
- El relanzamiento tardó ~9 min aunque todas las etapas vinieron de caché. Por lectura del código, una causa
  probable es que cada tarea recalcula el SHA256 de los ~1.1 GB de CSV al verificar la identidad de la
  corrida; no se midió.
- No se probaron aquí los otros casos del README (reanudación tras una tarea incompleta, cambio de datos o
  configuración con la misma etiqueta manual).
- Capturas: corrida grande en `docs/img/` (`airflow_graph`, `airflow_grid`, `mlflow_comparacion`, `mlflow_run_tags`); relanzamiento en `docs/img/airflow_idempotencia.png` (3 ejecuciones en éxito) y `docs/img/mlflow_idempotencia.png` (tag `airflow_attempt`).
