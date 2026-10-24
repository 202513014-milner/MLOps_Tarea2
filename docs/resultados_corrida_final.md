# Resultados de la corrida final — `auto-bde95c04df4aa4894f91`

Fuente: `data/output/runs/auto-bde95c04df4aa4894f91/` (`results_summary.json`, `compute_logs.json`, `report.md`, `backtesting/`). Tablas generadas desde esos archivos.

## 1. Resumen

- **Campeón:** `catb` (modo `parallel`), registrado en MLflow como `automl-venta-financiera` versión 1, alias `champion` (Primer campeón).
- Métrica primaria (captura en el top 10 %): **0.5257** · AUC validación 0.8554 · decay 1.80 % · lift 5.26 · KS 0.5556 · Gini 0.7109.
- Configuración: 10 % de muestra, 15 trials de Optuna por modelo, CV de 3 folds, semilla 42, split 70/30 por mes, ejecución en CPU.
- Backtesting: 18 artefactos evaluados (9 de HPO + 9 finales), todos con 4/4 olas válidas y **0 alertas de drift**.

## 2. Datos (meses 1–6, desarrollo)

Validación de datos: ok = True, errores = 0, advertencias = 0. Filas train: 127,229 · validación: 54,531 · predictores: 61.

| Archivo | Filas originales | Filas muestra | % positivos original | % positivos muestra |
|---|---:|---:|---:|---:|
| p1_extrac.csv | 328,488 | 32,848 | 3.42 | 3.42 |
| p2_extrac.csv | 329,324 | 32,932 | 3.12 | 3.12 |
| p3_extrac.csv | 322,356 | 32,235 | 3.52 | 3.52 |
| p4_extrac.csv | 304,813 | 30,481 | 2.95 | 2.95 |
| p5_extrac.csv | 295,732 | 29,573 | 2.94 | 2.95 |
| p6_extrac.csv | 236,914 | 23,691 | 3.84 | 3.84 |

Los meses p7–p10 no se cargaron durante entrenamiento ni selección (`future_files_loaded: false`).

## 3. Búsqueda de hiperparámetros (Optuna, TPE)

El AUC de CV es el de selección del HPO; no es una estimación independiente ni CV anidada.

| Modelo | Trials | AUC CV | Tiempo HPO (s) | Mejores hiperparámetros |
|---|---:|---:|---:|---|
| xgb | 15 | 0.8511 | 108 | n_estimators=222, max_depth=3, learning_rate=0.03919, subsample=0.7465, colsample_bytree=0.728 |
| lgbm | 15 | 0.8514 | 239 | n_estimators=297, num_leaves=15, learning_rate=0.025 |
| catb | 15 | 0.8515 | 384 | iterations=293, depth=4, learning_rate=0.05623 |
| logistic_regression | 15 | 0.8277 | 140 | C=21.42 |
| random_forest | 8 | 0.8416 | 672 | n_estimators=175, max_depth=20, min_samples_leaf=8 |
| ridge_classifier | 15 | 0.8265 | 30 | alpha=0.001022 |
| adaboost | 7 | 0.8368 | 670 | n_estimators=140, learning_rate=0.1596 |
| gradient_boosting_sklearn | 2 | 0.8426 | 852 | n_estimators=156, max_depth=5, learning_rate=0.1079 |
| extra_trees | 11 | 0.8228 | 677 | n_estimators=175, max_depth=20, min_samples_leaf=8 |

## 4. Modelos finales (validación, meses 1–6)

| Modelo | AUC train | AUC val | Decay % | Captura top-10 % | Lift | KS | ¿Pasa filtros? |
|---|---:|---:|---:|---:|---:|---:|---|
| xgb | 0.8684 | 0.8544 | 1.61 | 0.5173 | 5.17 | 0.5556 | sí |
| lgbm | 0.8912 | 0.8554 | 4.02 | 0.5213 | 5.21 | 0.5589 | sí |
| catb ★ | 0.8711 | 0.8554 | 1.80 | 0.5257 | 5.26 | 0.5556 | sí |
| logistic_regression | 0.8305 | 0.8315 | -0.12 | 0.4760 | 4.76 | 0.5113 | sí |
| random_forest | 0.9971 | 0.8462 | 15.13 | 0.5034 | 5.03 | 0.5522 | NO (decay > 10 %) |
| ridge_classifier | 0.8289 | 0.8298 | -0.11 | 0.4776 | 4.78 | 0.5091 | sí |
| adaboost | 0.8404 | 0.8394 | 0.11 | 0.4961 | 4.96 | 0.5373 | sí |
| gradient_boosting_sklearn | 0.9009 | 0.8495 | 5.70 | 0.5067 | 5.07 | 0.5536 | sí |
| extra_trees | 0.9833 | 0.8239 | 16.21 | 0.4642 | 4.64 | 0.4947 | NO (decay > 10 %) |

**Selección.** Se descartan `random_forest` y `extra_trees` por decay > 10 % (sobreajuste). Los modelos a menos de 0.005 de la mejor captura (lgbm, catb) se consideran empatados y se desempata por mayor AUC y luego menor tiempo; resultó `catb` (AUC 0.855442 frente a 0.855406 de `lgbm`, una diferencia mínima). En la práctica ambos son equivalentes: `lgbm` fue incluso ligeramente mejor en el AUC medio del backtesting (0.8457 vs 0.8434).

## 5. Benchmark secuencial vs paralelo

- Esta corrida se ejecutó solo en modo **paralelo** (`parallel.mode: parallel`, 2 workers): T_paralelo = 400.8 s.
- T_secuencial y **speed-up = null**: no se midieron porque no se corrió en modo `benchmark`. El README indica que un modo único no tiene speed-up. **Pendiente** si la rúbrica exige la comparación (`parallel.mode: benchmark`).

## 6. Backtesting (meses 7–10, sin reentrenar)

AUC medio de las 4 olas por modelo final (modo paralelo) y su referencia de validación:

| Modelo | AUC val (referencia) | AUC medio olas | Peor ola | Captura media | Alertas drift |
|---|---:|---:|---:|---:|---:|
| xgb | 0.8544 | 0.8435 | 0.8283 | 0.4788 | 0 |
| lgbm | 0.8554 | 0.8457 | 0.8306 | 0.4801 | 0 |
| catb ★ | 0.8554 | 0.8434 | 0.8285 | 0.4784 | 0 |
| logistic_regression | 0.8315 | 0.8134 | 0.8034 | 0.4172 | 0 |
| random_forest | 0.8462 | 0.8338 | 0.8193 | 0.4589 | 0 |
| ridge_classifier | 0.8298 | 0.8119 | 0.8032 | 0.4158 | 0 |
| adaboost | 0.8394 | 0.8255 | 0.8113 | 0.4557 | 0 |
| gradient_boosting_sklearn | 0.8495 | 0.8377 | 0.8231 | 0.4516 | 0 |
| extra_trees | 0.8239 | 0.8090 | 0.7965 | 0.4260 | 0 |

Olas del campeón `catb`:

| Ola | Mes | AUC | Decay % | Captura | Lift | Alerta |
|---:|---|---:|---:|---:|---:|---|
| T+1 | 2022-09 | 0.8285 | 3.15 | 0.4498 | 4.50 | False |
| T+2 | 2022-10 | 0.8476 | 0.92 | 0.4771 | 4.77 | False |
| T+3 | 2022-11 | 0.8436 | 1.38 | 0.4615 | 4.61 | False |
| T+4 | 2022-12 | 0.8539 | 0.18 | 0.5255 | 5.25 | False |

Un decay negativo es mejora respecto a validación. La tabla completa de 36 filas está en `report.md`. Hay variación entre meses (p. ej. T+1 más bajo y T+4 más alto) que esta evidencia no explica por sí sola.

## 7. Cómputo

Hardware visible para el contenedor: 16 CPU lógicas (8 físicas), 11.5 GB de RAM, GPU: no (modo `cpu_only`).

| Etapa | Duración (s) | CPU media % | RAM máx (MB) |
|---|---:|---:|---:|
| validate_data | 58.3 | 45.6 | 802 |
| prepare_data | 24.9 | 45.0 | 977 |
| select_hpo | 1.3 | 7.8 | 383 |
| benchmark | 503.8 | 88.7 | 928 |
| get_best_model | 0.7 | 7.5 | 383 |
| backtesting | 180.3 | 29.6 | 1018 |
| register_champion | 12.2 | 51.3 | 547 |
| hpo/xgb | 112.3 | 95.5 | 672 |
| hpo/lgbm | 249.8 | 96.1 | 697 |
| hpo/catb | 391.8 | 99.7 | 771 |
| hpo/logistic_regression | 147.1 | 92.2 | 696 |
| hpo/random_forest | 693.8 | 98.2 | 747 |
| hpo/ridge_classifier | 33.7 | 80.1 | 700 |
| hpo/adaboost | 680.8 | 99.3 | 643 |
| hpo/gradient_boosting_sklearn | 860.9 | 99.7 | 667 |
| hpo/extra_trees | 704.9 | 97.6 | 750 |
| generate_report | 5.9 | 9.7 | 272 |

CPU/RAM miden el proceso y sus hijos. Cada estimador usó 1 hilo.

## 8. Trazabilidad

| Campo | Valor |
|---|---|
| run_key | `auto-bde95c04df4aa4894f91` |
| git_commit | `1800f948f8d5cee6d0852541f83cb60704eebf07` |
| git_branch | `main` |
| git_dirty | `true` |
| code_hash | `504eb3c15deac3630a63781b208b6b55cca9458b42cf2a28b9c6ef881a210bf9` |
| data_hash | `8babef04aa4bee7a7bd1842f7c42e449078f15e279df54c7463a50165d6fdd49` |
| config_hash | `d99fa1b5cfb788e37ba1e7a8beca00a2359fc1e837f418687ef7d1bb2e137c3a` |
| docker_image | `automl-mlops:1800f948f8d5` |
| docker_image_id | `sha256:e9520382f679039dd41cb60ece2efa4b0eec24b0d53956ba09e8dc7beb89788d` |
| airflow_dag_id | `automl_training_pipeline` |
| airflow_dag_run_id | `manual__2026-10-24T13:52:05+00:00` |

`git_dirty = true`: la corrida se ejecutó con cambios sin commitear; el `code_hash` identifica el código efectivo.

## 9. Limitaciones y observaciones

- Muestra del 10 % y tope de 15 trials o 600 s por modelo: las métricas son de esta configuración, no de todo el dataset.
- **El límite de 600 s cortó la búsqueda en 4 modelos:** `gradient_boosting_sklearn` completó solo 2 trials, `adaboost` 7, `random_forest` 8 y `extra_trees` 11 (los demás completaron 15). Sus hiperparámetros salen de una búsqueda más corta, y el número de trials depende de la máquina. El campeón (`catb`) y los otros dos primeros (`lgbm`, `xgb`) sí completaron 15.
- Sin speed-up medido (ver sección 5).
- Todos los modelos finales corrieron en CPU; no se probó GPU.
- La idempotencia se comprobó relanzando esta misma corrida (ver `docs/evidencia_idempotencia.md`).
