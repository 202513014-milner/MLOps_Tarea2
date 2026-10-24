# Resultados de la corrida
Campeón: catb.
| Modelo | AUC train | AUC test | Decay | Captura top-k | Dispositivo | MB |
|---|---:|---:|---:|---:|---|---:|
| xgb | 0.8684 | 0.8544 | 1.6134 | 0.5173 | CPU | 0.2570 |
| lgbm | 0.8912 | 0.8554 | 4.0202 | 0.5213 | CPU | 0.5242 |
| catb | 0.8711 | 0.8554 | 1.8017 | 0.5257 | CPU | 0.1094 |
| logistic_regression | 0.8305 | 0.8315 | -0.1197 | 0.4760 | CPU | 0.0068 |
| random_forest | 0.9971 | 0.8462 | 15.1337 | 0.5034 | CPU | 54.8254 |
| ridge_classifier | 0.8289 | 0.8298 | -0.1131 | 0.4776 | CPU | 0.0068 |
| adaboost | 0.8404 | 0.8394 | 0.1140 | 0.4961 | CPU | 0.0927 |
| gradient_boosting_sklearn | 0.9009 | 0.8495 | 5.7039 | 0.5067 | CPU | 0.6332 |
| extra_trees | 0.9833 | 0.8239 | 16.2063 | 0.4642 | CPU | 79.6844 |

## Backtesting de modelos finales (sin reentrenar ni reseleccionar)
| Modelo | Ola | Mes | AUC | Captura | Lift | Alerta |
|---|---|---|---:|---:|---:|---|
| xgb | 1.0000 | 2022-09 | 0.8283 | 0.4608 | 4.6071 | False |
| xgb | 2.0000 | 2022-10 | 0.8487 | 0.4831 | 4.8316 | False |
| xgb | 3.0000 | 2022-11 | 0.8436 | 0.4546 | 4.5456 | False |
| xgb | 4.0000 | 2022-12 | 0.8535 | 0.5168 | 5.1676 | False |
| lgbm | 1.0000 | 2022-09 | 0.8306 | 0.4534 | 4.5336 | False |
| lgbm | 2.0000 | 2022-10 | 0.8531 | 0.4883 | 4.8835 | False |
| lgbm | 3.0000 | 2022-11 | 0.8438 | 0.4580 | 4.5796 | False |
| lgbm | 4.0000 | 2022-12 | 0.8552 | 0.5205 | 5.2050 | False |
| catb | 1.0000 | 2022-09 | 0.8285 | 0.4498 | 4.4968 | False |
| catb | 2.0000 | 2022-10 | 0.8476 | 0.4771 | 4.7710 | False |
| catb | 3.0000 | 2022-11 | 0.8436 | 0.4615 | 4.6136 | False |
| catb | 4.0000 | 2022-12 | 0.8539 | 0.5255 | 5.2548 | False |
| logistic_regression | 1.0000 | 2022-09 | 0.8034 | 0.4056 | 4.0557 | False |
| logistic_regression | 2.0000 | 2022-10 | 0.8205 | 0.4537 | 4.5372 | False |
| logistic_regression | 3.0000 | 2022-11 | 0.8115 | 0.3810 | 3.8088 | False |
| logistic_regression | 4.0000 | 2022-12 | 0.8182 | 0.4284 | 4.2835 | False |
| random_forest | 1.0000 | 2022-09 | 0.8193 | 0.4252 | 4.2518 | False |
| random_forest | 2.0000 | 2022-10 | 0.8367 | 0.4771 | 4.7710 | False |
| random_forest | 3.0000 | 2022-11 | 0.8326 | 0.4376 | 4.3756 | False |
| random_forest | 4.0000 | 2022-12 | 0.8467 | 0.4956 | 4.9559 | False |
| ridge_classifier | 1.0000 | 2022-09 | 0.8032 | 0.4032 | 4.0312 | False |
| ridge_classifier | 2.0000 | 2022-10 | 0.8203 | 0.4545 | 4.5458 | False |
| ridge_classifier | 3.0000 | 2022-11 | 0.8096 | 0.3821 | 3.8201 | False |
| ridge_classifier | 4.0000 | 2022-12 | 0.8143 | 0.4234 | 4.2337 | False |
| adaboost | 1.0000 | 2022-09 | 0.8113 | 0.4216 | 4.2150 | False |
| adaboost | 2.0000 | 2022-10 | 0.8303 | 0.4597 | 4.5978 | False |
| adaboost | 3.0000 | 2022-11 | 0.8236 | 0.4422 | 4.4209 | False |
| adaboost | 4.0000 | 2022-12 | 0.8367 | 0.4994 | 4.9933 | False |
| gradient_boosting_sklearn | 1.0000 | 2022-09 | 0.8231 | 0.4240 | 4.2395 | False |
| gradient_boosting_sklearn | 2.0000 | 2022-10 | 0.8452 | 0.4762 | 4.7623 | False |
| gradient_boosting_sklearn | 3.0000 | 2022-11 | 0.8361 | 0.4206 | 4.2056 | False |
| gradient_boosting_sklearn | 4.0000 | 2022-12 | 0.8464 | 0.4857 | 4.8563 | False |
| extra_trees | 1.0000 | 2022-09 | 0.7965 | 0.3995 | 3.9944 | False |
| extra_trees | 2.0000 | 2022-10 | 0.8113 | 0.4320 | 4.3207 | False |
| extra_trees | 3.0000 | 2022-11 | 0.8103 | 0.4240 | 4.2396 | False |
| extra_trees | 4.0000 | 2022-12 | 0.8178 | 0.4483 | 4.4828 | False |

## Recursos por entrenamiento final
| Modelo | Modo | CPU media % | CPU max % | RAM media MB | RAM max MB | Fit s | Hilos configurados |
|---|---|---:|---:|---:|---:|---:|---:|
| xgb | parallel | 100.5670 | 101.4000 | 576.3030 | 601.7930 | 5.0495 | 1.0000 |
| lgbm | parallel | 100.2400 | 100.6000 | 644.4270 | 659.7070 | 8.6622 | 1.0000 |
| catb | parallel | 101.0330 | 103.9000 | 728.0920 | 741.1370 | 11.8051 | 1.0000 |
| logistic_regression | parallel | 100.6500 | 100.9000 | 706.1480 | 744.4570 | 3.8871 | 1.0000 |
| random_forest | parallel | 100.5310 | 101.0000 | 742.7870 | 750.6720 | 57.4602 | 1.0000 |
| ridge_classifier | parallel | 101.9000 | 101.9000 | 692.1370 | 692.1370 | 0.6296 | 1.0000 |
| adaboost | parallel | 100.3570 | 100.9000 | 766.4020 | 768.9960 | 58.6345 | 1.0000 |
| gradient_boosting_sklearn | parallel | 100.4540 | 101.4000 | 768.5840 | 769.3480 | 223.8901 | 1.0000 |
| extra_trees | parallel | 100.4880 | 101.4000 | 796.1480 | 835.9530 | 30.7538 | 1.0000 |

CPU/RAM: process and children; GPU: device-global, not exclusive; CPU logical count is capacity, estimator_threads=1 is configured limit.

La selección usa validación p1–p6; p7–p10 se reportan sin cambiar al campeón.