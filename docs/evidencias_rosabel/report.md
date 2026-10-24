# Resultados de la corrida
Campeón: catb.
| Modelo | AUC train | AUC test | Decay | Captura top-k | Dispositivo | MB |
|---|---:|---:|---:|---:|---|---:|
| xgb | 0.8684 | 0.8544 | 1.6134 | 0.5173 | CPU | 0.2570 |
| lgbm | 0.8912 | 0.8554 | 4.0202 | 0.5213 | CPU | 0.5242 |
| catb | 0.8711 | 0.8554 | 1.8017 | 0.5257 | CPU | 0.1094 |
| logistic_regression | 0.8305 | 0.8315 | -0.1197 | 0.4760 | CPU | 0.0068 |
| random_forest | 0.9945 | 0.8464 | 14.8939 | 0.5106 | CPU | 54.3298 |
| ridge_classifier | 0.8289 | 0.8298 | -0.1131 | 0.4776 | CPU | 0.0068 |
| adaboost | 0.8442 | 0.8426 | 0.1957 | 0.5028 | CPU | 0.1142 |
| gradient_boosting_sklearn | 0.8640 | 0.8518 | 1.4113 | 0.5285 | CPU | 0.1770 |
| extra_trees | 0.9708 | 0.8257 | 14.9530 | 0.4681 | CPU | 104.6361 |

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
| random_forest | 1.0000 | 2022-09 | 0.8238 | 0.4265 | 4.2640 | False |
| random_forest | 2.0000 | 2022-10 | 0.8395 | 0.4779 | 4.7796 | False |
| random_forest | 3.0000 | 2022-11 | 0.8326 | 0.4490 | 4.4889 | False |
| random_forest | 4.0000 | 2022-12 | 0.8466 | 0.5031 | 5.0307 | False |
| ridge_classifier | 1.0000 | 2022-09 | 0.8032 | 0.4032 | 4.0312 | False |
| ridge_classifier | 2.0000 | 2022-10 | 0.8203 | 0.4545 | 4.5458 | False |
| ridge_classifier | 3.0000 | 2022-11 | 0.8096 | 0.3821 | 3.8201 | False |
| ridge_classifier | 4.0000 | 2022-12 | 0.8143 | 0.4234 | 4.2337 | False |
| adaboost | 1.0000 | 2022-09 | 0.8161 | 0.4240 | 4.2395 | False |
| adaboost | 2.0000 | 2022-10 | 0.8342 | 0.4693 | 4.6930 | False |
| adaboost | 3.0000 | 2022-11 | 0.8288 | 0.4501 | 4.5003 | False |
| adaboost | 4.0000 | 2022-12 | 0.8390 | 0.4832 | 4.8314 | False |
| gradient_boosting_sklearn | 1.0000 | 2022-09 | 0.8247 | 0.4559 | 4.5581 | False |
| gradient_boosting_sklearn | 2.0000 | 2022-10 | 0.8468 | 0.4866 | 4.8662 | False |
| gradient_boosting_sklearn | 3.0000 | 2022-11 | 0.8342 | 0.4297 | 4.2962 | False |
| gradient_boosting_sklearn | 4.0000 | 2022-12 | 0.8503 | 0.5093 | 5.0929 | False |
| extra_trees | 1.0000 | 2022-09 | 0.7993 | 0.3971 | 3.9699 | False |
| extra_trees | 2.0000 | 2022-10 | 0.8124 | 0.4329 | 4.3294 | False |
| extra_trees | 3.0000 | 2022-11 | 0.8115 | 0.4376 | 4.3756 | False |
| extra_trees | 4.0000 | 2022-12 | 0.8179 | 0.4496 | 4.4952 | False |

## Recursos por entrenamiento final
| Modelo | Modo | CPU media % | CPU max % | RAM media MB | RAM max MB | Fit s | Hilos configurados |
|---|---|---:|---:|---:|---:|---:|---:|
| xgb | parallel | 99.8500 | 100.2000 | 561.3850 | 599.6020 | 3.6108 | 1.0000 |
| lgbm | parallel | 100.2000 | 100.9000 | 638.2340 | 657.3320 | 6.5122 | 1.0000 |
| catb | parallel | 107.0400 | 133.4000 | 732.9380 | 750.2460 | 8.6759 | 1.0000 |
| logistic_regression | parallel | 101.5500 | 103.2000 | 707.9530 | 746.2660 | 2.1677 | 1.0000 |
| random_forest | parallel | 100.3050 | 101.8000 | 743.2020 | 747.1840 | 40.4963 | 1.0000 |
| ridge_classifier | parallel | 98.7000 | 98.7000 | 686.8870 | 686.8870 | 0.4879 | 1.0000 |
| adaboost | parallel | 100.2520 | 100.5000 | 760.8290 | 763.7230 | 52.8077 | 1.0000 |
| gradient_boosting_sklearn | parallel | 100.3120 | 104.7000 | 762.1440 | 764.0350 | 82.3805 | 1.0000 |
| extra_trees | parallel | 100.2880 | 101.3000 | 804.7500 | 857.6800 | 30.7198 | 1.0000 |

CPU/RAM: process and children; GPU: device-global, not exclusive; CPU logical count is capacity, estimator_threads=1 is configured limit.

La selección usa validación p1–p6; p7–p10 se reportan sin cambiar al campeón.