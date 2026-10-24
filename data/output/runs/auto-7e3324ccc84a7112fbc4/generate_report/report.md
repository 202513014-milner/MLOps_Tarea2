# Resultados de la corrida
Campeón: catb.
| Modelo | AUC train | AUC test | Decay | Captura top-k | Dispositivo | MB |
|---|---:|---:|---:|---:|---|---:|
| xgb | 1.0000 | 0.8138 | 18.6203 | 0.4302 | CPU | 0.9011 |
| lgbm | 0.9987 | 0.8290 | 16.9894 | 0.4749 | CPU | 0.5410 |
| catb | 0.8974 | 0.8468 | 5.6357 | 0.4860 | CPU | 0.0861 |
| logistic_regression | 0.8430 | 0.8148 | 3.3432 | 0.4190 | CPU | 0.0068 |
| random_forest | 0.9996 | 0.8483 | 15.1309 | 0.4916 | CPU | 7.0070 |
| ridge_classifier | 0.8428 | 0.8160 | 3.1796 | 0.4190 | CPU | 0.0068 |
| adaboost | 0.8511 | 0.8360 | 1.7830 | 0.4525 | CPU | 0.0718 |
| gradient_boosting_sklearn | 0.8649 | 0.8308 | 3.9462 | 0.4525 | CPU | 0.1628 |
| extra_trees | 0.9977 | 0.8210 | 17.7069 | 0.4637 | CPU | 11.6605 |

## Backtesting de modelos finales (sin reentrenar ni reseleccionar)
| Modelo | Ola | Mes | AUC | Captura | Lift | Alerta |
|---|---|---|---:|---:|---:|---|
| xgb | 1.0000 | 2022-09 | 0.7654 | 0.3049 | 3.0438 | False |
| xgb | 2.0000 | 2022-10 | 0.7988 | 0.3879 | 3.8745 | False |
| xgb | 3.0000 | 2022-11 | 0.7994 | 0.3636 | 3.6322 | False |
| xgb | 4.0000 | 2022-12 | 0.7931 | 0.3750 | 3.7535 | False |
| lgbm | 1.0000 | 2022-09 | 0.8055 | 0.3659 | 3.6526 | False |
| lgbm | 2.0000 | 2022-10 | 0.8302 | 0.3966 | 3.9606 | False |
| lgbm | 3.0000 | 2022-11 | 0.8187 | 0.3409 | 3.4052 | False |
| lgbm | 4.0000 | 2022-12 | 0.8117 | 0.4000 | 4.0038 | False |
| catb | 1.0000 | 2022-09 | 0.8203 | 0.3902 | 3.8961 | False |
| catb | 2.0000 | 2022-10 | 0.8350 | 0.4483 | 4.4772 | False |
| catb | 3.0000 | 2022-11 | 0.8302 | 0.4432 | 4.4268 | False |
| catb | 4.0000 | 2022-12 | 0.8141 | 0.4375 | 4.3791 | False |
| logistic_regression | 1.0000 | 2022-09 | 0.7999 | 0.3537 | 3.5308 | False |
| logistic_regression | 2.0000 | 2022-10 | 0.8159 | 0.4052 | 4.0467 | False |
| logistic_regression | 3.0000 | 2022-11 | 0.8131 | 0.3523 | 3.5187 | False |
| logistic_regression | 4.0000 | 2022-12 | 0.7945 | 0.3875 | 3.8787 | False |
| random_forest | 1.0000 | 2022-09 | 0.8251 | 0.3659 | 3.6526 | False |
| random_forest | 2.0000 | 2022-10 | 0.8269 | 0.4569 | 4.5633 | False |
| random_forest | 3.0000 | 2022-11 | 0.8155 | 0.4432 | 4.4268 | False |
| random_forest | 4.0000 | 2022-12 | 0.8052 | 0.4625 | 4.6294 | False |
| ridge_classifier | 1.0000 | 2022-09 | 0.8013 | 0.3659 | 3.6526 | False |
| ridge_classifier | 2.0000 | 2022-10 | 0.8188 | 0.4052 | 4.0467 | False |
| ridge_classifier | 3.0000 | 2022-11 | 0.8130 | 0.3523 | 3.5187 | False |
| ridge_classifier | 4.0000 | 2022-12 | 0.7974 | 0.4125 | 4.1289 | False |
| adaboost | 1.0000 | 2022-09 | 0.8083 | 0.3780 | 3.7743 | False |
| adaboost | 2.0000 | 2022-10 | 0.8173 | 0.4310 | 4.3050 | False |
| adaboost | 3.0000 | 2022-11 | 0.8135 | 0.4091 | 4.0863 | False |
| adaboost | 4.0000 | 2022-12 | 0.7930 | 0.3875 | 3.8787 | False |
| gradient_boosting_sklearn | 1.0000 | 2022-09 | 0.8161 | 0.4268 | 4.2613 | False |
| gradient_boosting_sklearn | 2.0000 | 2022-10 | 0.8152 | 0.4483 | 4.4772 | False |
| gradient_boosting_sklearn | 3.0000 | 2022-11 | 0.8024 | 0.4205 | 4.1998 | False |
| gradient_boosting_sklearn | 4.0000 | 2022-12 | 0.7905 | 0.3875 | 3.8787 | False |
| extra_trees | 1.0000 | 2022-09 | 0.8026 | 0.3659 | 3.6526 | False |
| extra_trees | 2.0000 | 2022-10 | 0.8201 | 0.3966 | 3.9606 | False |
| extra_trees | 3.0000 | 2022-11 | 0.8103 | 0.4318 | 4.3133 | False |
| extra_trees | 4.0000 | 2022-12 | 0.7765 | 0.3875 | 3.8787 | False |

## Recursos por entrenamiento final
| Modelo | Modo | CPU media % | CPU max % | RAM media MB | RAM max MB | Fit s | Hilos configurados |
|---|---|---:|---:|---:|---:|---:|---:|
| xgb | parallel | 99.7000 | 99.7000 | 419.1050 | 419.1050 | 1.3360 | 1.0000 |
| lgbm | parallel | 100.7000 | 100.7000 | 436.2270 | 436.2270 | 1.0443 | 1.0000 |
| catb | parallel | 98.9000 | 98.9000 | 457.6050 | 457.6050 | 1.3155 | 1.0000 |
| logistic_regression | parallel | 105.4000 | 105.4000 | 458.9060 | 458.9060 | 0.1445 | 1.0000 |
| random_forest | parallel | 100.2000 | 100.4000 | 459.2790 | 459.2970 | 3.0735 | 1.0000 |
| ridge_classifier | parallel | 92.2000 | 92.2000 | 465.5120 | 465.5120 | 0.0557 | 1.0000 |
| adaboost | parallel | 99.9000 | 100.4000 | 465.6370 | 465.6410 | 2.9788 | 1.0000 |
| gradient_boosting_sklearn | parallel | 100.2250 | 100.4000 | 465.9040 | 465.9450 | 7.5924 | 1.0000 |
| extra_trees | parallel | 100.9000 | 100.9000 | 466.0590 | 466.0590 | 1.4987 | 1.0000 |

CPU/RAM: process and children; GPU: device-global, not exclusive; CPU logical count is capacity, estimator_threads=1 is configured limit.

La selección usa validación p1–p6; p7–p10 se reportan sin cambiar al campeón.