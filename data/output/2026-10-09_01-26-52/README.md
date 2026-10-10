## Resultados reales de la corrida
Campeón inicial: xgb. Campeón temporal: adaboost.
Menor decay promedio entre candidatos que cumplen filtros P1 y no alertan en todas las olas; desempate por AUC promedio y AUC test.
| Modelo | AUC train | AUC test | Decay P1 % | AUC promedio BT | Decay promedio BT % |
|---|---:|---:|---:|---:|---:|
| xgb | 0.8707 | 0.8541 | 1.90 | 0.8431045590280964 | 3.1659830475638424 |
| lgbm | 0.8802 | 0.8536 | 3.02 | 0.8429406711750094 | 4.233240914392587 |
| catb | 0.8922 | 0.8545 | 4.23 | 0.8429827432292651 | 5.519432253958809 |
| logistic_regression | 0.8304 | 0.8314 | -0.12 | 0.8133456643727133 | 2.0585049331849663 |
| random_forest | 0.9979 | 0.8458 | 15.24 | 0.8351790412965604 | 16.30418785859188 |
| ridge_classifier | 0.8288 | 0.8298 | -0.11 | 0.8118038476988966 | 2.055741481568602 |
| adaboost | 0.8311 | 0.8293 | 0.22 | 0.8144691597605499 | 2.0036848867176724 |
| gradient_boosting_sklearn | 0.8746 | 0.8510 | 2.71 | 0.840481023393187 | 3.905094600570751 |
| extra_trees | 0.8400 | 0.8234 | 1.98 | 0.8002953076960214 | 4.728432967258407 |

### Olas de adaboost
| Mes / archivo | AUC | Decay % | Alerta drift |
|---|---:|---:|---|
| p7_extrac.csv | 0.8015826700659122 | 3.5541776091076027 | False |
| p8_extrac.csv | 0.8200191468289386 | 1.3359146278801666 | False |
| p9_extrac.csv | 0.8129767604161136 | 2.183249250425724 | False |
| p10_extrac.csv | 0.823298061731235 | 0.9413980594571958 | False |

| T secuencial s | T paralelo s | Speed-up |
|---:|---:|---:|
| 347.1733845999988 | 262.35374669999874 | 1.3233025598715435 |

Los cuatro meses participan en selección; no son un test independiente del campeón final.

![AUC por ola](auc_por_ola.png)