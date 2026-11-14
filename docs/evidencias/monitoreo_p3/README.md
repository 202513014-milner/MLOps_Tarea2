# Monitoreo P3: resultados p7–p10

## Resumen de las corridas

Los cuatro reportes corresponden al modelo registrado `automl-venta-financiera`, versión 1, con referencia `p6_extrac.csv` (236 914 filas). En los cuatro meses, el PSI/KL de la distribución de probabilidades permanece en nivel **estable** y `reentrena_modelo` es `false`.

| Mes | Estado general | Filas actuales | PSI prob. | KL prob. | Variables con mayor drift | ¿Reentrena? |
|---|---:|---:|---:|---:|---|---|
| p10 | alto | 317,378 | 0.0379 | 0.0197 | `ctd_camptototrcnl06m` PSI 2.252 (alto), `senthtml_1m` PSI 1.946 (alto), `grp_campecs06m` PSI 0.261 (alto) | No |
| p7 | moderado | 244,666 | 0.0027 | 0.0014 | `senthtml_1m` PSI 0.127 (moderado), `ctd_camptototrcnl06m` PSI 0.123 (moderado) | No |
| p8 | alto | 240,726 | 0.0092 | 0.0047 | `ctd_sms_received` PSI 0.283 (alto), `senthtml_1m` PSI 0.227 (moderado), `ctd_camptototrcnl06m` PSI 0.212 (moderado) | No |
| p9 | alto | 263,753 | 0.0191 | 0.0097 | `ctd_camptototrcnl06m` PSI 1.390 (alto), `senthtml_1m` PSI 0.446 (alto), `grp_campecs06m` PSI 0.341 (alto) | No |

## Interpretación

El estado general agrega los niveles de alerta de las variables predictoras y de la probabilidad tomando el nivel más alto. Por eso el estado puede ser moderado/alto aunque la probabilidad permanezca estable: las alertas de entrada reflejan cambios en las distribuciones de algunas variables; no implican por sí solas que haya empeorado la capacidad predictiva. Estos resultados no disparan reentrenamiento: los cuatro reportes registran `reentrena_modelo=false`.

- **p7:** estado moderado por drift moderado en `senthtml_1m` y `ctd_camptototrcnl06m`.
- **p8:** estado alto por `ctd_sms_received` (PSI ≈ 0.283); `senthtml_1m` y `ctd_camptototrcnl06m` quedan moderadas.
- **p9:** estado alto; `ctd_camptototrcnl06m`, `senthtml_1m` y `grp_campecs06m` son altas.
- **p10:** estado alto; `ctd_camptototrcnl06m` (PSI ≈ 2.252), `senthtml_1m` (≈ 1.946) y `grp_campecs06m` (≈ 0.261) son altas.

La regla agregada coincide con los niveles guardados en cada JSON. El PSI máximo de la probabilidad es ≈ 0.038 en p10.

## Evidencias incluidas

- `reports/`: los cuatro `monitoring_report.json` originales, uno por mes.
- `screenshots/mlflow_champion_v1.jpeg`: versión 1 del modelo con alias `champion`.
- `screenshots/mlflow_challenger_v2.jpeg`: versión 2 con alias `challenger`; evidencia de contexto del registro, no es el modelo usado en los reportes.
- `screenshots/airflow_inference_dag_run.jpeg`: vista del DAG de **inferencia** con tareas en `success`.
- `logs/log_scheduler.txt`: log completo del scheduler. Registra cuatro `DagRun Finished` de `automl_monitoring_pipeline`, todas en `state=success`; el extracto correspondiente está en `logs/airflow_monitoring_success_excerpt.txt`.
- `logs/git_log.txt`: commits de la rama temporal y de la documentación de evidencia.

## Evidencias aún necesarias para cerrar la revisión

La evidencia de Airflow sí está cubierta por `logs/log_scheduler.txt`: aparecen cuatro corridas exitosas del DAG `automl_monitoring_pipeline`. El log también conserva errores iniciales repetidos de lectura (`OSError` en `/opt/airflow/project`) anteriores a la recuperación del scheduler; las cuatro corridas exitosas posteriores quedan registradas.

Aún falta una captura o exportación de los runs de **monitoreo** en MLflow con las métricas PSI/KL. Las capturas MLflow incluidas muestran el registro del champion v1 y challenger v2, no los runs de monitoreo. Añadir esa evidencia antes de marcar el paso como completo en el PR.
