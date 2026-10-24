# Decisiones D1–D9

Evidencia: corrida real `auto-bde95c04df4aa4894f91` (muestra 10%, Docker/Airflow/MLflow, solo CPU). Resultados completos en `docs/resultados_corrida_10pct.md` y en `data/output/runs/auto-bde95c04df4aa4894f91/`.
Las pruebas sintéticas verifican comportamiento, no rendimiento del caso financiero.

| Decisión | Elección y justificación | Alternativa y compromiso | Evidencia (corrida real 10%) |
|---|---|---|---|
| D1 | Decay relativo 100*(ref-observado)/ref. Train→holdout para sobreajuste; holdout→ola para degradación temporal. Opción absoluta en puntos de AUC, umbrales en esa misma unidad. | No mezclar tipos ni referencias. Alerta de desempeño no prueba por sí sola data drift. | catb decay 1.80%; random_forest 15.13% y extra_trees 16.21% descartados (>10%); ridge −0.11%. Backtesting de catb sin alerta. |
| D2 | CPU portable por defecto; GPU opt-in con prueba real por backend y fallback. Modo sequential/parallel/benchmark configurable; un hilo por estimador. | Benchmark duplica entrenamiento y mide overhead; modo único reduce costo. Workers limitados por CPU, RAM requiere elección manual. | Modo parallel, 2 workers, solo CPU (16 hilos lógicos, sin GPU): entrenamiento final 401 s. Sin speed-up medido (no se corrió benchmark). |
| D3 | Una tarea por HPO de modelo; etapas compartidas para preparación, entrenamiento, backtesting y reporte. | Reintentar HPO no repite otros modelos; un fallo durante entrenamiento puede repetir el lote si aún no escribió el checkpoint. | 17 tareas: 9 HPO en paralelo y 8 etapas generales (captura en `docs/img/airflow_graph.png`). |
| D4 | Captura de positivos en top 10%, con lift y precisión complementarios. Capacidad de contacto es supuesto configurable. | Evita inventar márgenes/costos económicos; empates estables dependen del orden de filas, documentado. Scores no requieren calibración. | catb: captura top-10% 0.5257, lift 5.26, precisión 17.2% (tasa base ≈3.3%). Sin afirmar causalidad de campaña. |
| D5 | Optuna TPE, CV estratificada sobre entrenamiento; transformaciones dentro del Pipeline. GridSearch queda disponible. Presupuesto por trials/tiempo. | Mantiene P2; no es CV temporal ni anidada. Timeout modifica cantidad de trials y GPU no es bit a bit determinista. | 15 trials y 600 s por búsqueda. catb: 15 trials, 383.6 s, AUC CV 0.8515. gradient_boosting_sklearn solo 2 trials y adaboost 7 por el tiempo límite. |
| D6 | MLflow HTTP en contenedor, SQLite persistente y artefactos servidos; PostgreSQL separado para Airflow. Cliente/servidor MLflow 3.1.0. | SQLite adecuado para el pequeño grupo con concurrencia controlada; PostgreSQL sería preferible con muchos escritores. Versiones fijadas a la prueba, no afirmación de ser las más recientes. | Runs y tags en MLflow (capturas en `docs/img/mlflow_*.png`); imágenes construidas y ejecutadas en Docker Desktop. |
| D7 | Identidad por config/datos/commit/código efectivo/imagen; reusar runs sin eliminar, locks y marcadores. Etiqueta manual incompatible se rechaza. | Hash exacto implica I/O adicional. Filesystem local compartido; no promete exclusión distribuida. Artefactos completos se reutilizan, un reintento reanuda incompletos. | Dos ejecuciones del DAG con la misma configuración (`auto-bde95c04df4aa4894f91`): la repetición duró 3:14 frente a 1:39:40, reutilizó las etapas (`_SUCCESS.json`), no creó clave ni carpeta nuevas y añade un tag `airflow_attempt.*` en lugar de un run nuevo (`docs/evidencia_idempotencia.md`). |
| D8 | Filtros AUC/decay, luego captura top-k; empate práctico→AUC→tiempo→nombre. Selección solo en holdout p1–p6; todos los candidatos se evalúan después en p7–p10. | Se cambia P2 que incorporaba BT en selección para preservar evaluación temporal posterior. Sin candidato se reporta null y no se promueve. | Campeón catb (captura 0.5257); lgbm en empate práctico (0.5213). Versión 1 con alias `champion` (`promotion.json`, `champion.json`). |
| D9 | Producción: mensual tras llegada de datos/etiquetas suficientes; revisar degradación para decidir anticipar una corrida. Parcial manual schedule=null. | No implementar triggers automáticos; frecuencia debe contrastarse con desempeño real y demora de etiquetas. | Backtesting p7–p10 de catb: AUC 0.829–0.854 y captura 0.450–0.526 frente a 0.8554 y 0.5257 en holdout; sin alerta de decay. |

## Promoción

Versión idempotente con tag `version_key` en su creación; fuente es el run del modelo FINAL, no HPO.
Un candidato reemplaza al campeón si supera `registry.min_margin` en la métrica primaria, sobre una
misma identidad de evaluación (datos, semilla, split, muestra y definición de negocio). Si cambia el
conjunto evaluado, queda challenger y exige reevaluar ambos antes de promover; comparar números de
conjuntos distintos no es evidencia suficiente. Sin campeón previo se promueve el primer candidato válido.
Errores del servidor se propagan; no se interpretan como ausencia de campeón. Bloqueo local compartido.

## Monitoreo y honestidad de medición

CPU/RAM = proceso y descendientes. Conservar objetos Process evita reinicializar contadores de hijos.
Porcentaje CPU puede superar 100% al utilizar varios núcleos; `cpu_cores_equiv_avg/max` expresa núcleos
lógicos equivalentes de utilización, no identidad de núcleos físicos. `cpu_cores_logical` es capacidad;
`estimator_threads=1` es el límite configurado. GPU = dispositivo global y puede incluir otros procesos.
`execution_device` proviene del backend verificado; disponibilidad GPU no implica uso en una tarea CPU.
Muestreo puede omitir picos entre intervalos; RSS agregado puede contar páginas compartidas más de una vez.
Reporte incluye su propia fase de armado; excluye su subida final de logs y escritura del marcador.

## Cambios preservados

Nueve modelos, semilla, división 6/4, muestreo estratificado, pipelines de preprocesamiento, Optuna,
validación de fechas, ROC, Docker y alias. El entrenamiento normal no compara secuencial/paralelo salvo
modo benchmark. La ruta local principal ahora usa las mismas etapas de P3; la antigua está en main_p2.py.

## Evidencia pendiente de adjuntar

Capturas de Airflow y MLflow en `docs/img/`, URL del repositorio y PRs, y el commit exacto de la entrega (la corrida se ejecutó con cambios sin commit: `git_dirty=true` en `identity.json`).
