# Correcciones sobre la última versión de Kimberly

- Reutiliza la arquitectura, los nueve modelos, HPO y preprocesamiento originales.
- Modelos HPO y cada variante de entrenamiento tienen run propio: modelo/ROC/HP/métricas/cómputo/BT.
- Evalúa los 27 artefactos si se usa benchmark (9 HPO + 9 secuenciales + 9 paralelos); 18 en modo único.
  Todos usan idéntica muestra de cada ola; no hay fit ni reselección en backtesting.
- Monitoreo dentro de cada fit. GPU disponible/uso real separados. Reportes con tamaños del modelo final.
- Modos sequential/parallel/benchmark, workers=1 y CPU/GPU configurables.
- Reanuda runs y conserva historial; claves automáticas, locks, rechazo de etiquetas incompatibles.
- Registro usa artefacto final, tags de versión atómicos y comparación en mismo conjunto de evaluación.
- Sin campeón válido: backtesting/reporte continúan; se omite promoción.
- Reporte incluye recursos de generate_report, todas las olas y trazabilidad.
- Entrada local P3 comparte etapas con DAG; ruta P2 preservada mediante --legacy.
- Validación de diez meses consecutivos, fechas decimales y objetivo binario.
- Dependencias directas fijadas; cliente/servidor MLflow alineados; SQLAlchemy servidor/local fijado por incompatibilidad detectada.
- Pruebas antiguas redirigidas a integración vigente; documentación actualizada sin afirmar validación Docker/GPU.

Ver VALIDACION.md para evidencia real de pruebas. No se modificaron repositorios remotos ni se inventaron commits.
