# ============================================================================
# Paquete src — Módulos del sistema AutoML para Venta Financiera
# ============================================================================
# Este paquete contiene los módulos principales del pipeline AutoML de dos
# rondas para predicción de venta financiera (clasificación binaria):
#
#   - preprocessing.py : Carga y concatenación de CSVs particionados,
#                        muestreo estratificado, limpieza de columnas,
#                        imputación de nulos, encoding (OHE + LabelEncoding),
#                        conversión de tipos y split estratificado 70/30.
#
#   - training.py      : Definición de 9 modelos base (5 familias) con manejo
#                        de desbalance de clases (scale_pos_weight,
#                        class_weight='balanced', auto_class_weights).
#                        Ronda 1: entrena los 9 modelos, calcula AUC y decay.
#                        Ronda 2: selecciona los top 4 por AUC test (filtrados
#                        por decay <= 10% y AUC >= 0.55), evalúa estabilidad
#                        con validación cruzada estratificada 5-fold.
#                        Selección del campeón: mayor cv_auc_mean (desempate
#                        por menor cv_auc_std). Fallback: GetBestModel()
#                        con score compuesto si no hay candidatos en R2.
#
#   - evaluation.py    : Cálculo de métricas (AUC train/test, decay %),
#                        validación cruzada estratificada 5-fold para medir
#                        estabilidad (cv_auc_mean, cv_auc_std), y medición
#                        de tiempos de entrenamiento.
#
#   - registry.py      : Persistencia de artefactos: serialización de modelos
#                        (.pkl), metadata individual (.json) con métricas y
#                        versiones de librerías, y resumen consolidado
#                        (results_summary.json) por corrida con timestamp.
# ============================================================================
