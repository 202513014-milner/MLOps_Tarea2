# Inferencia mensual P3 (parte de Kim)

Rama `feature/p3-kim`, creada desde `refactor/limpieza-p3`. Incluye **solo** la inferencia mensual de p7 a p10,
el cálculo del TLV, la asignación de grupos de ejecución y el DAG de inferencia. El monitoreo (Milner) y las
validaciones de resultados (Rosabel) van en sus propias ramas.

## Qué agrega

| Archivo | Contenido |
|---|---|
| `src/inference.py` | Tareas del mes: `cargar_datos`, `preprocesar_datos`, `generar_scores`, `posprocesar`, `registrar_inferencia`. |
| `src/posprocessing.py` | `puntuar` (TLV) y `grupos_mes` (grupo de ejecución 1-10). |
| `dags/inference_dag.py` | DAG `automl_inference_pipeline`, un mes por ejecución. |
| `config/config.yaml` | Secciones `inference` y `postprocessing`. |
| `tests/test_inferencia.py` | 8 pruebas sin Docker, Airflow ni MLflow (champion falso). |

## Cómo correrlo

```bash
airflow dags trigger automl_inference_pipeline --conf '{"mes": "p7"}'   # p7, p8, p9 o p10
```

Entrada: `data/input/<mes>_extrac.csv`. El champion no se reentrena: se carga del Model Registry por el alias
configurado en `registry` y se rechaza si no es el algoritmo de `inference.expected_algorithm`.

```bash
python -m pytest tests/test_inferencia.py
```

## Qué hace cada paso

1. **cargar_datos**: lee el mes, descarta el target, valida columnas y fija la carpeta de la corrida.
2. **preprocesar_datos**: aplica el preprocesamiento ya ajustado del champion.
3. **generar_scores**: `predict_proba` del estimador; falla si algún score no es finito o sale de [0, 1].
4. **posprocesar**: calcula TLV y grupo de ejecución.
5. **registrar_inferencia**: run de MLflow (experimento `<exp>-inference`) con tags de modelo, mes y datos.

TLV = `prob × prob_value_contact × ln(monto + 1) × prob_frescura`, con `prob_frescura` según `grp_campecs06m`
(`postprocessing.freshness_prob`). Nulos de contacto se rellenan con `contact_fill`; monto nulo cuenta como 0.

## Salidas (contrato para monitoreo y validación)

Carpeta: `data/output/inference/<mes>/<clave>/`. La clave es un hash de mes, datos, versión del modelo y la
sección `postprocessing`: la misma entrada, modelo y configuración reutilizan carpeta y run de MLflow (idempotente).

| Archivo | Contenido |
|---|---|
| `scores_clientes.csv` | `codunicocli`, `prob`, `puntuacion_tlv`, `grupo_ejec_tlv` |
| `resumen_grupos.csv` | `grupo_ejec_tlv`, `clientes`, `pct` |
| `compute_logs.json` | tiempo, CPU y RAM por tarea |
| `inference_context.json` | mes, clave, versión del modelo, `inference_run_id`; punto de entrada para monitorear o validar |

Convención: `grupo_ejec_tlv` 1 = 5 % de mayor TLV y 10 = 3,5 % de menor TLV (cortes `postprocessing.dist_ge`).

## Fuera de esta rama

- Monitoreo (incluidos los deciles del score con cortes fijos de la referencia), PSI/KL, drift, alertas y DAG de monitoreo: rama de Milner. Necesita su propia sección `monitoring` en la config.
- Validaciones de probabilidades, TLV, deciles, integridad e idempotencia sobre estas salidas: rama de Rosabel.

## Límites

Las pruebas usan un champion falso y datos sintéticos. No se ejecutó el DAG en Airflow real (se comprobó su estructura con
un Airflow simulado) ni se corrió con el champion CatBoost registrado ni con los CSV reales p7 a p10.
