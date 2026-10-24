# Validación realizada

Fecha: 15 de octubre de 2026. Entorno local Linux, Python 3.12.14.

## Comprobado

- Integración CPU de las nueve etapas, nueve modelos y MLflow 3.1.0 real con SQLite.
- 35 runs: 9 modelos HPO, 18 variantes finales (sequential/parallel) y 8 etapas generales.
- 27 artefactos de modelo evaluados en cuatro olas cada uno, con modelos/ROC/metadata/backtesting/cómputo en MLflow.
- Prohibición de fit en backtesting verificada mediante mock que lanzaría excepción ante cualquier fit.
- 9 modelos en el reporte final, mismo artefacto para métricas y tamaño, recursos por entrenamiento.
- Segunda ejecución conserva exactamente el conjunto de IDs, incluyendo runs históricos (sin borrados).
- Reejecución de registro tras quitar marcador conserva la versión del campeón.
- Datos modificados cambian clave automática y una etiqueta manual incompatible se rechaza.
- 5 pruebas de contratos pasaron: métricas conocidas, sin campeón, falso uso GPU, fechas/archivo ausente y sintaxis.
- YAML parseable y resolución pip --dry-run de dependencias directas + Airflow 2.10.5 sin conflictos en el intérprete local.

Resultado compacto: docs/verification/integration_results.json.

## Límites explícitos

**El benchmark de integración se ejecutó con TEST_WORKERS=1.** Cubre ambas rutas y la persistencia,
pero no demuestra aceleración ni concurrencia multiproceso. Se intentó con 2 workers y este runtime
no permite a psutil resolver los PID de los procesos loky (NoSuchProcess). No se desactivó el monitoreo
para ocultarlo. En Windows/Docker deben ejecutar la prueba con 2 workers (valor predeterminado).

No hay Docker daemon disponible aquí: no se construyeron imágenes, no se inició scheduler/webserver
ni se ejecutó el DAG real. La prueba pip local (Python 3.12.14) no sustituye construir la imagen Python 3.11.
Tampoco se comprobó GPU ni el dataset real del curso. No se incluyen capturas o cifras del curso inventadas.

## Cómo repetir

```powershell
python -m pip install -r requirements.txt
python tests/test_contratos.py
python tests/test_integracion.py
```

La integración usa por defecto 2 workers y datos temporales. Para diagnóstico secuencial:

```powershell
$env:TEST_WORKERS="1"
python tests/test_integracion.py
Remove-Item Env:TEST_WORKERS
```

## Gate de entrega pendiente en la laptop

1. Build/up con Docker y servicios healthy; revisar logs e importación del DAG.
2. Corrida completa con config/config.yaml y reejecución con idénticos parámetros (hecho: ver docs/evidencia_idempotencia.md).
3. Probar 2 workers y, si se desea, GPU con backend realmente usado.
4. Corrida con CSV reales y parámetros finales; copiar tablas y capturas al repositorio.
5. Confirmar acceso docente, ramas/merges/tags y URL del repositorio P2.

No confundir aprobación de pruebas sintéticas con cumplimiento de evidencia de ejecución real.
