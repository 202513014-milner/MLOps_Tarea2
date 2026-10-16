# AutoML — Parcial de orquestación

Continuación de P2 y del avance `automl_p3_mlops_revisado_por_kim.zip`. Mantiene nueve modelos,
Optuna/GridSearch, preprocesamiento dentro de CV, Docker/Airflow/MLflow y captura top-k.
**Integrar en el repositorio de P2 conservando `.git`; no crear otro repositorio.**

## Inicio en Windows (16 GB RAM)

1. Abrir Docker Desktop (contenedores Linux/WSL2).
2. Copiar el contenido de esta carpeta a una rama del repositorio P2. Conservar `.git` y los CSV locales.
   Se renombró el DAG antiguo a `dags/training_dag.py`: eliminar `dags/automl_training_dag.py` si quedó de una copia anterior.
3. Colocar `p1_extrac.csv` ... `p10_extrac.csv` en `data/input/`. Dataset del curso:
   https://drive.google.com/drive/folders/1IKpzgvN9g9SxBO-dOGoWHmc_E-worDXU
   No subir los CSV a Git. Se exige `p_fecinformacion` YYYYMMDD y diez meses consecutivos.
4. Desde la carpeta que contiene `docker-compose.yml`:

```powershell
Copy-Item .env.example .env
python scripts/up.py
```

`up.py` registra commit/rama/imagen y ejecuta build/up. Alternativa:

```powershell
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail 80 airflow-init airflow-scheduler mlflow
```

Airflow: http://localhost:8080 (`admin` / `admin`, entorno académico local).
MLflow: http://localhost:5000. PostgreSQL es interno para Airflow; MLflow usa SQLite persistente.
No instalar estas aplicaciones en Windows por separado. La construcción descarga paquetes/imágenes y requiere internet.

## Primera corrida pequeña

En Airflow abrir `automl_training_pipeline` → Trigger DAG with config. Usar los parámetros:

```json
{"config_path":"config/smoke.yaml","run_key":"","force":false}
```

El perfil smoke usa 1% de muestra, 2 trials, ejecución secuencial y CPU. El pool HPO tiene 1 slot
por defecto en `config/config.yaml`. Verificar suficientes positivos para CV. El esquema/nulos/fechas
se validan en los archivos completos; la muestra pequeña reduce entrenamiento, no la lectura inicial.
No usar resultados smoke como métricas finales. Perfil normal: `config/config.yaml` (10%, 15 trials,
600 segundos por búsqueda; un trial en curso puede exceder el timeout). El presupuesto por tiempo
puede producir diferente cantidad de trials entre máquinas.

Para cambiar concurrencia del pool editar `airflow.pool_slots` y ejecutar:

```powershell
docker compose exec airflow-scheduler python scripts/configure_pool.py
```

`parallel.mode`: `sequential`, `parallel` o `benchmark`. Benchmark entrena en ambos modos y conserva
ambos artefactos, runs y recursos; el modelo de modo parallel es el candidato final. Un modo único
no tiene speed-up y lo reporta como null. `parallel.workers: 1` está permitido; `-1` usa CPU disponibles.
Para esta laptop empezar con 1 slot HPO y 1–2 workers, sin aumentar memoria de Docker inicialmente.

## GPU opcional

CPU es el valor inicial portable (`use_gpu: false`). Para probar GPU:

```powershell
python scripts/up.py --gpu
```

Activar además `use_gpu: true` en el perfil usado. La reserva GPU del Compose solo expone el hardware;
los backends se prueban con un fit. Si un backend no funciona, queda en CPU. El HPO se mantiene en CPU.
El tag `execution_device` identifica el dispositivo usado. La telemetría GPU es global del dispositivo,
no consumo exclusivo de un modelo. GPU/Windows/WSL2 aún deben comprobarse en la laptop.

## Ejecución local conservada

En Python compatible (se recomienda 3.11 para igualar la imagen):

```powershell
python -m pip install -r requirements.txt
python main.py --input data/input --config config/smoke.yaml
```

Esto ejecuta el pipeline P3 completo sin Airflow, con las mismas funciones de etapas.
Sin `MLFLOW_TRACKING_URI` utiliza SQLite local bajo `data/mlflow`; si se desea el servidor Docker,
configurar `$env:MLFLOW_TRACKING_URI="http://localhost:5000"`.
`python main.py --legacy --config config/config.yaml` conserva la ruta modular P2 y sus switches;
**no es la ruta de entrega P3**, porque no ejecuta el tracking/orquestación completos.
`main_p2.py` conserva la entrada previa; `src/parallelism.py` es un alias compatible de `src/parallel.py`.

## Qué se ejecuta

Validar → preparar → HPO por modelo → consolidar HPO → entrenar/benchmark → seleccionar en holdout
→ evaluar TODOS los artefactos en p7–p10 → registrar/promover → reporte.
XCom contiene solo identificadores. Los datasets/modelos viven en disco compartido/MLflow.
Cada modelo HPO y cada entrenamiento final tiene su run, parámetros, AUC/decay/negocio, modelo, ROC,
backtesting de cuatro olas y cómputo. Los modelos finales no se reentrenan en backtesting.
Si no hay candidato admisible, el pipeline informa `champion: null`, evalúa los candidatos y no promueve.

## Idempotencia (evidencia que deben capturar)

1. Ejecutar con `run_key` vacío y anotar carpeta `auto-*`, IDs y número de runs en MLflow.
2. Volver a disparar el DAG con idénticos parámetros, datos, código e imagen.
3. Todas las etapas completas se reutilizan. Los IDs y número de runs activos e históricos no cambian;
   se añaden tags de intento Airflow y se conserva el identificador de origen.
4. Una tarea incompleta reanuda el mismo run. No se borran runs previos.
5. Cambiar los datos/config/código/imagen produce otra clave automática y conserva la anterior.
   Si se fuerza una etiqueta manual para una identidad distinta, el programa se detiene: usar otra etiqueta.

`force` no destruye resultados completos: permite intentar de nuevo lo incompleto. Para un nuevo
experimento intencional, usar otra etiqueta manual o cambiar la configuración. Los locks requieren
un filesystem compartido (el volumen local del Compose); un despliegue distribuido requeriría locks externos.
No editar código/CSV durante una corrida; la protección de identidad detiene la ejecución si detecta cambios.
La clave incluye un hash del código efectivo además de Git, para detectar modificaciones sin commit.

## Resultados y evidencia

En `data/output/runs/<clave>/`:
- `results_summary.json`: métricas de los nueve modelos finales, tamaño/hash del MISMO artefacto,
  backtesting, selección, promoción y trazabilidad.
- `compute_logs.json`: recursos por etapa (incluida generate_report), HPO y entrenamiento/modo.
- `report.md`: tablas de modelos, 36 filas de backtesting final y recursos por entrenamiento.
- Subdirectorios por etapa, modelos y datos intermedios. MLflow conserva los artefactos pesados.

Copiar las tablas de la corrida final de `report.md` aquí antes de entregar. Añadir capturas reales
(DAG completo, comparación MLflow, idempotencia) en `docs/img/`. Conservar una carpeta de corrida con
sus JSON/tablas/gráficos; `.gitignore` excluye datasets, pickles y bundles MLflow. No publicar credenciales.
Los logs de Airflow, PostgreSQL y MLflow persisten en volúmenes. `docker compose down` los conserva;
**no usar `down -v` si quieren conservar experimentos**.

No se incluyen resultados del curso ni capturas fabricadas. Ver [VALIDACION.md](VALIDACION.md) para
el alcance exacto de las pruebas realizadas y [docs/decisiones_diseno.md](docs/decisiones_diseno.md).

## Pruebas

```powershell
python tests/test_integracion.py
python tests/test_contratos.py
```

La integración crea datos temporales sintéticos, prueba los nueve modelos, ambos modos, backtesting
sin fit, artefactos, reejecución, registro y protección ante cambios de datos. Requiere dependencias locales
completas o un entorno de pruebas con servidor MLflow. Ver límites de plataforma en VALIDACION.md.

## Git y entrega

Guardar P2 con tag `v2.0-p2`; una rama por contribución, commits reales de cada integrante, PR/merge a main.
Tag `v3.0-parcial` solo después de validar. El repo sigue privado, con acceso docente `manassesmauricio-cloud`.
La entrega es la URL del mismo repo P2, no este ZIP. Este paquete no incluye ni inventa historial Git.

Los notebooks heredados son material de P2; el reporte de P3 se consulta en data/output/runs/<clave>/report.md.
