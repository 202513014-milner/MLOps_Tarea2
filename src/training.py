"""
=============================================================================
Modulo de Entrenamiento — training.py
=============================================================================
Responsabilidad:
    Definir el diccionario de modelos, entrenar cada uno, calcular metricas,
    registrar metadata y seleccionar el modelo campeon en DOS RONDAS:

    RONDA 1 — Entrenamiento base (9 modelos):
        Entrena los 9 modelos con AUC-ROC y calcula el decay (sobreajuste).
        Filtra candidatos por: decay <= 10% y AUC test >= 0.55.

    RONDA 2 — Validacion cruzada (top 4 modelos):
        De los candidatos que pasaron los filtros de Ronda 1, se toman los
        TOP 4 por AUC test y se evalua su estabilidad con 5-fold Stratified
        Cross-Validation. El campeon se elige por mayor cv_auc_mean.

    CRITERIO: Solo los 4 mejores modelos pasan a Ronda 2 porque evaluar
    CV es costoso computacionalmente (entrena 5 veces por modelo) y los
    modelos con AUC test mas bajo tienen menor probabilidad de ser campeon.

Modelos implementados:
    === Modelos base del curso (3) — ya existian en el notebook ===
    1. XGBoost (xgb): Gradient boosting con regularizacion L1/L2.
    2. LightGBM (lgbm): Gradient boosting con histogram-based splitting.
    3. CatBoost (catb): Gradient boosting con manejo nativo de categoricas.

    === Modelos adicionales (6) — agregados para esta practica ===
    4. Regresion Logistica: modelo lineal con funcion sigmoide.
    5. Ridge Classifier: regresion lineal regularizada L2 con umbral.
    6. Random Forest: ensamble de arboles con bagging.
    7. AdaBoost: boosting adaptativo sobre arboles debiles.
    8. Gradient Boosting (sklearn): gradient boosting nativo de sklearn.
    9. Extra Trees: variante de Random Forest con splits aleatorios.

    Total: 9 modelos (3 base + 6 adicionales).

Manejo de desbalance de clases:
    El dataset tiene ~3.4% de clase positiva (desbalanceado). Para evitar
    que los modelos ignoren la clase minoritaria, se aplica ponderacion:
    - XGBoost y LightGBM: scale_pos_weight = ratio_desbalance
      (ratio = negativos / positivos = 28)
    - CatBoost: auto_class_weights='Balanced'
    - Sklearn (LR, RF, ExtraTrees): class_weight='balanced'
    Tecnica aprendida del notebook de clase 1 del profesor.

Arquitectura del diccionario de modelos:
    El AutoML usa un diccionario Python donde cada clave es el nombre corto
    del modelo y el valor es otro diccionario con:
    - 'model': instancia del estimador (sin entrenar).
    - 'params': hiperparametros (vacio = default).
    - 'performance': metricas (se llena despues del entrenamiento).
    - 'library_versions': versiones de librerias.
    - 'fit_kwargs': argumentos extra para el metodo fit().

    Este patron permite agregar un nuevo modelo con solo agregar una entrada
    al diccionario.

Librerias utilizadas y por que:
    - xgboost: implementacion optimizada de gradient boosting.
    - lightgbm: gradient boosting con tecnica "histogram-based".
    - catboost: gradient boosting de Yandex. Manejo nativo de categoricas.
    - scikit-learn (sklearn): libreria estandar de ML en Python.
    - numpy: operaciones vectorizadas sobre arrays numericos.
    - pandas: se importa para versioning.
=============================================================================
"""

# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np          # Operaciones numericas sobre arrays
import time                 # Medicion de tiempos de ejecucion

# --- Librerias de modelos ---
# Modelos base del curso (gradient boosting especializado)
import xgboost as xgb       # XGBoost: gradient boosting con regularizacion L1/L2
import lightgbm as lgb       # LightGBM: gradient boosting con histogram splitting
import catboost as catb       # CatBoost: gradient boosting de Yandex

# Modelos adicionales de scikit-learn (3 familias de algoritmos)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,       # Ensamble de arboles con bagging
    AdaBoostClassifier,           # Boosting adaptativo sobre stumps
    GradientBoostingClassifier,   # Gradient boosting nativo de sklearn
    ExtraTreesClassifier,         # Arboles con splits extra-aleatorios
)

# Modulos de soporte
import sklearn               # Se importa para obtener version (versioning)
import pandas as pd           # Se importa para obtener version (versioning)
import warnings
warnings.filterwarnings('ignore')  # Suprimir FutureWarnings de sklearn


# =============================================================================
# DETECCION DE GPU
# =============================================================================
# Se detecta GPU CUDA al inicio para configurar XGBoost y CatBoost
# automaticamente. Si hay GPU, se usa device='cuda' (XGB) y
# task_type='GPU' (CatBoost) para acelerar el entrenamiento.
# =============================================================================

def _detectar_gpu() -> bool:
    """
    Detecta si hay una GPU CUDA disponible para entrenamiento.

    Intenta dos metodos en orden de prioridad:
    1. PyTorch (torch.cuda.is_available) — mas confiable
    2. nvidia-smi (comando del sistema) — fallback

    Retorna
    -------
    bool
        True si se detecta GPU CUDA disponible, False en caso contrario.

    Criterio: Se detecta GPU para que XGBoost use device='cuda' y
    CatBoost use task_type='GPU', acelerando el entrenamiento.
    En Colab con GPU T4, el entrenamiento es ~10x mas rapido.
    """
    # Metodo 1: PyTorch — es el mas confiable porque verifica el driver CUDA
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[GPU] GPU detectada via PyTorch: {gpu_name}")
            return True
    except ImportError:
        pass  # PyTorch no instalado, intentar siguiente metodo

    # Metodo 2: nvidia-smi — comando del sistema que lista GPUs NVIDIA
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print("[GPU] GPU detectada via nvidia-smi")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass  # nvidia-smi no disponible

    print("[GPU] No se detecto GPU CUDA. Usando CPU.")
    return False


# Ejecutar deteccion de GPU al importar el modulo (se ejecuta una sola vez)
GPU_DISPONIBLE = _detectar_gpu()


# =============================================================================
# IMPORTS INTERNOS DEL PROYECTO
# =============================================================================
# Se importan despues de la deteccion de GPU para que el modulo cargue
# correctamente incluso si la deteccion tarda unos segundos.
# =============================================================================

from src.evaluation import calcular_metricas, medir_tiempo_entrenamiento, calcular_cv_stability
from src.registry import (
    crear_directorio_corrida,     # Crea carpeta con timestamp para esta corrida
    guardar_modelo,               # Serializa modelo a .pkl
    guardar_metadata,             # Guarda metricas + hiperparametros a .json
    guardar_resumen_consolidado,  # Guarda results_summary.json con todos los modelos
)


# =============================================================================
# DEFINICION DE MODELOS
# =============================================================================

def definir_modelos(random_seed: int, ratio_desbalance: int) -> dict:
    """
    Define el diccionario con los 9 modelos a entrenar en Ronda 1.

    Parametros
    ----------
    random_seed : int
        Semilla para reproducibilidad (viene del config.yaml).
    ratio_desbalance : int
        Ratio negativos/positivos para ponderacion de clases.
        Se calcula como: (y==0).sum() / (y==1).sum() = 28.
        Se aplica como:
        - scale_pos_weight en XGBoost y LightGBM
        - auto_class_weights='Balanced' en CatBoost
        - class_weight='balanced' en modelos sklearn

    Retorna
    -------
    dict
        Diccionario de 9 modelos con la estructura:
        {
            'nombre_modelo': {
                'model': instancia_sin_entrenar,
                'params': {},           # Se llena tras entrenar
                'performance': {},      # Se llena tras evaluar
                'library_versions': {}, # Versiones para trazabilidad
                'fit_kwargs': {},       # Args extra para fit()
            }
        }

    Hiperparametros elegidos y justificacion
    -----------------------------------------
    MODELOS BASE (del notebook del curso):
    - XGBoost: eval_metric='logloss' (funcion de perdida estandar para
      clasificacion binaria). scale_pos_weight=ratio para desbalance.
    - LightGBM: parametros default + scale_pos_weight=ratio.
    - CatBoost: auto_class_weights='Balanced' (equivalente CatBoost).

    MODELOS ADICIONALES:
    - Logistic Regression: C=1.0, max_iter=2000, solver='lbfgs'.
    - Ridge Classifier: alpha=1.0 (regularizacion L2 default).
    - Random Forest: 200 arboles, max_depth=10, class_weight='balanced'.
    - AdaBoost: 100 estimadores, learning_rate=0.1.
    - Gradient Boosting sklearn: 100 arboles, max_depth=4, subsample=0.8.
    - Extra Trees: 200 arboles, max_depth=10, class_weight='balanced'.
    """

    models = {
        # =================================================================
        # MODELOS BASE DEL CURSO (XGBoost, LightGBM, CatBoost)
        # Estos 3 ya estaban en el notebook de la clase S3.
        # Se mantienen con la configuracion del notebook base + ponderacion
        # de clases para manejar el desbalance del dataset.
        # =================================================================

        # --- XGBoost ---
        # Gradient boosting con regularizacion L1/L2.
        # eval_metric='logloss': funcion de perdida para clasificacion binaria.
        # scale_pos_weight: pondera la clase positiva (minoritaria) para que
        # el modelo no la ignore durante el entrenamiento.
        # device='cuda': se activa automaticamente si hay GPU disponible.
        'xgb': {
            'model': xgb.XGBClassifier(
                eval_metric='logloss',           # Perdida binaria (log loss)
                random_state=random_seed,        # Semilla para reproducibilidad
                n_jobs=-1,                       # Usar todos los cores de CPU
                verbosity=0,                     # Suprimir output de XGBoost
                scale_pos_weight=ratio_desbalance,  # Ponderacion de clase positiva
                **({'device': 'cuda'} if GPU_DISPONIBLE else {}),  # GPU si disponible
            ),
            'params': {},
            'performance': {},
            'library_versions': {
                'xgboost': xgb.__version__,
            },
            'fit_kwargs': {},
        },

        # --- LightGBM ---
        # Gradient boosting con histogram-based splitting.
        # Mas rapido que XGBoost para datasets grandes porque agrupa
        # valores continuos en bins (O(n_bins) en lugar de O(n_data)).
        # scale_pos_weight: misma tecnica de ponderacion que XGBoost.
        'lgbm': {
            'model': lgb.LGBMClassifier(
                random_state=random_seed,        # Semilla para reproducibilidad
                verbose=-1,                      # Suprimir output de LightGBM
                n_jobs=-1,                       # Usar todos los cores de CPU
                scale_pos_weight=ratio_desbalance,  # Ponderacion de clase positiva
            ),
            'params': {},
            'performance': {},
            'library_versions': {
                'lightgbm': lgb.__version__,
            },
            'fit_kwargs': {},
        },

        # --- CatBoost ---
        # Gradient boosting de Yandex con manejo nativo de categoricas.
        # auto_class_weights='Balanced': el equivalente CatBoost de
        # class_weight='balanced'. Calcula pesos automaticamente.
        # task_type='GPU': se activa si hay GPU disponible.
        'catb': {
            'model': catb.CatBoostClassifier(
                verbose=0,                       # Suprimir output de CatBoost
                random_seed=random_seed,         # Semilla para reproducibilidad
                auto_class_weights='Balanced',   # Ponderacion automatica de clases
                **({'task_type': 'GPU'} if GPU_DISPONIBLE else {}),  # GPU si disponible
            ),
            'params': {},
            'performance': {},
            'library_versions': {
                'catboost': catb.__version__,
            },
            'fit_kwargs': {},
        },

        # =================================================================
        # MODELOS ADICIONALES (6 modelos de 3 familias de algoritmos)
        # Se agregan para cubrir las familias sugeridas por el profesor:
        # modelos lineales, ensambles de arboles y boosting adaptativo.
        # =================================================================

        # --- Familia 1: Modelos Lineales ---
        # Sirven como baseline interpretable. Si un modelo lineal tiene
        # buen AUC, las relaciones en los datos son mayormente lineales.

        # Regresion Logistica: modelo lineal con funcion sigmoide.
        # class_weight='balanced': ajusta pesos de clase inversamente
        # proporcional a su frecuencia en el dataset.
        # max_iter=2000: iteraciones suficientes para converger con 60+ features.
        # solver='lbfgs': algoritmo quasi-Newton, eficiente para L2.
        'logistic_regression': {
            'model': LogisticRegression(
                C=1.0,                           # Inverso de fuerza de regularizacion
                max_iter=2000,                   # Iteraciones para convergencia
                solver='lbfgs',                  # Quasi-Newton (eficiente para L2)
                random_state=random_seed,        # Semilla para reproducibilidad
                class_weight='balanced',         # Ajuste por desbalance de clases
            ),
            'params': {},
            'performance': {},
            'library_versions': {
                'scikit-learn': sklearn.__version__,
            },
            'fit_kwargs': {},
        },

        # Ridge Classifier: regresion lineal regularizada L2 con umbral.
        # alpha=1.0: fuerza de regularizacion L2 (default de sklearn).
        # NOTA: No tiene predict_proba nativo. Se usa decision_function
        # para calcular AUC (evaluation.py lo detecta automaticamente).
        'ridge_classifier': {
            'model': RidgeClassifier(
                alpha=1.0,                       # Regularizacion L2
                class_weight='balanced',         # Ajuste por desbalance de clases
            ),
            'params': {},
            'performance': {},
            'library_versions': {
                'scikit-learn': sklearn.__version__,
            },
            'fit_kwargs': {},
        },

        # --- Familia 2: Ensambles de Arboles ---
        # Complementan a XGB/LGBM/CatBoost con diferentes estrategias.

        # Random Forest: ensamble de arboles con bagging (muestreo con
        # reemplazo). Cada arbol ve una muestra diferente del dataset.
        # n_estimators=200: mas arboles = mejor generalizacion.
        # max_depth=10: limitar profundidad reduce overfitting.
        'random_forest': {
            'model': RandomForestClassifier(
                n_estimators=200,                # 200 arboles en el bosque
                max_depth=10,                    # Profundidad maxima (anti-overfitting)
                random_state=random_seed,        # Semilla para reproducibilidad
                n_jobs=-1,                       # Paralelismo en todos los cores
                class_weight='balanced',         # Ajuste por desbalance de clases
            ),
            'params': {},
            'performance': {},
            'library_versions': {
                'scikit-learn': sklearn.__version__,
            },
            'fit_kwargs': {},
        },

        # AdaBoost: boosting adaptativo sobre arboles debiles (stumps).
        # Cada estimador se enfoca en los ejemplos que el anterior clasifico
        # mal. No tiene class_weight nativo.
        # n_estimators=100: numero de estimadores debiles.
        # learning_rate=0.1: tasa conservadora (mejor generalizacion).
        'adaboost': {
            'model': AdaBoostClassifier(
                n_estimators=100,                # 100 stumps secuenciales
                learning_rate=0.1,               # Tasa conservadora
                random_state=random_seed,        # Semilla para reproducibilidad
            ),
            'params': {},
            'performance': {},
            'library_versions': {
                'scikit-learn': sklearn.__version__,
            },
            'fit_kwargs': {},
        },

        # Gradient Boosting sklearn: implementacion nativa de sklearn.
        # Sirve como comparacion directa contra XGBoost/LightGBM.
        # subsample=0.8: stochastic gradient boosting (muestrea 80%
        # de los datos en cada etapa, reduce overfitting).
        'gradient_boosting_sklearn': {
            'model': GradientBoostingClassifier(
                n_estimators=100,                # 100 etapas de boosting
                max_depth=4,                     # Profundidad de cada arbol base
                learning_rate=0.1,               # Tasa de aprendizaje
                subsample=0.8,                   # Stochastic GB (80% de datos)
                random_state=random_seed,        # Semilla para reproducibilidad
            ),
            'params': {},
            'performance': {},
            'library_versions': {
                'scikit-learn': sklearn.__version__,
            },
            'fit_kwargs': {},
        },

        # Extra Trees: variante de Random Forest con splits aleatorios.
        # La diferencia clave: Extra Trees NO busca el mejor split,
        # elige splits al azar, lo que lo hace mas rapido y a veces
        # mas robusto al overfitting que Random Forest.
        'extra_trees': {
            'model': ExtraTreesClassifier(
                n_estimators=200,                # 200 arboles
                max_depth=10,                    # Profundidad maxima
                random_state=random_seed,        # Semilla para reproducibilidad
                n_jobs=-1,                       # Paralelismo en todos los cores
                class_weight='balanced',         # Ajuste por desbalance de clases
            ),
            'params': {},
            'performance': {},
            'library_versions': {
                'scikit-learn': sklearn.__version__,
            },
            'fit_kwargs': {},
        },

    }

    # Agregar versiones comunes a todos los modelos (trazabilidad MLOps)
    # Esto permite saber con que versiones exactas se entreno cada modelo,
    # lo cual es critico para reproducibilidad en produccion.
    common_versions = {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'python': __import__('sys').version.split()[0],
    }
    for ml_name in models:
        models[ml_name]['library_versions'].update(common_versions)

    return models


# =============================================================================
# ENTRENAMIENTO DE TODOS LOS MODELOS (RONDA 1)
# =============================================================================

def entrenar_todos_los_modelos(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    run_dir: str
) -> dict:
    """
    Entrena cada modelo del diccionario, calcula metricas y guarda artefactos.

    Esta funcion se usa en la RONDA 1 para entrenar los 9 modelos base.

    Parametros
    ----------
    models : dict
        Diccionario de modelos retornado por definir_modelos().
    X_train, y_train : np.ndarray
        Datos de entrenamiento.
    X_test, y_test : np.ndarray
        Datos de test.
    run_dir : str
        Directorio de la corrida para guardar artefactos (.pkl, .json).

    Retorna
    -------
    dict
        El mismo diccionario de modelos, ahora con 'performance' y 'params'
        rellenados con las metricas y los hiperparametros usados.

    Flujo por cada modelo
    ---------------------
    1. Entrenar con medir_tiempo_entrenamiento() -> tiempo en segundos.
    2. Calcular metricas con calcular_metricas() -> AUC train, AUC test, decay.
    3. Guardar modelo serializado (.pkl) con guardar_modelo().
    4. Guardar metadata (.json) con guardar_metadata().
    5. Imprimir resumen en consola.

    Manejo de errores
    -----------------
    Si un modelo falla (ej: no converge, se queda sin memoria), se captura
    la excepcion, se reporta el error y se continua con el siguiente.
    Criterio: el AutoML NO debe detenerse por un modelo individual que falle.
    """
    print("\n" + "=" * 70)
    print("  MODULO DE ENTRENAMIENTO — AutoML")
    print("=" * 70)
    print(f"  Modelos a entrenar: {len(models)}")
    print(f"  Directorio de salida: {run_dir}")
    print("=" * 70)

    # Iterar sobre cada modelo del diccionario y entrenarlo
    for ml_name, model_info in models.items():
        print(f"\n{'─' * 50}")
        print(f"  Entrenando: {ml_name}")
        print(f"{'─' * 50}")

        try:
            modelo = model_info['model']           # Instancia sin entrenar
            fit_kwargs = model_info.get('fit_kwargs', {})  # Args extra para fit()

            # PASO 1: Entrenar el modelo y medir cuanto tarda (en segundos)
            # medir_tiempo_entrenamiento() llama a modelo.fit() internamente
            training_time = medir_tiempo_entrenamiento(
                modelo, X_train, y_train, fit_kwargs
            )
            print(f"  Tiempo de entrenamiento: {training_time:.2f} segundos")

            # PASO 2: Calcular metricas de rendimiento (AUC train, test, decay)
            # calcular_metricas() usa predict_proba o decision_function segun
            # lo que el modelo soporte (ej: Ridge no tiene predict_proba)
            metricas = calcular_metricas(
                modelo, X_train, y_train, X_test, y_test
            )
            metricas['training_time_segs'] = training_time  # Agregar tiempo

            # Almacenar metricas y parametros en el diccionario del modelo
            models[ml_name]['performance'] = metricas
            models[ml_name]['params'] = modelo.get_params()

            # PASO 3: Guardar modelo serializado como archivo .pkl
            # Permite cargar el modelo entrenado sin reentrenar
            guardar_modelo(modelo, ml_name, run_dir)

            # PASO 4: Guardar metadata como archivo .json
            # Incluye: metricas, hiperparametros, versiones de librerias
            guardar_metadata(
                ml_name=ml_name,
                performance=metricas,
                hyperparameters=modelo.get_params(),
                library_versions=model_info['library_versions'],
                run_dir=run_dir,
            )

            # PASO 5: Imprimir resumen en consola para seguimiento
            print(f"  AUC Train: {metricas['auc_train']:.4f}")
            print(f"  AUC Test:  {metricas['auc_test']:.4f}")
            print(f"  Decay:     {metricas['decay_percent']:.2f}%")

        except Exception as e:
            # Si un modelo falla, registrar el error y continuar
            # Criterio: NO detenerse por un modelo individual
            print(f"  ⚠ ERROR en {ml_name}: {e}")
            models[ml_name]['performance'] = {
                'auc_train': 0.0,
                'auc_test': 0.0,
                'decay_percent': 100.0,
                'training_time_segs': 0.0,
                'error': str(e),
            }

    return models


# =============================================================================
# SELECCION DEL CAMPEON — GetBestModel() (FALLBACK de Ronda 1)
# =============================================================================

def _GetBestModelP1(models: dict, config: dict) -> tuple:
    """
    Selecciona el modelo campeon usando score compuesto (fallback de Ronda 1).

    Esta funcion se usa SOLO como fallback cuando ningun modelo pasa a
    Ronda 2. En el flujo normal, el campeon se elige en Ronda 2 por
    cv_auc_mean (mayor estabilidad en validacion cruzada).

    Parametros
    ----------
    models : dict
        Diccionario de modelos con metricas ya calculadas.
    config : dict
        Configuracion con umbrales:
        - max_decay_percent: maximo decay permitido (default: 10%).
        - min_auc_test: AUC minimo en test (default: 0.55).

    Retorna
    -------
    tuple : (nombre_campeon, modelo_campeon, performance_campeon)
        Si ningun modelo cumple los criterios, retorna (None, None, None).

    Criterios de seleccion (politica de eleccion)
    -----------------------------------------------
    1. Decay < 10%: descarta modelos con sobreajuste excesivo.
    2. AUC test > 0.55: descarta modelos sin capacidad predictiva.
    3. Score compuesto: AUC_test x (1 - decay/100) x factor_tiempo
       - AUC_test: capacidad predictiva en datos no vistos.
       - (1 - decay/100): penaliza sobreajuste.
       - factor_tiempo: penalizacion suave si tarda > 60s (max 5%).
    4. Desempate: menor decay (modelo mas estable).
    """
    max_decay = config.get('max_decay_percent', 10.0)
    min_auc = config.get('min_auc_test', 0.55)

    print("\n" + "=" * 70)
    print("  SELECCION DEL MODELO CAMPEON — GetBestModel() [Fallback R1]")
    print("=" * 70)
    print(f"  Criterios de filtrado:")
    print(f"    - Decay maximo permitido: {max_decay}%")
    print(f"    - AUC test minimo: {min_auc}")
    print(f"  Criterio de ranking: Score compuesto (AUC x estabilidad x eficiencia)")
    print("=" * 70)

    # PASO 1: Filtrar candidatos que cumplan los criterios minimos
    candidatos = []
    for ml_name, model_info in models.items():
        perf = model_info['performance']

        # Verificar que tenga metricas validas (no fallo en entrenamiento)
        if 'error' in perf:
            print(f"  ✗ {ml_name}: DESCARTADO (error en entrenamiento)")
            continue

        auc_test = perf['auc_test']
        decay = perf['decay_percent']
        training_time = perf.get('training_time_segs', 0)

        # Criterio 1: Decay debe ser menor al umbral (evitar sobreajuste)
        if decay > max_decay:
            print(f"  ✗ {ml_name}: DESCARTADO (decay={decay:.2f}% > {max_decay}%)")
            continue

        # Criterio 2: AUC test debe superar el minimo (capacidad predictiva)
        if auc_test < min_auc:
            print(f"  ✗ {ml_name}: DESCARTADO (AUC test={auc_test:.4f} < {min_auc})")
            continue

        # PASO 2: Calcular score compuesto para ranking
        # Componente de estabilidad: 1 - decay/100 (a menor decay, mejor)
        estabilidad = 1.0 - (decay / 100.0)

        # Componente de eficiencia: penalizacion suave por tiempo excesivo
        # Solo penaliza si toma > 60 segundos, maximo 5% de penalizacion
        # Criterio: la velocidad es relevante en produccion (reentreno mensual)
        if training_time > 60:
            factor_tiempo = max(0.95, 1.0 - (training_time - 60) / 1200.0)
        else:
            factor_tiempo = 1.0

        # Score compuesto = AUC x estabilidad x eficiencia
        score = auc_test * estabilidad * factor_tiempo

        candidatos.append({
            'ml_name': ml_name,
            'auc_test': auc_test,
            'decay': decay,
            'training_time': training_time,
            'score': score,
        })

        print(
            f"  ✓ {ml_name}: AUC={auc_test:.4f} | "
            f"Decay={decay:.2f}% | "
            f"Tiempo={training_time:.1f}s | "
            f"Score={score:.6f}"
        )

    # PASO 3: Seleccionar el campeon (mayor score, desempate por menor decay)
    if not candidatos:
        print("\n  ⚠ NINGUN MODELO CUMPLE LOS CRITERIOS.")
        print("  Acciones recomendadas:")
        print("    1. Revisar el preprocesamiento de datos.")
        print("    2. Ajustar hiperparametros de los modelos.")
        print("    3. Incrementar el umbral de decay en config.yaml.")
        print("    4. Revisar la calidad del dataset.")
        return None, None, None

    # Ordenar por score descendente; desempate por menor decay (mas estable)
    candidatos.sort(key=lambda x: (-x['score'], x['decay']))

    campeon = candidatos[0]
    ml_name = campeon['ml_name']
    modelo = models[ml_name]['model']
    performance = models[ml_name]['performance']

    print(f"\n  {'★' * 50}")
    print(f"  ★ MODELO CAMPEON (Fallback R1): {ml_name}")
    print(f"  ★ AUC Test:       {campeon['auc_test']:.4f}")
    print(f"  ★ Decay:          {campeon['decay']:.2f}%")
    print(f"  ★ Score compuesto:{campeon['score']:.6f}")
    print(f"  {'★' * 50}")

    return ml_name, modelo, performance


# =============================================================================
# FUNCION PRINCIPAL: run_training() — PIPELINE COMPLETO EN DOS RONDAS
# =============================================================================

def run_training(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict
) -> tuple:
    """
    Funcion principal del modulo: ejecuta el pipeline completo de
    entrenamiento en DOS RONDAS y selecciona el modelo campeon.

    Parametros
    ----------
    X_train, y_train : arrays
        Datos de entrenamiento (salida de preprocessing).
    X_test, y_test : arrays
        Datos de test (salida de preprocessing).
    config : dict
        Configuracion completa del sistema (desde config.yaml).

    Retorna
    -------
    tuple : (campeon_name, campeon_model, campeon_perf, run_dir, all_models)
        - campeon_name: nombre del modelo campeon seleccionado.
        - campeon_model: objeto estimador entrenado del campeon.
        - campeon_perf: dict con metricas del campeon.
        - run_dir: ruta al directorio de la corrida (con timestamp).
        - all_models: diccionario con TODOS los modelos y sus metricas.

    Flujo del pipeline en dos rondas
    ---------------------------------
    1. Crear directorio de corrida con timestamp.
    2. Calcular ratio de desbalance de clases.
    3. Definir diccionario de 9 modelos.
    4. RONDA 1: Entrenar los 9 modelos, calcular AUC + decay.
    5. Filtrar candidatos: decay <= 10% y AUC test >= 0.55.
    6. Seleccionar TOP 4 candidatos por AUC test.
    7. RONDA 2: Evaluar estabilidad de los top 4 con 5-fold CV.
    8. Seleccionar campeon: mayor cv_auc_mean (desempate: menor cv_auc_std).
    9. Guardar resumen consolidado (results_summary.json).

    Criterio de seleccion TOP 4 para Ronda 2
    ------------------------------------------
    Solo los 4 mejores modelos por AUC test pasan a Ronda 2 porque:
    - La validacion cruzada es costosa (entrena 5 veces por modelo).
    - Los modelos con AUC test inferior tienen menor probabilidad de
      ser el campeon, por lo que evaluar su CV no aporta valor.
    - 4 modelos es un balance razonable entre cobertura y eficiencia.
    """
    random_seed = config.get('random_seed', 42)
    output_dir = config.get('output_dir', 'data/output')

    # =========================================================================
    # PASO 1: Crear directorio de corrida con timestamp
    # Cada ejecucion genera su propia carpeta (ej: data/output/2026-09-18_14-30-00/)
    # para que los artefactos de corridas anteriores no se sobreescriban.
    # =========================================================================
    run_dir = crear_directorio_corrida(output_dir)

    # =========================================================================
    # PASO 2: Calcular ratio de desbalance de clases
    # ratio = (negativos) / (positivos)  = 28 para este dataset
    # Este ratio se usa para ponderar la clase minoritaria en los modelos.
    # Tecnica aprendida del notebook de clase 1 del profesor.
    # =========================================================================
    ratio_desbalance = int((y_train == 0).sum() / (y_train == 1).sum())
    print(f"\n[Training] Desbalance detectado: ratio = {ratio_desbalance}:1")
    print(f"[Training] Ponderacion de clases aplicada:")
    print(f"           XGBoost  -> scale_pos_weight={ratio_desbalance}")
    print(f"           LightGBM -> scale_pos_weight={ratio_desbalance}")
    print(f"           CatBoost -> auto_class_weights='Balanced'")

    # =========================================================================
    # PASO 3: Definir los 9 modelos con el ratio de desbalance
    # =========================================================================
    models = definir_modelos(random_seed, ratio_desbalance)
    print(f"\n[Training] {len(models)} modelos definidos para entrenamiento.")

    # =========================================================================
    # RONDA 1: Entrenamiento de los 9 modelos (AUC + Decay)
    # Se entrena cada modelo, se calcula AUC en train y test, y el decay
    # (diferencia relativa entre AUC train y AUC test, indicador de sobreajuste).
    # =========================================================================
    print("\n" + "=" * 70)
    print("  RONDA 1: Entrenamiento de los 9 modelos")
    print("=" * 70)

    models = entrenar_todos_los_modelos(
        models, X_train, y_train, X_test, y_test, run_dir
    )

    # Marcar todos los modelos como pertenecientes a Ronda 1
    for ml_name in models:
        models[ml_name]['performance']['ronda'] = 1

    # =========================================================================
    # FILTRAR CANDIDATOS PARA RONDA 2
    # Criterios: decay <= max_decay_percent Y auc_test >= min_auc_test
    # Luego: ordenar por AUC test descendente y tomar solo los TOP 4.
    # Criterio: solo los 4 mejores pasan porque CV es costoso y los de
    # menor AUC tienen menor probabilidad de ser campeon.
    # =========================================================================
    max_decay = config.get('max_decay_percent', 10.0)
    min_auc = config.get('min_auc_test', 0.55)
    top_n_r2 = 4  # Numero maximo de modelos que pasan a Ronda 2

    # Primero: filtrar modelos que cumplen los criterios minimos
    candidatos_filtrados = [
        name for name, info in models.items()
        if 'error' not in info['performance']                          # Sin errores
        and info['performance'].get('decay_percent', 100) <= max_decay  # Decay aceptable
        and info['performance'].get('auc_test', 0) >= min_auc           # AUC suficiente
    ]

    # Segundo: ordenar por AUC test descendente y tomar los top 4
    candidatos_filtrados.sort(
        key=lambda name: models[name]['performance']['auc_test'],
        reverse=True  # Mayor AUC primero
    )
    candidatos_r2 = candidatos_filtrados[:top_n_r2]  # Tomar solo los top 4

    print(f"\n[Training] Candidatos que pasan filtros (decay<={max_decay}%, AUC>={min_auc}): "
          f"{len(candidatos_filtrados)} modelo(s)")
    for c in candidatos_filtrados:
        marker = " <- TOP 4" if c in candidatos_r2 else ""
        auc = models[c]['performance']['auc_test']
        print(f"           - {c} (AUC test: {auc:.4f}){marker}")

    print(f"\n[Training] TOP {top_n_r2} seleccionados para Ronda 2 (CV): "
          f"{len(candidatos_r2)} modelo(s)")

    # =========================================================================
    # RONDA 2: Validacion Cruzada (CV Stability) para los TOP 4
    # Se evalua la estabilidad de cada modelo con 5-fold Stratified CV.
    # Criterio: un modelo estable tiene cv_auc_std bajo, lo que indica
    # que su rendimiento no depende de que datos caen en cada fold.
    # =========================================================================
    print("\n" + "=" * 70)
    print("  RONDA 2: Validacion Cruzada (CV Stability) — Top 4")
    print("=" * 70)

    for ml_name in candidatos_r2:
        print(f"\n{'─' * 50}")
        print(f"  CV para: {ml_name}")
        print(f"{'─' * 50}")

        try:
            # Obtener el modelo ya entrenado en Ronda 1
            modelo = models[ml_name]['model']

            # Ejecutar validacion cruzada estratificada con 5 folds
            # calcular_cv_stability() clona el modelo y lo entrena 5 veces,
            # cada vez con un fold diferente como validacion.
            cv_result = calcular_cv_stability(
                modelo, {}, X_train, y_train,
                n_folds=5, random_seed=random_seed
            )

            # Actualizar las metricas del modelo con los resultados de CV
            models[ml_name]['performance'].update(cv_result)
            models[ml_name]['performance']['ronda'] = 2  # Marcar como Ronda 2

            # Clasificar la estabilidad del modelo segun cv_auc_std
            std = cv_result['cv_auc_std']
            if std < 0.01:
                nivel = "Muy estable ✓✓"      # Excelente: rendimiento consistente
            elif std < 0.02:
                nivel = "Estable ✓"            # Bueno: variabilidad aceptable
            elif std < 0.03:
                nivel = "Moderado ~"           # Aceptable: algo sensible
            else:
                nivel = "Inestable ✗"          # Preocupante: depende del fold

            print(f"  CV AUC Mean: {cv_result['cv_auc_mean']:.4f}")
            print(f"  CV AUC Std:  {cv_result['cv_auc_std']:.4f}  -> {nivel}")

            # Actualizar la metadata en disco (.json) con los resultados de CV
            guardar_metadata(
                ml_name=ml_name,
                performance=models[ml_name]['performance'],
                hyperparameters=modelo.get_params(),
                library_versions=models[ml_name]['library_versions'],
                run_dir=run_dir,
            )

        except Exception as e:
            # Si falla el CV, el modelo se queda en Ronda 1 con CV = None
            print(f"  ⚠ ERROR en CV de {ml_name}: {e}")
            models[ml_name]['performance'].update({
                'cv_auc_mean': None,
                'cv_auc_std': None,
                'cv_auc_scores': [],
                'cv_n_folds': 5,
            })

    # =========================================================================
    # SELECCION DEL CAMPEON
    # Criterio principal: de los modelos con CV completado (Ronda 2),
    # elegir el de mayor cv_auc_mean (rendimiento promedio en los 5 folds).
    # Desempate: menor cv_auc_std (modelo mas estable).
    # Si no hay modelos R2, se usa GetBestModel() como fallback (score compuesto).
    # =========================================================================
    modelos_r2 = {
        name: info for name, info in models.items()
        if info['performance'].get('ronda') == 2
        and info['performance'].get('cv_auc_mean') is not None
    }

    if modelos_r2:
        # Seleccionar campeon de Ronda 2: mayor cv_auc_mean, desempate menor std
        campeon_name = max(
            modelos_r2.keys(),
            key=lambda n: (
                models[n]['performance']['cv_auc_mean'],      # Mayor AUC promedio CV
                -models[n]['performance'].get('cv_auc_std', 1.0)  # Menor variabilidad
            )
        )
        campeon_model = models[campeon_name]['model']
        campeon_perf = models[campeon_name]['performance']

        print(f"\n  {'★' * 50}")
        print(f"  ★ MODELO CAMPEON (Ronda 2 — CV): {campeon_name}")
        print(f"  ★ AUC Test:      {campeon_perf['auc_test']:.4f}")
        print(f"  ★ Decay:         {campeon_perf['decay_percent']:.2f}%")
        print(f"  ★ CV AUC Mean:   {campeon_perf['cv_auc_mean']:.4f}")
        print(f"  ★ CV AUC Std:    {campeon_perf['cv_auc_std']:.4f}")
        print(f"  {'★' * 50}")
    else:
        # FALLBACK: si ningun modelo paso a Ronda 2, usar GetBestModel()
        # que selecciona por score compuesto (AUC x estabilidad x eficiencia)
        print("\n  [Training] Sin candidatos R2. Seleccionando campeon desde Ronda 1...")
        campeon_name, campeon_model, campeon_perf = GetBestModel(models, config)

    # =========================================================================
    # GUARDAR RESUMEN CONSOLIDADO (results_summary.json)
    # Contiene: todos los modelos con sus metricas, ronda, CV, campeon, config.
    # =========================================================================
    dataset_info = {
        'n_rows_train': int(X_train.shape[0]),
        'n_rows_test': int(X_test.shape[0]),
        'n_rows_total': int(X_train.shape[0] + X_test.shape[0]),
        'n_features': int(X_train.shape[1]),
        'target_rate_train': float(round(y_train.mean() * 100, 4)),
        'target_rate_test': float(round(y_test.mean() * 100, 4)),
    }
    guardar_resumen_consolidado(models, campeon_name, config, run_dir, dataset_info)

    # Resumen final en consola
    if campeon_name:
        print(f"\n[Training] Modelo campeon '{campeon_name}' guardado en: {run_dir}")
    else:
        print("\n[Training] No se encontro un modelo campeon valido.")

    return campeon_name, campeon_model, campeon_perf, run_dir, models


def GetBestModel(models: dict, config: dict) -> tuple:
    """Compatibilidad P1: delega la selección extendida al módulo evaluation."""
    from src.evaluation import GetBestModel as select
    return select(models, config)
