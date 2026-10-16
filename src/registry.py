"""
=============================================================================
Módulo de Registro — registry.py
=============================================================================
Responsabilidad:
    Persistir los modelos entrenados y su metadata en disco.

Artefactos generados por modelo:
    1. Archivo .pkl (pickle): modelo serializado.
    2. Archivo .json (metadata): métricas, hiperparámetros, versiones,
       estabilidad por CV y timestamp.

Estructura del directorio de salida:
    data/output/YYYY-MM-DD_HH-MM-SS/
    ├── xgb_model.pkl
    ├── xgb_metadata.json
    ├── ...
    └── results_summary.json
=============================================================================
"""

# --------------------------------------------------------------------------
# Importaciones necesarias
# --------------------------------------------------------------------------

# 'os' se usa para manipular rutas de archivos y crear directorios en disco.
import os

# 'json' se usa para serializar diccionarios Python a archivos JSON legibles,
# que almacenan la metadata y el resumen consolidado de la corrida.
import json

# 'math' se usa para verificar NaN e Infinity antes de serializar a JSON.
# math.isnan() y math.isinf() detectan valores que producirían JSON inválido.
import math

# 'pickle' se usa para serializar objetos Python (los modelos entrenados)
# a formato binario, permitiendo reconstruirlos luego sin re-entrenar.
import pickle

# 'datetime' se usa para generar timestamps que identifican unívocamente
# cada corrida de entrenamiento y registran el momento exacto de creación.
from datetime import datetime


# ==========================================================================
# Función: crear_directorio_corrida
# ==========================================================================
def crear_directorio_corrida(output_dir: str) -> str:
    """
    Crea un directorio con timestamp único para almacenar los artefactos
    de una corrida de entrenamiento.

    Cada corrida genera su propio directorio aislado para evitar sobrescribir
    resultados anteriores y mantener trazabilidad temporal de los experimentos.
    El nombre usa el formato "YYYY-MM-DD_HH-MM-SS" que es legible, ordenable
    cronológicamente y seguro para nombres de carpeta en Windows/Linux (sin ":").

    Parámetros
    ----------
    output_dir : str
        Directorio base de salida donde se crea la subcarpeta de la corrida.
        Ej: 'data/output' (relativa a la raíz del proyecto).

    Retorna
    -------
    str
        Ruta absoluta del directorio de corrida creado,
        con formato: <output_dir>/YYYY-MM-DD_HH-MM-SS/
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[Registry] Directorio de corrida creado: {run_dir}")
    return run_dir


# ==========================================================================
# Función: guardar_modelo
# ==========================================================================
def guardar_modelo(modelo, ml_name: str, run_dir: str) -> str:
    """
    Serializa un modelo entrenado de machine learning a un archivo binario .pkl.

    El modelo entrenado se persiste en disco para poder cargarlo en producción
    o en evaluaciones futuras sin necesidad de re-entrenar (lo cual consumiría
    tiempo y recursos). El nombre del archivo incluye el identificador del
    modelo (ml_name) para distinguirlo de otros modelos en la misma corrida.

    Parámetros
    ----------
    modelo : estimator
        Modelo de ML entrenado (fitted). Debe ser compatible con pickle
        (scikit-learn, XGBoost, LightGBM y CatBoost lo son).
    ml_name : str
        Identificador corto del modelo. Ej: 'xgb', 'catb', 'rf'.
        El archivo se nombra <ml_name>_model.pkl.
    run_dir : str
        Directorio de la corrida actual donde se guardará el archivo.

    Retorna
    -------
    str
        Ruta absoluta del archivo .pkl generado.
    """
    filename = f"{ml_name}_model.pkl"
    filepath = os.path.join(run_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"[Registry] Modelo guardado: {filepath}")
    return filepath


# ==========================================================================
# Función auxiliar (privada): _limpiar_valor
# ==========================================================================
def _limpiar_valor(v):
    """
    Convierte un valor Python a un tipo nativo seguro para JSON (RFC 8259).

    Maneja tipos numpy, NaN, Infinity y estructuras dict/list/tuple de forma
    recursiva para garantizar que json.dump(..., allow_nan=False) no falle.

    json.dump por defecto produce JSON inválido cuando encuentra float('nan')
    o float('inf') (escribe "NaN" e "Infinity" sin comillas, violando RFC 8259).
    Usando allow_nan=False el dump lanza ValueError, que se previene limpiando
    todos los valores con esta función antes de serializar.

    Parámetros
    ----------
    v : any
        Valor a limpiar. Puede ser de cualquier tipo.

    Retorna
    -------
    any
        Valor convertido a un tipo nativo Python serializable a JSON:
        int, float, str, bool, None, list o dict.
        NaN e Infinity se convierten a None (JSON: null).
        Cualquier tipo no reconocido se convierte a str.
    """
    import numpy as np

    # Tipos numpy enteros (np.int8/16/32/64, np.uint8/16/32/64, etc.)
    if isinstance(v, np.integer):
        return int(v)

    # Tipos numpy de punto flotante (np.float16/32/64)
    # Se convierten a float nativo y luego caen al bloque float de Python.
    if isinstance(v, np.floating):
        v = float(v)

    # float nativo de Python (incluyendo los convertidos de np.floating)
    if isinstance(v, float):
        # NaN e Infinity producen JSON inválido. Se convierten a None (null).
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    # Arrays numpy: se convierten a lista y se limpian recursivamente.
    if isinstance(v, np.ndarray):
        return [_limpiar_valor(item) for item in v.tolist()]

    # Diccionarios anidados: se limpian recursivamente.
    if isinstance(v, dict):
        return {str(k2): _limpiar_valor(v2) for k2, v2 in v.items()}

    # Listas y tuplas: se limpian elemento por elemento.
    if isinstance(v, (list, tuple)):
        return [_limpiar_valor(item) for item in v]

    # Tipos básicos nativos de Python (ya serializables).
    if isinstance(v, (int, str, bool)) or v is None:
        return v

    # Fallback: intentar serializar tal cual; si falla, convertir a str.
    try:
        json.dumps(v, allow_nan=False)
        return v
    except (TypeError, ValueError, OverflowError):
        return str(v)


# ==========================================================================
# Función auxiliar (privada): _limpiar_params
# ==========================================================================
def _limpiar_params(params: dict) -> dict:
    """
    Limpia un diccionario de hiperparámetros para serialización JSON segura.

    Aplica _limpiar_valor recursivamente a todos los valores del diccionario,
    garantizando que el resultado sea compatible con json.dump(..., allow_nan=False).
    Convierte tipos numpy, elimina NaN/Infinity (→ None/null) y maneja
    estructuras anidadas dict/list/tuple.

    Los hiperparámetros de los modelos de ML pueden incluir tipos no serializables
    por JSON: objetos numpy (np.int64, np.float32), funciones, clases, etc.
    Sin esta limpieza, json.dump fallaría con TypeError o ValueError, interrumpiendo
    todo el registro de artefactos.

    Parámetros
    ----------
    params : dict
        Diccionario de hiperparámetros del modelo (obtenido con get_params()).

    Retorna
    -------
    dict
        Diccionario limpio con todos los valores convertidos a tipos nativos
        de Python compatibles con JSON estándar (RFC 8259).
    """
    return {str(k): _limpiar_valor(v) for k, v in params.items()}


# ==========================================================================
# Función: guardar_metadata
# ==========================================================================
def guardar_metadata(
    ml_name: str,
    performance: dict,
    hyperparameters: dict,
    library_versions: dict,
    run_dir: str,
) -> str:
    """
    Genera y guarda un archivo JSON con la metadata completa de un modelo entrenado.

    La metadata incluye métricas, hiperparámetros, versiones de librerías y
    timestamp. Es esencial para la reproducibilidad y auditoría de los modelos.
    Permite saber exactamente con qué configuración se entrenó cada modelo,
    qué rendimiento obtuvo y con qué versiones de software se generó.

    Estructura del JSON generado:
        {
            "ml_name": "catb",
            "performance": {
                "auc_train": 0.9135,
                "auc_test": 0.8479,
                "decay_percent": 7.18,
                "training_time_segs": 20.77,
                "cv_auc_mean": 0.854248,
                "cv_auc_std": 0.006063,
                "ronda": 2
            },
            "hyperparameters": { ... },
            "library_versions": { "catboost": "1.2.10" },
            "timestamp": "2026-09-19 01:38:25"
        }

    Parámetros
    ----------
    ml_name : str
        Identificador corto del modelo (ej: "xgb", "catb", "rf").
    performance : dict
        Diccionario con métricas de rendimiento (AUC, decay, CV, tiempo).
        Puede contener np.inf (decay cuando auc_train==0); se limpia a null.
    hyperparameters : dict
        Diccionario con los hiperparámetros usados en el entrenamiento.
        Puede contener tipos numpy no serializables; se limpian con _limpiar_params.
    library_versions : dict
        Diccionario con versiones de librerías para trazabilidad de reproducibilidad.
    run_dir : str
        Directorio de la corrida actual donde se guardará el archivo .json.

    Retorna
    -------
    str
        Ruta absoluta del archivo <ml_name>_metadata.json generado.
    """
    metadata = {
        'ml_name': ml_name,
        # performance puede contener np.inf (decay con auc_train==0).
        # Se limpia con _limpiar_valor para convertir Infinity → None (null).
        'performance': {k: _limpiar_valor(v) for k, v in performance.items()},
        # Hiperparámetros limpiados: tipos numpy → Python nativos.
        'hyperparameters': _limpiar_params(hyperparameters),
        'library_versions': library_versions,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    filename = f"{ml_name}_metadata.json"
    filepath = os.path.join(run_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        # allow_nan=False: garantiza JSON válido (RFC 8259). Lanza ValueError
        # si queda algún NaN o Infinity sin limpiar, actuando como red de seguridad.
        json.dump(metadata, f, indent=4, ensure_ascii=False, allow_nan=False)

    print(f"[Registry] Metadata guardada: {filepath}")
    return filepath


# ==========================================================================
# Función: guardar_resumen_consolidado
# ==========================================================================
def guardar_resumen_consolidado(
    models: dict,
    campeon_name: str,
    config: dict,
    run_dir: str,
    dataset_info: dict = None
) -> str:
    """
    Guarda un archivo JSON consolidado con los resultados de todos los modelos.

    Genera un único archivo "results_summary.json" que reúne los resultados
    de TODOS los modelos evaluados en la corrida, incluyendo el campeón,
    la configuración usada, información del dataset y métricas detalladas
    por modelo. Facilita la comparación entre modelos y sirve como registro
    histórico de cada experimento completo.

    Parámetros
    ----------
    models : dict
        Diccionario con todos los modelos y su información:
        {nombre: {'performance': {...}, ...}, ...}
    campeon_name : str
        Nombre del modelo seleccionado como campeón de la corrida.
    config : dict
        Configuración general del pipeline (semilla, test_size, umbrales, etc.).
    run_dir : str
        Directorio de la corrida actual donde se guardará results_summary.json.
    dataset_info : dict, optional
        Información del dataset (filas, columnas, distribución de clases, etc.).
        Si es None, se registra como diccionario vacío.

    Retorna
    -------
    str
        Ruta absoluta del archivo results_summary.json generado.
    """
    resumen = {
        'run_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'run_dir': run_dir,
        'config': {
            'random_seed': config.get('random_seed', 42),
            'test_size': config.get('test_size', 0.30),
            'max_decay_percent': config.get('max_decay_percent', 10.0),
            'min_auc_test': config.get('min_auc_test', 0.55),
        },
        'dataset': dataset_info or {},
        'campeon': campeon_name,
        'modelos': {}
    }

    for ml_name, model_info in models.items():
        perf = model_info.get('performance', {})
        cv_mean = perf.get('cv_auc_mean')
        cv_std  = perf.get('cv_auc_std')

        # round() puede producir float('nan') si el valor original es nan.
        # Se usa _limpiar_valor como seguridad adicional para decay_percent.
        decay_val = perf.get('decay_percent', 0.0)
        if isinstance(decay_val, float) and (math.isnan(decay_val) or math.isinf(decay_val)):
            decay_val = None
        else:
            decay_val = round(float(decay_val) if decay_val is not None else 0.0, 2)

        entry = {
            'ronda': perf.get('ronda', 1),
            'auc_train': round(float(perf.get('auc_train', 0.0)), 4),
            'auc_test': round(float(perf.get('auc_test', 0.0)), 4),
            'decay_percent': decay_val,
            'training_time_segs': round(float(perf.get('training_time_segs', 0.0)), 2),
            'cv_auc_mean': round(cv_mean, 6) if cv_mean is not None else None,
            'cv_auc_std':  round(cv_std,  6) if cv_std  is not None else None,
            'error': perf.get('error', None),
        }
        resumen['modelos'][ml_name] = entry

    filepath = os.path.join(run_dir, 'results_summary.json')
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(resumen, f, indent=4, ensure_ascii=False, allow_nan=False)

    print(f"[Registry] Resumen consolidado guardado: {filepath}")
    return filepath


def leer_json_compatible(path):
    """Lee JSON UTF-8/BOM y corridas antiguas Windows cp1252."""
    from pathlib import Path
    try:
        text = Path(path).read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        text = Path(path).read_text(encoding='cp1252')
    return json.loads(text)
