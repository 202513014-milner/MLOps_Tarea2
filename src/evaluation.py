"""
=============================================================================
Módulo de Evaluación — evaluation.py
=============================================================================
Responsabilidad:
    Calcular las métricas de rendimiento de cada modelo entrenado.

Métricas calculadas:
    - AUC-ROC (Area Under the Receiver Operating Characteristic Curve):
      Métrica principal para evaluar la capacidad discriminativa del modelo.
      Se eligió AUC-ROC porque es independiente del umbral de clasificación
      y resume el rendimiento del modelo en todos los puntos de corte posibles.
    - Decay (%): decaimiento relativo entre AUC de train y AUC de test.
      Permite detectar sobreajuste (overfitting): un decay alto indica que
      el modelo memoriza los datos de entrenamiento sin generalizar bien.
    - Tiempo de entrenamiento: en segundos. Permite comparar la eficiencia
      computacional entre modelos y determinar viabilidad en producción.
    - Validación cruzada (CV): AUC medio y desviación estándar con 5-fold
      stratified cross-validation para medir estabilidad del modelo.
      La estratificación garantiza que cada fold mantenga la misma proporción
      de clases que el dataset original, lo cual es crítico en datasets
      desbalanceados.

Librerías utilizadas:
    - sklearn.metrics.roc_auc_score: implementación estándar de AUC-ROC.
    - sklearn.model_selection.StratifiedKFold, cross_val_score: validación
      cruzada estratificada para evaluar estabilidad.
    - time: módulo estándar de Python para medir tiempos de ejecución.
=============================================================================
"""

# --- Importaciones ---

# 'time' se usa para medir el tiempo transcurrido durante el entrenamiento
# del modelo, capturando el reloj antes y después de .fit().
import time

# 'numpy' proporciona operaciones numéricas eficientes; aquí se usa para
# calcular media (np.mean), desviación estándar (np.std), y representar
# el valor infinito (np.inf) cuando no se puede calcular el decay.
import numpy as np

# 'roc_auc_score' calcula el Área Bajo la Curva ROC, que mide qué tan bien
# el modelo distingue entre las dos clases (positiva y negativa).
# Un valor de 1.0 indica discriminación perfecta; 0.5, aleatorio.
from sklearn.metrics import roc_auc_score

# 'StratifiedKFold' divide el dataset en k folds manteniendo la proporción
# de clases en cada fold (estratificación). Esto es esencial para
# clasificación binaria con desbalanceo, ya que cada fold es representativo.
# 'cross_val_score' es importado por compatibilidad aunque no se usa
# directamente en este módulo (se implementa validación cruzada manual
# para tener mayor control sobre el proceso).
from sklearn.model_selection import StratifiedKFold, cross_val_score


def calcular_metricas(
    modelo,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Calcula AUC en train y test, y decay.

    Parámetros:
        modelo:  modelo ya entrenado (fitted) de scikit-learn o compatible.
        X_train: matriz de características del conjunto de entrenamiento.
        y_train: vector de etiquetas reales del conjunto de entrenamiento.
        X_test:  matriz de características del conjunto de prueba.
        y_test:  vector de etiquetas reales del conjunto de prueba.

    Retorna:
        dict con claves:
            - 'auc_train':      AUC-ROC calculado sobre el set de entrenamiento.
            - 'auc_test':       AUC-ROC calculado sobre el set de prueba.
            - 'decay_percent':  decaimiento porcentual entre train y test.
              Fórmula: ((auc_train - auc_test) / auc_train) * 100.
              Un valor cercano a 0 indica buena generalización; un valor alto
              sugiere sobreajuste.

    Criterio de diseño:
        Se intenta primero 'predict_proba' (probabilidades calibradas) porque
        es el método estándar en la mayoría de clasificadores de sklearn.
        Si no existe, se recurre a 'decision_function' (distancia al
        hiperplano), que es lo que usan modelos como SVM lineal.
        Si ninguno está disponible, se lanza un error explícito.
    """

    # --- Obtención de puntuaciones de probabilidad ---
    # Se verifica qué método tiene el modelo para generar puntuaciones
    # continuas (necesarias para calcular AUC-ROC, que requiere scores
    # y no predicciones binarias).

    if hasattr(modelo, 'predict_proba'):
        # 'predict_proba' devuelve una matriz de forma (n_samples, n_classes).
        # Se toma la columna índice 1, que corresponde a la probabilidad
        # de la clase positiva (clase 1). Esto es una convención de sklearn.
        y_train_proba = modelo.predict_proba(X_train)[:, 1]
        y_test_proba = modelo.predict_proba(X_test)[:, 1]

    elif hasattr(modelo, 'decision_function'):
        # 'decision_function' devuelve un score continuo (distancia al
        # hiperplano de separación). Se usa en modelos que no proveen
        # probabilidades calibradas directamente, como SVC sin probability=True.
        # El score es directamente proporcional a la confianza de la clase
        # positiva, por lo que es compatible con roc_auc_score.
        y_train_proba = modelo.decision_function(X_train)
        y_test_proba = modelo.decision_function(X_test)

    else:
        # Si el modelo no ofrece ninguno de los dos métodos, no es posible
        # calcular AUC-ROC. Se lanza un error descriptivo indicando el nombre
        # del modelo que causó el problema.
        raise ValueError(
            f"El modelo {type(modelo).__name__} no tiene predict_proba "
            f"ni decision_function. No se puede calcular AUC."
        )

    # --- Cálculo de AUC-ROC en train y test ---
    # roc_auc_score compara las etiquetas reales contra las probabilidades
    # predichas. Internamente genera la curva ROC y calcula el área bajo ella.

    # AUC sobre datos de entrenamiento: indica qué tan bien el modelo se
    # ajusta a los datos con los que fue entrenado. Un valor muy alto (>0.99)
    # puede indicar sobreajuste.
    auc_train = roc_auc_score(y_train, y_train_proba)

    # AUC sobre datos de prueba: indica la capacidad real de generalización
    # del modelo ante datos no vistos durante el entrenamiento.
    auc_test = roc_auc_score(y_test, y_test_proba)

    # --- Cálculo del decay (decaimiento) ---
    # El decay mide el porcentaje de caída del rendimiento entre train y test.
    # Se calcula como la diferencia relativa respecto al AUC de train.
    # Criterio: si auc_train > 0, se puede dividir de forma segura.
    # Si auc_train fuera 0 (caso extremo, modelo completamente ineficaz),
    # se asigna infinito (np.inf) porque la división por cero no tiene sentido.
    if auc_train > 0:
        decay = ((auc_train - auc_test) / auc_train) * 100
    else:
        # np.inf señaliza que el decay no se puede determinar de manera
        # significativa porque el modelo no aprendió nada (AUC de train = 0).
        decay = np.inf

    # Se retorna un diccionario con los resultados redondeados.
    # round(x, 6) para AUC: 6 decimales ofrecen precisión suficiente para
    # comparaciones entre modelos sin generar ruido numérico excesivo.
    # round(x, 4) para decay: 4 decimales son suficientes para porcentajes.
    return {
        'auc_train': round(auc_train, 6),
        'auc_test': round(auc_test, 6),
        'decay_percent': round(decay, 4),
    }


def calcular_cv_stability(
    modelo_clase,
    modelo_params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42
) -> dict:
    """
    Calcula la estabilidad del modelo mediante validación cruzada estratificada.
    Usa StratifiedKFold para mantener la proporción del target en cada fold.

    Parámetros:
        modelo_clase:   instancia del modelo (no entrenada) de sklearn, que
                        será clonada para cada fold. Se clona para garantizar
                        que cada fold parta de un modelo limpio sin contaminación.
        modelo_params:  diccionario de hiperparámetros del modelo (no se usa
                        directamente aquí porque 'clone' copia los parámetros
                        del modelo original, pero se conserva en la interfaz
                        por consistencia con el pipeline).
        X_train:        matriz de características del conjunto de entrenamiento.
        y_train:        vector de etiquetas reales del conjunto de entrenamiento.
        n_folds:        número de folds para la validación cruzada. El valor
                        por defecto es 5, que es un estándar ampliamente usado
                        y ofrece un buen balance entre sesgo y varianza de la
                        estimación del rendimiento.
        random_seed:    semilla aleatoria para reproducibilidad. Se fija en 42
                        por convención para asegurar resultados idénticos en
                        cada ejecución del pipeline.

    Retorna:
        dict con claves:
            - 'cv_auc_mean':   AUC promedio a través de los k folds.
            - 'cv_auc_std':    desviación estándar del AUC entre folds.
              Una std baja indica que el modelo es estable y no depende
              fuertemente de la partición de datos.
            - 'cv_auc_scores': lista con el AUC individual de cada fold.
            - 'cv_n_folds':    número de folds usados (para trazabilidad).

    Criterio de diseño:
        Se implementa la validación cruzada manualmente (en lugar de usar
        cross_val_score) para tener control total sobre la clonación del
        modelo, la obtención de probabilidades, y el manejo de modelos
        que carecen de predict_proba/decision_function.
    """

    # Se importa 'clone' dentro de la función para evitar una dependencia
    # innecesaria a nivel de módulo. 'clone' crea una copia profunda del
    # estimador con los mismos hiperparámetros pero sin datos de entrenamiento,
    # garantizando que cada fold se entrena de forma independiente.
    from sklearn.base import clone

    # Se crea el generador de folds estratificados.
    # shuffle=True: se barajan los datos antes de dividir, para evitar
    # sesgos derivados del orden original de los datos.
    # random_state=random_seed: fija la semilla para que la partición sea
    # reproducible en cada ejecución.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    # Lista para acumular el AUC de cada fold.
    auc_scores = []

    # Se itera sobre cada fold. skf.split() genera los índices de
    # entrenamiento y validación para cada partición.
    # 'fold_idx' es el número de fold (0, 1, ..., n_folds-1), útil para
    # depuración o logging aunque aquí no se usa explícitamente.
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):

        # --- Partición de datos para el fold actual ---
        # Se verifica si X_train es un np.ndarray o un DataFrame de pandas.
        # Si es ndarray, se indexa con corchetes directamente.
        # Si es DataFrame, se usa .iloc para indexación posicional.
        # Esta dualidad hace la función compatible con ambos tipos de datos.
        X_fold_train = X_train[train_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[train_idx]
        y_fold_train = y_train[train_idx] if isinstance(y_train, np.ndarray) else y_train.iloc[train_idx]
        X_fold_val = X_train[val_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[val_idx]
        y_fold_val = y_train[val_idx] if isinstance(y_train, np.ndarray) else y_train.iloc[val_idx]

        # Se clona el modelo original para este fold. Esto asegura que cada
        # fold parte de un modelo con los mismos hiperparámetros pero sin
        # ningún ajuste previo. Sin clone, entrenar en un fold contaminaria
        # los pesos para los siguientes folds.
        modelo_fold = clone(modelo_clase)

        # Se entrena el modelo clonado con los datos de entrenamiento del fold.
        modelo_fold.fit(X_fold_train, y_fold_train)

        # --- Obtención de probabilidades para el fold de validación ---
        # Se sigue la misma lógica que en calcular_metricas:
        # primero predict_proba, luego decision_function como respaldo.
        if hasattr(modelo_fold, 'predict_proba'):
            # Se toma la probabilidad de la clase positiva (columna 1).
            y_val_proba = modelo_fold.predict_proba(X_fold_val)[:, 1]
        elif hasattr(modelo_fold, 'decision_function'):
            # Se usa el score de decisión como alternativa a probabilidades.
            y_val_proba = modelo_fold.decision_function(X_fold_val)
        else:
            # Si el modelo no tiene ninguno de los dos métodos, se omite
            # este fold con 'continue' en lugar de lanzar un error. Esto
            # permite que la validación cruzada continúe con los folds
            # restantes y no se detenga por un modelo problemático.
            continue

        # Se calcula el AUC-ROC para este fold de validación comparando
        # las etiquetas reales con las probabilidades predichas.
        auc = roc_auc_score(y_fold_val, y_val_proba)

        # Se agrega el AUC del fold a la lista, redondeado a 6 decimales
        # para consistencia con calcular_metricas.
        auc_scores.append(round(auc, 6))

    # Se retorna un diccionario con las estadísticas de la validación cruzada.
    return {
        # Media de los AUC de todos los folds: representa el rendimiento
        # esperado del modelo. Se convierte a float para serialización JSON.
        'cv_auc_mean': round(float(np.mean(auc_scores)), 6),

        # Desviación estándar de los AUC: mide la variabilidad del
        # rendimiento entre folds. Un valor bajo (<0.02) indica estabilidad.
        'cv_auc_std': round(float(np.std(auc_scores)), 6),

        # Lista completa de AUC por fold: permite inspeccionar si algún
        # fold tuvo un rendimiento anómalo (outlier).
        'cv_auc_scores': auc_scores,

        # Número de folds usado: se registra para trazabilidad y
        # reproducibilidad del experimento.
        'cv_n_folds': n_folds,
    }


def medir_tiempo_entrenamiento(modelo, X_train, y_train, fit_kwargs: dict = None) -> float:
    """
    Entrena el modelo y retorna el tiempo de entrenamiento en segundos.

    Parámetros:
        modelo:     instancia del modelo de sklearn (no entrenada). Se
                    entrenará dentro de esta función.
        X_train:    matriz de características del conjunto de entrenamiento.
        y_train:    vector de etiquetas reales del conjunto de entrenamiento.
        fit_kwargs: diccionario opcional de argumentos extra para .fit().
                    Permite pasar parámetros específicos del modelo, como
                    'eval_set' para XGBoost o 'sample_weight' para modelos
                    que lo soporten.

    Retorna:
        float: tiempo de entrenamiento en segundos, redondeado a 4 decimales.
        El modelo queda entrenado (fitted) in-place como efecto secundario.

    Criterio de diseño:
        Se usa time.time() (reloj de pared) en lugar de time.perf_counter()
        porque el tiempo de pared es más intuitivo para reportar al usuario
        y es suficiente para la granularidad requerida (segundos, no
        microsegundos). Se redondea a 4 decimales para evitar ruido.
    """

    # Si no se proporcionan argumentos extra para fit, se inicializa un
    # diccionario vacío para poder usar **fit_kwargs sin errores.
    if fit_kwargs is None:
        fit_kwargs = {}

    # Se captura el tiempo antes del entrenamiento (reloj de pared).
    start = time.time()

    # Se entrena el modelo. **fit_kwargs desempaqueta los argumentos
    # adicionales (si los hay) y los pasa a .fit() como parámetros nombrados.
    # El modelo se modifica in-place: después de esta línea, 'modelo'
    # contiene los pesos/parámetros aprendidos.
    modelo.fit(X_train, y_train, **fit_kwargs)

    # Se calcula el tiempo transcurrido restando el tiempo inicial del actual.
    elapsed = time.time() - start

    # Se retorna el tiempo redondeado a 4 decimales (precisión de 0.1 ms,
    # suficiente para comparar eficiencia entre modelos).
    return round(elapsed, 4)


def obtener_scores(model, X):
    """Scores continuos para AUC/KS, incluidas pipelines Ridge."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def GetBestModel(models, config):
    """Entrada de selección compatible con el enunciado; usa la política de negocio común."""
    from src.training import GetBestModel as select
    return select(models, config)
