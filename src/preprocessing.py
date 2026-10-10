"""
=============================================================================
Módulo de Preprocesamiento — preprocessing.py
=============================================================================
Responsabilidad:
    Recibir los CSV crudos del dataset de venta financiera y devolver los arrays
    (X_train, X_test, y_train, y_test) listos para entrenamiento.

Pasos del pipeline:
    1. Carga y concatenación de todos los CSV del directorio de entrada.
    2. Muestreo estratificado opcional.
    3. Eliminación de columnas identificadoras / exógenas.
    4. Imputación de valores nulos:
       - Numéricas: -9999999 (convención del notebook base).
       - Categóricas: 'SV' (Sin Valor).
    5. Encoding de variables categóricas:
       - OHE para ent_1erlntcrallsfm01.
       - Label Encoding para grp_campecs06m.
    6. Conversión de tipos a float32.
    7. Split estratificado en train/test (70/30).

Librerías: pandas, numpy, os, glob, sklearn.model_selection.train_test_split,
    sklearn.preprocessing.LabelEncoder
=============================================================================
"""

# ── Importación de librerías estándar ──────────────────────────────────────
# os y glob se usan para navegar el sistema de archivos y encontrar
# los CSV de entrada mediante patrones (wildcards).
import os
import glob

# ── Importación de librerías de terceros ───────────────────────────────────
# pandas: manipulación tabular (DataFrames).
# numpy: operaciones numéricas (no se usa explícitamente aquí, pero se
#         importa por si se necesita en operaciones vectorizadas internas).
import pandas as pd
import numpy as np

# ── Importación de utilidades de scikit-learn ──────────────────────────────
# train_test_split: divide datos en train/test manteniendo proporciones
#                   de la variable objetivo (stratify).
# LabelEncoder: convierte categorías textuales a enteros consecutivos
#               (0, 1, 2, …), necesario para modelos que no aceptan texto.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ==========================================================================
# FUNCIÓN 1: cargar_datos
# ==========================================================================
# QUÉ HACE: Lee todos los archivos CSV de un directorio y los une en un
#            único DataFrame.
# POR QUÉ: Los datos originales vienen particionados en múltiples archivos
#           (p1_extrac.csv … p10_extrac.csv). Se necesita un DataFrame
#           consolidado para aplicar el pipeline completo.
# CRITERIO: Se usa glob con patrón '*.csv' para capturar cualquier CSV
#           presente. Se ordena alfabéticamente para reproducibilidad.
# ==========================================================================
def cargar_datos(input_dir: str) -> pd.DataFrame:
    """
    Lee todos los archivos CSV de un directorio y los une en un único DataFrame.

    El dataset de venta financiera se distribuye en múltiples particiones CSV
    (p1_extrac.csv … p10_extrac.csv). Esta función las detecta por extensión
    .csv, las ordena alfabéticamente para reproducibilidad, las carga una a
    una y las concatena verticalmente en un único DataFrame consolidado.

    Parámetros
    ----------
    input_dir : str
        Ruta al directorio que contiene los archivos CSV de entrada.
        Ej: 'data/input' (relativa al directorio raíz del proyecto).

    Retorna
    -------
    pd.DataFrame
        DataFrame consolidado con todas las filas de todos los CSV
        encontrados, con índice reiniciado desde 0.

    Lanza
    -----
    FileNotFoundError
        Si no se encuentra ningún archivo .csv en input_dir.
    """
    # Construir el patrón de búsqueda: "<directorio>/*.csv"
    patron_csv = os.path.join(input_dir, '*.csv')

    # Obtener la lista de archivos que coinciden, ordenada alfabéticamente
    # para garantizar un orden determinista en cada ejecución.
    archivos_csv = sorted(glob.glob(patron_csv))

    # Validación: si no hay archivos CSV, lanzar error descriptivo
    # para que el usuario sepa exactamente qué falta y dónde colocarlos.
    if len(archivos_csv) == 0:
        raise FileNotFoundError(
            f"No se encontraron archivos CSV en: {input_dir}\n"
            f"Coloque los archivos CSV (p1_extrac.csv, ..., p10_extrac.csv) "
            f"en el directorio data/input/"
        )

    # Informar al usuario qué directorio se está procesando y cuántos
    # archivos se encontraron (trazabilidad del pipeline).
    print(f"[Preprocessing] Directorio de entrada: {input_dir}")
    print(f"[Preprocessing] Archivos CSV encontrados: {len(archivos_csv)}")

    # Lista donde se acumularán los DataFrames parciales antes de concatenar.
    lista_dfs = []

    # Iterar sobre cada archivo CSV encontrado.
    for archivo in archivos_csv:
        # Extraer solo el nombre del archivo (sin la ruta) para el log.
        nombre = os.path.basename(archivo)

        # Leer el CSV completo en un DataFrame parcial.
        df_parcial = pd.read_csv(archivo)

        # Mostrar las dimensiones de cada archivo para verificar la carga.
        print(f"  - {nombre}: {df_parcial.shape[0]:,} filas × {df_parcial.shape[1]} columnas")

        # Agregar el DataFrame parcial a la lista.
        lista_dfs.append(df_parcial)

    # Concatenar todos los DataFrames verticalmente (axis=0 por defecto).
    # ignore_index=True reinicia el índice desde 0 para evitar duplicados.
    df = pd.concat(lista_dfs, ignore_index=True)

    # Mostrar las dimensiones del dataset consolidado y su uso de memoria.
    # Esto ayuda a detectar si el dataset es demasiado grande para la RAM.
    print(f"[Preprocessing] Dataset consolidado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    print(f"[Preprocessing] Memoria total: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return df


# ==========================================================================
# FUNCIÓN 2: muestreo_estratificado
# ==========================================================================
# QUÉ HACE: Reduce el tamaño del dataset tomando una fracción aleatoria,
#            pero manteniendo la misma proporción de clases del target.
# POR QUÉ: Permite hacer pruebas rápidas con un subconjunto representativo
#           sin sesgar la distribución de la variable objetivo.
# CRITERIO: Se usa train_test_split con stratify para garantizar que el
#           porcentaje de positivos/negativos se conserve en la muestra.
#           Si sample_fraction es None o >= 1.0, se usa el 100% de la data.
# ==========================================================================
def muestreo_estratificado(
    df: pd.DataFrame,
    target_col: str,
    sample_fraction,
    random_seed: int
) -> pd.DataFrame:
    """
    Reduce el tamaño del DataFrame tomando una fracción estratificada del target.

    Permite hacer pruebas rápidas con un subconjunto representativo del dataset
    completo (~3.2M filas) sin sesgar la distribución de la variable objetivo.
    Si sample_fraction es None o >= 1.0, se usa el 100 % de la data original.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame completo cargado desde los CSV.
    target_col : str
        Nombre de la columna objetivo (variable a predecir). Se usa para la
        estratificación, garantizando que la muestra mantenga la misma
        proporción de positivos/negativos que el dataset original.
    sample_fraction : float or None
        Fracción del dataset a conservar (0.0 < sample_fraction < 1.0).
        Ejemplo: 0.10 toma el 10 % de las filas manteniendo la proporción del
        target. Si es None o >= 1.0, se devuelve el DataFrame sin modificar.
    random_seed : int
        Semilla aleatoria para reproducibilidad del muestreo.

    Retorna
    -------
    pd.DataFrame
        DataFrame muestreado con índice reiniciado, o el DataFrame original
        si sample_fraction es None o >= 1.0.
    """
    # Si no se especifica fracción o es >= 1.0, no se hace muestreo.
    # Esto permite desactivar el muestreo sin cambiar el código.
    if sample_fraction is None or sample_fraction >= 1.0:
        print(f"[Preprocessing] Muestreo: desactivado (usando 100% de la data)")
        return df

    # Guardar la cantidad original de filas para reportar la reducción.
    n_original = len(df)

    # Usar train_test_split solo para obtener la fracción deseada.
    # stratify=df[target_col] asegura que la proporción de la clase
    # objetivo se mantenga idéntica en la muestra.
    # Se descarta la segunda partición (_) porque no se necesita.
    df_muestra, _ = train_test_split(
        df, train_size=sample_fraction, random_state=random_seed, stratify=df[target_col],
    )

    # Reiniciar el índice para que sea consecutivo (0, 1, 2, …).
    # drop=True evita que el índice anterior se convierta en columna.
    df_muestra = df_muestra.reset_index(drop=True)

    # Reportar la reducción y verificar que la proporción del target
    # se mantuvo entre el dataset original y la muestra.
    print(f"[Preprocessing] Muestreo estratificado: {sample_fraction*100:.0f}% de la data")
    print(f"  - Original:  {n_original:,} filas")
    print(f"  - Muestra:   {len(df_muestra):,} filas")
    print(f"  - % target original: {df[target_col].mean()*100:.2f}%")
    print(f"  - % target muestra:  {df_muestra[target_col].mean()*100:.2f}%")

    return df_muestra


# ==========================================================================
# FUNCIÓN 3: eliminar_columnas
# ==========================================================================
# QUÉ HACE: Remueve del DataFrame las columnas que no deben participar
#            en el entrenamiento del modelo.
# POR QUÉ: Columnas como identificadores de cliente, fechas de extracción
#           o variables exógenas no aportan información predictiva y pueden
#           causar data leakage o sobreajuste.
# CRITERIO: Se recibe una lista explícita (cols_excluir) definida en la
#           configuración. Solo se eliminan las que realmente existen en
#           el DataFrame (errors='ignore' como respaldo adicional).
# ==========================================================================
def eliminar_columnas(df: pd.DataFrame, cols_excluir: list) -> pd.DataFrame:
    """
    Remueve del DataFrame las columnas identificadoras y variables exógenas.

    Columnas como IDs de cliente (codunicocli), fechas de extracción
    (fch_creacion, p_fecinformacion) o particiones internas no aportan
    capacidad predictiva y pueden causar data leakage o sobreajuste.
    La lista de columnas a excluir se define en config.yaml (cols_excluir).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame del que se eliminarán las columnas.
    cols_excluir : list of str
        Lista de nombres de columnas a eliminar. Solo se eliminan las que
        existan en df; las inexistentes se ignoran sin lanzar error.

    Retorna
    -------
    pd.DataFrame
        DataFrame sin las columnas especificadas.
    """
    # Filtrar: quedarse solo con las columnas que efectivamente existen
    # en el DataFrame, para evitar errores si alguna ya fue removida.
    cols_presentes = [c for c in cols_excluir if c in df.columns]

    # Informar cuáles columnas se van a eliminar.
    print(f"[Preprocessing] Eliminando {len(cols_presentes)} columnas: {cols_presentes}")

    # Eliminar las columnas. errors='ignore' es un respaldo: si alguna
    # columna no existe, simplemente la ignora sin lanzar excepción.
    df = df.drop(columns=cols_presentes, errors='ignore')

    # Mostrar la forma resultante del DataFrame tras la eliminación.
    print(f"[Preprocessing] Shape tras eliminar columnas: {df.shape}")

    return df


# ==========================================================================
# FUNCIÓN 4: imputar_nulos
# ==========================================================================
# QUÉ HACE: Rellena los valores faltantes (NaN) en el DataFrame.
# POR QUÉ: La mayoría de modelos de ML no aceptan valores nulos. Es
#           necesario imputarlos antes del entrenamiento.
# CRITERIO:
#   - Variables numéricas: se imputan con -9999999 (convención del notebook
#     base). Este valor extremo permite que el modelo lo trate como una
#     categoría especial de "dato ausente" sin confundirlo con valores reales.
#   - Variables categóricas (tipo object): se imputan con 'SV' (Sin Valor),
#     una etiqueta explícita que indica la ausencia de información.
# ==========================================================================
def imputar_nulos(
    df: pd.DataFrame,
    cols_cat_ohe: list,
    cols_cat_label: list
) -> pd.DataFrame:
    """
    Rellena los valores faltantes (NaN) del DataFrame con centinelas convencionales.

    La mayoría de modelos de ML no aceptan NaN como entrada. Se aplican dos
    estrategias diferenciadas según el tipo de columna:
    - Variables numéricas: se imputan con -9999999 (valor extremo fuera del
      rango natural de las features), permitiendo al modelo tratarlo como
      una categoría especial de "dato ausente".
    - Variables categóricas (dtype object): se imputan con 'SV' (Sin Valor),
      etiqueta explícita que indica ausencia de información textual.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con posibles valores NaN.
    cols_cat_ohe : list of str
        Columnas categóricas que recibirán One-Hot Encoding. Se usan para
        identificar qué columnas son categóricas (y excluirlas de la
        imputación numérica).
    cols_cat_label : list of str
        Columnas categóricas que recibirán Label Encoding. Mismo propósito
        que cols_cat_ohe para la clasificación de columnas.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame sin valores NaN.
    """
    # Crear copia para no modificar el DataFrame original (evitar side effects).
    df = df.copy()

    # Unir todas las columnas categóricas (OHE + Label) en un conjunto
    # para poder identificar las columnas numéricas por exclusión.
    todas_cat = set(cols_cat_ohe + cols_cat_label)

    # Columnas numéricas: aquellas que NO están en la lista de categóricas
    # Y cuyo dtype NO es 'object' (texto). Estas se imputan con -9999999.
    cols_numericas = [c for c in df.columns if c not in todas_cat and df[c].dtype != 'object']

    # Columnas de tipo object: son las categóricas detectadas por su dtype.
    # Estas se imputan con 'SV'.
    cols_object = [c for c in df.columns if df[c].dtype == 'object']

    # Contar el total de nulos numéricos ANTES de imputar (para el reporte).
    nulos_antes = df[cols_numericas].isna().sum().sum()

    # Imputar cada columna numérica con -9999999.
    for col in cols_numericas:
        df[col] = df[col].fillna(-9999999)
    print(f"[Preprocessing] Imputados {nulos_antes} nulos numéricos con -9999999")

    # Imputar cada columna de tipo object con 'SV' solo si tiene nulos.
    for col in cols_object:
        n_nulos = df[col].isna().sum()
        if n_nulos > 0:
            df[col] = df[col].fillna('SV')
            print(f"[Preprocessing] Imputados {n_nulos} nulos en '{col}' con 'SV'")

    return df


# ==========================================================================
# FUNCIÓN 5: aplicar_encoding
# ==========================================================================
# QUÉ HACE: Transforma las variables categóricas en representaciones
#            numéricas que los modelos de ML puedan procesar.
# POR QUÉ: Los algoritmos de ML (XGBoost, LogReg, etc.) requieren
#           entradas numéricas. Las variables de texto deben codificarse.
# CRITERIO:
#   - One-Hot Encoding (OHE) para cols_cat_ohe (ej: ent_1erlntcrallsfm01):
#     Se agrupan las categorías poco frecuentes bajo 'OTRO', conservando
#     solo 'INTERBANK' como categoría principal. Esto reduce la
#     dimensionalidad y evita columnas dummy con muy pocos casos.
#   - Label Encoding para cols_cat_label (ej: grp_campecs06m):
#     Convierte cada categoría a un entero (0, 1, 2, …). Es adecuado
#     para variables ordinales o cuando el modelo (como XGBoost) puede
#     manejar este tipo de codificación sin asumir orden.
# ==========================================================================
def aplicar_encoding(
    df: pd.DataFrame,
    cols_cat_ohe: list,
    cols_cat_label: list
) -> pd.DataFrame:
    """
    Transforma variables categóricas en representaciones numéricas para ML.

    Se aplican dos estrategias de encoding según la naturaleza de cada columna:

    One-Hot Encoding (OHE) para cols_cat_ohe (ej: ent_1erlntcrallsfm01):
        Las categorías poco frecuentes se agrupan bajo 'OTRO', conservando
        solo 'INTERBANK' como categoría principal. Esto reduce dimensionalidad
        y evita columnas dummy con muy pocos casos positivos.

    Label Encoding para cols_cat_label (ej: grp_campecs06m):
        Cada categoría se convierte a un entero (0, 1, 2, …). Adecuado para
        variables ordinales o cuando el modelo puede manejar esta codificación
        sin asumir un orden artificial (como XGBoost o CatBoost).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las columnas categóricas a codificar.
    cols_cat_ohe : list of str
        Columnas a codificar con One-Hot Encoding.
    cols_cat_label : list of str
        Columnas a codificar con Label Encoding.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame con las columnas categóricas reemplazadas por
        sus representaciones numéricas. Las columnas OHE originales se
        eliminan y se agregan sus columnas dummy.
    """
    # Crear copia para no modificar el DataFrame original.
    df = df.copy()

    # ── Aplicar One-Hot Encoding (OHE) ────────────────────────────────
    for col in cols_cat_ohe:
        # Si la columna no existe en el DataFrame, saltar a la siguiente.
        if col not in df.columns:
            continue

        # Definir las categorías principales que se conservarán como
        # columnas independientes. Las demás se agrupan como 'OTRO'.
        categorias_principales = ['INTERBANK']

        # Rellenar nulos con 'SV' antes de agrupar.
        df[col] = df[col].fillna('SV')

        # Reemplazar cualquier valor que NO esté en categorias_principales
        # con la etiqueta 'OTRO'. Esto reduce la cantidad de dummies.
        df.loc[~df[col].isin(categorias_principales), col] = 'OTRO'

        # Informar las categorías resultantes antes de crear las dummies.
        print(f"[Preprocessing] OHE en '{col}': categorías = {df[col].unique().tolist()}")

        # Crear las columnas dummy (una por categoría).
        dummies = pd.get_dummies(df[col], prefix=col)

        # Eliminar la columna original (ya fue reemplazada por las dummies).
        df = df.drop(columns=[col])

        # Agregar las columnas dummy al DataFrame.
        df = pd.concat([df, dummies], axis=1)

    # ── Aplicar Label Encoding ────────────────────────────────────────
    for col in cols_cat_label:
        # Si la columna no existe en el DataFrame, saltar.
        if col not in df.columns:
            continue

        # Crear una instancia de LabelEncoder.
        le = LabelEncoder()

        # Convertir la columna a string (por si tiene valores mixtos)
        # y aplicar fit_transform: ajusta las clases y transforma en un paso.
        df[col] = le.fit_transform(df[col].astype(str))

        # Informar cuántas clases únicas se codificaron.
        print(f"[Preprocessing] Label Encoding en '{col}': {len(le.classes_)} clases")

    return df


# ==========================================================================
# FUNCIÓN 6: convertir_tipos
# ==========================================================================
# QUÉ HACE: Convierte los tipos de datos de las columnas a formatos más
#            eficientes en memoria (float32 en lugar de float64).
# POR QUÉ: float64 usa el doble de memoria que float32. Para datasets
#           grandes, esta conversión puede reducir el consumo de RAM a
#           la mitad sin pérdida significativa de precisión.
# CRITERIO:
#   - Target: se convierte a int (0 o 1, clasificación binaria).
#   - Booleanas: se convierten a int (0/1) para compatibilidad con modelos.
#   - Enteras (int16/32/64): se convierten a float32.
#   - float64: se convierte a float32 y se redondea a 4 decimales.
# ==========================================================================
def convertir_tipos(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Convierte los tipos de datos de las columnas a formatos más eficientes.

    float64 usa el doble de memoria que float32. Para un dataset de ~3.2M filas
    y ~62 features, esta conversión puede reducir el consumo de RAM en ~50 %
    sin pérdida significativa de precisión para los modelos entrenados.

    Criterios de conversión por tipo:
    - target_col: → int (clasificación binaria requiere etiquetas enteras 0/1).
    - bool: → int (0/1) para compatibilidad con modelos que no aceptan bool.
    - int16/32/64: → float32 (los modelos de ML esperan entrada en float).
    - float64: → float32 redondeado a 4 decimales para reducir memoria y
      eliminar ruido numérico irrelevante.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con todas las transformaciones previas ya aplicadas.
    target_col : str
        Nombre de la columna objetivo, que se convierte a int en lugar de float.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame con tipos optimizados.
    """
    # Crear copia para no modificar el DataFrame original.
    df = df.copy()

    for col in df.columns:
        if col == target_col:
            df[col] = df[col].astype(int)
        elif df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
        elif df[col].dtype in ['int16', 'int32', 'int64']:
            df[col] = df[col].astype('float32')
        elif df[col].dtype in ['float64']:
            df[col] = df[col].astype('float32').round(4)

    print(f"[Preprocessing] Tipos convertidos. Memoria: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


# ==========================================================================
# FUNCIÓN 7: split_train_test
# ==========================================================================
# QUÉ HACE: Divide el DataFrame en conjuntos de entrenamiento (train)
#            y prueba (test), separando features (X) del target (y).
# POR QUÉ: Es fundamental evaluar el modelo con datos que NO vio durante
#           el entrenamiento. Sin esta separación, no se puede medir la
#           capacidad de generalización del modelo.
# CRITERIO: Se usa split estratificado (stratify=y) para que ambos
#           conjuntos mantengan la misma proporción de clases positivas
#           y negativas. El tamaño del test se define en la configuración
#           (por defecto 30%). La semilla (random_seed) garantiza
#           reproducibilidad.
# ==========================================================================
def split_train_test(
    df: pd.DataFrame,
    target_col: str,
    test_size: float,
    random_seed: int
):
    """
    Divide el DataFrame en conjuntos de entrenamiento y prueba estratificados.

    Es fundamental evaluar el modelo con datos que no vio durante el
    entrenamiento. El split estratificado garantiza que ambos conjuntos
    (train y test) mantengan la misma proporción de clases positivas y
    negativas que el dataset original, lo cual es crítico dado el fuerte
    desbalance del dataset (~3.3 % de positivos).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame completo con features y target, ya preprocesado.
    target_col : str
        Nombre de la columna objetivo. Se separa de las features antes del split.
    test_size : float
        Proporción del dataset reservada para test (0.0 a 1.0).
        Ejemplo: 0.30 reserva el 30 % para test (split 70/30).
    random_seed : int
        Semilla aleatoria para reproducibilidad del split.

    Retorna
    -------
    tuple : (X_train, X_test, y_train, y_test)
        X_train : pd.DataFrame — features de entrenamiento
        X_test  : pd.DataFrame — features de prueba
        y_train : pd.Series — etiquetas de entrenamiento
        y_test  : pd.Series — etiquetas de prueba
    """
    # Obtener la lista de columnas que son features (todo excepto el target).
    feature_cols = [c for c in df.columns if c != target_col]

    # Separar el DataFrame en matriz de features (X) y vector objetivo (y).
    X = df[feature_cols]
    y = df[target_col]

    # Realizar el split estratificado.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    # Reportar las dimensiones resultantes y las proporciones de la
    # clase positiva en cada conjunto para verificar la estratificación.
    print(f"[Preprocessing] Split completado:")
    print(f"  X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"  % positivos train: {y_train.mean()*100:.2f}%")
    print(f"  % positivos test:  {y_test.mean()*100:.2f}%")
    print(f"  Features: {len(feature_cols)}")

    return X_train, X_test, y_train, y_test


# ==========================================================================
# FUNCIÓN 8 (PRINCIPAL): run_preprocessing
# ==========================================================================
# QUÉ HACE: Orquesta todo el pipeline de preprocesamiento de principio a
#            fin, ejecutando cada paso en el orden correcto.
# POR QUÉ: Centraliza la lógica del pipeline en una sola función que
#           recibe un diccionario de configuración. Esto permite reutilizar
#           el pipeline con diferentes parámetros sin modificar el código.
# CRITERIO: El orden de ejecución es crítico (ver docstring completo).
# ==========================================================================
def run_preprocessing(config: dict):
    """
    Orquesta el pipeline completo de preprocesamiento de principio a fin.

    Recibe el diccionario de configuración (leído de config.yaml) y ejecuta
    los 7 pasos del pipeline en el orden correcto: carga → muestreo →
    eliminación de columnas → imputación → encoding → conversión de tipos →
    split train/test. Devuelve directamente los arrays listos para entrenar.

    El orden de ejecución es crítico:
    1. Cargar datos → se necesitan los datos antes de cualquier operación.
    2. Muestreo → reducir ANTES de procesar ahorra tiempo de cómputo.
    3. Eliminar columnas → quitar lo innecesario ANTES de imputar/codificar.
    4. Imputar nulos → los NaN deben resolverse ANTES del encoding.
    5. Encoding → convertir categorías a números ANTES de tipificar.
    6. Convertir tipos → optimizar memoria DESPUÉS de que todo sea numérico.
    7. Split → dividir AL FINAL, con los datos completamente listos.

    Parámetros
    ----------
    config : dict
        Diccionario de configuración con las claves:
        - 'input_dir' (str): ruta al directorio con los CSV.
        - 'target_col' (str): nombre de la columna objetivo.
        - 'test_size' (float): proporción para test (ej: 0.30).
        - 'random_seed' (int): semilla de reproducibilidad.
        - 'sample_fraction' (float, opcional): fracción a usar (None = 100 %).
        - 'cols_excluir' (list, opcional): columnas a eliminar.
        - 'cols_categoricas_ohe' (list, opcional): columnas para OHE.
        - 'cols_categoricas_label' (list, opcional): columnas para Label Enc.

    Retorna
    -------
    tuple : (X_train, X_test, y_train, y_test)
        Arrays preprocesados listos para entrenamiento y evaluación.
    """
    # Imprimir encabezado visual para delimitar la sección de preprocesamiento
    # en los logs del pipeline completo.
    print("=" * 70)
    print("  MÓDULO DE PREPROCESAMIENTO")
    print("=" * 70)

    # ── Extraer parámetros del diccionario de configuración ───────────
    input_dir       = config['input_dir']
    cols_excluir    = config.get('cols_excluir', [])
    cols_cat_ohe    = config.get('cols_categoricas_ohe', [])
    cols_cat_label  = config.get('cols_categoricas_label', [])
    target_col      = config['target_col']
    test_size       = config['test_size']
    random_seed     = config['random_seed']
    sample_fraction = config.get('sample_fraction', None)

    # ── Ejecución secuencial del pipeline ─────────────────────────────
    df = cargar_datos(input_dir)
    df = muestreo_estratificado(df, target_col, sample_fraction, random_seed)
    df = eliminar_columnas(df, cols_excluir)
    df = imputar_nulos(df, cols_cat_ohe, cols_cat_label)
    df = aplicar_encoding(df, cols_cat_ohe, cols_cat_label)
    df = convertir_tipos(df, target_col)
    X_train, X_test, y_train, y_test = split_train_test(df, target_col, test_size, random_seed)

    print(f"\n[Preprocessing] Pipeline completado exitosamente.")
    print("=" * 70)

    return X_train, X_test, y_train, y_test


# Práctica 2: carga exclusiva de los seis archivos de desarrollo.
def cargar_desarrollo_temporal(config):
    from pathlib import Path
    root = Path(config['input_dir'])
    temporal = config['temporal']
    train_files = temporal['development_files']
    future_files = temporal['backtesting_files']
    if len(train_files) != 6 or len(future_files) != 4 or set(train_files) & set(future_files):
        raise ValueError('Se requieren seis archivos de desarrollo y cuatro futuros disjuntos.')
    if len(set(train_files + future_files)) != 10:
        raise ValueError('Los diez nombres deben ser únicos.')
    for name in train_files + future_files:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    fraction = config.get('sample_fraction', 0.10)
    if not 0 < fraction <= 1:
        raise ValueError('sample_fraction debe estar en (0, 1].')
    target = config['target_col']
    seed = config.get('random_seed', 42)
    samples, report = [], []
    period = temporal.get('period_col')
    periods = []
    for index, name in enumerate(train_files):
        df = pd.read_csv(root / name, sep=config.get('csv_sep', ','),
                         encoding=config.get('csv_encoding', 'utf-8'))
        if target not in df or df[target].isna().any() or set(df[target].unique()) != {0, 1}:
            raise ValueError(f'{name}: target debe contener ambas clases 0 y 1 sin nulos.')
        if period:
            if period not in df or df[period].isna().any() or df[period].nunique() != 1:
                raise ValueError(f'{name}: debe contener exactamente un periodo en {period}.')
            periods.append(str(df[period].iloc[0]))
        selected = muestreo_estratificado(df, target, fraction, seed + index)
        report.append({'file': name, 'rows_original': len(df), 'rows_sample': len(selected),
                       'positive_rate_original': float(df[target].mean()),
                       'positive_rate_sample': float(selected[target].mean())})
        selected = selected.copy()
        selected['_source_month'] = index
        samples.append(selected)
    if periods and periods != sorted(set(periods)):
        raise ValueError('Los periodos de desarrollo no están en orden creciente.')
    data = pd.concat(samples, ignore_index=True)
    # Holdout estratificado por mes y clase, siempre dentro de meses 1–6.
    train_parts, val_parts = [], []
    for month, group in data.groupby('_source_month'):
        a, b = train_test_split(group, test_size=config.get('test_size', .30),
                               stratify=group[target], random_state=seed)
        train_parts.append(a); val_parts.append(b)
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    excluded = set(config.get('cols_excluir', [])) | {target, '_source_month'}
    if period:
        excluded.add(period)
    # Periodo e identificadores no son predictores.
    excluded.add('p_codmes')
    features = [c for c in data.columns if c not in excluded]
    if not features:
        raise ValueError('No quedaron predictores.')
    manifest = {'development': report, 'development_periods': periods,
                'reserved_backtesting_files': future_files,
                'future_files_loaded': False, 'sample_fraction': fraction,
                'feature_columns': features, 'rows_train': len(train), 'rows_validation': len(val),
                'chronology': 'Orden explícito declarado en config; confirmar meses reales del curso.'}
    return train[features], val[features], train[target], val[target], manifest
