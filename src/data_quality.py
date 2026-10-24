
"""Validación de esquemas de los CSV de desarrollo temporal."""

from pathlib import Path
import pandas as pd


def validar_esquema_temporal(config: dict) -> dict:
    """Verifica que los CSV de desarrollo tengan las mismas columnas."""

    carpeta = Path(config["input_dir"])
    archivos = config["temporal"]["development_files"]
    objetivo = config["target_col"]

    if not archivos:
        raise ValueError("No se definieron archivos de desarrollo.")

    columnas_referencia = None
    reporte = {}

    for nombre in archivos:
        ruta = carpeta / nombre

        if not ruta.is_file():
            raise FileNotFoundError(f"No existe el archivo: {ruta}")

        df = pd.read_csv(
            ruta,
            sep=config.get("csv_sep", ","),
            encoding=config.get("csv_encoding", "utf-8"),
            nrows=100
        )

        if df.empty:
            raise ValueError(f"{nombre}: archivo sin registros de muestra.")

        if objetivo not in df.columns:
            raise ValueError(
                f"{nombre}: falta la variable objetivo '{objetivo}'."
            )

        columnas = set(df.columns)

        if columnas_referencia is None:
            columnas_referencia = columnas
        elif columnas != columnas_referencia:
            faltantes = sorted(columnas_referencia - columnas)
            adicionales = sorted(columnas - columnas_referencia)

            raise ValueError(
                f"{nombre}: esquema incompatible. "
                f"Faltantes: {faltantes}. Adicionales: {adicionales}"
            )

        reporte[nombre] = {
            "columnas": len(df.columns),
            "columnas_validas": True,
            "filas_inspeccionadas": len(df)
        }

    return reporte
