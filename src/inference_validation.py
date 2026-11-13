
"""Validación de resultados de inferencia - Práctica 3."""

import pandas as pd


class InferenceValidationError(ValueError):
    """Error de calidad en los resultados de inferencia."""


def validar_probabilidades(df: pd.DataFrame, columna="probabilidad"):
    """Verifica probabilidades válidas entre 0 y 1."""

    if df.empty:
        raise InferenceValidationError(
            "El archivo de inferencia no contiene registros."
        )

    if columna not in df.columns:
        raise InferenceValidationError(
            f"No existe la columna '{columna}'."
        )

    probabilidades = pd.to_numeric(
        df[columna], errors="coerce"
    )

    if probabilidades.isna().any():
        raise InferenceValidationError(
            "Existen probabilidades nulas o no numéricas."
        )

    fuera_rango = ~probabilidades.between(0, 1)

    if fuera_rango.any():
        raise InferenceValidationError(
            f"Se encontraron {fuera_rango.sum()} probabilidades fuera de rango."
        )

    return {
        "ok": True,
        "filas_validadas": len(df),
        "probabilidad_minima": float(probabilidades.min()),
        "probabilidad_maxima": float(probabilidades.max()),
    }
