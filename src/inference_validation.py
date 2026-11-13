
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

import numpy as np

FACTORES_FRESHNESS = {
    "G1": 0.066,
    "G2": 0.028,
    "G3": 0.022,
    "G4": 0.008,
}


def validar_tlv(df: pd.DataFrame, tolerancia=1e-8):
    """Verifica que el TLV corresponda a la fórmula de negocio."""

    columnas = [
        "probabilidad",
        "prob_value_contact",
        "monto",
        "grp_campecs06m",
        "tlv",
    ]

    faltantes = [c for c in columnas if c not in df.columns]

    if faltantes:
        raise InferenceValidationError(
            f"Faltan columnas: {faltantes}"
        )

    validar_probabilidades(df, "probabilidad")
    validar_probabilidades(df, "prob_value_contact")

    monto = pd.to_numeric(df["monto"], errors="coerce")
    tlv = pd.to_numeric(df["tlv"], errors="coerce")

    if monto.isna().any() or (~np.isfinite(monto)).any() or (monto < 0).any():
        raise InferenceValidationError("Existen montos inválidos.")

    if tlv.isna().any() or (~np.isfinite(tlv)).any():
        raise InferenceValidationError("Existen valores TLV inválidos.")

    grupos = df["grp_campecs06m"].fillna("OTROS").astype(str).str.upper()
    freshness = grupos.map(FACTORES_FRESHNESS).fillna(0.004)

    esperado = (
        pd.to_numeric(df["probabilidad"])
        * pd.to_numeric(df["prob_value_contact"])
        * np.log1p(monto)
        * freshness
    )

    diferencias = np.abs(tlv - esperado)

    if (diferencias > tolerancia).any():
        raise InferenceValidationError(
            f"TLV incorrecto en {int((diferencias > tolerancia).sum())} registros."
        )

    return {
        "ok": True,
        "filas_validadas": len(df),
        "diferencia_maxima": float(diferencias.max()),
    }
