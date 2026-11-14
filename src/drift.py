"""Métricas de drift con distribuciones fijas calculadas desde una referencia."""

import numpy as np
import pandas as pd

OTHER = "__P3_OTHER__"
MISSING = "__P3_MISSING__"


def psi_kl(conteos_referencia, conteos_actuales, epsilon=1e-6):
    """Calcula PSI y KL(referencia || actual) para dos vectores de conteos."""
    ref = np.asarray(conteos_referencia, dtype=float)
    act = np.asarray(conteos_actuales, dtype=float)

    if ref.ndim != 1 or act.ndim != 1 or ref.size != act.size:
        raise ValueError("Las distribuciones deben ser vectores de igual longitud.")
    if ref.size == 0 or (ref < 0).any() or (act < 0).any():
        raise ValueError("Los conteos deben ser no negativos y no vacíos.")
    if ref.sum() == 0 or act.sum() == 0:
        raise ValueError("No se puede comparar una distribución sin observaciones.")

    p = ref / ref.sum()
    q = act / act.sum()

    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()

    psi = np.sum((q - p) * np.log(q / p))
    kl = np.sum(p * np.log(p / q))
    return float(psi), float(kl)


def _numericos(valores):
    arr = pd.to_numeric(pd.Series(valores), errors="coerce").to_numpy(dtype=float)
    validos = np.isfinite(arr)
    return arr[validos], int((~validos).sum())


def _conteos_numericos(valores, cortes):
    x, nulos = _numericos(valores)
    cortes = np.asarray(cortes, dtype=float)
    codigos = np.searchsorted(cortes, x, side="right")
    bins = np.bincount(codigos, minlength=len(cortes) + 1)
    return np.append(bins, nulos)


def ajustar_referencia_numerica(valores, bins=10):
    """Fija cortes por cuantiles de la referencia y guarda su distribución."""
    x, nulos = _numericos(valores)

    if len(x):
        cuantiles = np.quantile(x, np.linspace(0, 1, max(2, int(bins)) + 1))
        cortes = np.unique(cuantiles[1:-1])
        cortes = cortes[(cortes > x.min()) & (cortes < x.max())]
    else:
        cortes = np.array([], dtype=float)

    return {
        "tipo": "numerica",
        "cortes": cortes.tolist(),
        "distribucion": _conteos_numericos(valores, cortes).tolist(),
        "n_referencia": int(len(x) + nulos),
    }


def _etiqueta_categoria(valor):
    if pd.isna(valor):
        return MISSING
    return str(valor)


def _conteos_categoricos(valores, categorias):
    conocidas = set(categorias) - {OTHER, MISSING}
    etiquetas = [
        (etiqueta if etiqueta == MISSING or etiqueta in conocidas else OTHER)
        for etiqueta in map(_etiqueta_categoria, valores)
    ]
    return (
        pd.Series(etiquetas)
        .value_counts()
        .reindex(categorias, fill_value=0)
        .to_numpy(dtype=float)
    )


def ajustar_referencia_categorica(valores, max_categorias=20):
    """Fija las categorías frecuentes de la referencia; agrupa las demás en OTHER."""
    etiquetas = pd.Series(list(map(_etiqueta_categoria, valores)))
    frecuentes = (
        etiquetas[etiquetas != MISSING]
        .value_counts()
        .head(int(max_categorias))
        .index
        .tolist()
    )
    categorias = frecuentes + [OTHER, MISSING]

    return {
        "tipo": "categorica",
        "categorias": categorias,
        "distribucion": _conteos_categoricos(valores, categorias).tolist(),
        "n_referencia": int(len(etiquetas)),
    }


def comparar_con_referencia(perfil, valores_actuales, epsilon=1e-6):
    """Compara una variable mensual contra el perfil fijo de referencia."""
    if perfil["tipo"] == "numerica":
        actuales = _conteos_numericos(valores_actuales, perfil["cortes"])
    elif perfil["tipo"] == "categorica":
        actuales = _conteos_categoricos(valores_actuales, perfil["categorias"])
    else:
        raise ValueError(f"Tipo de variable desconocido: {perfil['tipo']}")

    psi, kl = psi_kl(perfil["distribucion"], actuales, epsilon)
    return {
        "psi": psi,
        "kl": kl,
        "distribucion_actual": actuales.tolist(),
        "n_actual": int(actuales.sum()),
    }


def nivel_alerta(valor, moderado=0.10, alto=0.25):
    """Clasifica una métrica como estable, moderada o alta."""
    if valor >= alto:
        return "alto"
    if valor >= moderado:
        return "moderado"
    return "estable"