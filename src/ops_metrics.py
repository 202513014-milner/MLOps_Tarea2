"""Métrica de negocio (D4) y tipo de decay (D1).

Decisión del grupo (ver docs/decisiones_diseno.md):
- La métrica primaria de negocio es la **captura de ventas en el top-k% de scores**
  (``capture_rate_top_k``): de todas las ventas reales, qué fracción cae dentro del
  k% de clientes con mayor score. Es lo que importa cuando el contact center solo
  puede llamar a una fracción limitada de la cartera.
- También se reportan ``lift_top_k`` y ``precision_top_k``.
- El decay puede ser relativo (default, el de la P2) o absoluto.
"""
import numpy as np


def business_metrics(y_true, scores, top_fraction=0.10):
    """Métricas de negocio sobre el top-k% de scores. ``scores`` pueden ser no-probabilidades."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    if not 0 < top_fraction <= 1:
        raise ValueError('top_fraction debe estar en (0, 1].')
    if len(y) != len(s) or not np.isfinite(s).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('Etiquetas/scores inválidos.')
    n = len(y)
    if n == 0 or y.sum() == 0:
        return {'capture_rate_top_k': None, 'lift_top_k': None, 'precision_top_k': None,
                'top_fraction': top_fraction}
    k = max(1, int(round(n * top_fraction)))
    # Orden estable descendente: los empates no dependen del orden de memoria.
    order = np.argsort(-s, kind='mergesort')[:k]
    hits = int(y[order].sum())
    base_rate = y.mean()
    precision = hits / k
    return {'capture_rate_top_k': hits / int(y.sum()),
            'lift_top_k': precision / base_rate,
            'precision_top_k': precision,
            'top_fraction': top_fraction}


def calcular_decay(reference, value, kind='relative'):
    """Decay entre una métrica de referencia y otra observada.

    relative: 100 * (reference - value) / reference   (en %, el de la P1/P2)
    absolute: 100 * (reference - value)               (en puntos de AUC)
    Un valor negativo indica mejora.
    """
    if reference is None or value is None:
        return None
    if kind == 'absolute':
        return 100.0 * (reference - value)
    if kind != 'relative':
        raise ValueError("decay.type debe ser 'relative' o 'absolute'.")
    if reference == 0:
        return None
    return 100.0 * (reference - value) / reference


def decay_kind(config):
    return (config.get('decay') or {}).get('type', 'relative')


def top_fraction(config):
    return float((config.get('business_metric') or {}).get('top_fraction', 0.10))


def primary_metric_name(config):
    return (config.get('business_metric') or {}).get('primary', 'capture_rate_top_k')
