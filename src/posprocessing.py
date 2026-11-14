"""Posprocesamiento prescriptivo (puntuación TLV y grupos de ejecución). Parámetros en config.yaml."""
import numpy as np
import pandas as pd


def puntuar(df, scores, cfg):
    """Agrega prob, prob_frescura, puntuacion_tlv; no asigna grupos."""
    p = cfg['postprocessing']
    out = df.copy()
    out['prob'] = np.asarray(scores, float)
    fr = p['freshness_prob']
    out['prob_frescura'] = out[p['freshness_col']].map(fr).fillna(fr['default']).astype(float)
    contact = pd.to_numeric(out[p['contact_col']], errors='coerce').fillna(p['contact_fill'])
    monto = pd.to_numeric(out[p['amount_col']], errors='coerce').fillna(0).clip(lower=0)
    out['puntuacion_tlv'] = out['prob'] * contact * np.log(monto + 1) * out['prob_frescura']
    return out


def grupos_mes(puntuacion, cfg):
    """Grupo 1 (5 % de mayor puntuación) a 10, con los cortes dist_ge del mes.

    Se usa rank(method='first') porque muchos clientes empatan en puntuación baja
    y pd.qcut fallaría con cortes repetidos.
    """
    q = cfg['postprocessing']['dist_ge']
    return pd.qcut(puntuacion.rank(method='first'), q=q, labels=list(range(10, 0, -1))).astype(int)
