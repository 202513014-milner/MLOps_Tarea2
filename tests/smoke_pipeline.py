"""Prueba de integración del pipeline P3 con datos sintéticos (sin Airflow ni Docker).

Verifica: compuerta de validación, las 8 etapas, trazabilidad en MLflow (commit, hash de datos,
imagen Docker, config, ejecución del DAG), logs de cómputo, idempotencia (segunda ejecución salta;
reintento tras fallo no duplica runs), GetBestModel por métrica de negocio y promoción en el registro.

Ejecutar:  python tests/smoke_pipeline.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def hacer_datos(directorio, filas=3000, seed=0, corromper=None):
    rng = np.random.default_rng(seed)
    for i in range(1, 11):
        x1, x2, x3 = rng.normal(size=(3, filas))
        logit = -2.6 + 1.1 * x1 - 0.8 * x2 + 0.5 * x3 * (i / 10)  # leve deriva en el tiempo
        y = (rng.random(filas) < 1 / (1 + np.exp(-logit))).astype(int)
        df = pd.DataFrame({'partition': i, 'tip_doc': 1, 'key_value': np.arange(filas), 'codunicocli': np.arange(filas),
                           'prob_value_contact': rng.random(filas), 'fch_creacion': '2026-01-01',
                           'p_fecinformacion': 20260000 + i * 100 + 28, 'monto': x1 * 100 + 500, 'ingreso_neto': x2 * 50 + 2000,
                           'tea': x3, 'ruido': rng.normal(size=filas),
                           'ent_1erlntcrallsfm01': rng.choice(['A', 'B', 'C', None], filas),
                           'grp_campecs06m': rng.choice(['G1', 'G2', 'G3'], filas), 'target': y})
        df.loc[rng.random(filas) < 0.1, 'ingreso_neto'] = np.nan
        if corromper == i:
            df['target'] = 0  # tasa de positivos 0 -> la compuerta debe rechazarlo
        df.to_csv(Path(directorio) / f'p{i}_extrac.csv', index=False)


def configuracion(tmp):
    from src.stages import cargar_config
    over = {'input_dir': str(Path(tmp) / 'input'), 'output_dir': str(Path(tmp) / 'output'),
            'sample_fraction': 0.5, 'hpo': {'cv_folds': 2, 'verbose': 0, 'optuna': {'n_trials': 3, 'timeout_seconds': 60}},
            'parallel': {'workers': 2}, 'backtesting': {'sample_fraction': 0.5},
            'compute_logging': {'interval_seconds': 0.5},
            'validation': {'min_rows': 500, 'min_positive_rate': 0.005}}
    return cargar_config('config/config.yaml', over)


def main():
    """Ejecuta la prueba de integración vigente; hacer_datos sigue disponible."""
    from test_integracion import run
    run()

if __name__=='__main__':main()
