
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_quality import validar_esquema_temporal


class TestDataQuality(unittest.TestCase):

    def test_esquema_correcto(self):
        """Dos archivos con las mismas columnas deben pasar."""
        with tempfile.TemporaryDirectory() as carpeta:
            archivos = ["p1.csv", "p2.csv"]

            for nombre in archivos:
                pd.DataFrame({
                    "edad": [25, 35],
                    "ingreso": [1000, 2000],
                    "target": [0, 1]
                }).to_csv(Path(carpeta) / nombre, index=False)

            config = {
                "input_dir": carpeta,
                "target_col": "target",
                "temporal": {"development_files": archivos}
            }

            resultado = validar_esquema_temporal(config)
            self.assertEqual(len(resultado), 2)

    def test_esquema_incompatible(self):
        """Dos archivos con columnas diferentes deben fallar."""
        with tempfile.TemporaryDirectory() as carpeta:
            pd.DataFrame({
                "edad": [25],
                "target": [0]
            }).to_csv(Path(carpeta) / "p1.csv", index=False)

            pd.DataFrame({
                "ingreso": [1000],
                "target": [1]
            }).to_csv(Path(carpeta) / "p2.csv", index=False)

            config = {
                "input_dir": carpeta,
                "target_col": "target",
                "temporal": {
                    "development_files": ["p1.csv", "p2.csv"]
                }
            }

            with self.assertRaisesRegex(ValueError, "esquema incompatible"):
                validar_esquema_temporal(config)


if __name__ == "__main__":
    unittest.main()
