
import unittest
import pandas as pd

from src.inference_validation import (
    validar_probabilidades,
    InferenceValidationError,
)


class TestInferenciaP3(unittest.TestCase):

    def test_probabilidades_validas(self):
        df = pd.DataFrame({
            "probabilidad": [0.0, 0.25, 0.8, 1.0]
        })
        resultado = validar_probabilidades(df)

        self.assertTrue(resultado["ok"])
        self.assertEqual(resultado["filas_validadas"], 4)

    def test_probabilidades_fuera_de_rango(self):
        df = pd.DataFrame({
            "probabilidad": [-0.2, 0.5, 1.3]
        })
        with self.assertRaises(InferenceValidationError):
            validar_probabilidades(df)

    def test_probabilidades_nulas(self):
        df = pd.DataFrame({
            "probabilidad": [0.2, None, 0.9]
        })
        with self.assertRaises(InferenceValidationError):
            validar_probabilidades(df)

    def test_columna_inexistente(self):
        df = pd.DataFrame({
            "score": [0.4, 0.7]
        })
        with self.assertRaises(InferenceValidationError):
            validar_probabilidades(df)


if __name__ == "__main__":
    unittest.main()
