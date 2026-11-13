
import unittest
import pandas as pd
import numpy as np

from src.inference_validation import (
    validar_probabilidades,
    validar_tlv,
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
    
    def test_tlv_correcto(self):
        df = pd.DataFrame({
            "probabilidad": [0.8, 0.6, 0.4, 0.7, 0.5],
            "prob_value_contact": [0.9] * 5,
            "monto": [1000] * 5,
            "grp_campecs06m": ["G1", "G2", "G3", "G4", "OTROS"],
        })

        factores = np.array([0.066, 0.028, 0.022, 0.008, 0.004])

        df["tlv"] = (
            df["probabilidad"]
            * df["prob_value_contact"]
            * np.log1p(df["monto"])
            * factores
        )

        resultado = validar_tlv(df)

        self.assertTrue(resultado["ok"])
        self.assertEqual(resultado["filas_validadas"], 5)

    def test_tlv_incorrecto(self):
        df = pd.DataFrame({
            "probabilidad": [0.8],
            "prob_value_contact": [0.9],
            "monto": [1000],
            "grp_campecs06m": ["G1"],
            "tlv": [999.0],
        })

        with self.assertRaises(InferenceValidationError):
            validar_tlv(df)



if __name__ == "__main__":
    unittest.main()
