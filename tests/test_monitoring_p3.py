import json
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.drift import (
    ajustar_referencia_categorica,
    ajustar_referencia_numerica,
    comparar_con_referencia,
    nivel_alerta,
    psi_kl,
)


def test_psi_kl_cero_con_distribuciones_iguales():
    psi, kl = psi_kl([50, 50], [50, 50])
    assert psi == pytest.approx(0.0)
    assert kl == pytest.approx(0.0)


def test_psi_kl_detectan_cambio():
    psi, kl = psi_kl([80, 20], [20, 80])
    assert psi > 0
    assert kl > 0


def test_niveles_de_alerta():
    assert nivel_alerta(0.05, moderado=0.10, alto=0.25) == "estable"
    assert nivel_alerta(0.10, moderado=0.10, alto=0.25) == "moderado"
    assert nivel_alerta(0.25, moderado=0.10, alto=0.25) == "alto"


def test_comparacion_numerica_y_categorica():
    perfil_num = ajustar_referencia_numerica([1, 2, 3, 4], bins=2)
    igual = comparar_con_referencia(perfil_num, [1, 2, 3, 4])
    distinto = comparar_con_referencia(perfil_num, [8, 9, 10, 11])

    assert igual["psi"] == pytest.approx(0.0)
    assert igual["kl"] == pytest.approx(0.0)
    assert distinto["psi"] > 0
    assert distinto["kl"] > 0

    perfil_cat = ajustar_referencia_categorica(["A", "A", "B", "B"])
    resultado_cat = comparar_con_referencia(
        perfil_cat, ["A", "A", "B", "B"]
    )
    assert resultado_cat["psi"] == pytest.approx(0.0)
    assert resultado_cat["kl"] == pytest.approx(0.0)


def test_monitoreo_procesa_contexto_y_registra_mlflow(tmp_path, monkeypatch):
    import src.inference as inference
    import src.monitoring_p3 as monitoring

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    inference_dir = output_dir / "inference" / "p7" / "corrida-test"
    input_dir.mkdir()
    inference_dir.mkdir(parents=True)

    pd.DataFrame({"x": [1, 2, 3, 4], "target": [0, 1, 0, 1]}).to_csv(
        input_dir / "p6_extrac.csv", index=False
    )
    pd.DataFrame({"x": [8, 9, 10]}).to_pickle(inference_dir / "datos.pkl")
    pd.DataFrame({"prob": [0.80, 0.85, 0.90]}).to_csv(
        inference_dir / "scores_clientes.csv", index=False
    )

    contexto = {
        "mes": "p7",
        "key": "corrida-test",
        "dir": str(inference_dir),
        "model_name": "champion-test",
        "model_version": "1",
        "data_hash": "hash-test",
        "inference_run_id": "inference-run-test",
    }
    (inference_dir / "inference_context.json").write_text(
        json.dumps(contexto), encoding="utf-8"
    )

    class Preprocesador:
        llamado = False

        def transform(self, datos):
            self.llamado = True
            return datos[["x"]].to_numpy()

    class Estimador:
        def predict_proba(self, matriz):
            prob = np.clip(np.asarray(matriz)[:, 0] / 10.0, 0.01, 0.99)
            return np.column_stack([1 - prob, prob])

    preprocesador = Preprocesador()
    champion = SimpleNamespace(
        feature_names_in_=["x"],
        named_steps={
            "preprocessing": preprocesador,
            "model": Estimador(),
        },
    )
    monkeypatch.setattr(inference, "cargar_champion", lambda cfg: champion)

    @contextmanager
    def run_falso(*args, **kwargs):
        yield SimpleNamespace(info=SimpleNamespace(run_id="monitor-run-test"))

    monkeypatch.setattr(monitoring, "run_rastreado", run_falso)

    cliente = SimpleNamespace(set_tag=Mock())
    mlflow_falso = types.ModuleType("mlflow")
    mlflow_falso.log_metrics = Mock()
    mlflow_falso.log_artifact = Mock()

    tracking_falso = types.ModuleType("mlflow.tracking")
    tracking_falso.MlflowClient = lambda: cliente
    mlflow_falso.tracking = tracking_falso

    monkeypatch.setitem(sys.modules, "mlflow", mlflow_falso)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking_falso)

    cfg = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "target_col": "target",
        "monitoring": {
            "reference_files": ["p6_extrac.csv"],
            "bins": 2,
            "max_categories": 10,
            "epsilon": 1e-6,
            "psi": {"warn": 0.10, "alarm": 0.25},
            "kl": {"warn": 0.10, "alarm": 0.25},
        },
    }

    resultado = monitoring.monitorear_desde_contexto(
        cfg, mes="p7", key="corrida-test"
    )

    reporte = json.loads(
        (inference_dir / "monitoring_report.json").read_text(encoding="utf-8")
    )

    assert preprocesador.llamado
    assert resultado["monitoring_run_id"] == "monitor-run-test"
    assert reporte["reentrena_modelo"] is False
    assert reporte["probabilidad"]["psi"] > 0
    assert reporte["probabilidad"]["kl"] > 0
    assert reporte["probabilidad"]["nivel"] == "alto"
    mlflow_falso.log_metrics.assert_called_once()
    assert {"psi_prob", "kl_prob"} <= set(
        mlflow_falso.log_metrics.call_args.args[0]
    )
    assert cliente.set_tag.call_args.args == (
        "inference-run-test",
        "monitoring_run_id",
        "monitor-run-test",
    )