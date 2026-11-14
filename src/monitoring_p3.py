"""Monitoreo de drift para una inferencia mensual, sin reentrenar el champion."""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.drift import (
    ajustar_referencia_categorica,
    ajustar_referencia_numerica,
    comparar_con_referencia,
    nivel_alerta,
)
from src.tracking import run_rastreado


def _nivel_mayor(niveles):
    orden = {"estable": 0, "moderado": 1, "alto": 2}
    return max(niveles, key=lambda n: orden[n]) if niveles else "estable"


def _perfil(valores, bins, max_categorias):
    if (
        pd.api.types.is_numeric_dtype(valores)
        and not pd.api.types.is_bool_dtype(valores)
    ):
        return ajustar_referencia_numerica(valores, bins=bins)

    return ajustar_referencia_categorica(
        valores,
        max_categorias=max_categorias,
    )


def monitorear_desde_contexto(
    cfg,
    mes,
    key,
    inference_run_id=None,
):
    """Lee los artefactos de inferencia y genera el reporte de monitoreo."""
    carpeta = Path(cfg["output_dir"]) / "inference" / mes / key
    archivo_contexto = carpeta / "inference_context.json"
    archivo_datos = carpeta / "datos.pkl"
    archivo_scores = carpeta / "scores_clientes.csv"

    faltantes = [
        str(p)
        for p in (archivo_contexto, archivo_datos, archivo_scores)
        if not p.exists()
    ]
    if faltantes:
        raise FileNotFoundError(
            "Faltan archivos de la inferencia: " + ", ".join(faltantes)
        )

    contexto = json.loads(archivo_contexto.read_text(encoding="utf-8"))
    inference_run_id = inference_run_id or contexto.get("inference_run_id")
    if not inference_run_id:
        raise ValueError("No se encontró el inference_run_id de esta corrida.")

    datos = pd.read_pickle(archivo_datos)
    scores = pd.read_csv(archivo_scores)

    if "prob" not in scores:
        raise ValueError("scores_clientes.csv debe contener la columna 'prob'.")

    ajustes = cfg.get("monitoring", {})
    bins = int(ajustes.get("bins", 10))
    max_categorias = int(ajustes.get("max_categories", 20))
    epsilon = float(ajustes.get("epsilon", 1e-6))
    umbral_psi = ajustes.get("psi", {"warn": 0.10, "alarm": 0.25})
    umbral_kl = ajustes.get("kl", {"warn": 0.10, "alarm": 0.25})

    # Las columnas predictoras se toman del champion registrado.
    # Solo se carga el modelo existente; nunca se llama a fit().
    from src.inference import cargar_champion

    champion = cargar_champion(cfg)
    columnas = list(champion.feature_names_in_)

    faltan_datos = [c for c in columnas if c not in datos.columns]
    if faltan_datos:
        raise ValueError(
            "Faltan predictores en datos.pkl: " + ", ".join(faltan_datos)
        )

    archivos_referencia = ajustes.get("reference_files", ["p6_extrac.csv"])
    referencias = []
    for nombre in archivos_referencia:
        ruta = Path(cfg["input_dir"]) / nombre
        if not ruta.exists():
            raise FileNotFoundError(f"No existe el archivo de referencia: {ruta}")
        referencias.append(
            pd.read_csv(
                ruta,
                sep=cfg.get("csv_sep", ","),
                encoding=cfg.get("csv_encoding", "utf-8"),
                low_memory=False,
            )
        )

    referencia = pd.concat(referencias, ignore_index=True)
    referencia = referencia.drop(
        columns=[cfg.get("target_col", "target")],
        errors="ignore",
    )

    faltan_ref = [c for c in columnas if c not in referencia.columns]
    if faltan_ref:
        raise ValueError(
            "Faltan predictores en los CSV de referencia: "
            + ", ".join(faltan_ref)
        )

    # Compara cada predictor usando cortes fijos obtenidos de la referencia.
    variables = {}
    for columna in columnas:
        perfil = _perfil(referencia[columna], bins, max_categorias)
        resultado = comparar_con_referencia(
            perfil,
            datos[columna],
            epsilon=epsilon,
        )
        resultado["nivel_psi"] = nivel_alerta(
            resultado["psi"],
            moderado=float(umbral_psi["warn"]),
            alto=float(umbral_psi["alarm"]),
        )
        resultado["nivel_kl"] = nivel_alerta(
            resultado["kl"],
            moderado=float(umbral_kl["warn"]),
            alto=float(umbral_kl["alarm"]),
        )
        resultado["nivel"] = _nivel_mayor(
            [resultado["nivel_psi"], resultado["nivel_kl"]]
        )
        variables[columna] = resultado

    # Compara la distribución de probabilidades con las probabilidades del
    # champion sobre los archivos de referencia. Esto tampoco entrena el modelo.
    referencia_prob = champion.predict_proba(referencia[columnas])[:, 1]
    perfil_prob = ajustar_referencia_numerica(referencia_prob, bins=bins)
    resultado_prob = comparar_con_referencia(
        perfil_prob,
        scores["prob"],
        epsilon=epsilon,
    )
    resultado_prob["nivel_psi"] = nivel_alerta(
        resultado_prob["psi"],
        moderado=float(umbral_psi["warn"]),
        alto=float(umbral_psi["alarm"]),
    )
    resultado_prob["nivel_kl"] = nivel_alerta(
        resultado_prob["kl"],
        moderado=float(umbral_kl["warn"]),
        alto=float(umbral_kl["alarm"]),
    )
    resultado_prob["nivel"] = _nivel_mayor(
        [resultado_prob["nivel_psi"], resultado_prob["nivel_kl"]]
    )

    niveles = [v["nivel"] for v in variables.values()]
    niveles.append(resultado_prob["nivel"])
    estado = _nivel_mayor(niveles)

    reporte = {
        "mes": mes,
        "key": key,
        "fecha_monitoreo_utc": datetime.now(timezone.utc).isoformat(),
        "estado_general": estado,
        "reentrena_modelo": False,
        "model_name": contexto.get("model_name"),
        "model_version": contexto.get("model_version"),
        "inference_run_id": inference_run_id,
        "referencias": archivos_referencia,
        "filas_referencia": int(len(referencia)),
        "filas_actuales": int(len(datos)),
        "variables_predictoras": variables,
        "probabilidad": resultado_prob,
    }

    ruta_reporte = carpeta / "monitoring_report.json"
    ruta_reporte.write_text(
        json.dumps(reporte, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Registra el monitoreo como un run separado y lo enlaza con inferencia.
    import mlflow
    from mlflow.tracking import MlflowClient

    metricas = {
        "psi_prob": float(resultado_prob["psi"]),
        "kl_prob": float(resultado_prob["kl"]),
        "filas_actuales": float(len(datos)),
        "variables_altas": float(
            sum(v["nivel"] == "alto" for v in variables.values())
        ),
        "variables_moderadas": float(
            sum(v["nivel"] == "moderado" for v in variables.values())
        ),
    }
    tags = {
        "mes": mes,
        "inference_run_id": str(inference_run_id),
        "estado_general": estado,
        "model_version": str(contexto.get("model_version", "desconocida")),
        "reentrena_modelo": "false",
    }

    with run_rastreado(
        cfg,
        etapa="monitoring",
        run_key=key,
        data_hash=contexto.get("data_hash"),
        tags_extra=tags,
    ) as run:
        monitoring_run_id = run.info.run_id
        mlflow.log_metrics(metricas)
        mlflow.log_artifact(str(ruta_reporte), artifact_path="monitoring")

    MlflowClient().set_tag(inference_run_id, "monitoring_run_id", monitoring_run_id)

    return {
        "mes": mes,
        "key": key,
        "estado_general": estado,
        "monitoring_run_id": monitoring_run_id,
        "reporte": str(ruta_reporte),
    }