"""Punto de entrada local del pipeline de entrenamiento.

Utiliza las mismas etapas que el DAG de Airflow.
"""

import argparse

from src import stages


def main():
    """Ejecuta el pipeline completo con la configuración indicada."""
    parser = argparse.ArgumentParser(
        description="Ejecutar el pipeline de entrenamiento AutoML"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta del archivo de configuración",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Directorio de entrada de los datos",
    )
    parser.add_argument(
        "--run-key",
        default=None,
        help="Identificador de la ejecución",
    )

    args = parser.parse_args()

    overrides = {"input_dir": args.input} if args.input else {}
    cfg = stages.cargar_config(args.config, overrides)

    run_key = args.run_key or stages.run_key_por_defecto(cfg)

    for etapa in stages.ETAPAS:
        if etapa == "hpo_model":
            for modelo in stages.MODELOS:
                stages.ejecutar(etapa, cfg, run_key, modelo)
        else:
            stages.ejecutar(etapa, cfg, run_key)

    print("Resultados:", stages.raiz_corrida(cfg, run_key))


if __name__ == "__main__":
    main()
