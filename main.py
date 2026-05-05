"""
main.py — Punto de entrada principal del pipeline.

Uso básico:
    python main.py                          # entrena con sklearn (por defecto)
    python main.py --source sklearn         # descarga desde sklearn
    python main.py --source uci             # descarga desde UCI
    python main.py --retrain                # fuerza reentrenamiento
    python main.py --no-train               # modo inferencia (sin entrenar)
    python main.py --config mi_config.yaml  # config personalizada
    python main.py --start-mlflow           # inicia MLFlow UI automáticamente
"""

import os
import sys
import logging
import argparse
import subprocess
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a"),
    ],
)

# Crear directorio de logs
os.makedirs("logs", exist_ok=True)

log = logging.getLogger(__name__)

# Añadir src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def start_mlflow_server(tracking_uri: str = "http://127.0.0.1:5000"):
    """Inicia MLFlow UI en background."""
    port = tracking_uri.split(":")[-1]
    log.info("Iniciando MLFlow UI en %s …", tracking_uri)

    def _run():
        subprocess.run(
            ["mlflow", "ui", "--port", port, "--host", "127.0.0.1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    time.sleep(3)          # esperar a que el servidor levante
    log.info("MLFlow UI disponible en: %s", tracking_uri)


def main():
    parser = argparse.ArgumentParser(
        description="Cancer Detection Pipeline — Wisconsin Breast Cancer Dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", choices=["sklearn", "uci", "csv"],
        default=None,
        help="Fuente del dataset (sobreescribe config.yaml)",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Fuerza reentrenamiento del modelo",
    )
    parser.add_argument(
        "--no-train", action="store_true",
        help="Salta entrenamiento (usa modelo registrado en MLFlow)",
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Ruta al archivo de configuración",
    )
    parser.add_argument(
        "--start-mlflow", action="store_true",
        help="Inicia MLFlow UI antes de ejecutar el pipeline",
    )
    args = parser.parse_args()

    # Verificar que el archivo de config existe
    if not os.path.exists(args.config):
        log.error("Archivo de configuración no encontrado: %s", args.config)
        sys.exit(1)

    # Sobreescribir fuente en config si se especificó en CLI
    if args.source:
        import yaml
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["data"]["source"] = args.source
        with open(args.config, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        log.info("Fuente de datos actualizada en config: %s", args.source)

    # Iniciar MLFlow UI si se solicitó
    if args.start_mlflow:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        start_mlflow_server(cfg["mlflow"]["tracking_uri"])

    # Importar y ejecutar pipeline
    from src.pipeline import run_pipeline

    metrics = run_pipeline(
        config_path   = args.config,
        force_retrain = args.retrain,
        skip_train    = args.no_train,
    )

    # Exit code 0 si F1 > 0.85, sino 1 (útil para CI/CD)
    f1 = metrics.get("f1_weighted", 0)
    exit_code = 0 if f1 >= 0.85 else 1
    log.info("Pipeline finalizado. F1 = %.4f — exit code: %d", f1, exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
