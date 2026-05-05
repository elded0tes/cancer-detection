"""
pipeline.py
───────────
Orquesta el pipeline completo:
  - Descarga → Preprocesamiento → Train/Load → Evaluación → MLFlow

Cuando retrain=False, carga el modelo registrado desde MLFlow y omite
la fase de entrenamiento (modo producción / inferencia).

Uso:
    python src/pipeline.py               # entrena (según config.yaml)
    python src/pipeline.py --no-train    # usa modelo existente
    python src/pipeline.py --retrain     # fuerza reentrenamiento
"""

import os
import sys
import logging
import argparse
import datetime
import yaml
import mlflow
import mlflow.sklearn

# Para importar módulos del paquete src
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download_data
import preprocess
import train as train_module
import evaluate

log = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_model_from_mlflow(cfg: dict):
    """Carga la última versión del modelo registrado desde MLFlow."""
    mlflow_cfg = cfg["mlflow"]
    model_name = cfg["model"]["registered_model_name"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])

    model_uri = f"models:/{model_name}/latest"
    log.info("Cargando modelo desde MLFlow: %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)
    return model


def run_pipeline(config_path: str = "config.yaml",
                 force_retrain: bool = False,
                 skip_train: bool = False) -> dict:
    """
    Ejecuta el pipeline completo.

    Parámetros
    ----------
    config_path   : ruta al archivo de configuración
    force_retrain : sobreescribe config.yaml retrain=False
    skip_train    : modo producción — salta entrenamiento completamente

    Retorna
    -------
    dict con métricas finales de evaluación
    """
    pipeline_start = datetime.datetime.now()
    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║   CANCER DETECTION PIPELINE — INICIO %-12s ║",
             pipeline_start.strftime("%H:%M:%S"))
    log.info("╚══════════════════════════════════════════════════╝")

    cfg = load_config(config_path)

    # ── PASO 1: Descarga ───────────────────────────────────────────────────
    log.info("▶  PASO 1 — Descarga del dataset")
    if not os.path.exists(cfg["data"]["output_path"]):
        download_data.main()
    else:
        log.info("   Dataset ya existe: %s", cfg["data"]["output_path"])

    # ── PASO 2: Preprocesamiento ──────────────────────────────────────────
    log.info("▶  PASO 2 — Preprocesamiento")
    X_train, X_test, y_train, y_test, scaler = preprocess.run(config_path)

    # ── PASO 3: Entrenamiento o carga de modelo ───────────────────────────
    if skip_train:
        log.info("▶  PASO 3 — Cargando modelo (skip_train=True)")
        try:
            model = load_model_from_mlflow(cfg)
            run_id = "loaded"
            cv_summary = {}
        except Exception as e:
            log.warning("No se pudo cargar desde MLFlow (%s). Entrenando …", e)
            model, cv_summary, run_info = train_module.train(
                X_train, y_train, config_path, force_retrain=True
            )
            run_id = run_info["run_id"]
    else:
        log.info("▶  PASO 3 — Entrenamiento del modelo")
        model, cv_summary, run_info = train_module.train(
            X_train, y_train, config_path, force_retrain=force_retrain
        )
        run_id = run_info["run_id"]

    # ── PASO 4: Evaluación ────────────────────────────────────────────────
    log.info("▶  PASO 4 — Evaluación y métricas")

    # Abrir run de MLFlow para loguear las métricas de test
    mlflow_cfg = cfg["mlflow"]
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(
        run_name=f"{mlflow_cfg['run_name']}_eval",
        tags={"stage": "evaluation", "parent_run_id": run_id}
    ) as eval_run:
        metrics = evaluate.run(
            model, X_test, y_test,
            run_id=eval_run.info.run_id,
            config_path=config_path,
        )

    # ── Resumen final ─────────────────────────────────────────────────────
    pipeline_end = datetime.datetime.now()
    total_secs   = (pipeline_end - pipeline_start).total_seconds()

    log.info("╔══════════════════════════════════════════════════╗")
    log.info("║   PIPELINE COMPLETADO  (%.2f s)                  ║", total_secs)
    log.info("╠══════════════════════════════════════════════════╣")
    for k, v in metrics.items():
        if k != "plots" and isinstance(v, float):
            log.info("║   %-28s %.4f           ║", k, v)
    log.info("╠══════════════════════════════════════════════════╣")
    log.info("║   MLFlow UI: %-35s ║", mlflow_cfg["tracking_uri"])
    log.info("╚══════════════════════════════════════════════════╝")

    return metrics


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="Cancer Detection Pipeline")
    parser.add_argument("--retrain",   action="store_true",
                        help="Forza reentrenamiento aunque exista modelo")
    parser.add_argument("--no-train",  action="store_true",
                        help="Salta entrenamiento, usa modelo de MLFlow")
    parser.add_argument("--config",    default="config.yaml",
                        help="Ruta al archivo de configuración")
    args = parser.parse_args()

    run_pipeline(
        config_path   = args.config,
        force_retrain = args.retrain,
        skip_train    = args.no_train,
    )
