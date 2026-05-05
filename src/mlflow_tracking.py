"""
mlflow_tracking.py
==================
Registra en MLflow:
  - Parámetros del modelo
  - Métricas del test y cross-validation
  - Artifacts: gráficas PNG, modelo serializado
  - Firma del modelo (Model Signature)
  - Timestamps de inicio/fin del entrenamiento
  - Instrucciones para levantar la UI
"""

import os
import datetime
import yaml
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Setup del experimento
# ─────────────────────────────────────────────

def setup_mlflow(cfg: dict) -> None:
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    experiment_name = cfg["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    print(f"\n[MLFLOW] Tracking URI   : {tracking_uri}")
    print(f"[MLFLOW] Experimento    : {experiment_name}")
    print(f"[MLFLOW] Para ver la UI ejecuta:")
    print(f"         mlflow ui --port 5000")
    print(f"         → http://127.0.0.1:5000\n")


# ─────────────────────────────────────────────
# Registro del run
# ─────────────────────────────────────────────

def log_run(
    model,
    X_train,
    y_train,
    X_test,
    metrics: dict,
    cv_metrics: dict,
    cm_path: str,
    roc_path: str,
    cv_path: str,
    train_start: datetime.datetime,
    train_end: datetime.datetime,
    feature_names: list,
    config_path: str = "config.yaml",
) -> str:
    """
    Registra todo el run en MLflow y retorna el run_id.
    """
    cfg = load_config(config_path)
    ml_cfg = cfg["mlflow"]
    model_cfg = cfg["model"]

    # Nombre del run con timestamp
    run_name = (
        f"{ml_cfg['run_name_prefix']}_"
        f"{model_cfg['algorithm']}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"[MLFLOW] Run iniciado : {run_id}")

        # ── Tags ──────────────────────────────────────────────
        mlflow.set_tags({
            "algorithm":   model_cfg["algorithm"],
            "dataset":     "Wisconsin Breast Cancer",
            "uci_id":      str(cfg["dataset"].get("uci_id", "")),
            "author":      "cancer-detection-pipeline",
        })

        # ── Timestamps de entrenamiento ───────────────────────
        if train_start and train_end:
            duration = (train_end - train_start).total_seconds()
            mlflow.log_params({
                "train_start_datetime": train_start.isoformat(),
                "train_end_datetime":   train_end.isoformat(),
                "train_duration_sec":   round(duration, 3),
            })

        # ── Parámetros del modelo ─────────────────────────────
        algo = model_cfg["algorithm"]
        hp = model_cfg["hyperparameters"].get(algo, {})
        mlflow.log_params({f"model_{k}": v for k, v in hp.items()})
        mlflow.log_params({
            "algorithm":        algo,
            "test_size":        cfg["dataset"]["test_size"],
            "random_state":     cfg["dataset"]["random_state"],
            "normalization":    cfg["preprocessing"]["normalization"],
            "cv_folds":         cfg["model"]["cross_validation"].get("folds", 5),
            "n_features":       len(feature_names),
            "n_train_samples":  len(X_train),
            "n_test_samples":   len(X_test),
        })

        # ── Métricas del test ─────────────────────────────────
        mlflow.log_metrics(metrics)
        print(f"[MLFLOW] Métricas de test registradas: {len(metrics)}")

        # ── Métricas de Cross-Validation ──────────────────────
        if cv_metrics:
            mlflow.log_metrics(cv_metrics)
            print(f"[MLFLOW] Métricas de CV registradas: {len(cv_metrics)}")

        # ── Artifacts: Gráficas ───────────────────────────────
        for path, label in [
            (cm_path,  "confusion_matrix"),
            (roc_path, "roc_curve"),
            (cv_path,  "cv_scores"),
        ]:
            if path and os.path.exists(path):
                mlflow.log_artifact(path, artifact_path="plots")
                print(f"[MLFLOW] Artifact '{label}' → {path}")

        # ── Firma del modelo (Model Signature) ───────────────
        try:
            import pandas as pd
            X_sample = pd.DataFrame(X_train[:5], columns=feature_names)
            y_sample = model.predict(X_train[:5])
            signature = infer_signature(X_sample, y_sample)
        except Exception:
            signature = None

        # ── Log del modelo ────────────────────────────────────
        if ml_cfg.get("log_model", True):
            model_path = cfg["model"]["saved_model_path"]
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                registered_model_name=(
                    ml_cfg["registered_model_name"]
                    if ml_cfg.get("register_model", False) else None
                ),
                input_example=(
                    pd.DataFrame(X_train[:3], columns=feature_names)
                    if signature else None
                ),
            )
            print(f"[MLFLOW] Modelo registrado en artifact 'model'")

            # También guardar el .pkl como artifact adicional
            if os.path.exists(model_path):
                mlflow.log_artifact(model_path, artifact_path="pkl")

        # ── Resumen en consola ────────────────────────────────
        print(f"\n[MLFLOW] ✅ Run completado")
        print(f"[MLFLOW]    Run ID      : {run_id}")
        print(f"[MLFLOW]    Experimento : {ml_cfg['experiment_name']}")
        print(f"[MLFLOW]    UI          : {cfg['mlflow']['tracking_uri']}")

    return run_id


# ─────────────────────────────────────────────
# Helper — levantar MLflow UI
# ─────────────────────────────────────────────

def print_mlflow_instructions(cfg: dict) -> None:
    uri = cfg["mlflow"]["tracking_uri"]
    print("\n" + "=" * 60)
    print("  CÓMO VER LOS RESULTADOS EN MLFLOW")
    print("=" * 60)
    print(f"  1. En una terminal separada ejecuta:")
    print(f"       mlflow ui --port 5000")
    print(f"  2. Abre tu navegador en:")
    print(f"       {uri}")
    print(f"  3. Selecciona el experimento:")
    print(f"       '{cfg['mlflow']['experiment_name']}'")
    print("=" * 60)
