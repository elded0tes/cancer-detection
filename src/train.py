"""
train.py
────────
Entrenamiento y re-entrenamiento del modelo de detección de cáncer.

Características:
  - Soporte para RandomForest, LogisticRegression, SVM, GradientBoosting
  - Cross-Validation (StratifiedKFold) con métricas por fold
  - Registro de tiempo de inicio/fin (datetime)
  - Guardado local del modelo con joblib
  - Integración con MLFlow (parámetros, métricas de CV)

Uso:
    python src/train.py
    python src/train.py --retrain   # fuerza reentrenamiento
"""

import os
import logging
import argparse
import datetime
import yaml
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

log = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "RandomForest": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "SVM": SVC,
    "GradientBoosting": GradientBoostingClassifier,
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_model(model_name: str, params: dict):
    """Instancia el modelo configurado."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{model_name}' no soportado. "
            f"Opciones: {list(MODEL_REGISTRY)}"
        )
    ModelClass = MODEL_REGISTRY[model_name]
    # SVM necesita probability=True para ROC
    if model_name == "SVM":
        params = {**params, "probability": True}
    return ModelClass(**params)


def cross_validate_model(model, X_train, y_train, folds: int = 5) -> dict:
    """
    Realiza StratifiedKFold cross-validation.
    Retorna métricas promedio y por fold.
    """
    log.info("Cross-validation — %d folds …", folds)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
        "roc_auc": "roc_auc",
    }

    results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,
    )

    summary = {}
    for metric in scoring:
        vals = results[f"test_{metric}"]
        summary[metric] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "per_fold": vals.tolist(),
        }
        log.info("CV %s — mean: %.4f ± %.4f", metric,
                 summary[metric]["mean"], summary[metric]["std"])

    return summary


def train(X_train, y_train, config_path: str = "config.yaml",
          force_retrain: bool = False) -> tuple:
    """
    Entrena o re-entrena el modelo.
    Retorna (model, cv_summary, run_info).
    """
    cfg = load_config(config_path)
    model_cfg  = cfg["model"]
    mlflow_cfg = cfg["mlflow"]
    paths_cfg  = cfg["paths"]

    retrain = force_retrain or model_cfg.get("retrain", True)
    model_path = os.path.join(
        paths_cfg["models_dir"],
        f"{model_cfg['name']}_model.pkl"
    )

    # ── Registro de tiempos ───────────────────────────────────────────────────
    start_dt = datetime.datetime.now()
    log.info("╔══ ENTRENAMIENTO INICIADO ══ %s", start_dt.isoformat())

    # ── Configurar MLFlow ─────────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    with mlflow.start_run(run_name=mlflow_cfg["run_name"]) as run:
        run_id = run.info.run_id
        log.info("MLFlow Run ID: %s", run_id)

        # Loguear parámetros del modelo y configuración
        mlflow.log_params({
            "model_name":   model_cfg["name"],
            "retrain":      retrain,
            "cv_folds":     model_cfg["cross_val_folds"],
            "test_size":    cfg["data"]["test_size"],
            "random_state": cfg["data"]["random_state"],
            "scaler":       cfg["preprocessing"]["scaler"],
            **{f"param_{k}": v for k, v in model_cfg["params"].items()},
        })

        # ── Construir modelo ──────────────────────────────────────────────────
        model = build_model(model_cfg["name"], model_cfg["params"])

        # ── Cross-Validation ──────────────────────────────────────────────────
        cv_summary = cross_validate_model(
            model, X_train, y_train, folds=model_cfg["cross_val_folds"]
        )

        # Loguear métricas de CV en MLFlow
        for metric, vals in cv_summary.items():
            mlflow.log_metric(f"cv_{metric}_mean", vals["mean"])
            mlflow.log_metric(f"cv_{metric}_std",  vals["std"])
            for i, v in enumerate(vals["per_fold"]):
                mlflow.log_metric(f"cv_{metric}_fold_{i+1}", v)

        # ── Entrenar en datos completos ───────────────────────────────────────
        if retrain or not os.path.exists(model_path):
            log.info("Entrenando modelo %s …", model_cfg["name"])
            model.fit(X_train, y_train)
            os.makedirs(paths_cfg["models_dir"], exist_ok=True)
            joblib.dump(model, model_path)
            log.info("Modelo guardado en: %s", model_path)
            mlflow.log_param("model_action", "trained")
        else:
            log.info("Cargando modelo existente: %s", model_path)
            model = joblib.load(model_path)
            mlflow.log_param("model_action", "loaded")

        # ── Tiempo de fin ─────────────────────────────────────────────────────
        end_dt = datetime.datetime.now()
        duration = (end_dt - start_dt).total_seconds()
        log.info("╚══ ENTRENAMIENTO FINALIZADO ══ %s (%.2f s)", end_dt.isoformat(), duration)

        mlflow.log_params({
            "training_start": start_dt.isoformat(),
            "training_end":   end_dt.isoformat(),
            "training_duration_sec": round(duration, 2),
        })

        # ── Registrar modelo (firma) en MLFlow ────────────────────────────────
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model       = model,
            artifact_path  = mlflow_cfg["artifact_path"],
            signature      = signature,
            registered_model_name = model_cfg["registered_model_name"],
        )

        run_info = {
            "run_id":   run_id,
            "start_dt": start_dt.isoformat(),
            "end_dt":   end_dt.isoformat(),
            "duration": duration,
            "model_path": model_path,
        }

    return model, cv_summary, run_info


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true")
    args = parser.parse_args()
    log.info("Ejecutar train.py desde main.py para datos completos.")
