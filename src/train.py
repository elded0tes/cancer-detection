"""
train.py
========
Entrenamiento, re-entrenamiento y carga del modelo de detección de cáncer.

Flujo:
  - Si config.model.retrain = true  → entrena/re-entrena y guarda
  - Si config.model.retrain = false → carga modelo existente del disco
  - Cross-validation (k-fold estratificado)
  - Test final sobre el conjunto holdout
  - Registra fechas de inicio/fin del entrenamiento
"""

import os
import joblib
import datetime
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline as SklearnPipeline


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Construcción del estimador
# ─────────────────────────────────────────────

def build_estimator(cfg: dict):
    """Devuelve el clasificador configurado en config.yaml."""
    algorithm = cfg["model"]["algorithm"]
    hp = cfg["model"]["hyperparameters"].get(algorithm, {})

    if algorithm == "random_forest":
        return RandomForestClassifier(**hp)
    elif algorithm == "svm":
        return SVC(probability=True, **hp)
    elif algorithm == "logistic_regression":
        return LogisticRegression(**hp)
    else:
        raise ValueError(f"[ERROR] Algoritmo desconocido: '{algorithm}'")


# ─────────────────────────────────────────────
# Cross-Validation
# ─────────────────────────────────────────────

def run_cross_validation(estimator, X_train, y_train, cfg: dict) -> dict:
    cv_cfg = cfg["model"]["cross_validation"]
    if not cv_cfg.get("enabled", True):
        return {}

    folds = cv_cfg.get("folds", 5)
    scoring = cv_cfg.get("scoring", "f1")

    print(f"\n[CV] Ejecutando {folds}-Fold Cross-Validation (scoring={scoring}) …")
    skf = StratifiedKFold(n_splits=folds, shuffle=True,
                          random_state=cfg["dataset"]["random_state"])

    cv_results = cross_validate(
        estimator, X_train, y_train,
        cv=skf,
        scoring=["accuracy", "f1", "precision", "recall", "roc_auc"],
        return_train_score=True,
        n_jobs=-1,
    )

    summary = {}
    for metric in ["test_accuracy", "test_f1", "test_precision",
                   "test_recall", "test_roc_auc"]:
        vals = cv_results[metric]
        clean_key = metric.replace("test_", "cv_")
        summary[f"{clean_key}_mean"] = float(np.mean(vals))
        summary[f"{clean_key}_std"]  = float(np.std(vals))
        print(f"  {clean_key:<22} {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    return summary


# ─────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────

def train_model(X_train, y_train, config_path: str = "config.yaml"):
    """
    Entrena el modelo y retorna:
        model, cv_metrics, train_start, train_end
    """
    cfg = load_config(config_path)

    # ── Retrain=False → cargar modelo existente ──
    if not cfg["model"].get("retrain", True):
        model_path = cfg["model"]["saved_model_path"]
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[ERROR] retrain=false pero no existe {model_path}"
            )
        print(f"\n[TRAIN] Cargando modelo desde {model_path} …")
        model = joblib.load(model_path)
        return model, {}, None, None

    # ── Retrain=True → entrenar ──
    estimator = build_estimator(cfg)
    print(f"\n[TRAIN] Algoritmo seleccionado: {cfg['model']['algorithm'].upper()}")

    # Cross-validation ANTES del fit final
    cv_metrics = run_cross_validation(estimator, X_train, y_train, cfg)

    # Registro de tiempo
    train_start = datetime.datetime.now()
    print(f"\n[TRAIN] Inicio entrenamiento: {train_start.isoformat()}")

    # Entrenamiento completo sobre todo X_train
    estimator.fit(X_train, y_train)

    train_end = datetime.datetime.now()
    duration = (train_end - train_start).total_seconds()
    print(f"[TRAIN] Fin entrenamiento   : {train_end.isoformat()}")
    print(f"[TRAIN] Duración            : {duration:.2f} s")

    # Guardar modelo en disco
    os.makedirs(cfg["output"]["models_dir"], exist_ok=True)
    model_path = cfg["model"]["saved_model_path"]
    joblib.dump(estimator, model_path)
    print(f"[TRAIN] Modelo guardado en  : {model_path}")

    return estimator, cv_metrics, train_start, train_end


# ─────────────────────────────────────────────
# Predicción / Test
# ─────────────────────────────────────────────

def predict(model, X_test):
    """
    Retorna:
        y_pred         → etiquetas predichas
        y_pred_proba   → probabilidades (clase positiva)
    """
    y_pred = model.predict(X_test)
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        # SVC sin probability=True
        y_pred_proba = model.decision_function(X_test)
    return y_pred, y_pred_proba
