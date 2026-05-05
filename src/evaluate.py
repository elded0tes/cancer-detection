"""
evaluate.py
───────────
Cálculo y registro completo de métricas de evaluación:
  - Precision, Recall, F1-Score (por clase y ponderado)
  - Confusion Matrix  (con gráfico guardado)
  - ROC Curve + AUC   (con gráfico guardado)
  - Classification Report
  - Log de métricas y artefactos en MLFlow
"""

import os
import logging
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)

log = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Classification report ─────────────────────────────────────────────────────
def compute_metrics(y_test, y_pred, y_prob=None) -> dict:
    """Calcula todas las métricas de clasificación."""
    metrics = {
        "accuracy":           accuracy_score(y_test, y_pred),
        "precision_weighted": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall_weighted":    recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_weighted":        f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "precision_macro":    precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro":       recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_macro":           f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
        except Exception:
            pass

    for k, v in metrics.items():
        log.info("%-30s %.4f", k, v)

    return metrics


# ── Confusion Matrix ──────────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, plots_dir: str,
                          run_id: str = "") -> str:
    cm = confusion_matrix(y_test, y_pred)
    labels = ["Benigno (0)", "Maligno (1)"]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor="white",
    )
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title("Matriz de Confusión", fontsize=13, fontweight="bold")
    plt.tight_layout()

    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, f"confusion_matrix_{run_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix guardada: %s", path)
    return path


# ── ROC Curve ────────────────────────────────────────────────────────────────
def plot_roc_curve(y_test, y_prob, plots_dir: str,
                   run_id: str = "") -> str:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc_val = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4C72B0", lw=2,
            label=f"ROC AUC = {roc_auc_val:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.fill_between(fpr, tpr, alpha=0.08, color="#4C72B0")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("Tasa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Tasa de Verdaderos Positivos", fontsize=12)
    ax.set_title("Curva ROC", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()

    path = os.path.join(plots_dir, f"roc_curve_{run_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ROC curve guardada: %s", path)
    return path


# ── Classification report plot ────────────────────────────────────────────────
def plot_classification_report(y_test, y_pred, plots_dir: str,
                                run_id: str = "") -> str:
    report = classification_report(
        y_test, y_pred,
        target_names=["Benigno", "Maligno"],
        output_dict=True,
    )
    df_report = pd.DataFrame(report).transpose().round(3)
    log.info("Classification Report:\n%s", df_report.to_string())

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    table = ax.table(
        cellText=df_report.values,
        colLabels=df_report.columns,
        rowLabels=df_report.index,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Classification Report", fontsize=13,
                 fontweight="bold", pad=20)
    plt.tight_layout()

    path = os.path.join(plots_dir, f"classification_report_{run_id}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Classification report guardado: %s", path)
    return path


# ── Pipeline de evaluación ────────────────────────────────────────────────────
def run(model, X_test, y_test, run_id: str = "",
        config_path: str = "config.yaml") -> dict:
    """
    Evalúa el modelo, genera gráficos y los registra en MLFlow.
    Retorna diccionario con todas las métricas.
    """
    cfg       = load_config(config_path)
    plots_dir = cfg["paths"]["plots_dir"]

    log.info("═" * 55)
    log.info("EVALUACIÓN DEL MODELO")

    # Predicciones
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)

    # Métricas numéricas
    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Gráficos
    cm_path     = plot_confusion_matrix(y_test, y_pred, plots_dir, run_id)
    roc_path    = None
    report_path = plot_classification_report(y_test, y_pred, plots_dir, run_id)

    if y_prob is not None:
        roc_path = plot_roc_curve(y_test, y_prob, plots_dir, run_id)

    # ── Log en MLFlow ────────────────────────────────────────────────────────
    # Se asume que hay un run activo (iniciado desde pipeline.py)
    try:
        active_run = mlflow.active_run()
        if active_run is None:
            mlflow_cfg = cfg["mlflow"]
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
            mlflow.set_experiment(mlflow_cfg["experiment_name"])
            ctx = mlflow.start_run(run_id=run_id or None,
                                   run_name=mlflow_cfg["run_name"])
        else:
            ctx = None

        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(cm_path,     artifact_path="plots")
        mlflow.log_artifact(report_path, artifact_path="plots")
        if roc_path:
            mlflow.log_artifact(roc_path, artifact_path="plots")

        log.info("Métricas y artefactos registrados en MLFlow.")

        if ctx is not None:
            ctx.__exit__(None, None, None)

    except Exception as e:
        log.warning("No se pudo conectar a MLFlow: %s", e)

    metrics["plots"] = {
        "confusion_matrix": cm_path,
        "roc_curve": roc_path,
        "classification_report": report_path,
    }
    log.info("═" * 55)
    return metrics
