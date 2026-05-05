"""
evaluate.py
===========
Cálculo y visualización de métricas:
  - Precision, Recall, F1-Score (por clase y macro/weighted)
  - Confusion Matrix  → heatmap PNG
  - ROC Curve         → curva AUC PNG
  - Classification Report completo
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")          # sin display GUI
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

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


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Métricas escalares
# ─────────────────────────────────────────────

def compute_metrics(y_test, y_pred, y_pred_proba) -> dict:
    metrics = {
        "accuracy":          float(accuracy_score(y_test, y_pred)),
        "precision_macro":   float(precision_score(y_test, y_pred, average="macro",   zero_division=0)),
        "precision_weighted":float(precision_score(y_test, y_pred, average="weighted",zero_division=0)),
        "recall_macro":      float(recall_score(y_test, y_pred, average="macro",      zero_division=0)),
        "recall_weighted":   float(recall_score(y_test, y_pred, average="weighted",   zero_division=0)),
        "f1_macro":          float(f1_score(y_test, y_pred, average="macro",          zero_division=0)),
        "f1_weighted":       float(f1_score(y_test, y_pred, average="weighted",       zero_division=0)),
        "roc_auc":           float(roc_auc_score(y_test, y_pred_proba)),
    }

    # Por clase
    prec_c = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_c  = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_c   = f1_score(y_test, y_pred, average=None, zero_division=0)
    classes = ["Benign (B=0)", "Malignant (M=1)"]
    for i, cls in enumerate(classes):
        key = cls.split()[0].lower()
        metrics[f"precision_{key}"] = float(prec_c[i]) if i < len(prec_c) else 0.0
        metrics[f"recall_{key}"]    = float(rec_c[i])  if i < len(rec_c)  else 0.0
        metrics[f"f1_{key}"]        = float(f1_c[i])   if i < len(f1_c)   else 0.0

    return metrics


def print_classification_report(y_test, y_pred) -> None:
    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    report = classification_report(
        y_test, y_pred,
        target_names=["Benigno (B)", "Maligno (M)"],
        digits=4
    )
    print(report)


# ─────────────────────────────────────────────
# Confusion Matrix
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred,
                          plots_dir: str = "plots") -> str:
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "confusion_matrix.png")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benigno", "Maligno"],
        yticklabels=["Benigno", "Maligno"],
        linewidths=0.5, ax=ax
    )
    ax.set_title("Matriz de Confusión", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_xlabel("Predicho", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [PLOT] Confusion Matrix → {path}")
    return path


# ─────────────────────────────────────────────
# ROC Curve
# ─────────────────────────────────────────────

def plot_roc_curve(y_test, y_pred_proba,
                   plots_dir: str = "plots") -> str:
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "roc_curve.png")

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#E63946", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="#A8DADC", lw=1.5, linestyle="--",
            label="Clasificador aleatorio")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title("Curva ROC — Detección de Cáncer", fontsize=14,
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [PLOT] ROC Curve → {path}")
    return path


# ─────────────────────────────────────────────
# Métricas CV plot
# ─────────────────────────────────────────────

def plot_cv_scores(cv_metrics: dict, plots_dir: str = "plots") -> str:
    if not cv_metrics:
        return ""
    os.makedirs(plots_dir, exist_ok=True)
    path = os.path.join(plots_dir, "cv_scores.png")

    metrics_to_plot = {
        k: v for k, v in cv_metrics.items() if k.endswith("_mean")
    }
    stds = {
        k.replace("_mean", "_std"): cv_metrics.get(k.replace("_mean", "_std"), 0)
        for k in metrics_to_plot
    }

    labels = [k.replace("cv_", "").replace("_mean", "") for k in metrics_to_plot]
    means  = list(metrics_to_plot.values())
    errors = [stds.get(k.replace("_mean", "_std"), 0) for k in metrics_to_plot]

    x = np.arange(len(labels))
    colors = ["#1D3557", "#457B9D", "#A8DADC", "#E63946", "#F4A261"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, means, yerr=errors, color=colors[:len(labels)],
                  capsize=5, alpha=0.9, edgecolor="white", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Validation Scores (mean ± std)", fontsize=13,
                 fontweight="bold")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  [PLOT] CV Scores → {path}")
    return path


# ─────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────

def evaluate(y_test, y_pred, y_pred_proba,
             cv_metrics: dict,
             config_path: str = "config.yaml"):
    """
    Ejecuta evaluación completa y retorna:
        metrics (dict), cm_path, roc_path, cv_path
    """
    cfg = load_config(config_path)
    plots_dir = cfg["output"]["plots_dir"]

    print_classification_report(y_test, y_pred)

    metrics = compute_metrics(y_test, y_pred, y_pred_proba)

    print("\n[METRICS] Resumen:")
    for k, v in metrics.items():
        print(f"  {k:<30} {v:.4f}")

    # Gráficas
    cm_path  = plot_confusion_matrix(y_test, y_pred, plots_dir)
    roc_path = plot_roc_curve(y_test, y_pred_proba, plots_dir)
    cv_path  = plot_cv_scores(cv_metrics, plots_dir)

    return metrics, cm_path, roc_path, cv_path
