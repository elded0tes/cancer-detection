"""
pipeline.py
===========
Punto de entrada principal del pipeline de detección de cáncer.

Uso:
    python pipeline.py                   # Usa config.yaml por defecto
    python pipeline.py --config config.yaml
    python pipeline.py --retrain false   # Carga modelo existente
    python pipeline.py --source sklearn  # Cambia fuente del dataset

Flags:
    --config   PATH     Ruta al archivo de configuración (default: config.yaml)
    --retrain  BOOL     Sobrescribe config.model.retrain  (true/false)
    --source   STR      Sobrescribe config.dataset.source (ucimlrepo/sklearn/csv)
    --uci-id   INT      Sobrescribe config.dataset.uci_id
"""

import argparse
import sys
import os
import yaml
import datetime

# ─── Asegurar que src/ esté en el path ───────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.data_ingestion   import ingest_data
from src.preprocessing    import preprocess
from src.train            import train_model, predict
from src.evaluate         import evaluate
from src.mlflow_tracking  import setup_mlflow, log_run, print_mlflow_instructions


# ─────────────────────────────────────────────
# Argumentos CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pipeline de Detección de Cáncer — Wisconsin Dataset"
    )
    parser.add_argument("--config",   type=str,  default="config.yaml")
    parser.add_argument("--retrain",  type=str,  default=None,
                        help="true/false — sobrescribe config.model.retrain")
    parser.add_argument("--source",   type=str,  default=None,
                        help="ucimlrepo | sklearn | csv")
    parser.add_argument("--uci-id",   type=int,  default=None,
                        help="ID UCI del dataset (ej. 17, 15, 16)")
    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # Cargar config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Sobrescribir con flags CLI
    if args.retrain is not None:
        cfg["model"]["retrain"] = args.retrain.lower() == "true"
    if args.source is not None:
        cfg["dataset"]["source"] = args.source
    if args.uci_id is not None:
        cfg["dataset"]["uci_id"] = args.uci_id

    # Guardar config temporalmente para que los módulos la lean
    with open(args.config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("\n" + "█" * 60)
    print("  🔬 PIPELINE — DETECCIÓN DE CÁNCER DE MAMA")
    print("  Wisconsin Breast Cancer Dataset")
    print("  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("█" * 60)

    pipeline_start = datetime.datetime.now()

    # ════════════════════════════════════════
    # PASO 1 — Ingesta de datos
    # ════════════════════════════════════════
    print("\n▶ PASO 1/4 — Descarga e ingesta del dataset")
    df = ingest_data(args.config)

    # ════════════════════════════════════════
    # PASO 2 — Preprocesamiento
    # ════════════════════════════════════════
    print("\n▶ PASO 2/4 — Preprocesamiento")
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(
        df, args.config
    )

    # ════════════════════════════════════════
    # PASO 3 — Entrenamiento / Carga del modelo
    # ════════════════════════════════════════
    print("\n▶ PASO 3/4 — Entrenamiento del modelo")
    model, cv_metrics, train_start, train_end = train_model(
        X_train, y_train, args.config
    )

    y_pred, y_pred_proba = predict(model, X_test)

    # ════════════════════════════════════════
    # PASO 4 — Evaluación + MLflow
    # ════════════════════════════════════════
    print("\n▶ PASO 4/4 — Evaluación y registro MLflow")

    metrics, cm_path, roc_path, cv_path = evaluate(
        y_test, y_pred, y_pred_proba, cv_metrics, args.config
    )

    # Setup MLflow y registrar run
    setup_mlflow(cfg)
    run_id = log_run(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        metrics=metrics,
        cv_metrics=cv_metrics,
        cm_path=cm_path,
        roc_path=roc_path,
        cv_path=cv_path,
        train_start=train_start,
        train_end=train_end,
        feature_names=feature_names,
        config_path=args.config,
    )

    # ════════════════════════════════════════
    # Resumen final
    # ════════════════════════════════════════
    pipeline_end = datetime.datetime.now()
    total = (pipeline_end - pipeline_start).total_seconds()

    print("\n" + "█" * 60)
    print("  ✅ PIPELINE COMPLETADO")
    print(f"  Duración total : {total:.2f} segundos")
    print(f"  MLflow Run ID  : {run_id}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted)  : {metrics['f1_weighted']:.4f}")
    print(f"  ROC AUC        : {metrics['roc_auc']:.4f}")
    print("█" * 60)

    print_mlflow_instructions(cfg)


if __name__ == "__main__":
    main()
