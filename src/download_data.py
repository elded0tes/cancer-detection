"""
download_data.py
────────────────
Descarga el dataset Wisconsin Breast Cancer desde:
  - sklearn.datasets (rápido, sin dependencias externas)
  - UCI ML Repository  (requiere ucimlrepo)
  - CSV local

Uso:
    python src/download_data.py --source sklearn
    python src/download_data.py --source uci
"""

import os
import argparse
import logging
import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_from_sklearn() -> pd.DataFrame:
    """Descarga Wisconsin desde sklearn — listo para usar sin Internet."""
    from sklearn.datasets import load_breast_cancer
    log.info("Cargando dataset Wisconsin desde sklearn.datasets …")
    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    # sklearn usa 0=malignant, 1=benign → convertimos a M/B para consistencia
    df["diagnosis"] = df["target"].map({0: "M", 1: "B"})
    df.drop(columns=["target"], inplace=True)
    log.info("Dataset cargado: %d filas, %d columnas", *df.shape)
    return df


def download_from_uci(uci_id: int = 17) -> pd.DataFrame:
    """Descarga Wisconsin desde el UCI ML Repository (requiere ucimlrepo)."""
    from ucimlrepo import fetch_ucirepo
    log.info("Descargando dataset desde UCI ML Repository (id=%d) …", uci_id)
    dataset = fetch_ucirepo(id=uci_id)
    X = dataset.data.features
    y = dataset.data.targets
    df = pd.concat([X, y], axis=1)
    # Normalizar nombre de columna objetivo
    if "Diagnosis" in df.columns:
        df.rename(columns={"Diagnosis": "diagnosis"}, inplace=True)
    log.info("Dataset descargado: %d filas, %d columnas", *df.shape)
    return df


def download_from_csv(csv_path: str) -> pd.DataFrame:
    """Carga desde un CSV local."""
    log.info("Cargando dataset desde CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("CSV cargado: %d filas, %d columnas", *df.shape)
    return df


def main():
    parser = argparse.ArgumentParser(description="Descarga el dataset Wisconsin")
    parser.add_argument("--source", choices=["sklearn", "uci", "csv"],
                        default=None, help="Fuente de datos (sobreescribe config.yaml)")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = args.source or cfg["data"]["source"]
    output_path = cfg["data"]["output_path"]

    if source == "sklearn":
        df = download_from_sklearn()
    elif source == "uci":
        df = download_from_uci(cfg["data"]["uci_id"])
    elif source == "csv":
        df = download_from_csv(cfg["data"]["csv_path"])
    else:
        raise ValueError(f"Fuente desconocida: {source}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    log.info("Dataset guardado en: %s", output_path)
    return df


if __name__ == "__main__":
    main()
