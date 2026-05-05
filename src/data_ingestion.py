"""
data_ingestion.py
=================
Descarga el dataset de Cáncer de Wisconsin desde:
  - ucimlrepo  (fuente recomendada, ID 17 = Diagnostic)
  - sklearn    (datasets.load_breast_cancer)
  - csv local  (ruta definida en config)
"""

import os
import pandas as pd
import yaml
from sklearn import datasets as sk_datasets


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_from_ucimlrepo(uci_id: int) -> pd.DataFrame:
    """
    Descarga el dataset directamente desde el repositorio UCI via ucimlrepo.
    ID 17  → Wisconsin Diagnostic Breast Cancer (WDBC)
    ID 15  → Wisconsin Original Breast Cancer
    ID 16  → Wisconsin Prognostic Breast Cancer
    """
    try:
        from ucimlrepo import fetch_ucirepo
        print(f"[INFO] Descargando dataset UCI ID={uci_id} …")
        repo = fetch_ucirepo(id=uci_id)
        X = repo.data.features
        y = repo.data.targets
        df = pd.concat([X, y], axis=1)
        print(f"[INFO] Dataset descargado: {df.shape[0]} filas × {df.shape[1]} columnas")
        print(f"[INFO] Variables objetivo → {y.columns.tolist()}")
        return df
    except Exception as e:
        raise RuntimeError(f"[ERROR] ucimlrepo falló: {e}")


def download_from_sklearn() -> pd.DataFrame:
    """Carga el dataset de sklearn como alternativa."""
    print("[INFO] Cargando Wisconsin Breast Cancer desde sklearn …")
    data = sk_datasets.load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # sklearn usa 0=malignant, 1=benign — invertimos para M=1, B=0
    df["target"] = df["target"].map({0: "M", 1: "B"})
    df.rename(columns={"target": "diagnosis"}, inplace=True)
    print(f"[INFO] Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def load_from_csv(csv_path: str) -> pd.DataFrame:
    """Carga un CSV local."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"[ERROR] No se encontró el archivo: {csv_path}")
    print(f"[INFO] Cargando CSV desde {csv_path} …")
    df = pd.read_csv(csv_path)
    print(f"[INFO] CSV cargado: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df


def save_raw_data(df: pd.DataFrame, path: str = "data/raw_data.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Datos crudos guardados en {path}")


def ingest_data(config_path: str = "config.yaml") -> pd.DataFrame:
    """
    Punto de entrada principal.
    Lee el config y descarga / carga el dataset según 'source'.
    """
    cfg = load_config(config_path)
    ds_cfg = cfg["dataset"]
    source = ds_cfg.get("source", "ucimlrepo")

    if source == "ucimlrepo":
        df = download_from_ucimlrepo(ds_cfg["uci_id"])
    elif source == "sklearn":
        df = download_from_sklearn()
    elif source == "csv":
        df = load_from_csv(ds_cfg["csv_path"])
    else:
        raise ValueError(f"[ERROR] Fuente desconocida: '{source}'. Usa 'ucimlrepo', 'sklearn' o 'csv'.")

    save_raw_data(df, "data/raw_data.csv")
    return df


if __name__ == "__main__":
    df = ingest_data()
    print(df.head())
