"""
preprocess.py
─────────────
Limpieza y transformación completa del dataset Wisconsin:
  1. data info      — dtypes, shape, resumen
  2. data describe  — estadísticas descriptivas
  3. value counts   — distribución de clases
  4. null removal   — estrategias: mean / median / drop
  5. caracteres     — columnas con nombres inválidos, espacios
  6. encoding       — M → 1, B → 0
  7. normalization  — StandardScaler / MinMaxScaler / RobustScaler
  8. train/test split
"""

import logging
import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

log = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── 1. Data Info ──────────────────────────────────────────────────────────────
def data_info(df: pd.DataFrame) -> None:
    log.info("═" * 55)
    log.info("DATA INFO")
    log.info("Shape : %s", df.shape)
    log.info("Columns :\n%s", list(df.columns))
    log.info("Dtypes :\n%s", df.dtypes.to_string())
    log.info("Memory : %.2f KB", df.memory_usage(deep=True).sum() / 1024)
    log.info("═" * 55)


# ── 2. Data Describe ──────────────────────────────────────────────────────────
def data_describe(df: pd.DataFrame) -> pd.DataFrame:
    desc = df.describe(include="all")
    log.info("DATA DESCRIBE :\n%s", desc.to_string())
    return desc


# ── 3. Value Counts ───────────────────────────────────────────────────────────
def data_value_counts(df: pd.DataFrame, target_col: str) -> pd.Series:
    counts = df[target_col].value_counts()
    log.info("VALUE COUNTS — %s :\n%s", target_col, counts.to_string())
    return counts


# ── 4. Drop specified columns ─────────────────────────────────────────────────
def drop_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    existing = [c for c in cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        log.info("Columnas eliminadas : %s", existing)
    return df


# ── 5. Clean column names (espacios, caracteres raros) ───────────────────────
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r"[^a-z0-9_]", "_", regex=True)
                  .str.replace(r"_+", "_", regex=True)
                  .str.strip("_")
    )
    log.info("Columnas normalizadas.")
    return df


# ── 6. Null handling ──────────────────────────────────────────────────────────
def handle_nulls(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()

    if total_nulls == 0:
        log.info("Sin valores nulos.")
        return df

    log.info("Nulos por columna:\n%s", null_counts[null_counts > 0].to_string())

    if strategy == "drop":
        df = df.dropna()
        log.info("Filas eliminadas con nulos. Nuevo shape: %s", df.shape)
    elif strategy in ("mean", "median"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        fill_values = (
            df[numeric_cols].mean() if strategy == "mean"
            else df[numeric_cols].median()
        )
        df[numeric_cols] = df[numeric_cols].fillna(fill_values)
        df = df.fillna(df.mode().iloc[0])   # no-numéricas con moda
        log.info("Nulos imputados con estrategia: %s", strategy)
    else:
        raise ValueError(f"Estrategia desconocida: {strategy}")

    return df


# ── 7. Encode target  M → 1, B → 0 ──────────────────────────────────────────
def encode_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if df[target_col].dtype == object:
        mapping = {"M": 1, "B": 0, "Malignant": 1, "Benign": 0,
                   "malignant": 1, "benign": 0}
        df[target_col] = df[target_col].map(mapping)
        log.info("Target codificado: M→1, B→0")
    return df


# ── 8. Normalization ──────────────────────────────────────────────────────────
SCALERS = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def normalize(X_train: pd.DataFrame, X_test: pd.DataFrame,
              scaler_name: str = "standard"):
    ScalerClass = SCALERS.get(scaler_name, StandardScaler)
    scaler = ScalerClass()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    log.info("Normalización aplicada: %s", scaler_name)
    return (
        pd.DataFrame(X_train_scaled, columns=X_train.columns),
        pd.DataFrame(X_test_scaled,  columns=X_test.columns),
        scaler,
    )


# ── Pipeline completo ─────────────────────────────────────────────────────────
def run(config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    data_cfg  = cfg["data"]
    prep_cfg  = cfg["preprocessing"]

    # Cargar datos
    df = pd.read_csv(data_cfg["output_path"])

    # Análisis exploratorio
    data_info(df)
    data_describe(df)
    data_value_counts(df, data_cfg["target_column"])

    # Limpieza
    df = drop_columns(df, prep_cfg.get("drop_columns", []))
    df = clean_column_names(df)
    df = handle_nulls(df, prep_cfg.get("fill_strategy", "mean"))

    # Re-ajustar nombre de target tras clean_column_names
    target_col = data_cfg["target_column"].lower()

    # Codificar target
    df = encode_target(df, target_col)

    # Split
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size   = data_cfg["test_size"],
        random_state= data_cfg["random_state"],
        stratify    = y,
    )
    log.info("Split — train: %d, test: %d", len(X_train), len(X_test))

    # Normalizar
    X_train, X_test, scaler = normalize(
        X_train, X_test, prep_cfg.get("scaler", "standard")
    )

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    run()
