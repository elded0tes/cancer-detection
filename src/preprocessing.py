"""
preprocessing.py
================
Pipeline completo de preprocesamiento:
  1. Data Info      → tipos, nulos, shape
  2. Data Describe  → estadísticas descriptivas
  3. Value Counts   → distribución de clases
  4. Data Cleaning  → duplicados, nulos, caracteres especiales, columnas basura
  5. Encoding       → variable objetivo a numérica (M=1, B=0)
  6. Normalization  → StandardScaler / MinMaxScaler
  7. Train/Test split
"""

import os
import re
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ─────────────────────────────────────────────
# Utilidades
# ─────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────
# Paso 1 — Data Info
# ─────────────────────────────────────────────

def data_info(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  DATA INFO")
    print("=" * 60)
    print(f"  Shape          : {df.shape}")
    print(f"  Columnas       : {df.columns.tolist()}")
    print(f"  Tipos de datos :")
    print(df.dtypes.to_string())
    print(f"\n  Nulos por columna:")
    print(df.isnull().sum().to_string())
    print("=" * 60)


# ─────────────────────────────────────────────
# Paso 2 — Data Describe
# ─────────────────────────────────────────────

def data_describe(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("  DATA DESCRIBE")
    print("=" * 60)
    print(df.describe(include="all").to_string())
    print("=" * 60)


# ─────────────────────────────────────────────
# Paso 3 — Value Counts (distribución de clases)
# ─────────────────────────────────────────────

def data_value_counts(df: pd.DataFrame, target_col: str) -> None:
    print("\n" + "=" * 60)
    print(f"  VALUE COUNTS — {target_col}")
    print("=" * 60)
    if target_col in df.columns:
        counts = df[target_col].value_counts()
        pcts = df[target_col].value_counts(normalize=True) * 100
        summary = pd.DataFrame({"count": counts, "percent (%)": pcts.round(2)})
        print(summary.to_string())
    else:
        print(f"  [WARN] Columna '{target_col}' no encontrada")
    print("=" * 60)


# ─────────────────────────────────────────────
# Paso 4 — Data Cleaning
# ─────────────────────────────────────────────

def remove_drop_columns(df: pd.DataFrame, drop_cols: list) -> pd.DataFrame:
    """Elimina columnas innecesarias (id, Unnamed, etc.)."""
    existing = [c for c in drop_cols if c in df.columns]
    if existing:
        df = df.drop(columns=existing)
        print(f"  [CLEAN] Columnas eliminadas: {existing}")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    print(f"  [CLEAN] Duplicados eliminados: {n_removed}")
    return df


def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina filas con nulos. Si una columna tiene >50% nulos, la elimina."""
    null_pct = df.isnull().mean()
    high_null_cols = null_pct[null_pct > 0.5].index.tolist()
    if high_null_cols:
        df = df.drop(columns=high_null_cols)
        print(f"  [CLEAN] Columnas con >50% nulos eliminadas: {high_null_cols}")
    n_before = len(df)
    df = df.dropna()
    print(f"  [CLEAN] Filas con nulos eliminadas: {n_before - len(df)}")
    return df


def clean_special_characters(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia caracteres especiales en columnas de texto / objeto."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"[^\w\s\.\-]", "", regex=True)  # quita chars raros
            .str.upper()
        )
    print("  [CLEAN] Caracteres especiales limpiados en columnas texto")
    return df


def coerce_numeric_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Convierte a numérico las columnas que deberían serlo (ignora target)."""
    feature_cols = [c for c in df.columns if c != target_col]
    for col in feature_cols:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    print("  [CLEAN] Columnas numéricas forzadas a tipo numérico")
    return df


def clean_data(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pp = cfg["preprocessing"]
    ds = cfg["dataset"]

    print("\n" + "=" * 60)
    print("  DATA CLEANING")
    print("=" * 60)

    df = remove_drop_columns(df, pp.get("drop_columns", []))
    df = clean_special_characters(df)
    df = coerce_numeric_columns(df, ds["target_column"])

    if pp.get("remove_duplicates", True):
        df = remove_duplicates(df)
    if pp.get("remove_nulls", True):
        df = remove_nulls(df)

    print(f"  [CLEAN] Shape final: {df.shape}")
    print("=" * 60)
    return df


# ─────────────────────────────────────────────
# Paso 5 — Encoding
# ─────────────────────────────────────────────

def encode_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Convierte M → 1, B → 0 (o usa LabelEncoder si hay más clases)."""
    if df[target_col].dtype == object:
        mapping = {"M": 1, "B": 0, "MALIGNANT": 1, "BENIGN": 0}
        df[target_col] = df[target_col].map(mapping)
        # Si hay valores que no mapearon → LabelEncoder como fallback
        if df[target_col].isnull().any():
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))
        print(f"  [ENCODE] '{target_col}' codificado → M=1, B=0")
    return df


# ─────────────────────────────────────────────
# Paso 6 — Normalización
# ─────────────────────────────────────────────

def normalize_features(X_train: np.ndarray,
                        X_test: np.ndarray,
                        method: str = "standard"):
    """
    Ajusta el scaler sobre X_train y transforma X_train y X_test.
    Retorna (X_train_scaled, X_test_scaled, scaler).
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        print("  [NORM] Sin normalización aplicada")
        return X_train, X_test, None

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    print(f"  [NORM] Normalización '{method}' aplicada")
    return X_train_s, X_test_s, scaler


# ─────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame,
               config_path: str = "config.yaml"):
    """
    Ejecuta el pipeline completo de preprocesamiento y retorna:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    cfg = load_config(config_path)
    target_col = cfg["dataset"]["target_column"]

    # 1 — Info / Describe / Value Counts
    data_info(df)
    data_describe(df)
    data_value_counts(df, target_col)

    # 2 — Cleaning
    df = clean_data(df, cfg)

    # 3 — Encoding
    if cfg["preprocessing"].get("encode_target", True):
        df = encode_target(df, target_col)

    # 4 — Split features / target
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values.astype(int)
    feature_names = df.drop(columns=[target_col]).columns.tolist()

    # 5 — Train/Test split
    test_size = cfg["dataset"]["test_size"]
    random_state = cfg["dataset"]["random_state"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"\n  [SPLIT] Train={len(y_train)} | Test={len(y_test)}")

    # 6 — Normalización
    norm_method = cfg["preprocessing"].get("normalization", "standard")
    X_train, X_test, scaler = normalize_features(X_train, X_test, norm_method)

    # Guardar datos procesados
    os.makedirs("data", exist_ok=True)
    pd.DataFrame(X_train, columns=feature_names).to_csv("data/X_train.csv", index=False)
    pd.DataFrame(X_test, columns=feature_names).to_csv("data/X_test.csv", index=False)
    pd.Series(y_train, name=target_col).to_csv("data/y_train.csv", index=False)
    pd.Series(y_test, name=target_col).to_csv("data/y_test.csv", index=False)
    print("  [SAVE] Datos procesados guardados en data/")

    return X_train, X_test, y_train, y_test, feature_names, scaler
