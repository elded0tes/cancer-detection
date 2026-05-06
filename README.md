# Cancer Detection Pipeline 🔬

Pipeline de Machine Learning para detección de cáncer de mama usando el
**Wisconsin Breast Cancer Dataset**, con seguimiento completo mediante **MLFlow**
y automatización con **GitHub Actions**.

---

## Estructura del proyecto

```
cancer-detection-pipeline/
├── .github/
│   └── workflows/
│       └── pipeline.yml        # CI/CD con GitHub Actions
├── data/                       # Dataset descargado (gitignored)
├── models/                     # Modelos entrenados (joblib)
├── plots/                      # Gráficos generados
├── logs/                       # Logs de ejecución
├── src/
│   ├── __init__.py
│   ├── download_data.py        # Descarga Wisconsin (sklearn/UCI/CSV)
│   ├── preprocess.py           # Limpieza, normalización, split
│   ├── train.py                # Entrenamiento + cross-validation
│   ├── evaluate.py             # Métricas + gráficos + MLFlow
│   └── pipeline.py             # Orquestador del pipeline
├── main.py                     # Punto de entrada CLI
├── config.yaml                 # Configuración centralizada
└── requirements.txt
```

---

## Requisitos

- Python >= 3.9
- pip

```bash
pip install -r requirements.txt
```

---

## Inicio rápido

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/cancer-detection-pipeline.git
cd cancer-detection-pipeline
```

### 2. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Iniciar MLFlow UI (en otra terminal)

```bash
mlflow ui --port 5001
# Abre: http://127.0.0.1:5001
```

### 4. Ejecutar el pipeline completo

```bash
# Fuente: sklearn (sin Internet)
python main.py --source sklearn

# Fuente: UCI ML Repository
python main.py --source uci

# Forzar reentrenamiento
python main.py --retrain

# Modo producción (sin entrenar, carga modelo de MLFlow)
python main.py --no-train

# Iniciar MLFlow UI automáticamente
python main.py --start-mlflow
```

---

## Módulos

### `download_data.py`

Descarga el dataset Wisconsin Breast Cancer desde tres fuentes:

| Fuente | Descripción |
|--------|-------------|
| `sklearn` | `sklearn.datasets.load_breast_cancer` — sin Internet |
| `uci` | UCI ML Repository vía `ucimlrepo` (dataset ID 17) |
| `csv` | CSV local configurado en `config.yaml` |

### `preprocess.py`

Preprocesamiento completo:
- **data_info** — shape, dtypes, memoria
- **data_describe** — estadísticas descriptivas
- **value_counts** — distribución de clases (M/B)
- **null handling** — imputación por media/mediana o eliminación de filas
- **clean_column_names** — normalización de nombres (espacios, caracteres raros)
- **encode_target** — M→1, B→0
- **normalize** — StandardScaler / MinMaxScaler / RobustScaler
- **train_test_split** — estratificado

### `train.py`

Entrenamiento con:
- Modelos soportados: `RandomForest`, `LogisticRegression`, `SVM`, `GradientBoosting`
- **StratifiedKFold cross-validation** (n_folds configurable)
- Registro de **start_datetime** y **end_datetime** con duración
- Guardado local con `joblib`
- Registro en MLFlow: parámetros, métricas CV por fold, firma del modelo

### `evaluate.py`

Métricas y gráficos:
- **Precision, Recall, F1-Score** (weighted y macro)
- **Confusion Matrix** → `plots/confusion_matrix_<run_id>.png`
- **ROC Curve + AUC** → `plots/roc_curve_<run_id>.png`
- **Classification Report** → `plots/classification_report_<run_id>.png`
- Registro de métricas y artefactos en MLFlow

### `pipeline.py`

Orquestador:
- Ejecuta todos los pasos en secuencia
- Soporta modo **retrain** y modo **inferencia** (sin reentrenar)
- Imprime resumen final con métricas y URL de MLFlow

---

## Configuración (`config.yaml`)

```yaml
data:
  source: "sklearn"       # sklearn | uci | csv
  test_size: 0.2

model:
  name: "RandomForest"    # RandomForest | LogisticRegression | SVM | GradientBoosting
  retrain: true
  cross_val_folds: 5
  params:
    n_estimators: 200
    max_depth: 8

mlflow:
  tracking_uri: "http://127.0.0.1:5001"
  experiment_name: "cancer-detection"
```

---

## CI/CD con GitHub Actions

El workflow `.github/workflows/pipeline.yml` se ejecuta automáticamente en:
- Push a `main` o `develop`
- Pull Request a `main`
- Manualmente vía `workflow_dispatch`

Los artefactos generados (plots, modelo, logs) se suben como **GitHub Artifacts**
y están disponibles 30 días.

---

## MLFlow

Abre la UI en `http://127.0.0.1:5001` para ver:
- **Experimentos** con todas las ejecuciones
- **Parámetros** del modelo y preprocesamiento
- **Métricas** de CV (por fold y promedio) y de test
- **Artefactos**: plots de ROC, Confusion Matrix, modelo serializado
- **Model Registry**: versiones del modelo registradas

---

## Dataset — Wisconsin Breast Cancer

| Atributo | Valor |
|----------|-------|
| Instancias | 569 |
| Features | 30 (núcleo celular) |
| Clases | Maligno (M) / Benigno (B) |
| Balance | 37% M / 63% B |
| Fuente | UCI ID 17 / sklearn |
