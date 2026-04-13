import os
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pymongo import MongoClient

from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve

# Modelos opcionales
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False

from ml_toolkit import (
    EDAExplorer,
    DataPreparer,
    SupervisedRunner,
    ModelEvaluator,
)

warnings.filterwarnings("ignore")

# =========================================================
# CONFIGURACIÓN GENERAL
# =========================================================
TARGET_COL = "group"          # <-- CAMBIA ESTO si tu target tiene otro nombre
POS_LABEL = 1                # Clase positiva: conversión
DROP_COLS = ["_id", "id", "ID", "patient_id"]   # Ajusta según tu colección
EXCLUDE_COLS = ["initial_edss", "final_edss"]
TRAIN_SIZE = 0.80
RANDOM_STATE = 42
N_SPLITS = 5
OUTPUT_DIR = Path("artifacts_ms")

# Mapeo del target.
# Ajusta según cómo venga tu variable objetivo en Mongo.
TARGET_MAP = {
    1: 1,
    2: 0,
    "1": 1,
    "2": 0,
    "CDMS": 1,
    "Non-CDMS": 0,
    "non-CDMS": 0,
    "conversion": 1,
    "no_conversion": 0,
    "yes": 1,
    "no": 0,
    "Yes": 1,
    "No": 0,
}

# Scoring principal para escoger mejor modelo
MAIN_SCORE = "ROC_AUC_Pos"


# =========================================================
# UTILIDADES
# =========================================================
def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo .env en: {env_path.resolve()}")

    with open(env_path, "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def load_data_from_mongo() -> pd.DataFrame:
    load_env_file()

    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME", "ms_data")
    collection_name = os.getenv("MONGO_COLLECTION_MODEL", "cis_model")

    if not mongo_uri:
        raise ValueError("No existe MONGO_URI en el .env")

    client = MongoClient(mongo_uri)
    try:
        db = client[db_name]
        collection = db[collection_name]
        docs = list(collection.find({}, {"_id": 0}))
        if not docs:
            raise ValueError("La colección está vacía o no devolvió documentos.")
        df = pd.DataFrame(docs)
        return df
    finally:
        client.close()


def validate_loaded_data(df: pd.DataFrame) -> None:
    print("\n=== VALIDACIÓN BÁSICA ===")
    print("Shape:", df.shape)
    print("\nColumnas:")
    print(df.columns.tolist())
    print("\nTipos:")
    print(df.dtypes)
    print("\nNulos por columna:")
    print(df.isna().sum())


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    # Normaliza nombres
    eda = EDAExplorer.from_df(df)
    eda.normalizar_columnas()
    df = eda.df

    # Eliminar columnas que no deben modelarse o que el usuario pidió excluir
    cols_to_drop = DROP_COLS + EXCLUDE_COLS
    drop_existing = [col for col in cols_to_drop if col in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"No se encontró la columna objetivo '{TARGET_COL}'. "
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    # Mapear target a binario
    df[TARGET_COL] = df[TARGET_COL].map(lambda x: TARGET_MAP.get(x, x))

    # Validación del target
    target_unique = pd.Series(df[TARGET_COL]).dropna().unique().tolist()
    if not set(target_unique).issubset({0, 1}):
        raise ValueError(
            f"El target '{TARGET_COL}' no quedó binario. Valores encontrados: {target_unique}"
        )

    # No crear variables adicionales: se conserva exactamente la estructura original del dataset
    df_model = df.copy()

    # Eliminar filas con target nulo
    df_model = df_model.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    # Convertir a numérico todas las features del modelo
    feature_cols = [c for c in df_model.columns if c != TARGET_COL]
    for col in feature_cols:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    # El target también debe quedar numérico
    df_model[TARGET_COL] = pd.to_numeric(df_model[TARGET_COL], errors="coerce")

    # Protección mínima contra nulos remanentes
    df_model = df_model.dropna().reset_index(drop=True)

    final_features = [c for c in df_model.columns if c != TARGET_COL]
    return df_model, final_features


def build_models() -> dict:
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            random_state=RANDOM_STATE
        ),
        "SVM": SVC(
            probability=True,
            random_state=RANDOM_STATE
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE
        )

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=-1
        )

    return models


def get_class_weight_for_model(model_name: str):
    # Puedes dejar balanced para los modelos que lo soportan.
    # XGBoost usa scale_pos_weight y tu toolkit ya lo calcula si encuentra ese parámetro.
    if model_name in {"LogisticRegression", "RandomForest", "SVM", "XGBoost", "LightGBM"}:
        return "balanced"
    return None


def get_sampling_method():
    # Como dijiste que la limpieza la hacen antes, dejo None por defecto.
    # Si luego quieres probar balanceo extra:
    # return "oversample"
    # return "undersample"
    # return "smote_tomek"
    return None


def fit_and_evaluate_models(
    df_model: pd.DataFrame,
    features: list[str],
    models: dict
) -> tuple[pd.DataFrame, dict]:
    results_rows = []
    detailed_results = {}

    for model_name, model in models.items():
        print(f"\nEntrenando: {model_name}")

        preparer = DataPreparer(
            train_size=TRAIN_SIZE,
            random_state=RANDOM_STATE,
            scale_X=True
        )

        runner = SupervisedRunner(
            df=df_model,
            target=TARGET_COL,
            model=clone(model),
            task="classification",
            features=features,
            preparer=preparer,
            encode_target=False,
            pos_label=POS_LABEL,
            class_weight=get_class_weight_for_model(model_name),
            sampling_method=get_sampling_method(),
        )

        holdout_metrics = runner.evaluate()
        cv_metrics = runner.evaluate_cv(n_splits=N_SPLITS, shuffle=True)

        merged = {
            "model": model_name,
            **{f"holdout_{k}": v for k, v in holdout_metrics.items() if k != "ConfusionMatrix"},
            **{f"cv_{k}": v for k, v in cv_metrics.items()},
        }

        results_rows.append(merged)

        detailed_results[model_name] = {
            "runner": runner,
            "holdout_metrics": holdout_metrics,
            "cv_metrics": cv_metrics,
        }

    results_df = pd.DataFrame(results_rows)
    return results_df, detailed_results


def pick_best_model(results_df: pd.DataFrame, score_col: str = f"cv_{MAIN_SCORE}") -> str:
    if score_col not in results_df.columns:
        raise ValueError(f"No existe la columna de score '{score_col}' en results_df.")

    best_idx = results_df[score_col].astype(float).idxmax()
    return results_df.loc[best_idx, "model"]


def tune_best_model(best_model_name: str, runner: SupervisedRunner) -> dict:
    model_spaces = {}

    if best_model_name == "LogisticRegression":
        model_spaces["LogisticRegression"] = {
            "estimator": LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            "param_grid": {
                "C": [0.01, 0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"],
            }
        }

    elif best_model_name == "RandomForest":
        model_spaces["RandomForest"] = {
            "estimator": RandomForestClassifier(random_state=RANDOM_STATE),
            "param_grid": {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 5, 10],
                "min_samples_split": [2, 5, 10],
            }
        }

    elif best_model_name == "SVM":
        model_spaces["SVM"] = {
            "estimator": SVC(probability=True, random_state=RANDOM_STATE),
            "param_grid": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
            }
        }

    elif best_model_name == "XGBoost" and XGBOOST_AVAILABLE:
        model_spaces["XGBoost"] = {
            "estimator": XGBClassifier(
                eval_metric="logloss",
                random_state=RANDOM_STATE
            ),
            "param_grid": {
                "n_estimators": [200, 300],
                "max_depth": [3, 4, 6],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.8, 0.9, 1.0],
            }
        }

    elif best_model_name == "LightGBM" and LIGHTGBM_AVAILABLE:
        model_spaces["LightGBM"] = {
            "estimator": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            "param_grid": {
                "n_estimators": [200, 300],
                "num_leaves": [15, 31, 63],
                "learning_rate": [0.03, 0.05, 0.1],
            }
        }

    else:
        print(f"No hay tuning configurado para {best_model_name}.")
        return {}

    evaluator = runner.build_evaluator(scoring="roc_auc", cv=N_SPLITS)
    tuning_results = evaluator.exhaustive_search(model_spaces)

    return tuning_results


def evaluate_fitted_model(runner: SupervisedRunner, model) -> dict:
    # Forzamos el split si aún no existe
    if not runner._prepared:
        runner._prepare()

    X_train_fit, y_train_fit = runner._apply_sampling(runner.X_train, runner.y_train)
    model = runner._apply_class_balancing(model, y_train_fit, runner.class_weight)
    model.fit(X_train_fit, y_train_fit)

    y_pred = model.predict(runner.X_test)
    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(runner.X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(runner.X_test)

    metrics = {
        "Accuracy": float(accuracy_score(runner.y_test, y_pred)),
        "Precision": float(precision_score(runner.y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(runner.y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(runner.y_test, y_pred, zero_division=0)),
        "ConfusionMatrix": confusion_matrix(runner.y_test, y_pred).tolist(),
    }

    if y_score is not None:
        metrics["ROC_AUC"] = float(roc_auc_score(runner.y_test, y_score))

    return {
        "model": model,
        "metrics": metrics,
        "X_test": runner.X_test.copy(),
        "y_test": pd.Series(runner.y_test).copy(),
        "feature_names": list(runner.X_test.columns),
    }



def save_roc_curve_data(best_eval: dict) -> None:
    model = best_eval["model"]
    X_test = best_eval["X_test"]
    y_test = best_eval["y_test"]

    y_score = None
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)

    if y_score is None:
        print("No se pudo generar roc_curve_data.csv: el modelo no expone scores.")
        return

    fpr, tpr, _ = roc_curve(y_test, y_score)

    roc_df = pd.DataFrame({
        "fpr": fpr,
        "tpr": tpr
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    roc_df.to_csv(OUTPUT_DIR / "roc_curve_data.csv", index=False)
    print("roc_curve_data.csv guardado correctamente.")


def extract_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        if np.ndim(coef) == 2:
            importances = np.abs(coef[0])
        else:
            importances = np.abs(coef)

    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi


def save_artifacts(
    df_raw: pd.DataFrame,
    df_model: pd.DataFrame,
    feature_cols: list[str],
    results_df: pd.DataFrame,
    best_model_name: str,
    best_eval: dict,
    feature_importance_df: pd.DataFrame,
    tuning_results: dict | None = None
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Tablas principales
    df_raw.head(50).to_csv(OUTPUT_DIR / "raw_preview.csv", index=False)
    df_model.head(50).to_csv(OUTPUT_DIR / "model_preview.csv", index=False)
    results_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)
    feature_importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    # Mejor modelo
    joblib.dump(best_eval["model"], OUTPUT_DIR / "best_model.pkl")

    # Features esperadas por el modelo
    with open(OUTPUT_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    # Métricas del mejor modelo
    with open(OUTPUT_DIR / "best_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model_name": best_model_name,
                "metrics": best_eval["metrics"],
            },
            f,
            ensure_ascii=False,
            indent=2
        )

    # Ejemplos de input para Streamlit
    sample_input = pd.DataFrame(columns=feature_cols)
    sample_input.to_csv(OUTPUT_DIR / "sample_input_template.csv", index=False)

    # Tuning opcional
    if tuning_results:
        serializable_tuning = {}
        for model_name, result in tuning_results.items():
            serializable_tuning[model_name] = {
                "best_params": result.get("best_params", {}),
                "best_score": result.get("best_score", None),
            }
        with open(OUTPUT_DIR / "tuning_results.json", "w", encoding="utf-8") as f:
            json.dump(serializable_tuning, f, ensure_ascii=False, indent=2)

    print(f"\nArtefactos guardados en: {OUTPUT_DIR.resolve()}")


def main():
    # 1) Cargar datos
    df_raw = load_data_from_mongo()
    validate_loaded_data(df_raw)

    # 2) Preprocesamiento
    df_model, feature_cols = preprocess_dataframe(df_raw)

    print("\n=== DATASET LISTO PARA MODELADO ===")
    print("Shape:", df_model.shape)
    print("Target distribution:")
    print(df_model[TARGET_COL].value_counts(dropna=False))

    # 3) Modelos
    models = build_models()
    results_df, detailed_results = fit_and_evaluate_models(df_model, feature_cols, models)

    print("\n=== RESUMEN DE MÉTRICAS ===")
    print(results_df)

    # 4) Escoger mejor modelo según cv_ROC_AUC_Pos
    best_model_name = pick_best_model(results_df, score_col=f"cv_{MAIN_SCORE}")
    print(f"\nMejor modelo seleccionado: {best_model_name}")

    best_runner = detailed_results[best_model_name]["runner"]

    # 5) Tuning opcional del mejor modelo
    tuning_results = tune_best_model(best_model_name, best_runner)

    if tuning_results and best_model_name in tuning_results:
        tuned_model = tuning_results[best_model_name]["estimator"]
        print(f"\nSe aplicó tuning a: {best_model_name}")
    else:
        tuned_model = clone(models[best_model_name])
        print(f"\nSe usará el modelo base de: {best_model_name}")

    # 6) Evaluación final del mejor modelo
    best_eval = evaluate_fitted_model(best_runner, tuned_model)
    save_roc_curve_data(best_eval)

    print("\n=== MÉTRICAS FINALES DEL MEJOR MODELO ===")
    print(best_eval["metrics"])

    # 7) Importancia de variables
    fi_df = extract_feature_importance(best_eval["model"], best_eval["feature_names"])
    print("\n=== TOP FEATURES ===")
    print(fi_df.head(15))

    # 8) Guardar artefactos para Streamlit
    save_artifacts(
        df_raw=df_raw,
        df_model=df_model,
        feature_cols=feature_cols,
        results_df=results_df,
        best_model_name=best_model_name,
        best_eval=best_eval,
        feature_importance_df=fi_df,
        tuning_results=tuning_results
    )


if __name__ == "__main__":
    main()