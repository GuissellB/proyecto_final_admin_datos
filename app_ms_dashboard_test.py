import json
import os
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient

APP_TITLE = "CIS to Multiple Sclerosis Conversion Prediction"
APP_SUBTITLE = "Clinical machine learning analytics and prediction support"
ARTIFACTS_DIR = Path("artifacts_ms")
TARGET_COL = "group"
DROP_COLS = ["_id", "id", "ID", "patient_id"]
EXCLUDE_COLS = ["initial_edss", "final_edss", "mri_lesion_count", "has_any_mri_lesion", "evoked_positive_count", "has_any_evoked_positive", "age_group_24_or_less", "age_group_25_to_34", "age_group_35_to_44", "age_group_45_or_more",
]
TARGET_MAP = {
    1: 1, 2: 0, "1": 1, "2": 0,
    "CDMS": 1, "Non-CDMS": 0, "non-CDMS": 0,
    "conversion": 1, "no_conversion": 0,
    "yes": 1, "no": 0, "Yes": 1, "No": 0,
    True: 1, False: 0,
}
NAV_ITEMS = [
    "Overview",
    "Dataset",
    "Model Performance",
    "Explainability",
    "Patient Prediction",
    "Insights",
]

FIELD_SPECS = {
    "age": {
        "label": "Age (years)",
        "type": "int_range",
        "min_value": 0,
        "max_value": 120,
        "default": 30,
        "help": "Allowed range: 0 to 120 years.",
    },
    "schooling": {
        "label": "Schooling (years)",
        "type": "int_range",
        "min_value": 0,
        "max_value": 40,
        "default": 12,
    },
    "gender": {
        "label": "Gender",
        "type": "coded_select",
        "options": {1: "Male", 2: "Female"},
    },
    "breastfeeding": {
        "label": "Breastfeeding",
        "type": "coded_select",
        "options": {1: "Yes", 2: "No", 3: "Unknown"},
    },
    "varicella": {
        "label": "Varicella",
        "type": "coded_select",
        "options": {1: "Positive", 2: "Negative", 3: "Unknown"},
    },
    "initial_symptoms": {
        "label": "Initial Symptoms",
        "type": "coded_select",
        "options": {
            1: "Visual",
            2: "Sensory",
            3: "Motor",
            4: "Other",
            5: "Visual and sensory",
            6: "Visual and motor",
            7: "Visual and other",
            8: "Sensory and motor",
            9: "Sensory and other",
            10: "Motor and other",
            11: "Visual, sensory and motor",
            12: "Visual, sensory and other",
            13: "Visual, motor and other",
            14: "Sensory, motor and other",
            15: "Visual, sensory, motor and other",
        },
    },
    "mono_or_polysymptomatic": {
        "label": "Mono / Polysymptomatic",
        "type": "coded_select",
        "options": {1: "Monosymptomatic", 2: "Polysymptomatic", 3: "Unknown"},
    },
    "oligoclonal_bands": {
        "label": "Oligoclonal Bands",
        "type": "coded_select",
        "options": {0: "Negative", 1: "Positive", 2: "Unknown"},
    },
    "llssep": {
        "label": "LLSSEP",
        "type": "coded_select",
        "options": {0: "Negative", 1: "Positive"},
    },
    "ulssep": {
        "label": "ULSSEP",
        "type": "coded_select",
        "options": {0: "Negative", 1: "Positive"},
    },
    "vep": {
        "label": "VEP",
        "type": "coded_select",
        "options": {0: "Negative", 1: "Positive"},
    },
    "baep": {
        "label": "BAEP",
        "type": "coded_select",
        "options": {0: "Negative", 1: "Positive"},
    },
    "periventricular_mri": {
        "label": "Periventricular MRI",
        "type": "coded_select",
        "options": {0: "Negative", 1: "Positive"},
    },
    "cortical_mri": {
        "label": "Cortical MRI",
        "type": "coded_select",
        "options": {0: "Negative", 1: "Positive"},
    },
    "infratentorial_mri": {
        "label": "Infratentorial MRI",
        "type": "coded_select",
        "options": {0: "Negative", 1: "Positive"},
    },
    "spinal_cord_mri": {
        "label": "Spinal Cord MRI",
        "type": "coded_select",
        "options": {0: "Negative", 1: "Positive"},
    },
}

FIELD_ALIASES = {
    "sex": "gender",
    "initial_symptom": "initial_symptoms",
    "initial_symptoms_1": "initial_symptoms",
    "first_symptom": "initial_symptoms",
    "mono_polysymptomatic": "mono_or_polysymptomatic",
    "mono_or_polysymtomatic": "mono_or_polysymptomatic",
    "oligoclonal_band": "oligoclonal_bands",
    "periventricular": "periventricular_mri",
    "cortical": "cortical_mri",
    "infratentorial": "infratentorial_mri",
    "spinal_cord": "spinal_cord_mri",
}


st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()


def safe_read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).replace('"', '').strip() for c in out.columns]
    return out


def clean_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"[^0-9a-zA-Z_]+", "_", str(c)).strip("_") for c in out.columns]
    return out


def map_target(series: pd.Series) -> pd.Series:
    return series.map(lambda x: TARGET_MAP.get(x, x))


@st.cache_data(show_spinner=False)
def load_data_from_mongo() -> pd.DataFrame:
    load_env_file()
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        return pd.DataFrame()

    db_name = os.getenv("MONGO_DB_NAME", "ms_data")
    collection_name = os.getenv("MONGO_COLLECTION_MODEL", "cis_model")

    client = MongoClient(mongo_uri)
    try:
        docs = list(client[db_name][collection_name].find({}, {"_id": 0}))
        return pd.DataFrame(docs) if docs else pd.DataFrame()
    finally:
        client.close()


@st.cache_data(show_spinner=False)
def load_previews() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(ARTIFACTS_DIR / "raw_preview.csv") if (ARTIFACTS_DIR / "raw_preview.csv").exists() else pd.DataFrame()
    model = pd.read_csv(ARTIFACTS_DIR / "model_preview.csv") if (ARTIFACTS_DIR / "model_preview.csv").exists() else pd.DataFrame()
    return raw, model


@st.cache_resource(show_spinner=False)
def load_model_and_metadata() -> tuple[Any, list[str], dict[str, Any]]:
    model = joblib.load(ARTIFACTS_DIR / "best_model.pkl") if (ARTIFACTS_DIR / "best_model.pkl").exists() else None
    feature_columns = safe_read_json(ARTIFACTS_DIR / "feature_columns.json", [])
    best_metrics = safe_read_json(ARTIFACTS_DIR / "best_model_metrics.json", {})
    return model, feature_columns, best_metrics


@st.cache_data(show_spinner=False)
def load_support_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_df = pd.read_csv(ARTIFACTS_DIR / "metrics_summary.csv") if (ARTIFACTS_DIR / "metrics_summary.csv").exists() else pd.DataFrame()
    fi_df = pd.read_csv(ARTIFACTS_DIR / "feature_importance.csv") if (ARTIFACTS_DIR / "feature_importance.csv").exists() else pd.DataFrame()
    return metrics_df, fi_df


@st.cache_data(show_spinner=False)
def load_roc_curve_data() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "roc_curve_data.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def get_raw_df() -> pd.DataFrame:
    mongo_df = load_data_from_mongo()
    if not mongo_df.empty:
        return normalize_columns(mongo_df)
    raw_preview, _ = load_previews()
    return normalize_columns(raw_preview) if not raw_preview.empty else pd.DataFrame()



def get_model_df() -> pd.DataFrame:
    _, model_preview = load_previews()
    return model_preview.copy() if not model_preview.empty else pd.DataFrame()


def canonical_field_name(col_name: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z_]+", "_", str(col_name).strip().lower()).strip("_")

    if normalized in FIELD_ALIASES:
        return FIELD_ALIASES[normalized]

    if normalized.startswith("initial_symptom"):
        return "initial_symptoms"

    return normalized


def prettify_label(col_name: str) -> str:
    return str(col_name).replace("_", " ").strip().title()


def get_series_default(series: pd.Series, fallback: Any) -> Any:
    if series is None or series.empty:
        return fallback
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return series.iloc[0]


def render_patient_field(col_name: str, series: pd.Series, key_prefix: str = "pred") -> Any:
    canonical = canonical_field_name(col_name)
    spec = FIELD_SPECS.get(canonical)
    widget_key = f"{key_prefix}_{col_name}"

    if spec and spec.get("type") == "coded_select":
        options_map = spec["options"]
        valid_codes = list(options_map.keys())
        default_code = get_series_default(series[series.isin(valid_codes)] if not series.empty else series, valid_codes[0])
        default_code = default_code if default_code in valid_codes else valid_codes[0]
        labels = [options_map[code] for code in valid_codes]
        selected_label = st.selectbox(
            spec.get("label", prettify_label(col_name)),
            options=labels,
            index=valid_codes.index(default_code),
            key=widget_key,
            help=spec.get("help"),
        )
        reverse_map = {label: code for code, label in options_map.items()}
        return reverse_map[selected_label]

    if spec and spec.get("type") == "int_range":
        min_value = int(spec.get("min_value", 0))
        max_value = int(spec.get("max_value", 100))
        default_value = int(spec.get("default", min_value))
        if not series.empty:
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric_series.empty:
                default_value = int(np.clip(round(float(numeric_series.median())), min_value, max_value))
        return st.number_input(
            spec.get("label", prettify_label(col_name)),
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=1,
            key=widget_key,
            help=spec.get("help"),
        )

    if spec and spec.get("type") == "float_range":
        min_value = float(spec.get("min_value", 0.0))
        max_value = float(spec.get("max_value", 10.0))
        step = float(spec.get("step", 0.5))
        default_value = float(spec.get("default", min_value))
        if not series.empty:
            numeric_series = pd.to_numeric(series, errors="coerce").dropna()
            if not numeric_series.empty:
                default_value = float(np.clip(float(numeric_series.median()), min_value, max_value))
                default_value = round(default_value / step) * step
        return st.number_input(
            spec.get("label", prettify_label(col_name)),
            min_value=min_value,
            max_value=max_value,
            value=float(default_value),
            step=step,
            key=widget_key,
            help=spec.get("help"),
        )

    if pd.api.types.is_numeric_dtype(series):
        numeric_series = pd.to_numeric(series, errors="coerce").dropna()
        value = float(numeric_series.median()) if not numeric_series.empty else 0.0

        is_integer_like = False
        if not numeric_series.empty:
            is_integer_like = bool((numeric_series % 1 == 0).all())
        else:
            is_integer_like = float(value).is_integer()

        if is_integer_like:
            value = int(round(value))
            return st.number_input(
                prettify_label(col_name),
                value=value,
                step=1,
                key=widget_key,
            )

        return st.number_input(
            prettify_label(col_name),
            value=float(value),
            step=0.1,
            key=widget_key,
        )

    options = [str(v) for v in series.astype(str).dropna().unique().tolist()] or [""]
    return st.selectbox(
        prettify_label(col_name),
        options=options,
        index=0,
        key=widget_key,
    )



def prepare_prediction_input(raw_input: dict[str, Any], raw_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([raw_input])
    df = normalize_columns(df)

    cols_to_drop = DROP_COLS + EXCLUDE_COLS
    drop_existing = [c for c in cols_to_drop if c in df.columns]
    if drop_existing:
        df = df.drop(columns=drop_existing)
    if TARGET_COL in df.columns:
        df = df.drop(columns=[TARGET_COL])

    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0

    extra_cols = [c for c in df.columns if c not in feature_columns]
    if extra_cols:
        df = df.drop(columns=extra_cols)

    df = df[feature_columns]
    return df



def metric_col(metrics_df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in metrics_df.columns:
            return candidate
    lowered = {c.lower(): c for c in metrics_df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None



def friendly_model_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame(columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])

    model_col = "model" if "model" in metrics_df.columns else metrics_df.columns[0]
    acc = metric_col(metrics_df, ["cv_Accuracy", "holdout_Accuracy", "Accuracy"])
    prec = metric_col(metrics_df, ["cv_Precision_Pos", "holdout_Precision_Pos", "Precision", "Precision_Pos"])
    rec = metric_col(metrics_df, ["cv_Recall_Pos", "holdout_Recall_Pos", "Recall", "Recall_Pos"])
    f1 = metric_col(metrics_df, ["cv_F1_Pos", "holdout_F1_Pos", "F1", "F1_Pos"])
    auc = metric_col(metrics_df, ["cv_ROC_AUC_Pos", "holdout_ROC_AUC_Pos", "ROC_AUC", "ROC_AUC_Pos"])

    out = pd.DataFrame({
        "Model": metrics_df[model_col].astype(str),
        "Accuracy": pd.to_numeric(metrics_df[acc], errors="coerce") if acc else np.nan,
        "Precision": pd.to_numeric(metrics_df[prec], errors="coerce") if prec else np.nan,
        "Recall": pd.to_numeric(metrics_df[rec], errors="coerce") if rec else np.nan,
        "F1": pd.to_numeric(metrics_df[f1], errors="coerce") if f1 else np.nan,
        "ROC-AUC": pd.to_numeric(metrics_df[auc], errors="coerce") if auc else np.nan,
    })
    return out



def build_kpis(raw_df: pd.DataFrame, model_table: pd.DataFrame, best_metrics: dict[str, Any]) -> dict[str, Any]:
    total = len(raw_df) if not raw_df.empty else 0
    conversion_cases = 0
    non_conversion_cases = 0
    if not raw_df.empty and TARGET_COL in raw_df.columns:
        y = map_target(raw_df[TARGET_COL])
        vc = y.value_counts(dropna=False)
        conversion_cases = int(vc.get(1, 0))
        non_conversion_cases = int(vc.get(0, 0))

    best_model = None
    best_auc = None
    best_recall = None
    if not model_table.empty:
        auc_idx = model_table["ROC-AUC"].astype(float).idxmax()
        best_model = model_table.loc[auc_idx, "Model"]
        best_auc = float(model_table.loc[auc_idx, "ROC-AUC"])
        best_recall = float(model_table["Recall"].max())

    

    return {
        "total_patients": total,
        "conversion_cases": conversion_cases,
        "non_conversion_cases": non_conversion_cases,
        "best_roc_auc": best_auc,
        "best_recall": best_recall,
        "best_model": best_model,
    }



def infer_feature_categories(raw_df: pd.DataFrame) -> pd.DataFrame:
    if raw_df.empty:
        return pd.DataFrame(columns=["category", "count", "feature_list"])

    cols = [c for c in raw_df.columns if c not in DROP_COLS + EXCLUDE_COLS + [TARGET_COL]]
    groups = {
        "Demographic": [],
        "Clinical": [],
        "MRI Features": [],
        "Laboratory": [],
        "Other": [],
    }

    for col in cols:
        name = col.lower()
        if any(k in name for k in ["age", "sex", "gender", "family"]):
            groups["Demographic"].append(col)
        elif any(k in name for k in ["mri", "lesion", "t2", "spinal", "brain", "gd", "periventricular", "cortical", "infratentorial"]):
            groups["MRI Features"].append(col)
        elif any(k in name for k in ["oligo", "band", "csf", "lab"]):
            groups["Laboratory"].append(col)
        elif any(k in name for k in ["edss", "symptom", "duration", "clinical", "attack", "score", "mono", "poly"]):
            groups["Clinical"].append(col)
        else:
            groups["Other"].append(col)

    rows = []
    for group, features in groups.items():
        if features:
            rows.append({
                "category": group,
                "count": len(features),
                "feature_list": features
            })

    return pd.DataFrame(rows)

def top_predictor_names(fi_df: pd.DataFrame, n: int = 3) -> list[str]:
    if fi_df.empty or "feature" not in fi_df.columns:
        return []
    return fi_df.sort_values("importance", ascending=False)["feature"].head(n).tolist()



def risk_label(prob: float) -> tuple[str, str]:
    if prob >= 0.70:
        return "High Risk", "high"
    if prob >= 0.40:
        return "Medium Risk", "medium"
    return "Low Risk", "low"


CUSTOM_CSS = """
<style>
    :root {
        --bg: #f3f5f7;
        --sidebar: #082b4c;
        --sidebar-accent: #0a4f63;
        --card: #ffffff;
        --text: #0b2f56;
        --muted: #73839a;
        --teal: #13b8aa;
        --teal-dark: #0ca89b;
        --blue: #2d6df6;
        --orange: #ff6b00;
        --line: #e7ebef;
        --danger: #ff4d4f;
        --success: #00a86b;
    }
    .stApp {background: var(--bg);}
    .block-container {padding-top: 3.8rem; padding-bottom: 1.5rem; max-width: 1400px;}
    [data-testid="stSidebar"] {background: var(--sidebar);}
    [data-testid="stSidebar"] * {
    color: white !important;
    }

    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div {
        color: white !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background: transparent;
        border-radius: 16px;
        padding: 14px 14px;
        border: 1px solid transparent;
        color: #d7e3f0 !important;
        margin: 0;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label p,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label span,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label div {
        color: #d7e3f0 !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(15,115,130,.55), rgba(6,74,89,.9));
        border-color: rgba(31, 209, 196, .22);
        color: white !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) p,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) span,
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) div {
        color: white !important;
        opacity: 1 !important;
    }

    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] p {
        font-size: 1.10rem;
        font-weight: 600;
        margin: 0;
        color: #d7e3f0 !important;
    }
    [data-testid="stSidebar"] > div:first-child {padding-top: 0.8rem;}
    .sidebar-brand {padding: 0.7rem 0.3rem 1.1rem 0.3rem; color: white;}
    .brand-title {font-size: 1.95rem; font-weight: 800; line-height: 1.05; margin: 0;}
    .brand-sub {color: #6affef; font-size: 1.0rem; margin-top: 0.25rem;}
    [data-testid="stSidebar"] .stRadio > label {display: none;}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {gap: 0.65rem;}
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
        background: transparent;
        border-radius: 16px;
        padding: 14px 14px;
        border: 1px solid transparent;
        color: #d7e3f0;
        margin: 0;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(90deg, rgba(15,115,130,.55), rgba(6,74,89,.9));
        border-color: rgba(31, 209, 196, .22);
        color: white;
    }
    [data-testid="stSidebar"] .stRadio div[role="radiogroup"] p {
        font-size: 1.10rem; font-weight: 600; margin: 0;
    }
    .header-wrap {
        background: transparent;
        padding: 0 0 .35rem 0;
        margin-bottom: .55rem;
    }
    .header-row {display:flex; align-items:center; justify-content:flex-start; gap:16px;}
    .header-title {font-size: 2.3rem; font-weight: 800; color: var(--text); line-height:1.05; margin:0;}
    .header-sub {font-size: 1.05rem; color: var(--muted); margin-top: .2rem;}
    .section-title {font-size: 1.95rem; color: var(--text); font-weight: 800; margin: .2rem 0 1rem 0;}
    .card {
        background: var(--card); border: 1px solid var(--line); border-radius: 22px;
        padding: 1.35rem 1.45rem; box-shadow: 0 4px 18px rgba(20,34,56,.05);
        margin-bottom: 1rem;
    }
    .kpi-card {min-height: 184px; display:flex; justify-content:space-between; align-items:flex-start;}
    .kpi-title {font-size: 1.05rem; color:#51657d; margin-bottom:.8rem;}
    .kpi-value {font-size: 2.7rem; line-height:1; color:var(--text); font-weight: 500; margin-bottom:.45rem;}
    .kpi-note {font-size: 0.95rem; color:#96a3b3;}
    .icon-pill {
        width: 62px; height: 62px; border-radius: 16px; display:flex; align-items:center; justify-content:center;
        font-size: 1.8rem; color:white; font-weight:700;
    }
    .icon-blue {background:#2d6df6;}
    .icon-orange {background:#ff6b00;}
    .icon-teal {background:#13b8aa;}
    .icon-cyan {background:#12a8a4;}
    .subcard-title {font-size: 1.1rem; font-weight: 700; color:#334e6f; margin-bottom: .75rem;}
    .feature-row {display:flex; justify-content:space-between; gap:10px; padding: .8rem 0; border-bottom: 1px solid #edf1f4;}
    .feature-row:last-child {border-bottom:none;}
    .feature-name {font-size:1rem; font-weight:700; color:#183b63;}
    .feature-desc {font-size:.95rem; color:#7b8c9f; margin-top:.15rem;}
    .feature-count {font-size:1.05rem; color:#06a4a0; align-self:center; white-space:nowrap;}
    .soft-table table {border-collapse: collapse; width: 100%;}
    .soft-table th, .soft-table td {padding: 14px 16px; border-bottom: 1px solid #edf1f4; text-align:left;}
    .soft-table th {color:#415d7f; font-size: 1rem;}
    .soft-table td {font-size: 1.02rem; color:#17385d;}
    .best-chip {display:inline-block; background:#dff8f2; color:#079c92; border-radius:999px; padding: .18rem .6rem; font-size:.9rem; margin-left:.45rem;}
    .matrix-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: .3rem;
    }

    div[data-testid="stForm"]:has(.prediction-button-hook) div[data-testid="stFormSubmitButton"] > button {
    border-radius: 16px !important;
    border: none !important;
    background: #13b8aa !important;
    color: white !important;
    font-weight: 800 !important;
    min-height: 58px !important;
    width: 100% !important;
    font-size: 1.15rem !important;
    }

    div[data-testid="stForm"]:has(.prediction-button-hook) div[data-testid="stFormSubmitButton"] > button:hover {
        background: #0ca89b !important;
        color: white !important;
    }

    div[data-testid="stForm"]:has(.prediction-button-hook) div[data-testid="stFormSubmitButton"] > button:focus,
    div[data-testid="stForm"]:has(.prediction-button-hook) div[data-testid="stFormSubmitButton"] > button:active {
        background: #13b8aa !important;
        color: white !important;
        box-shadow: none !important;
    }
    .matrix-cell {
        border-radius: 14px;
        border: 2px solid;
        min-height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        padding: 8px;
    }

    .matrix-value {
        font-size: 2.1rem;
        line-height: 1;
        margin-bottom: .25rem;
    }
    .m-green {background:#eef8f2; border-color:#97e6b7; color:#008f4f;}
    .m-orange {background:#fbf6ee; border-color:#f4c487; color:#f06c00;}
    .m-red {background:#f9eef0; border-color:#f0b5ba; color:#d30000;}
    .m-teal {background:#eef9f7; border-color:#8ae3d1; color:#007d8c;}
    .shap-row {margin: 1rem 0;}
    .shap-top {display:flex; justify-content:space-between; margin-bottom:.35rem; color:#1a3b61; font-size:1rem;}
    .shap-bar {height: 12px; width:100%; background:#eef0f3; border-radius:999px; overflow:hidden;}
    .shap-fill {height:100%; background:#13b8aa; border-radius:999px;}
    .form-section label {font-weight: 700; color:#2f4d6f;}
    .prediction-class {font-size: 2.45rem; color: var(--text); margin: .65rem 0 1.2rem 0;}
    .prob-value {font-size: 2.25rem; color: var(--text); margin: .3rem 0 .55rem 0;}
    .prob-track {height: 16px; background:#eff1f4; border-radius:999px; overflow:hidden;}
    .prob-fill {height:100%; background:#13b8aa; border-radius:999px;}
    .risk-badge {display:inline-block; border-radius: 16px; padding: .7rem 1rem; font-size: 1.05rem; font-weight: 700; border:1px solid; margin-top: .45rem;}
    .risk-high {background:#fff0f0; color:#db3030; border-color:#f2b0b0;}
    .risk-medium {background:#fff5ea; color:#dd6d00; border-color:#f5c995;}
    .risk-low {background:#effbf6; color:#0a9960; border-color:#a6e6c9;}
    .disclaimer {margin-top: 1.35rem; background:#edf5ff; color:#295bbf; border:1px solid #9ec2ff; border-radius: 16px; padding: 1rem 1.1rem; font-size:1rem;}
    .insight-title {font-size: 1.55rem; color: var(--text); font-weight:800; margin-bottom: .9rem;}
    .insight-card {display:flex; gap:1rem; align-items:flex-start; min-height: 140px;}
    .insight-icon {width:48px; height:48px; border-radius: 14px; display:flex; align-items:center; justify-content:center; font-size:1.35rem;}
    .bg-mint {background:#dff8f2; color:#0b9a8f;}
    .bg-blue {background:#e8f0ff; color:#2b68f5;}
    .bg-peach {background:#f9ead7; color:#f16d00;}
    .bg-gray {background:#eff1f4; color:#6f7782;}
    .insight-head {font-size:1.05rem; color:#0c2441; font-weight:800; margin-bottom:.45rem;}
    .insight-body {font-size:1rem; color:#4e6177; line-height:1.55;}
    .stNumberInput input, .stSelectbox div[data-baseweb="select"], .stTextInput input {
        border-radius: 16px !important; min-height: 52px;
    }
    .stButton > button {
        border-radius: 16px; border:none; background:#13b8aa; color:white; font-weight:800;
        min-height:58px; width:100%; font-size:1.15rem;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid var(--line) !important;
        border-radius: 22px !important;
        padding: 1.2rem 1.25rem !important;
        box-shadow: 0 4px 18px rgba(20,34,56,.05) !important;
        margin-bottom: 1rem !important;
        outline: none !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        background: transparent !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background: transparent !important;
        border: none !important;
        border-radius: 0 !important;
        padding: 0 !important;
        box-shadow: none !important;
        margin-bottom: 0 !important;
    }
    .white-panel {
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 1.35rem 1.45rem 1.35rem 1.45rem;
    box-shadow: 0 4px 18px rgba(20,34,56,.05);
    margin-bottom: 1rem;
    }
    
    .features-wrap {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.35rem 1.45rem 1.35rem 1.45rem;
        box-shadow: 0 4px 18px rgba(20,34,56,.05);
        margin-bottom: 1rem;
    }

    .features-block {
        padding: 0.9rem 0;
        border-bottom: 1px solid #edf1f4;
    }

    .features-block:last-child {
        border-bottom: none;
    }

    .features-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        margin-bottom: 0.45rem;
    }

    .features-cat {
        font-size: 1.05rem;
        font-weight: 800;
        color: #183b63;
    }

    .prediction-panel {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.35rem 1.45rem;
        box-shadow: 0 4px 18px rgba(20,34,56,.05);
        margin-bottom: 1rem;
    }

    .prediction-placeholder {
        color: #6d8097;
        font-size: 1rem;
        line-height: 1.6;
        padding-top: .35rem;
    }

    .prediction-divider {
        height: 1rem;
    }

    .features-count {
        font-size: 1.05rem;
        color: #06a4a0;
        white-space: nowrap;
        font-weight: 600;
    }

    .features-list {
        font-size: 0.98rem;
        color: #6d8097;
        line-height: 1.65;
        word-break: break-word;
    }
    

</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

raw_df = get_raw_df()
model_df = get_model_df()
model, feature_columns, best_metrics = load_model_and_metadata()
metrics_df, fi_df = load_support_tables()
roc_df = load_roc_curve_data()
model_table = friendly_model_table(metrics_df)
kpis = build_kpis(raw_df, model_table, best_metrics)
feature_categories = infer_feature_categories(raw_df)

with st.sidebar:
    st.markdown('<div class="sidebar-brand"><div class="brand-title">MS Predictor</div><div class="brand-sub">Clinical ML Analytics</div></div>', unsafe_allow_html=True)
    page = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")

st.markdown(
    f'''<div class="header-wrap"><div class="header-row"><div><div class="header-title">{APP_TITLE}</div><div class="header-sub">{APP_SUBTITLE}</div></div></div></div>''',
    unsafe_allow_html=True,
)

def render_overview() -> None:
    st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
    card_specs = [
        ("Total Patients", kpis["total_patients"], f'{kpis["total_patients"]} records available' if kpis["total_patients"] else 'No dataset loaded', "👥", "icon-blue"),
        ("Conversion Cases", kpis["conversion_cases"], f'{(100*kpis["conversion_cases"]/kpis["total_patients"]):.1f}% conversion rate' if kpis["total_patients"] else 'Conversion class', "↗", "icon-orange"),
        ("Non-Conversion Cases", kpis["non_conversion_cases"], f'{(100*kpis["non_conversion_cases"]/kpis["total_patients"]):.1f}% stable CIS' if kpis["total_patients"] else 'Stable class', "↘", "icon-teal"),
        ("Best ROC-AUC", f'{kpis["best_roc_auc"]:.2f}' if kpis["best_roc_auc"] is not None else "-", f'{kpis["best_model"] or "Best model"} performance', "🏅", "icon-cyan"),
        ("Best Recall", f'{kpis["best_recall"]:.2f}' if kpis["best_recall"] is not None else "-", 'Minimizing false negatives', "∿", "icon-blue"),
        ("Best Model", str(kpis["best_model"] or "-").replace("_", " "), 'Highest overall metrics', "🧠", "icon-orange"),
    ]

    for row_start in (0, 3):
        cols = st.columns(3, gap="large")
        for col, spec in zip(cols, card_specs[row_start:row_start+3]):
            title, value, note, icon, icon_class = spec
            with col:
                st.markdown(
                    f'''<div class="card kpi-card"><div><div class="kpi-title">{title}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div><div class="icon-pill {icon_class}">{icon}</div></div>''',
                    unsafe_allow_html=True,
                )



def render_dataset() -> None:
    st.markdown('<div class="section-title">Dataset Overview</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 1.1], gap="large")

    with left:
        st.markdown('<div class="card"><div class="subcard-title">Class Distribution</div>', unsafe_allow_html=True)
        if not raw_df.empty and TARGET_COL in raw_df.columns:
            y = map_target(raw_df[TARGET_COL])
            counts = pd.DataFrame({
                "Class": ["Non-Conversion", "Conversion"],
                "value": [int((y == 0).sum()), int((y == 1).sum())],
            })
            fig = go.Figure()
            fig.add_bar(x=counts["Class"], y=counts["value"], marker_color=["#23b6ab", "#082b4c"], width=0.55)
            fig.update_layout(
                height=340, margin=dict(l=30, r=15, t=10, b=10), plot_bgcolor="white", paper_bgcolor="white",
                yaxis=dict(gridcolor="#e9eef2", zeroline=False), xaxis=dict(showgrid=False), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No class distribution available yet.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if not feature_categories.empty:
            blocks = []

            for _, row in feature_categories.iterrows():
                feature_list = row["feature_list"]
                if isinstance(feature_list, list):
                    features_text = ",".join(feature_list)
                else:
                    features_text = str(feature_list)

                blocks.append(f"""
    <div class="features-block">
        <div class="features-head">
            <div class="features-cat">{row['category']}</div>
            <div class="features-count">{row['count']} features</div>
        </div>
        <div class="features-list">{features_text}</div>
    </div>
    """)

            html = f"""
    <div class="features-wrap">
        <div class="subcard-title">Feature Categories</div>
        {''.join(blocks)}
    </div>
    """
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.markdown(
                """
    <div class="features-wrap">
        <div class="subcard-title">Feature Categories</div>
        <div class="features-list">Feature categories will appear once the dataset is loaded.</div>
    </div>
    """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="card"><div class="subcard-title">Sample Patient Data</div>', unsafe_allow_html=True)
    if not raw_df.empty:
        preview = raw_df.head(8).copy()
        if TARGET_COL in preview.columns:
            preview[TARGET_COL] = map_target(preview[TARGET_COL]).map({1: "Yes", 0: "No"}).fillna(preview[TARGET_COL])
        st.dataframe(preview, use_container_width=True, hide_index=True)
    else:
        st.info("No dataset preview available.")
    st.markdown('</div>', unsafe_allow_html=True)



def render_model_performance() -> None:
    st.markdown('<div class="section-title">Model Performance Comparison</div>', unsafe_allow_html=True)
    if not model_table.empty:
        best_idx = model_table["ROC-AUC"].astype(float).idxmax()
        styled_rows = []
        for idx, row in model_table.iterrows():
            best_chip = ' <span class="best-chip">Best</span>' if idx == best_idx else ''
            styled_rows.append(
                f"<tr style=\"background:{'#eef7f5' if idx == best_idx else 'white'}\"><td><strong>{row['Model']}</strong>{best_chip}</td><td>{row['Accuracy']:.2f}</td><td>{row['Precision']:.2f}</td><td>{row['Recall']:.2f}</td><td>{row['F1']:.2f}</td><td>{row['ROC-AUC']:.2f}</td></tr>"
            )

        html_table = """
        <div class="card">
            <div class="soft-table"><table>
            <thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC-AUC</th></tr></thead>
            <tbody>
        """ + "".join(styled_rows) + "</tbody></table></div></div>"

        st.markdown(html_table, unsafe_allow_html=True)
    else:
        st.info("No model metrics found. Run the training script first.")

    st.markdown('<div class="card"><div class="subcard-title">Model Comparison by Key Metrics</div>', unsafe_allow_html=True)
    if not model_table.empty:
        melted = model_table.melt(id_vars="Model", value_vars=["ROC-AUC", "Recall", "F1"], var_name="Metric", value_name="Score")
        color_map = {"ROC-AUC": "#082b4c", "Recall": "#23b6ab", "F1": "#14a7db"}
        fig = px.bar(melted, x="Model", y="Score", color="Metric", barmode="group", color_discrete_map=color_map)
        fig.update_layout(height=340, margin=dict(l=20, r=20, t=8, b=10), paper_bgcolor="white", plot_bgcolor="white")
        fig.update_yaxes(range=[max(0.0, float(melted["Score"].min()) - 0.1), 1.0], gridcolor="#e9eef2")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)



def render_explainability() -> None:
    st.markdown('<div class="section-title">Model Explainability</div>', unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(f'<div class="card"><div class="subcard-title">Feature Importance ({kpis["best_model"] or "Best Model"})</div>', unsafe_allow_html=True)
        if not fi_df.empty and {"feature", "importance"}.issubset(fi_df.columns):
            top_df = fi_df.sort_values("importance", ascending=True).tail(8)
            fig = px.bar(top_df, x="importance", y="feature", orientation="h")
            fig.update_traces(marker_color="#23b6ab")
            fig.update_layout(height=360, margin=dict(l=20, r=10, t=10, b=10), paper_bgcolor="white", plot_bgcolor="white", showlegend=False)
            fig.update_xaxes(gridcolor="#e9eef2")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance will appear after training and artifact export.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        cv_auc = float(kpis["best_roc_auc"]) if kpis["best_roc_auc"] is not None else 0.50

        st.markdown(
            f'<div class="card"><div class="subcard-title">ROC Curve (CV AUC = {cv_auc:.2f})</div>',
            unsafe_allow_html=True
        )

        if not roc_df.empty and {"fpr", "tpr"}.issubset(roc_df.columns):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=roc_df["fpr"],
                y=roc_df["tpr"],
                mode="lines",
                line=dict(color="#082b4c", width=3),
                name=str(kpis["best_model"] or "Model")
            ))
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="#cfd6de", width=2, dash="dash"),
                name="Random Classifier"
            ))
            fig.update_layout(
                height=360,
                margin=dict(l=20, r=10, t=10, b=10),
                paper_bgcolor="white",
                plot_bgcolor="white"
            )
            fig.update_xaxes(title="False Positive Rate", gridcolor="#e9eef2")
            fig.update_yaxes(title="True Positive Rate", gridcolor="#e9eef2")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC curve data not found. Add artifacts_ms/roc_curve_data.csv with columns fpr and tpr.")

        st.markdown('</div>', unsafe_allow_html=True)

    left2, right2 = st.columns(2, gap="large")
    with left2:
        st.markdown('<div class="card"><div class="subcard-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = None
        if isinstance(best_metrics, dict):
            cm = best_metrics.get("metrics", {}).get("ConfusionMatrix")
        if isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2:
            tn, fp = cm[0]
            fn, tp = cm[1]
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        st.markdown(
            f'''<div class="matrix-grid">
                <div class="matrix-cell m-green"><div class="matrix-value">{tn}</div><div>True Negative</div></div>
                <div class="matrix-cell m-orange"><div class="matrix-value">{fp}</div><div>False Positive</div></div>
                <div class="matrix-cell m-red"><div class="matrix-value">{fn}</div><div>False Negative</div></div>
                <div class="matrix-cell m-teal"><div class="matrix-value">{tp}</div><div>True Positive</div></div>
            </div>''',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right2:
        if not fi_df.empty and {"feature", "importance"}.issubset(fi_df.columns):
            top_df = fi_df.sort_values("importance", ascending=False).head(5).copy()
            total = top_df["importance"].sum() or 1
            top_df["pct"] = top_df["importance"] / total * 100

            shap_blocks = []
            for _, row in top_df.iterrows():
                shap_blocks.append(
                    f'''<div class="shap-row">
                            <div class="shap-top">
                                <span>{row["feature"]}</span>
                                <span>{row["pct"]:.1f}%</span>
                            </div>
                            <div class="shap-bar">
                                <div class="shap-fill" style="width:{min(100, row["pct"]):.1f}%"></div>
                            </div>
                        </div>'''
                )

            shap_html = f'''
            <div class="card">
                <div class="subcard-title">SHAP Value Contributions</div>
                {"".join(shap_blocks)}
            </div>
            '''
            st.markdown(shap_html, unsafe_allow_html=True)
        else:
            st.markdown(
                '''
                <div class="card">
                    <div class="subcard-title">SHAP Value Contributions</div>
                    <div style="color:#6d8097;">Contribution bars require feature importance output.</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )

def render_prediction() -> None:
    st.markdown('<div class="section-title">Patient Conversion Prediction</div>', unsafe_allow_html=True)
    left, right = st.columns([1.05, 1.1], gap="large")

    predictor_candidates = [
        c for c in raw_df.columns
        if c not in DROP_COLS + EXCLUDE_COLS + [TARGET_COL]
    ] if not raw_df.empty else []

    pred = None
    prob = None
    risk_text = None
    risk_kind = None
    error_message = None

    with left:
        form_values: dict[str, Any] = {}

        if predictor_candidates:
            with st.form("patient_prediction_form", clear_on_submit=False):
                st.markdown('<div class="subcard-title">Patient Variables</div>', unsafe_allow_html=True)
                cols = st.columns(2, gap="large")

                for idx, col_name in enumerate(predictor_candidates):
                    series = raw_df[col_name].dropna() if col_name in raw_df.columns else pd.Series(dtype=float)
                    with cols[idx % 2]:
                        form_values[col_name] = render_patient_field(col_name, series)

                st.markdown('<div class="prediction-button-hook"></div>', unsafe_allow_html=True)
                predict_clicked = st.form_submit_button("↗  Generate Prediction")
        else:
            st.info("No raw dataset available. Prediction form needs the Mongo dataset or raw preview.")
            predict_clicked = False

    if predict_clicked and model is not None and feature_columns and form_values:
        try:
            X_input = prepare_prediction_input(form_values, raw_df, feature_columns)
            pred = int(model.predict(X_input)[0])

            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X_input)[0][1])
            elif hasattr(model, "decision_function"):
                score = float(model.decision_function(X_input)[0])
                prob = 1 / (1 + np.exp(-score))
            else:
                prob = 1.0 if pred == 1 else 0.0

            risk_text, risk_kind = risk_label(prob)

        except Exception as exc:
            error_message = f"Prediction could not be generated: {exc}"

    with right:
        if error_message:
            st.markdown('<div class="prediction-panel"><div class="subcard-title">Prediction Results</div></div>', unsafe_allow_html=True)
            st.error(error_message)

        elif pred is None or prob is None:
            st.markdown(
                '''<div class="prediction-panel">
                    <div class="subcard-title">Prediction Results</div>
                    <div class="prediction-placeholder">
                        Complete the patient variables and click <strong>Generate Prediction</strong> to display the estimated conversion risk.
                    </div>
                </div>''',
                unsafe_allow_html=True,
            )

        else:
            prob_pct = prob * 100
            pred_label = "Conversion" if pred == 1 else "Non-Conversion"
            st.markdown(
                f'''<div class="prediction-panel">
                    <div class="subcard-title">Prediction Results</div>
                    <div style="color:#61758c;font-size:1.0rem;">Predicted Class</div>
                    <div class="prediction-class">{pred_label}</div>
                    <div style="color:#61758c;font-size:1.0rem;">Conversion Probability</div>
                    <div class="prob-value">{prob_pct:.1f}%</div>
                    <div class="prob-track"><div class="prob-fill" style="width:{prob_pct:.1f}%"></div></div>
                    <div class="prediction-divider"></div>
                    <div style="color:#61758c;font-size:1.0rem;">Risk Level</div>
                    <div class="risk-badge risk-{risk_kind}">&#9679;&nbsp;&nbsp;{risk_text}</div>
                    <div class="disclaimer">&#9432;&nbsp;&nbsp;This prediction is for academic and decision-support purposes only and does not replace clinical judgment.</div>
                </div>''',
                unsafe_allow_html=True,
            )

def render_insights() -> None:
    st.markdown('<div class="section-title">Key Insights & Conclusions</div>', unsafe_allow_html=True)
    top_feats = top_predictor_names(fi_df, 3)
    best_model = str(kpis["best_model"] or "Best model")
    best_auc = f'{float(kpis["best_roc_auc"]):.2f}' if kpis["best_roc_auc"] is not None else "-"
    best_recall = f'{float(kpis["best_recall"]):.2f}' if kpis["best_recall"] is not None else "-"

    row1 = st.columns(2, gap="large")
    with row1[0]:
        st.markdown(
            f'''<div class="card insight-card"><div class="insight-icon bg-mint">🏅</div><div><div class="insight-head">Best Performing Model</div><div class="insight-body">{best_model} achieved the strongest overall discrimination with ROC-AUC of {best_auc} and recall of {best_recall}.</div></div></div>''',
            unsafe_allow_html=True,
        )
    with row1[1]:
        feat_text = ", ".join(top_feats) if top_feats else "Top predictors will appear after feature importance export."
        st.markdown(
            f'''<div class="card insight-card"><div class="insight-icon bg-blue">🧠</div><div><div class="insight-head">Most Influential Predictors</div><div class="insight-body">{feat_text}</div></div></div>''',
            unsafe_allow_html=True,
        )

    row2 = st.columns(2, gap="large")
    with row2[0]:
        st.markdown(
            '''<div class="card insight-card"><div class="insight-icon bg-peach">📊</div><div><div class="insight-head">Model Prioritization</div><div class="insight-body">Recall was prioritized to reduce false negatives in clinical decision-support scenarios, while maintaining competitive ROC-AUC and F1.</div></div></div>''',
            unsafe_allow_html=True,
        )
    with row2[1]:
        st.markdown(
            '''<div class="card insight-card"><div class="insight-icon bg-gray">❕</div><div><div class="insight-head">Overall Interpretation</div><div class="insight-body">The model shows promising discriminative performance on the available dataset. External validation on independent cohorts is still recommended before real clinical use.</div></div></div>''',
            unsafe_allow_html=True,
        )


if page == "Overview":
    render_overview()
elif page == "Dataset":
    render_dataset()
elif page == "Model Performance":
    render_model_performance()
elif page == "Explainability":
    render_explainability()
elif page == "Patient Prediction":
    render_prediction()
elif page == "Insights":
    render_insights()