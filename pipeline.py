import os
import pandas as pd
from pathlib import Path
from time import perf_counter
from pymongo import MongoClient


# ──────────────────────────────────────────────
# CARGA DE VARIABLES DE ENTORNO
# ──────────────────────────────────────────────

def load_env_file(path: Path | None = None) -> None:
    env_path = path or (Path(__file__).resolve().parent / ".env")
    if not env_path.exists():
        print(f"[AVISO] No se encontró .env en {env_path}")
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_required_env(var_name: str) -> str:
    value = os.getenv(var_name)
    if not value:
        raise ValueError(f"[ERROR] Variable de entorno requerida no definida: {var_name}")
    return value


load_env_file()

MONGO_URI = get_required_env("MONGO_URI")
DB_NAME   = os.getenv("MONGO_DB_NAME", "ms_data")
COL_RAW   = os.getenv("MONGO_COLLECTION_RAW", "cis_raw")
COL_MODEL = os.getenv("MONGO_COLLECTION_MODEL", "cis_model")
CSV_PATH  = os.getenv(
    "CSV_PATH",
    str(Path(__file__).resolve().parents[1] / "data" /
        "conversion_predictors_of_clinically_isolated_syndrome_to_multiple_sclerosis.csv")
)


# ──────────────────────────────────────────────
# ETAPA 1 — CARGAR CSV A cis_raw
# ──────────────────────────────────────────────

def cargar_csv(csv_path: str = CSV_PATH) -> int:
    print("\n" + "="*50)
    print("ETAPA 1 — Cargando CSV a MongoDB (cis_raw)")
    print("="*50)
    start = perf_counter()
    client = None

    try:
        # Verificar CSV
        if not Path(csv_path).exists():
            raise FileNotFoundError(
                f"No se encontró el CSV en: {csv_path}\n"
                "Descárguelo de Kaggle y colóquelo en la carpeta data/"
            )
        print(f"[OK] CSV encontrado: {csv_path}")

        # Leer CSV
        df = pd.read_csv(csv_path)
        print(f"[OK] CSV leído: {len(df)} filas, {len(df.columns)} columnas")
        print(f"     Columnas: {list(df.columns)}")

        # Insertar en MongoDB
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        coleccion = client[DB_NAME][COL_RAW]
        coleccion.delete_many({})
        print(f"[OK] Colección '{COL_RAW}' limpiada")

        resultado = coleccion.insert_many(df.to_dict(orient="records"))
        total = len(resultado.inserted_ids)
        print(f"[OK] {total} documentos insertados en '{COL_RAW}'")
        return total

    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        if client:
            client.close()
        print(f"[OK] Tiempo etapa 1: {perf_counter() - start:.3f}s")


# ──────────────────────────────────────────────
# ETAPA 2 — TRANSFORMAR Y CARGAR A cis_model
# ──────────────────────────────────────────────

def transformar_y_cargar() -> int:
    print("\n" + "="*50)
    print("ETAPA 2 — Transformando y cargando (cis_model_ready)")
    print("="*50)
    start = perf_counter()
    client = None

    try:
        # Leer desde cis_raw
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        datos = list(client[DB_NAME][COL_RAW].find({}, {"_id": 0}))

        if not datos:
            raise ValueError("cis_raw está vacía. Corra primero la Etapa 1.")

        df = pd.DataFrame(datos)
        print(f"[OK] Leídos {len(df)} documentos desde '{COL_RAW}'")

        # ── LIMPIEZA ──────────────────────────────────

        # 1. Normalizar tipos numéricos
        columnas_numericas = [
            "Unnamed: 0", "Gender", "Age", "Schooling", "Breastfeeding",
            "Varicella", "Initial_Symptom", "Mono_or_Polysymptomatic",
            "Oligoclonal_Bands", "LLSSEP", "ULSSEP", "VEP", "BAEP",
            "Periventricular_MRI", "Cortical_MRI", "Infratentorial_MRI",
            "Spinal_Cord_MRI", "Initial_EDSS", "Final_EDSS", "group"
        ]
        for col in columnas_numericas:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        print("[OK] Tipos normalizados a numérico")

        # 2. Eliminar duplicados
        duplicados = df.duplicated().sum()
        df.drop_duplicates(inplace=True)
        print(f"[OK] Duplicados eliminados: {duplicados}")

        # 3. Eliminar columna índice sin valor analítico
        if "Unnamed: 0" in df.columns:
            df.drop(columns=["Unnamed: 0"], inplace=True)
            print("[OK] Columna 'Unnamed: 0' eliminada")

        # 4. Eliminar columnas con leakage
        leakage = [c for c in ["Initial_EDSS", "Final_EDSS"] if c in df.columns]
        if leakage:
            df.drop(columns=leakage, inplace=True)
            print(f"[OK] Columnas leakage eliminadas: {leakage}")

        # 5. Imputar nulos en Schooling con mediana
        if "Schooling" in df.columns:
            nulos = df["Schooling"].isnull().sum()
            df["Schooling"] = df["Schooling"].fillna(df["Schooling"].median())
            print(f"[OK] Schooling: {nulos} nulos imputados con mediana")

        # 6. Imputar nulos en Initial_Symptom con -1
        if "Initial_Symptom" in df.columns:
            nulos = df["Initial_Symptom"].isnull().sum()
            df["Initial_Symptom"] = df["Initial_Symptom"].fillna(-1)
            print(f"[OK] Initial_Symptom: {nulos} nulos imputados con -1")

        # Reporte de nulos restantes
        nulos_restantes = df.isnull().sum()
        nulos_restantes = nulos_restantes[nulos_restantes > 0]
        if not nulos_restantes.empty:
            print(f"[AVISO] Nulos restantes:\n{nulos_restantes}")
        else:
            print("[OK] Sin nulos restantes")

        # ── FEATURE ENGINEERING ───────────────────────

        # 7. Conteo de lesiones MRI
        cols_mri = [c for c in
                    ["Periventricular_MRI", "Cortical_MRI",
                     "Infratentorial_MRI", "Spinal_Cord_MRI"]
                    if c in df.columns]
        if cols_mri:
            df["mri_lesion_count"]   = df[cols_mri].sum(axis=1)
            df["has_any_mri_lesion"] = (df["mri_lesion_count"] > 0).astype(int)
            print(f"[OK] Features MRI creadas: mri_lesion_count, has_any_mri_lesion")

        # 8. Conteo de pruebas evocadas positivas
        cols_evocados = [c for c in
                         ["LLSSEP", "ULSSEP", "VEP", "BAEP"]
                         if c in df.columns]
        if cols_evocados:
            df["evoked_positive_count"]   = df[cols_evocados].sum(axis=1)
            df["has_any_evoked_positive"] = (df["evoked_positive_count"] > 0).astype(int)
            print(f"[OK] Features evocados creadas: evoked_positive_count, has_any_evoked_positive")

        # 9. Grupos de edad con one-hot encoding
        if "Age" in df.columns:
            df["age_group"] = pd.cut(
                df["Age"],
                bins=[0, 24, 34, 44, 120],
                labels=["<=24", "25-34", "35-44", "45+"],
                include_lowest=True
            )
            df["age_group"] = df["age_group"].astype("object").fillna("Missing")
            df = pd.get_dummies(df, columns=["age_group"], drop_first=False, dtype=int)
            print("[OK] Feature age_group creada con one-hot encoding")

        print(f"\n[RESUMEN] Dataset final: {len(df)} filas, {len(df.columns)} columnas")
        print(f"     Columnas finales: {list(df.columns)}")

        # ── Guardar en cis_model_ready ────────────────
        coleccion_modelo = client[DB_NAME][COL_MODEL]
        coleccion_modelo.delete_many({})
        print(f"[OK] Colección '{COL_MODEL}' limpiada")

        resultado = coleccion_modelo.insert_many(df.to_dict(orient="records"))
        total = len(resultado.inserted_ids)
        print(f"[OK] {total} documentos insertados en '{COL_MODEL}'")
        return total

    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        if client:
            client.close()
        print(f"[OK] Tiempo etapa 2: {perf_counter() - start:.3f}s")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*50)
    print("PIPELINE ETL — Esclerosis Múltiple (CIS → CDMS)")
    print("="*50)
    start_total = perf_counter()

    total_crudos  = cargar_csv(CSV_PATH)
    total_modelo  = transformar_y_cargar()

    print("\n" + "="*50)
    print("PIPELINE COMPLETADO")
    print(f"  cis_raw:         {total_crudos} documentos")
    print(f"  cis_model: {total_modelo} documentos")
    print(f"  Tiempo total:    {perf_counter() - start_total:.3f}s")
    print("="*50)