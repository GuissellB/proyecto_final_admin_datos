# Proyecto Final de Administracion de Datos

Este proyecto implementa un pipeline ETL en Python para trabajar con un dataset clinico sobre la conversion de **CIS (Clinically Isolated Syndrome)** a **esclerosis multiple**. El proceso toma un archivo CSV, lo carga en MongoDB y genera una version limpia y transformada para analisis o modelado.

## Que hace el proyecto

El flujo principal esta en [`pipeline.py`](/d:/Repositorios/proyecto_final_admin_datos/pipeline.py) y realiza dos etapas:

1. Carga el dataset original en una coleccion cruda de MongoDB (`cis_raw`).
2. Limpia y transforma los datos para dejarlos listos en una coleccion final (`cis_model`).

Durante la transformacion se aplican tareas como:

- conversion de columnas a formato numerico
- eliminacion de duplicados
- eliminacion de la columna `Unnamed: 0`
- exclusion de `Initial_EDSS` y `Final_EDSS` por leakage
- imputacion de valores faltantes
- creacion de variables derivadas de grupos de edad

## Dataset

El dataset usado en el proyecto proviene de Kaggle:

- fuente: `desalegngeb/conversion-predictors-of-cis-to-multiple-sclerosis`
- archivo original: `conversion_predictors_of_clinically_isolated_syndrome_to_multiple_sclerosis.csv`
- enlace: `https://www.kaggle.com/datasets/desalegngeb/conversion-predictors-of-cis-to-multiple-sclerosis`

En este repositorio el archivo se encuentra como:

`data/ms_dataset.csv`

## Como ejecutarlo

1. Instala las dependencias:

```powershell
pip install -r requirements.txt
```

2. Configura un archivo `.env` en la raiz con valores como estos:

```env
MONGO_URI=<tu_uri_de_mongodb>
MONGO_DB_NAME=ms_data
MONGO_COLLECTION_RAW=cis_raw
MONGO_COLLECTION_MODEL=cis_model
CSV_PATH=data/ms_dataset.csv
```

3. Ejecuta el pipeline:

```powershell
python pipeline.py
```

Si usas entorno virtual en Windows:

```powershell
.venv\Scripts\python.exe pipeline.py
```

## Resultado

Al ejecutar el pipeline se generan dos colecciones en MongoDB:

- `cis_raw`: datos originales cargados desde el CSV
- `cis_model`: datos limpios y preparados para etapas posteriores

El script tambien muestra en consola un resumen con la cantidad de registros procesados y el tiempo de ejecucion.

## Archivos de apoyo

- [`report.md`](/d:/Repositorios/proyecto_final_admin_datos/report.md): explica los hallazgos del analisis exploratorio y las decisiones del pipeline
- [`notebooks/ms_eda.ipynb`](/d:/Repositorios/proyecto_final_admin_datos/notebooks/ms_eda.ipynb): notebook de exploracion del dataset
