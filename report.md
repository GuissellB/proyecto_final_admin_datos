# Reporte de Exploracion y Orquestacion: CIS a Esclerosis Multiple

## Objetivo
Este reporte resume los hallazgos del analisis exploratorio realizado en [ms_eda.ipynb](/d:/Repositorios/proyecto_final_admin_datos/notebooks/ms_eda.ipynb).

El foco de esta etapa fue entender el dataset, revisar su calidad, detectar variables relevantes o problematicas y documentar como esos hallazgos se reflejaron en [pipeline.py](/d:/Repositorios/proyecto_final_admin_datos/pipeline.py).

## Dataset
- Archivo analizado: `data/conversion_predictors_of_clinically_isolated_syndrome_to_multiple_sclerosis.csv`
- Observaciones: `273`
- Columnas originales: `20`
- Target original: `group`
  - `1 = CDMS`
  - `2 = non-CDMS`

## Hallazgos de calidad
- `Unnamed: 0` parece ser solo un indice tecnico.
- `Initial_EDSS` y `Final_EDSS` tienen muchos faltantes.
- `Schooling` e `Initial_Symptom` tienen pocos faltantes reales.
- `Breastfeeding`, `Varicella`, `Mono_or_Polysymptomatic` y `Oligoclonal_Bands` incluyen categoria `unknown`.

Estas categorias `unknown` no representan un estado clinico real, sino datos no disponibles.

## Variables que llamaron la atencion
Las columnas que merecieron mas revision fueron:

- `Initial_EDSS`
- `Final_EDSS`
- `Schooling`
- `Initial_Symptom`
- `Breastfeeding`
- `Varicella`
- `Mono_or_Polysymptomatic`
- `Oligoclonal_Bands`
- variables MRI

Los motivos fueron faltantes, presencia de `unknown`, posible relacion con el target y relevancia clinica.

## Relacion con el target
La relacion con `group` se reviso con:

- tablas de proporcion para variables categoricas
- correlacion numerica para variables numericas o binarias
- mutual information como medida complementaria

Los resultados sugieren que:

- las variables MRI aportan señal importante
- `Age` tambien parece relevante
- `Oligoclonal_Bands` merece atencion por su relacion con `group`

## Leakage
El hallazgo mas importante fue el comportamiento de:

- `Initial_EDSS`
- `Final_EDSS`

Estas columnas tienen muchos faltantes y, ademas, el patron de faltantes depende demasiado del target. Esto genera **leakage**, es decir, una pista artificial que podria hacer que el modelo aprenda el resultado de forma engañosa.

Conclusion:

- pueden comentarse en el EDA
- no deberian usarse para modelado

## Transformaciones implementadas en el pipeline
Con base en los hallazgos del EDA, el archivo [pipeline.py](/d:/Repositorios/proyecto_final_admin_datos/pipeline.py) ya implementa estas decisiones:

- lectura del CSV y carga cruda en `cis_raw`
- normalizacion de tipos numericos
- eliminacion de duplicados exactos
- eliminacion de `Unnamed: 0`
- exclusion de `Initial_EDSS` y `Final_EDSS` por leakage
- imputacion de `Schooling` con mediana
- imputacion de `Initial_Symptom` con `-1`
- creacion de `age_group` con one-hot encoding
- carga final en `cis_model`

## Feature engineering aplicado / propuesto
La variable derivada que se mantiene en el pipeline es:

- `age_group`

Esta variable resume la edad en grupos y se expande con one-hot encoding para facilitar etapas posteriores de modelado.

En la version actual del pipeline, las columnas derivadas de `age_group` quedan con nombres compatibles para modelado:

- `age_group_24_or_less`
- `age_group_25_to_34`
- `age_group_35_to_44`
- `age_group_45_or_more`

## Decisiones metodologicas
Con base en este EDA y en la orquestacion actual, las decisiones principales son:

- eliminar `Unnamed: 0`
- conservar `group` como target
- excluir `Initial_EDSS` y `Final_EDSS` del dataset de modelo
- imputar `Schooling` e `Initial_Symptom` si se requiere
- mantener por ahora la codificacion numerica original, incluyendo `unknown`
- crear features simples de edad

## Cierre
Este trabajo permitio entender el dataset, identificar problemas de calidad, detectar leakage y traducir esos hallazgos en un pipeline que deja una coleccion `cis_model` preparada para etapas posteriores de modelado.
