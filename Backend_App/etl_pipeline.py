# -*- coding: utf-8 -*-
"""
HydroTrace AI - ETL Pipeline
Autor: Proyecto Universitario CDMX
Descripcion: Carga, limpia y une los 4 datasets fuente,
             calcula variables derivadas y genera dataset_maestro.csv
"""

import os
import unicodedata
import pandas as pd


# ---------------------------------------------
# CONFIGURACION DE RUTAS
# ---------------------------------------------
DATA_DIR   = "data"
OUTPUT_DIR = "data"

RUTAS = {
    "consumo"    : os.path.join(DATA_DIR, "consumo_agua_historico_2019.csv"),
    "demografico": os.path.join(DATA_DIR, "c_demograficas_total_alcaldia.csv"),
    "reportes"   : os.path.join(DATA_DIR, "reportes_agua_hist.csv"),
    "superficie" : os.path.join(DATA_DIR, "superficie_alcaldias.csv"),
}

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "dataset_maestro.csv")


# ---------------------------------------------
# FUNCION DE NORMALIZACION DE ALCALDIAS
# Elimina acentos, pone mayusculas y
# quita espacios extra para un merge limpio.
# ---------------------------------------------
def normalizar_alcaldia(serie):
    """Normaliza la columna alcaldia: mayusculas, sin acentos, sin espacios."""
    def limpiar(texto):
        if not isinstance(texto, str):
            return texto
        texto = unicodedata.normalize("NFD", texto)
        texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
        return texto.upper().strip()
    return serie.apply(limpiar)


# ---------------------------------------------
# CARGA DE DATOS
# ---------------------------------------------
print("[1/5] Cargando archivos fuente...")

df_consumo     = pd.read_csv(RUTAS["consumo"],     encoding="utf-8")
df_demografico = pd.read_csv(RUTAS["demografico"], encoding="utf-8")
df_reportes    = pd.read_csv(RUTAS["reportes"],    encoding="utf-8")
df_superficie  = pd.read_csv(RUTAS["superficie"],  encoding="utf-8")

print("  - consumo      : " + str(df_consumo.shape))
print("  - demografico  : " + str(df_demografico.shape))
print("  - reportes     : " + str(df_reportes.shape))
print("  - superficie   : " + str(df_superficie.shape))


# ---------------------------------------------
# LIMPIEZA DE COLUMNA alcaldia EN CADA DF
# ---------------------------------------------
print("\n[2/5] Normalizando columna alcaldia...")

for df in [df_consumo, df_demografico, df_reportes, df_superficie]:
    df["alcaldia"] = normalizar_alcaldia(df["alcaldia"])


# ---------------------------------------------
# AGRUPACION DE REPORTES POR ALCALDIA
# Se suman todas las quejas/reportes numericos.
# ---------------------------------------------
print("[3/5] Agrupando reportes por alcaldia...")

# ESTA ES LA LÍNEA MÁGICA (Solo cuenta las filas, no suma textos)
df_reportes_agg = df_reportes.groupby("alcaldia").size().reset_index(name="total_reportes")

# ESTE SE QUEDA EXACTAMENTE IGUAL (Porque el consumo sí son puros números)
df_consumo_agg = (
    df_consumo
    .groupby("alcaldia")["consumo_total"]
    .sum()
    .reset_index()
)

# ---------------------------------------------
# MERGE PROGRESIVO (LEFT JOIN sobre superficie)
# superficie -> demografico -> consumo -> reportes
# ---------------------------------------------
print("[4/5] Uniendo datasets...")

df_maestro = df_superficie.copy()
df_maestro = df_maestro.merge(df_demografico,  on="alcaldia", how="left")
df_maestro = df_maestro.merge(df_consumo_agg,  on="alcaldia", how="left")
df_maestro = df_maestro.merge(df_reportes_agg, on="alcaldia", how="left")

print("  - Shape (antes de KPIs): " + str(df_maestro.shape))


# ---------------------------------------------
# CALCULO DE VARIABLES DERIVADAS
# consumo_per_capita   : consumo_total / poblacion
# densidad_poblacional : poblacion / superficie_km2
# ---------------------------------------------
print("[5/5] Calculando KPIs derivados...")

df_maestro["consumo_per_capita"] = (
    df_maestro["consumo_total"] / df_maestro["poblacion"]
).round(4)

df_maestro["densidad_poblacional"] = (
    df_maestro["poblacion"] / df_maestro["superficie_km2"]
).round(4)

cols_num = df_maestro.select_dtypes(include="number").columns
df_maestro[cols_num] = df_maestro[cols_num].fillna(0)


# ---------------------------------------------
# EXPORTACION
# ---------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
df_maestro.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print("\n[OK] Dataset maestro guardado en: " + OUTPUT_PATH)
print("     Columnas: " + str(list(df_maestro.columns)))
print("     Shape   : " + str(df_maestro.shape))
print(df_maestro.head().to_string())