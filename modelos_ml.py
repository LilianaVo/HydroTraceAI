# -*- coding: utf-8 -*-
"""
HydroTrace AI — Modelos de Machine Learning
Autor: Proyecto Universitario CDMX
Descripción: Aplica K-Means, Isolation Forest y Regresión Lineal
             al dataset_maestro.csv y exporta resultados_anomalias.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "dataset_maestro.csv")
OUTPUT_PATH = os.path.join("data", "resultados_anomalias.csv")

# Semilla para reproducibilidad
SEED = 42

# Features para los modelos no supervisados
FEATURES = ["consumo_per_capita", "densidad_poblacional", "total_reportes"]

# Umbral de anomalía Isolation Forest (cuanto más negativo, más anómalo)
CONTAMINACION = 0.20   # Se espera ~20 % de alcaldías con comportamiento irregular


# ─────────────────────────────────────────────
# CARGA Y VALIDACIÓN
# ─────────────────────────────────────────────
print("[1/5] Cargando dataset maestro...")
df = pd.read_csv(DATA_PATH, encoding="utf-8")

# Verificar que existan las columnas necesarias
for col in FEATURES:
    if col not in df.columns:
        raise ValueError(f"Columna faltante en dataset_maestro.csv: '{col}'")

print(f"  • Shape: {df.shape}")
print(f"  • Alcaldías: {df['alcaldia'].tolist()}")


# ─────────────────────────────────────────────
# PREPROCESAMIENTO — ESCALADO
# Se escalan las features para que los modelos
# no sean sensibles a diferencias de magnitud.
# ─────────────────────────────────────────────
print("\n[2/5] Escalando features...")

X = df[FEATURES].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ─────────────────────────────────────────────
# MODELO 1: K-MEANS CLUSTERING
# Agrupa alcaldías por comportamiento similar.
# Se usa k=3 para obtener grupos: normal,
# moderado y crítico.
# ─────────────────────────────────────────────
print("[3/5] Entrenando K-Means (k=3)...")

kmeans = KMeans(n_clusters=3, random_state=SEED, n_init="auto")
df["cluster_kmeans"] = kmeans.fit_predict(X_scaled)

# Etiquetar clusters en función del centroide de consumo_per_capita escalado
centroides = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=FEATURES
)
orden_consumo = centroides["consumo_per_capita"].rank().astype(int)
mapa_cluster = {
    idx: nombre
    for idx, nombre in zip(
        orden_consumo.index,
        ["Bajo Consumo", "Consumo Moderado", "Alto Consumo"][: len(orden_consumo)]
    )
}
df["cluster_nombre"] = df["cluster_kmeans"].map(mapa_cluster)

print(f"  • Distribución de clusters:\n{df['cluster_nombre'].value_counts()}")


# ─────────────────────────────────────────────
# MODELO 2: ISOLATION FOREST — DETECCIÓN DE ANOMALÍAS
# Detecta alcaldías donde el consumo per cápita
# es desproporcionado frente a su densidad
# poblacional (señal de extracción irregular).
# ─────────────────────────────────────────────
print("[4/5] Entrenando Isolation Forest...")

iso_forest = IsolationForest(
    contamination=CONTAMINACION,
    random_state=SEED,
    n_estimators=200,
)
df["anomalia_if"] = iso_forest.fit_predict(X_scaled)
# sklearn devuelve -1 (anómalo) o 1 (normal)
# Convertimos a flag legible
df["es_anomalia"] = df["anomalia_if"].map({-1: True, 1: False})

# Score de anomalía (más negativo = más sospechoso)
df["score_anomalia"] = iso_forest.decision_function(X_scaled).round(4)

n_anomalas = df["es_anomalia"].sum()
print(f"  • Alcaldías anómalas detectadas: {n_anomalas}/{len(df)}")


# ─────────────────────────────────────────────
# MODELO 3: REGRESIÓN LINEAL MÚLTIPLE
# Calcula el "Consumo Teórico Esperado" dado
# la densidad poblacional y total de reportes.
# La diferencia entre real y esperado es un
# indicador de fuga o extracción ilegal.
# ─────────────────────────────────────────────
print("[5/5] Entrenando Regresión Lineal (Consumo Teórico)...")

features_regresion = ["densidad_poblacional", "total_reportes"]
X_reg = df[features_regresion]
y_reg = df["consumo_per_capita"]

modelo_reg = LinearRegression()
modelo_reg.fit(X_reg, y_reg)

df["consumo_teorico_esperado"] = modelo_reg.predict(X_reg).round(4)
df["desviacion_consumo"] = (
    df["consumo_per_capita"] - df["consumo_teorico_esperado"]
).round(4)

mae = mean_absolute_error(y_reg, df["consumo_teorico_esperado"])
r2  = r2_score(y_reg, df["consumo_teorico_esperado"])
print(f"  • R²  del modelo: {r2:.4f}")
print(f"  • MAE del modelo: {mae:.4f}")

# Coeficientes del modelo para interpretabilidad
coef_df = pd.DataFrame(
    {"feature": features_regresion, "coeficiente": modelo_reg.coef_}
)
print(f"  • Coeficientes:\n{coef_df.to_string(index=False)}")


# ─────────────────────────────────────────────
# CLASIFICACIÓN DE RIESGO FINAL
# Combina los señales de Isolation Forest,
# la desviación del consumo y el cluster
# para asignar un nivel de riesgo por alcaldía.
# ─────────────────────────────────────────────

def calcular_riesgo(row: pd.Series) -> str:
    """
    Lógica de negocio para clasificar el riesgo de extracción ilegal:
      - Alto  : anomalía detectada Y desviación positiva alta
      - Medio : anomalía detectada O desviación alta sin anomalía
      - Bajo  : sin señales de alerta
    """
    desviacion_alta = row["desviacion_consumo"] > df["desviacion_consumo"].quantile(0.70)

    if row["es_anomalia"] and desviacion_alta:
        return "Alto"
    elif row["es_anomalia"] or desviacion_alta:
        return "Medio"
    else:
        return "Bajo"


df["Riesgo"] = df.apply(calcular_riesgo, axis=1)

print("\n  • Distribución de Riesgo:")
print(df["Riesgo"].value_counts().to_string())


# ─────────────────────────────────────────────
# EXPORTACIÓN DE RESULTADOS
# ─────────────────────────────────────────────
columnas_salida = [
    "alcaldia",
    "superficie_km2",
    "vocacion_principal",
    "poblacion",
    "consumo_total",
    "consumo_per_capita",
    "consumo_teorico_esperado",
    "desviacion_consumo",
    "densidad_poblacional",
    "total_reportes",
    "cluster_kmeans",
    "cluster_nombre",
    "es_anomalia",
    "score_anomalia",
    "Riesgo",
]

# Sólo incluir columnas que existan (robustez)
columnas_salida = [c for c in columnas_salida if c in df.columns]
df[columnas_salida].to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

print(f"\n✅ Resultados guardados en: {OUTPUT_PATH}")
print(df[["alcaldia", "Riesgo", "score_anomalia", "cluster_nombre"]].to_string(index=False))