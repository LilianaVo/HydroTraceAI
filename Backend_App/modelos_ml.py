"""
PROYECTO: HydroTrace AI - Módulo de Machine Learning
RESPONSABLE: Irving Morales e Ileana
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Importamos las utilidades de validación del main (pm-support)
from main import grafica_metodo_codo, visualizar_clusters_2d, reporte_desempeno_negocio

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dataset_maestro.csv"))
OUTPUT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "resultados_anomalias.csv"))

def ejecutar_pipeline_ml():
    print("[IA] Iniciando proceso de modelado...")
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: No existe el dataset maestro en {DATA_PATH}. Bolívar, ¡muévete!")
        return

    df = pd.read_csv(DATA_PATH)

    # 1. SELECCIÓN Y ESCALAMIENTO (YA HECHO AL 100%)
    features = ['consumo_per_capita', 'densidad_poblacional', 'total_reportes']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2. CLUSTERING K-MEANS (AL 70%)
    # Irving: Yo ya te puse el modelo base. Tú justifica el 'n_clusters' usando la gráfica del codo.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
    
    # Llamamos a tu apoyo visual
    grafica_metodo_codo(X_scaled)
    visualizar_clusters_2d(X_scaled, df['cluster_kmeans'])

    # 3. DETECCIÓN DE ANOMALÍAS (AL 70%)
    # Irving: Ajusta 'contamination' basado en lo que investigaste de SACMEX (¿qué % es huachicol?).
    iso = IsolationForest(contamination=0.2, random_state=42)
    df['es_anomalia'] = iso.fit_predict(X_scaled) # -1 es anomalía

    # 4. REGRESIÓN LINEAL (AL 70%)
    # Objetivo: Predecir el consumo 'normal'. Irving: Explica los coeficientes en el doc.
    reg = LinearRegression()
    X_reg = df[['poblacion', 'densidad_poblacional']]
    y_reg = df['consumo_total']
    
    reg.fit(X_reg, y_reg)
    df['consumo_teorico_esperado'] = reg.predict(X_reg)
    df['desviacion_consumo'] = df['consumo_total'] - df['consumo_teorico_esperado']

    # 5. LÓGICA DE RIESGO (YA HECHO PARA EL DASHBOARD)
    def clasificar_riesgo(row):
        if row['es_anomalia'] == -1 and row['desviacion_consumo'] > 0:
            return "Alto"
        elif row['es_anomalia'] == -1 or row['desviacion_consumo'] > 500:
            return "Medio"
        return "Bajo"

    df['riesgo'] = df.apply(clasificar_riesgo, axis=1)

    # 6. REPORTE DE DESEMPEÑO (LA VALIDACIÓN ACADÉMICA)
    reporte_desempeno_negocio(df)

    # GUARDAR RESULTADOS
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Modelos ejecutados. Resultados en {OUTPUT_PATH}")

if __name__ == "__main__":
    ejecutar_pipeline_ml()