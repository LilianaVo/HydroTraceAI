"""
================================================================================
PROYECTO : HydroTrace AI - Ciencia de Datos UNAM
MODULO   : modelos_ml.py
AUTORES  : Ileana Lee (Lead Scientist) & Irving Morales
OBJETIVO : Segmentación, detección de anomalías y regresión para diagnóstico
           de fugas y huachicol de agua a nivel colonia en la CDMX.

NOTAS TÉCNICAS (PARA IRVING):
  - El backend de matplotlib está en 'Agg' para compatibilidad con servidores.
  - Los umbrales de diagnóstico se calculan dinámicamente con percentiles.
  - TODO: Completar las métricas de validación y análisis de clusters.
================================================================================
"""

import os
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Backend seguro para ejecuciones en nube/servidor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH = "dataset_maestro_colonia_final.csv"
OUTPUT_PATH = "resultados_finales_IA.csv"

# ──────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y CALIBRACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def cargar_y_preparar():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: No se encontró {DATA_PATH}. Ileana, ¿corriste el ETL?")
        return None, None
    
    df = pd.read_csv(DATA_PATH)
    features = ['consumo_per_capita', 'densidad_poblacional', 'uso_suelo_num', 'idsm']
    df_model = df.dropna(subset=features).copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[features])
    
    return df_model, X_scaled

def calibrar_umbrales(df):
    """
    TODO @Irving: Explica por qué usamos percentiles (75 y 10) 
    en lugar de valores fijos como '15 reportes'.
    """
    umbrales = {
        'alto_reporte': np.percentile(df['total_reportes'], 75),  # Top 25% de quejas
        'bajo_reporte': np.percentile(df['total_reportes'], 10)   # 10% que menos se queja
    }
    return umbrales

# ──────────────────────────────────────────────────────────────────────────────
# 2. SEGMENTACIÓN (K-MEANS)
# ──────────────────────────────────────────────────────────────────────────────

def ejecutar_clustering(df, X_scaled):
    print("\n[IA] Iniciando Clustering...")
    
    # TODO @Irving: Implementar el 'Método del Codo' y guardar la gráfica 
    # como 'metodo_codo.png'. Justifica el número de clusters.
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster_perfil'] = kmeans.fit_predict(X_scaled)
    
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 3. DETECCIÓN DE ANOMALÍAS (ISOLATION FOREST)
# ──────────────────────────────────────────────────────────────────────────────

def detectar_anomalias(df, X_scaled):
    print("[IA] Analizando anomalías con Isolation Forest...")
    
    # TODO @Irving: Ajusta 'contamination' basado en pérdidas de SEGUIAGUA.
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    df['es_anomalia'] = iso_forest.fit_predict(X_scaled) 
    
    # TODO @Irving: Genera el scatter plot 'mapa_anomalias_analitico.png'.
    
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 4. REGRESIÓN: CONSUMO ESPERADO
# ──────────────────────────────────────────────────────────────────────────────

def estimar_consumo_base(df):
    print("[IA] Entrenando Regresión Múltiple...")
    X = df[['pob', 'uso_suelo_num', 'superficie_km2_calculada']]
    y = df['consumo_total']
    
    model = LinearRegression()
    model.fit(X, y)
    df['consumo_esperado'] = model.predict(X)
    
    # TODO @Irving: Imprimir R2 y MAE. 
    # Analiza por qué el R2 es bajo (0.14) y si es aceptable para el proyecto.
    
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 5. DIAGNÓSTICO INTEGRADO (LÓGICA SEGUIAGUA)
# ──────────────────────────────────────────────────────────────────────────────

def clasificar_riesgo(row, umbrales):
    exceso = row['consumo_total'] - row['consumo_esperado']
    
    # CRÍTICO: Anomalía + Exceso + Reportes arriba del percentil 75
    if row['es_anomalia'] == -1 and exceso > 0 and row['total_reportes'] >= umbrales['alto_reporte']:
        return "CRÍTICO (Posible Fuga de Red)"
    
    # SOSPECHOSO: Anomalía + Exceso + Reportes abajo del percentil 10 (Nadie avisa)
    if row['es_anomalia'] == -1 and exceso > 0 and row['total_reportes'] <= umbrales['bajo_reporte']:
        return "SOSPECHOSO (Posible Huachicol)"
        
    return "NORMAL"

# ──────────────────────────────────────────────────────────────────────────────
# 6. EJECUCIÓN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_final, X_s = cargar_y_preparar()
    if df_final is not None:
        u = calibrar_umbrales(df_final)
        df_final = ejecutar_clustering(df_final, X_s)
        df_final = detectar_anomalias(df_final, X_s)
        df_final = estimar_consumo_base(df_final)
        
        print("[IA] Generando etiquetas de diagnóstico...")
        df_final['diagnostico_final'] = df_final.apply(lambda r: clasificar_riesgo(r, u), axis=1)
        
        df_final.to_csv(OUTPUT_PATH, index=False)
        print(f"🚀 Pipeline completado. Resultados en {OUTPUT_PATH}")