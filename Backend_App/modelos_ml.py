"""
================================================================================
PROYECTO : HydroTrace AI - Ciencia de Datos UNAM
MODULO   : modelos_ml.py
AUTORES  : Ileana Lee & Irving Morales
OBJETIVO : Segmentación, detección de anomalías y regresión para diagnóstico
           de fugas y huachicol de agua a nivel colonia en la CDMX.

INSTRUCCIONES PARA IRVING:
  - Completar los bloques marcados con 'TODO'.
  - Calibrar parámetros de modelos basados en la investigación de SEGUIAGUA.
  - Generar métricas de validación para el reporte financiero.
================================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH = "dataset_maestro_colonia_final.csv"
OUTPUT_PATH = "resultados_finales_IA.csv"

# Configuración de estilo para que las gráficas no se vean "equis"
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# ──────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y ESCALAMIENTO (YA LISTO)
# ──────────────────────────────────────────────────────────────────────────────

def cargar_y_preparar():
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: No se encontró {DATA_PATH}. Ileana, ¿corriste el ETL?")
        return None, None
    
    df = pd.read_csv(DATA_PATH)
    
    # Variables críticas para el perfilamiento de la colonia
    features = ['consumo_per_capita', 'densidad_poblacional', 'uso_suelo_num', 'idsm']
    
    # Limpieza de seguridad
    df_model = df.dropna(subset=features).copy()
    
    # TODO @Irving: Explica por qué es obligatorio usar StandardScaler en K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[features])
    
    return df_model, X_scaled

# ──────────────────────────────────────────────────────────────────────────────
# 2. SEGMENTACIÓN (K-MEANS)
# ──────────────────────────────────────────────────────────────────────────────

def ejecutar_clustering(df, X_scaled):
    print("\n[IA] Iniciando Clustering...")
    
    # TODO @Irving: Implementar el 'Método del Codo' (WCSS) para 
    # encontrar el número óptimo de clusters (K). 
    # Tip: Prueba un rango de 1 a 10 y grafica la inercia.
    
    # --- ESPACIO PARA TU CÓDIGO DEL CODO ---
    
    # Aplicamos K-Means (Ajusta n_clusters según tu análisis del codo)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster_perfil'] = kmeans.fit_predict(X_scaled)
    
    print("Segmentación completada.")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 3. DETECCIÓN DE ANOMALÍAS (ISOLATION FOREST)
# ──────────────────────────────────────────────────────────────────────────────

def detectar_anomalias(df, X_scaled):
    print("[IA] Buscando anomalías en el consumo...")
    
    # TODO @Irving: Investiga el % de pérdida de agua reportado por SEGUIAGUA
    # y ajusta el parámetro 'contamination' en consecuencia.
    contamination_val = 0.15 
    
    iso_forest = IsolationForest(contamination=contamination_val, random_state=42)
    df['es_anomalia'] = iso_forest.fit_predict(X_scaled) # -1 = Raro, 1 = Normal
    
    # TODO @Irving: Genera un Scatter Plot (consumo vs densidad) 
    # resaltando las anomalías en un color distinto (ej. Rojo).
    
    print("Análisis de anomalías finalizado.")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 4. REGRESIÓN: EL CONSUMO TEÓRICO
# ──────────────────────────────────────────────────────────────────────────────

def estimar_consumo_base(df):
    print("[IA] Calculando consumo esperado mediante Regresión...")
    
    # Variables que deberían explicar el consumo "normal"
    X = df[['pob', 'uso_suelo_num', 'superficie_km2_calculada']]
    y = df['consumo_total']
    
    model = LinearRegression()
    model.fit(X, y)
    
    df['consumo_esperado'] = model.predict(X)
    
    # TODO @Irving: Calcula R2 y MAE. 
    # Estos valores son clave para que Ana (Finanzas) sepa qué tan confiable es el ROI.
    
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 5. DIAGNÓSTICO INTEGRADO (LÓGICA DE NEGOCIO)
# ──────────────────────────────────────────────────────────────────────────────

def clasificar_riesgo(row):
    """
    Combina los 3 modelos para dar un diagnóstico humano.
    Ileana: Esta parte define nuestra ventaja competitiva.
    """
    diff = row['consumo_total'] - row['consumo_esperado']
    
    # TODO @Irving & Ileana: Refinar los umbrales de 'exceso' y 'reportes'
    if row['es_anomalia'] == -1 and diff > 0 and row['total_reportes'] > 15:
        return "CRÍTICO (Posible Fuga)"
    
    if row['es_anomalia'] == -1 and diff > 0 and row['total_reportes'] <= 2:
        return "SOSPECHOSO (Posible Huachicol)"
        
    return "NORMAL"

# ──────────────────────────────────────────────────────────────────────────────
# 6. EJECUCIÓN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df, X_s = cargar_y_preparar()
    if df is not None:
        df = ejecutar_clustering(df, X_s)
        df = detectar_anomalias(df, X_s)
        df = estimar_consumo_base(df)
        
        print("[IA] Generando etiquetas finales...")
        df['diagnostico_final'] = df.apply(clasificar_riesgo, axis=1)
        
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"Proceso terminado. Resultados en {OUTPUT_PATH}")