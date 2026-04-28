"""
PROYECTO: HydroTrace AI - Ciencia de Datos UNAM
DESCRIPCIÓN: Implementación y validación de modelos predictivos y de detección.
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
from sklearn.metrics import (
    mean_absolute_error, r2_score, confusion_matrix, 
    silhouette_score, classification_report
)
from ml_utils import grafica_metodo_codo, visualizar_clusters_2d, reporte_desempeno_negocio

# Configuración de rutas
DATA_PATH = os.path.join("data", "dataset_maestro.csv")
OUTPUT_PATH = os.path.join("data", "resultados_anomalias.csv")

def cargar_datos():
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encontró el dataset en {DATA_PATH}")
        return None
    return pd.read_csv(DATA_PATH)

# =============================================================================
# BLOQUE 1: MODELADO (TRABAJO DE IRVING)
# =============================================================================

def aplicar_clustering(df):
    """K-Means: Agrupamiento de alcaldías por perfil socio-hídrico."""
    print("[ML] Entrenando K-Means...")
    
    # TODO: selecciona las columnas para el agrupamiento
    features = ['consumo_per_capita', 'densidad_poblacional']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # TODO: justifica el número de clusters (¿Por qué 3?)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster_kmeans'] = kmeans.fit_predict(X_scaled)
    
    return df, X_scaled

def detectar_anomalias(df):
    """Isolation Forest: Detección de posibles fugas o huachicol."""
    print("[ML] Entrenando Isolation Forest...")
    
    # TODO: justifica el parámetro 'contamination' basado en el análisis de Bolívar
    iso = IsolationForest(contamination=0.2, random_state=42)
    
    # -1 es anomalía, 1 es normal
    df['es_anomalia'] = iso.fit_predict(df[['consumo_per_capita', 'total_reportes']])
    return df

# =============================================================================
# BLOQUE 2: MARCO DE VALIDACIÓN (APOYO DE ILEANA)
# =============================================================================

def validar_modelo_academico(df, X_scaled):
    """
    FUNCIONES DE VALIDACIÓN: Esto es lo que pide el profe.
    Irving debe interpretar estos resultados en el PDF.
    """
    print("\n" + "="*30)
    print(" REPORTE DE VALIDACIÓN TÉCNICA ")
    print("="*30)

    # 1. Validación de Clustering (Silhouette Score)
    score_s = silhouette_score(X_scaled, df['cluster_kmeans'])
    print(f"• Silhouette Score (K-Means): {score_s:.4f}")
    # Nota para Irving: Cerca de 1 es perfecto, cerca de 0 es traslape.

    # 2. Simulación de Matriz de Confusión
    # Dado que es aprendizaje no supervisado, comparamos con un 'Ground Truth' 
    # basado en reportes históricos para ver si la IA detecta lo que ya sabemos.
    y_true = (df['total_reportes'] > df['total_reportes'].median()).astype(int)
    y_pred = (df['es_anomalia'] == -1).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Visualización de la Matriz (Para el documento)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomalía'], 
                yticklabels=['Real Sano', 'Real Fuga'])
    plt.title('Matriz de Confusión: Detección de Anomalías')
    plt.savefig('graficas_reporte/matriz_confusion_ia.png')
    print("• Matriz de Confusión generada en 'graficas_reporte/'")
    
    print("\n• Reporte de Clasificación:")
    print(classification_report(y_true, y_pred))

def ejecutar_pipeline_ml():
    df = cargar_datos()
    if df is None: return

    # Correr modelos
    df, X_scaled = aplicar_clustering(df)
    df = detectar_anomalias(df)
    
    # --- LÓGICA DE RIESGO ---
    df['Riesgo'] = np.where(df['es_anomalia'] == -1, "Alto", "Bajo")

    # Correr validaciones
    validar_modelo_academico(df, X_scaled)

    # Exportar
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Resultados finales listos para Ileana en: {OUTPUT_PATH}")

if __name__ == "__main__":
    # Crear carpeta de gráficas si no existe
    if not os.path.exists('graficas_reporte'): os.makedirs('graficas_reporte')
    ejecutar_pipeline_ml()