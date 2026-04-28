"""
PROYECTO: HydroTrace AI - Ciencia de Datos UNAM
DESCRIPCIÓN: Implementación de algoritmos para detección de anomalías y predicción.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Configuración de rutas
DATA_PATH = os.path.join("data", "dataset_maestro.csv")
OUTPUT_PATH = os.path.join("data", "resultados_anomalias.csv")

def cargar_datos():
    """Carga el dataset maestro generado por el ETL."""
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encontró el archivo en {DATA_PATH}")
        return None
    return pd.read_csv(DATA_PATH)

# =============================================================================
# IRVING: IMPLEMENTAR LA LÓGICA MATEMÁTICA
# =============================================================================

def aplicar_clustering(df):
    """
    TAREA 1: K-Means (Unidad 5 - Aprendizaje No Supervisado)
    Objetivo: Agrupar alcaldías por perfiles de consumo y densidad.
    """
    print("[ML] Iniciando K-Means...")
    
    # TODO: Irving, selecciona las features adecuadas (ej: 'consumo_per_capita', 'densidad_poblacional')
    # features = ...
    
    # TODO: Escalar los datos con StandardScaler()
    
    # TODO: Configurar y entrenar el modelo KMeans
    # kmeans = KMeans(n_clusters=3, random_state=42)
    
    # Retorna el dataframe con la columna 'cluster' agregada
    return df

def detectar_anomalias(df):
    """
    TAREA 2: Isolation Forest (Detección de Outliers)
    Objetivo: Identificar zonas con consumos 'extraños' o 'atípicos'.
    """
    print("[ML] Iniciando Isolation Forest...")
    
    # TODO: Configurar Isolation Forest. 
    # TIP: Ajusta el parámetro 'contamination' para controlar la sensibilidad.
    # iso_forest = IsolationForest(contamination=0.2, random_state=42)
    
    # TODO: Entrenar y predecir. Guarda los resultados en 'es_anomalia' y 'score_anomalia'.
    
    return df

def predecir_consumo_esperado(df):
    """
    TAREA 3: Regresión Lineal (Unidad 4 - Modelos Supervisados)
    Objetivo: Calcular cuánto DEBERÍA consumir una zona según su población.
    """
    print("[ML] Iniciando Regresión Lineal...")
    
    # TODO: Definir X (independientes: poblacion, densidad) y y (dependiente: consumo_total)
    
    # TODO: Entrenar el modelo LinearRegression()
    
    # TODO: Calcular 'consumo_teorico_esperado' y la 'desviacion_consumo' (Real - Esperado)
    
    return df

def ejecutar_pipeline_ml():
    """Director de orquesta del modelado."""
    df = cargar_datos()
    if df is None: return

    # Ejecución secuencial de los modelos
    df = aplicar_clustering(df)
    df = detectar_anomalias(df)
    df = predecir_consumo_esperado(df)

    # --- LÓGICA DE CLASIFICACIÓN DE RIESGO (ILEANA) ---
    # Para que eldashboard funcione de una
    def asignar_riesgo(row):
        # Un ejemplo simple de lógica de decisión
        if row.get('es_anomalia', 0) == -1 and row.get('desviacion_consumo', 0) > 0:
            return "Alto"
        elif row.get('es_anomalia', 0) == -1 or row.get('desviacion_consumo', 0) > 1000:
            return "Medio"
        return "Bajo"

    if 'es_anomalia' in df.columns:
        df['Riesgo'] = df.apply(asignar_riesgo, axis=1)
    
    # Exportación final
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[ÉXITO] Resultados exportados a {OUTPUT_PATH}")

if __name__ == "__main__":
    ejecutar_pipeline_ml()