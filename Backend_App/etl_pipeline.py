"""
PROYECTO: HydroTrace AI - Ciencia de Datos UNAM
DESCRIPCIÓN: Pipeline de extracción, transformación y carga (ETL).
             Une los 4 datasets fuente y genera el 'dataset_maestro.csv'.
             Incluye auditoría de veracidad y limpieza de indeterminaciones.
"""

import os
import unicodedata
import pandas as pd
import numpy as np

# ---------------------------------------------
# CONFIGURACIÓN DE RUTAS Y ENTORNO
# ---------------------------------------------
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

RUTAS = {
    "consumo": os.path.join(DATA_DIR, "consumo_agua_historico_2019.csv"),
    "demografico": os.path.join(DATA_DIR, "c_demograficas_total_alcaldia.csv"),
    "reportes": os.path.join(DATA_DIR, "reportes_agua_hist.csv"),
    "superficie": os.path.join(DATA_DIR, "superficie_alcaldias.csv"),
}
OUTPUT_PATH = os.path.join(DATA_DIR, "dataset_maestro.csv")

# ---------------------------------------------
# AUDITORÍA Y CALIDAD (Data Profiling) 
# ---------------------------------------------

def auditar_datos(df, nombre_dataset, columnas_criticas=None):
    #Audita los datos buscando nulos, negativos y ceros sospechosos.
    print(f"\n[AUDITORÍA] Analizando dataset: {nombre_dataset}...")
    
    # 1. Verificar Nulos
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        print(f"  ⚠ ADVERTENCIA: Se detectaron {nulos} valores nulos.")
    
    # 2. Verificar Negativos (Veracidad) 
    cols_numericas = df.select_dtypes(include=['number']).columns
    for col in cols_numericas:
        negativos = (df[col] < 0).sum()
        if negativos > 0:
            raise ValueError(f"  X ERROR CRÍTICO: La columna '{col}' tiene {negativos} valores negativos.")
    
    # 3. Detector de "Ceros Sospechosos" (Específico para el Merge) 
    if columnas_criticas:
        for col in columnas_criticas:
            if col in df.columns:
                ceros = (df[col] == 0).sum()
                if ceros > 0:
                    alcaldias_afectadas = df[df[col] == 0]['alcaldia'].tolist()
                    print(f"  ! ALERTA DE INTEGRACIÓN: '{col}' tiene valor 0 en: {alcaldias_afectadas}")

    print(f"  ✓ Auditoría completa: Dataset '{nombre_dataset}' verificado.")


# ---------------------------------------------
# TAREAS: LIMPIEZA Y TRANSFORMACIÓN 
# ---------------------------------------------

def normalizar_texto(texto):
    #Normalización avanzada para homologar nombres de alcaldias de la CDMX.
    if not isinstance(texto, str): return texto
    
    # 1. Limpieza básica (acentos, mayúsculas) 
    texto = unicodedata.normalize('NFD', texto)
    texto = "".join([c for c in texto if not unicodedata.combining(c)])
    texto = texto.strip().upper()
    
    # 2. Mapeo de nombres largos a estándar corto (asegura que el MERGE sea exitoso)
    mapeo_alcaldias = {
        "CUAJIMALPA DE MORELOS": "CUAJIMALPA",
        "LA MAGDALENA CONTRERAS": "MAGDALENA CONTRERAS",
        "GUSTAVO A MADERO": "GUSTAVO A. MADERO", # A veces falta el punto
    }
    
    # Si el nombre está en nuestro mapeo, lo cambiamos al corto
    return mapeo_alcaldias.get(texto, texto)

def cargar_y_limpiar_consumo():
    #Carga y agrupa el consumo total por alcaldía
    print("[ETL] Procesando datos de consumo...")
    df = pd.read_csv(RUTAS["consumo"])
    
    df['alcaldia'] = df['alcaldia'].apply(normalizar_texto)
    df_agrupado = df.groupby('alcaldia')['consumo_total'].sum().reset_index()
  
    auditar_datos(df_agrupado, "Consumo 2019")
    return df_agrupado

def cargar_y_limpiar_demograficos():
    #Carga datos poblacionales del INEGI
    print("[ETL] Procesando datos demográficos...")
    df = pd.read_csv(RUTAS["demografico"])
    
    df['alcaldia'] = df['alcaldia'].apply(normalizar_texto)
    
    auditar_datos(df, "Demografía")
    return df

def cargar_y_limpiar_reportes():
    #Filtra y cuenta reportes de fugas y falta de agua
    print("[ETL] Procesando datos de reportes de SACMEX...")
    df = pd.read_csv(RUTAS["reportes"])
    df['alcaldia'] = df['alcaldia'].apply(normalizar_texto)
    
    # Filtrar solo por fallas relacionadas con el proyecto
    df_filtrado = df[df['tipo_de_falla'].str.contains('Fuga|Falta', case=False, na=False)]
    df_conteo = df_filtrado.groupby('alcaldia').size().reset_index(name='total_reportes')
    auditar_datos(df_conteo, "Reportes")
    return df_conteo

def integrar_dataset_maestro():
    #Realiza el merge multidimensional y calcula KPIs 
    print("\n[ETL] Iniciando integración final (Merge)...")
    
    #1. Cargar superficie como base territorial
    df_maestro = pd.read_csv(RUTAS["superficie"])
    df_maestro['alcaldia'] = df_maestro['alcaldia'].apply(normalizar_texto)
    
    #2.  Unir todos los datasets usando 'alcaldia' como llave
    df_maestro = df_maestro.merge(cargar_y_limpiar_consumo(), on='alcaldia', how='left')
    df_maestro = df_maestro.merge(cargar_y_limpiar_demograficos(), on='alcaldia', how='left')
    df_maestro = df_maestro.merge(cargar_y_limpiar_reportes(), on='alcaldia', how='left')
    
    #3. Manejo inicial de nulos (llenar con 0 donde no hubo registros)
    df_maestro['total_reportes'] = df_maestro['total_reportes'].fillna(0)
      
    #4. Auditoría de Integración
    # Vigilamos que tras el merge no existan ceros en columnas vitales
    auditar_datos(df_maestro, "Dataset Maestro Final", columnas_criticas=['consumo_total', 'poblacion'])
    
    # 5. Cálculo de KPIs 
    print("[ETL] Calculando métricas de negocio...")
    df_maestro['consumo_per_capita'] = df_maestro['consumo_total'] / df_maestro['poblacion']
    df_maestro['densidad_poblacional'] = df_maestro['poblacion'] / df_maestro['superficie_km2']
    return df_maestro

def ejecutar_pipeline():
    #Ejecuta todo el proceso de ingeniería de datos.
    try:
        df_final = integrar_dataset_maestro()
        
        if df_final is not None:
            df_final.to_csv(OUTPUT_PATH, index=False)
            print(f"\n[ÉXITO] Dataset maestro generado en: {OUTPUT_PATH}")
            print(f"Dimensiones finales: {df_final.shape}")
            return df_final.head()
    except Exception as e:
        print(f"\n[ERROR CRÍTICO] El pipeline falló: {e}")

if __name__ == "__main__":
    ejecutar_pipeline()