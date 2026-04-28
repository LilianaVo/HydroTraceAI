"""
PROYECTO: HydroTrace AI - Ciencia de Datos UNAM
DESCRIPCIÓN: Pipeline de extracción, transformación y carga (ETL).
             Une los 4 datasets fuente y genera el 'dataset_maestro.csv'.
"""

import os
import unicodedata
import pandas as pd

# ---------------------------------------------
# CONFIGURACIÓN DE RUTAS
# ---------------------------------------------
DATA_DIR = "data"
RUTAS = {
    "consumo": os.path.join(DATA_DIR, "consumo_agua_historico_2019.csv"),
    "demografico": os.path.join(DATA_DIR, "c_demograficas_total_alcaldia.csv"),
    "reportes": os.path.join(DATA_DIR, "reportes_agua_hist.csv"),
    "superficie": os.path.join(DATA_DIR, "superficie_alcaldias.csv"),
}
OUTPUT_PATH = os.path.join(DATA_DIR, "dataset_maestro.csv")

# ---------------------------------------------
# TAREAS: LIMPIEZA Y TRANSFORMACIÓN
# ---------------------------------------------

def normalizar_texto(texto):
    """
    TAREA 1: Normalización (Unidad 2 - Calidad de Datos)
    Objetivo: Eliminar acentos, convertir a mayúsculas y quitar espacios extra.
    Esto es CRÍTICO para que los merge entre alcaldías no fallen.
    """
    if not isinstance(texto, str): return texto
    
    # TODO: Implementar lógica para quitar acentos usando unicodedata
    # TODO: Convertir a MAYÚSCULAS y quitar espacios en blanco laterales
    
    return texto

def cargar_y_limpiar_consumo():
    """
    TAREA 2: Procesamiento de Consumo
    Objetivo: Cargar el CSV de consumo, normalizar alcaldías y agrupar por zona.
    """
    print("[ETL] Limpiando datos de consumo...")
    # df = pd.read_csv(RUTAS["consumo"])
    
    # TODO: Aplicar normalizar_texto a la columna de alcaldía
    # TODO: Agrupar por alcaldía y sumar el consumo total
    
    return None # Retornar DataFrame procesado

def cargar_y_limpiar_demograficos():
    """
    TAREA 3: Procesamiento Demográfico
    Objetivo: Cargar población y normalizar.
    """
    print("[ETL] Limpiando datos demográficos...")
    
    # TODO: Cargar CSV y normalizar nombres de alcaldías
    
    return None

def cargar_y_limpiar_reportes():
    """
    TAREA 4: Procesamiento de Reportes
    Objetivo: Contar cuántas fallas/fugas se reportaron por alcaldía.
    """
    print("[ETL] Limpiando datos de reportes...")
    
    # TODO: Cargar CSV, normalizar y hacer un count() por alcaldía
    
    return None

def integrar_dataset_maestro():
    """
    TAREA 5: Integración Final (Merge)
    Objetivo: Unir los 4 DataFrames en uno solo usando 'alcaldia' como llave.
    """
    print("[ETL] Iniciando integración (Merge multidimensional)...")
    
    # TODO: Cargar superficie_alcaldias.csv como base
    # TODO: Hacer merge sucesivo con consumo, demográficos y reportes
    
    # --- CÁLCULO DE KPIs DERIVADOS ---
    print("[ETL] Calculando variables derivadas...")
    
    # TODO: Calcular 'consumo_per_capita' (consumo_total / poblacion)
    # TODO: Calcular 'densidad_poblacional' (poblacion / superficie_km2)
    
    # TODO: Manejo de valores nulos (fillna con 0 o la media según criterio)
    
    # return df_maestro
    return None

def ejecutar_pipeline():
    """Ejecuta todo el proceso de ingeniería de datos."""
    df_final = integrar_dataset_maestro()
    
    if df_final is not None:
        df_final.to_csv(OUTPUT_PATH, index=False)
        print(f"\n[ÉXITO] Dataset maestro generado en: {OUTPUT_PATH}")
        print(f"Columnas finales: {list(df_final.columns)}")
    else:
        print("\n[ERROR] El pipeline no devolvió datos. Revisa las funciones de Bolívar.")

if __name__ == "__main__":
    ejecutar_pipeline()