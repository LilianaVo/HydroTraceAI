"""
Módulo de Extracción y Limpieza (ETL)
Responsable: Bolívar García
Instrucciones: Debes cargar los CSVs y devolver un dataframe unificado.
"""

import pandas as pd

def auditoria_datos(df):
    """
    TAREA: Implementar lógica de la Unidad 2 (Calidad de Datos).
    Detectar nulos, duplicados y valores atípicos iniciales.
    """
    pass

def normalizar_alcaldias(df):
    """
    TAREA: Estandarizar nombres (Mayúsculas, sin acentos).
    """
    pass

def generar_dataset_maestro():
    """
    TAREA: Hacer el merge de consumo, demografía y superficie.
    Guardar el resultado en 'data/dataset_maestro.csv'.
    """
    print("Ejecutando limpieza de datos...")
    # Tu código aquí
    pass

if __name__ == "__main__":
    generar_dataset_maestro()
    
    # Y más funciones...