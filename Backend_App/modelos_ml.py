"""
================================================================================
PROYECTO : HydroTrace AI - Ciencia de Datos UNAM
MODULO   : modelos_ml.py
AUTORES  : Irving Morales & Ileana Lee
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
DATA_DIR = Path("data")
GRAFICAS_DIR = Path("graficas_reporte")

DATA_DIR.mkdir(exist_ok=True)
GRAFICAS_DIR.mkdir(exist_ok=True)

DATA_PATH = DATA_DIR / "dataset_maestro_colonia_final.csv"
OUTPUT_PATH = DATA_DIR / "resultados_finales_IA.csv"

# ──────────────────────────────────────────────────────────────────────────────
# 1. CARGA Y CALIBRACIÓN
# ──────────────────────────────────────────────────────────────────────────────

def cargar_y_preparar():
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encontró {DATA_PATH}.")
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
        'bajo_reporte': np.percentile(df['total_reportes'], 10),   # 10% que menos se queja
        'alta_falta_agua': np.percentile(df['reportes_falta_agua'], 75)
    }
    # Se utilizan percentiles dinámicos en lugar de valores fijos
    # para adaptar automáticamente los umbrales a la distribución
    # real de reportes ciudadanos en la CDMX.
    # Si usamos valores fijos como 15 reportes es peligroso porque 
    return umbrales

# ──────────────────────────────────────────────────────────────────────────────
# 2. SEGMENTACIÓN (K-MEANS)
# ──────────────────────────────────────────────────────────────────────────────

def ejecutar_clustering(df, X_scaled):
    print("\n[IA] Iniciando Clustering...")
    
    ##TODO @Irving: Implementar el 'Método del Codo' y guardar la gráfica
    ## como 'metodo_codo.png'. Justifica el número de clusters.
    inercia = []

    K_range = range(1, 11)

    for k in K_range:
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(X_scaled)
        inercia.append(modelo.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(K_range, inercia, marker='o')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.grid(True)

    plt.savefig(GRAFICAS_DIR / "metodo_codo.png")
    plt.close()
    
    # El Método del Codo muestra que a partir de K=4
    # la reducción de inercia comienza a estabilizarse,
    # indicando un balance entre compactación y sobresegmentación.
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)

    df['cluster_perfil'] = kmeans.fit_predict(X_scaled)

    print(f"Clusters generados: {kmeans.n_clusters}")

    print("\nDistribución de colonias por cluster:")
    print(df['cluster_perfil'].value_counts())
    
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 3. DETECCIÓN DE ANOMALÍAS (ISOLATION FOREST)
# ──────────────────────────────────────────────────────────────────────────────

def detectar_anomalias(df, X_scaled):
    print("[IA] Analizando anomalías con Isolation Forest...")
    
    # TODO @Irving: Ajusta 'contamination' basado en pérdidas de SEGUIAGUA.
    # SEGUIAGUA y reportes de Agua No Contabilizada (ANC)
    # estiman pérdidas hídricas cercanas al 30-40% en CDMX,
    # incluyendo fugas físicas, errores operativos y pérdidas comerciales.
    
    # Sin embargo, no toda el ANC representa eventos críticos.
    # Para aislar únicamente comportamientos altamente atípicos
    # asociados a fugas mayores o posibles tomas clandestinas,
    # se utiliza contamination=0.15 (15%) como criterio conservador
    # de anomalías críticas dentro del universo total de pérdidas.
    
    # Esto evita sobredetectar colonias normales como anomalías.

    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    df['es_anomalia'] = iso_forest.fit_predict(X_scaled) 
    
    anomalias = (df['es_anomalia'] == -1).sum()

    print(f"Colonias anómalas detectadas: {anomalias}")
    
    # TODO @Irving: Genera el scatter plot 'mapa_anomalias_analitico.png'.
    
    plt.figure(figsize=(10,6))

    sns.scatterplot(
        x=df['consumo_per_capita'],
        y=df['total_reportes'],
        hue=df['es_anomalia'],
        palette={1:'blue', -1:'red'}
    )
    plt.title('Mapa Analítico de Anomalías')
    plt.xlabel('Consumo Per Cápita')
    plt.ylabel('Total de Reportes')

    plt.savefig(GRAFICAS_DIR / "mapa_anomalias_analitico.png")
    plt.close()
    ##########################################################

    plt.figure(figsize=(10,6))

    sns.scatterplot(
        x=df['densidad_poblacional'],
        y=df['consumo_per_capita'],
        hue=df['es_anomalia'],
        palette={1:'blue', -1:'red'}
    )

    plt.title('Consumo Per Cápita vs Densidad Poblacional')
    plt.xlabel('Densidad Poblacional')
    plt.ylabel('Consumo Per Cápita')

    plt.savefig(GRAFICAS_DIR / "scatter_consumo_densidad.png")
    plt.close()

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
    predicciones = model.predict(X)
    # Un R² moderado/bajo es esperado debido a la complejidad
    # multifactorial del consumo hídrico urbano.
    # El objetivo del modelo no es predecir exactamente el consumo,
    # sino establecer una línea base estadística para detectar
    # desviaciones anómalas potencialmente asociadas a fugas
    # o extracción irregular de agua.
    r2 = r2_score(y, predicciones)
    mae = mean_absolute_error(y, predicciones)

    print(f"R² del modelo: {r2:.4f}")
    print(f"MAE del modelo: {mae:.2f}")

    print("\nCoeficientes del modelo:")

    for variable, coef in zip(X.columns, model.coef_):
        print(f"{variable}: {coef:.4f}")

    # El MAE representa el error promedio absoluto entre
    # el consumo real y el consumo estimado por el modelo.
    # Esta métrica permite estimar el margen mínimo de error
    # financiero para análisis de ahorro y rentabilidad.
    
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

    # Consumo menor al esperado + muchas quejas de falta de agua
    if (
        exceso < 0
        and row['reportes_falta_agua'] >= umbrales['alta_falta_agua']
    ):
        return "DEFICIENCIA (Posible Baja Presión o Desabasto)"

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
        #El método del codo indicó K=4 como punto óptimo de estabilización de la inercia. 
        # Sin embargo, uno de los clusters contiene una sola colonia, lo que sugiere la 
        # existencia de un perfil urbano altamente atípico dentro del dataset.
        print("Generando etiquetas de diagnóstico...")
        df_final['diagnostico_final'] = df_final.apply(lambda r: clasificar_riesgo(r, u), axis=1)
        
        # Limpieza final de columna crítica
        df_final['diagnostico_final'] = (df_final['diagnostico_final'].astype(str).str.strip()
)

        print("\n[VALIDACIÓN] Diagnósticos generados:")

        print(df_final['diagnostico_final'].value_counts())

        # Verificar valores nulos
        nulos = df_final['diagnostico_final'].isnull().sum()

        print(f"Valores nulos en diagnostico_final: {nulos}")


        df_final.to_csv(OUTPUT_PATH, index=False)
        print(f"Pipeline completado. Resultados en {OUTPUT_PATH}")