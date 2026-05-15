"""
================================================================================
PROYECTO : HydroTrace AI - Ciencia de Datos UNAM
MODULO   : modelos_ml.py
AUTORES  : Irving Morales & Ileana Lee
OBJETIVO : Segmentación, detección de anomalías y regresión para diagnóstico
           de fugas y huachicol de agua a nivel colonia en la CDMX.

NOTAS TÉCNICAS:
  - El backend de matplotlib está en 'Agg' para compatibilidad con servidores.
  - Los umbrales de diagnóstico se calculan dinámicamente con percentiles.
  - K=4 elegido por interpretabilidad de negocio (ver comentario en ejecutar_clustering).
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
    Usa percentiles en lugar de valores fijos para adaptar los umbrales
    automáticamente a la distribución real del dataset.

    Un valor fijo como '15 reportes' sería arbitrario e inútil aquí:
    la mediana real del dataset es ~227 reportes y el mínimo es 5,
    por lo que un umbral fijo de 15 clasificaría casi todo como anómalo
    o casi nada, dependiendo del sentido. Los percentiles se recalibran
    solos si el dataset crece o cambia de año.
    """
    umbrales = {
        'alto_reporte'   : np.percentile(df['total_reportes'], 75),   # Top 25% de quejas
        'bajo_reporte'   : np.percentile(df['total_reportes'], 10),   # 10% que menos se queja
        'alta_falta_agua': np.percentile(df['reportes_falta_agua'], 75)
    }
    print(f"[UMBRALES] Reportes CRITICO   (p75): {umbrales['alto_reporte']:.0f}")
    print(f"[UMBRALES] Reportes SOSPECHOSO (p10): {umbrales['bajo_reporte']:.0f}")
    print(f"[UMBRALES] Falta agua DEFICIENCIA (p75): {umbrales['alta_falta_agua']:.0f}")
    return umbrales

# ──────────────────────────────────────────────────────────────────────────────
# 2. SEGMENTACIÓN (K-MEANS)
# ──────────────────────────────────────────────────────────────────────────────

def ejecutar_clustering(df, X_scaled):
    print("\n[IA] Iniciando Clustering...")
    
    inercia = []

    K_range = range(1, 11)

    for k in K_range:
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(X_scaled)
        inercia.append(modelo.inertia_)

    plt.figure(figsize=(8,5))
    plt.plot(K_range, inercia, marker='o')
    plt.axvline(x=4, color='red', linestyle='--', linewidth=1.5, label='K elegido = 4')
    plt.xlabel('Número de clusters')
    plt.ylabel('Inercia')
    plt.title('Método del Codo')
    plt.legend()
    plt.grid(True)

    plt.savefig(GRAFICAS_DIR / "metodo_codo.png")
    plt.close()

    # K=4 se eligió por interpretabilidad de negocio, no únicamente por la curva.
    # La gráfica muestra descenso continuo hasta K=6-7, por lo que no hay un codo
    # matemáticamente claro en K=4. Sin embargo, K=4 corresponde a los 4 perfiles
    # urbanos reales de la CDMX: residencial, comercial, industrial y vulnerable/irregular.
    # Aumentar K produciría segmentos sin interpretación operativa útil para SACMEX.
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
    
    iso_forest = IsolationForest(contamination=0.15, random_state=42)
    df['es_anomalia']    = iso_forest.fit_predict(X_scaled)
    # Score continuo: más negativo = más anómalo. Útil para rankear colonias por riesgo.
    df['anomalia_score'] = iso_forest.decision_function(X_scaled)

    anomalias = (df['es_anomalia'] == -1).sum()

    print(f"Colonias anómalas detectadas: {anomalias}")
    
    # SEGUIAGUA estima pérdidas hídricas del 30-40% en CDMX (fugas, errores operativos
    # y pérdidas comerciales). Se usa contamination=0.15 como criterio conservador
    # para aislar únicamente los eventos más atípicos, evitando sobredetección.
    
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(
        x=df['consumo_per_capita'],
        y=df['total_reportes'],
        hue=df['es_anomalia'],
        palette={1: 'blue', -1: 'red'},
        alpha=0.75,
        s=80,
        ax=ax,
    )

    # Recortar ambos ejes al percentil 95 para que el outlier no aplaste la vista.
    x_lim = df['consumo_per_capita'].quantile(0.95)
    y_lim = df['total_reportes'].quantile(0.95)
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)

    # Etiquetar las colonias anomalas mas relevantes dentro del area visible
    anomalos_visibles = df[
        (df['es_anomalia'] == -1) &
        (df['consumo_per_capita'] <= x_lim) &
        (df['total_reportes'] <= y_lim)
    ].nlargest(6, 'consumo_per_capita')

    for _, fila in anomalos_visibles.iterrows():
        ax.annotate(
            fila['colonia'],
            xy=(fila['consumo_per_capita'], fila['total_reportes']),
            xytext=(6, 4),
            textcoords='offset points',
            fontsize=7.5,
            color='#8B0000',
        )

    ax.set_title('Mapa Analítico de Anomalías', fontsize=13, fontweight='bold')
    ax.set_xlabel('Consumo Per Cápita (m3/hab)')
    ax.set_ylabel('Total de Reportes Ciudadanos')

    # Nota aclaratoria sobre puntos fuera del rango visible
    n_fuera = ((df['consumo_per_capita'] > x_lim) | (df['total_reportes'] > y_lim)).sum()
    if n_fuera > 0:
        ax.annotate(
            f'* {n_fuera} punto(s) fuera del rango visible (outliers extremos)',
            xy=(0.01, 0.01), xycoords='axes fraction',
            fontsize=8, color='gray', style='italic',
        )

    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "mapa_anomalias_analitico.png", dpi=150)
    plt.close(fig)

    plt.figure(figsize=(10,6))

    sns.scatterplot(
        x=df['densidad_poblacional'],
        y=df['consumo_per_capita'],
        hue=df['es_anomalia'],
        palette={1:'blue', -1:'red'}
    )
    # Se recorta el eje X al percentil 95 para evitar que el outlier de
    # alta densidad (~85,000 hab/km²) comprima la visualización del resto.
    # El punto extremo sigue existiendo en los datos; solo se ajusta la vista.
    plt.xlim(0, df['densidad_poblacional'].quantile(0.95))
    plt.title('Consumo Per Cápita vs Densidad Poblacional')
    plt.xlabel('Densidad Poblacional (hab/km²)')
    plt.ylabel('Consumo Per Cápita (m³/hab)')

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
    # Diferencia entre lo que se consume y lo que el modelo estima como normal.
    # Valor positivo = consumo por encima de lo esperado (posible fuga o extracción).
    # Valor negativo = consumo por debajo (posible desabasto).
    df['exceso_consumo'] = df['consumo_total'] - df['consumo_esperado']

    # Un R² moderado/bajo es esperado debido a la complejidad
    # multifactorial del consumo hídrico urbano.
    # El objetivo del modelo no es predecir exactamente el consumo,
    # sino establecer una línea base estadística para detectar
    # desviaciones anómalas potencialmente asociadas a fugas
    # o extracción irregular de agua.
    r2  = r2_score(y, df['consumo_esperado'])
    mae = mean_absolute_error(y, df['consumo_esperado'])

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
    exceso = row['exceso_consumo']   # Ya calculado en estimar_consumo_base
    
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

        print("Generando etiquetas de diagnóstico...")
        df_final['diagnostico_final'] = df_final.apply(lambda r: clasificar_riesgo(r, u), axis=1)
        df_final['diagnostico_final'] = df_final['diagnostico_final'].astype(str).str.strip()

        print("\n[VALIDACIÓN] Diagnósticos generados:")
        print(df_final['diagnostico_final'].value_counts())

        nulos = df_final['diagnostico_final'].isnull().sum()
        print(f"Valores nulos en diagnostico_final: {nulos}")

        # Ordenar por severidad para que el CSV sea directamente legible en Excel.
        # Dentro de cada nivel, las colonias de mayor consumo aparecen primero.
        orden_severidad = {
            "CRÍTICO (Posible Fuga de Red)"                 : 0,
            "SOSPECHOSO (Posible Huachicol)"                : 1,
            "DEFICIENCIA (Posible Baja Presión o Desabasto)": 2,
            "NORMAL"                                        : 3,
        }
        df_final['_orden'] = df_final['diagnostico_final'].map(orden_severidad)
        df_final = df_final.sort_values(
            ['_orden', 'consumo_total'], ascending=[True, False]
        ).drop(columns='_orden')

        df_final.to_csv(OUTPUT_PATH, index=False)
        print(f"Pipeline completado. Resultados en {OUTPUT_PATH}")