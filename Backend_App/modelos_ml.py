"""
================================================================================
HydroTrace AI — Módulo de Machine Learning
modelos_ml.py

Autores : Irving Morales & Ileana Lee
Materia : Ciencia de Datos en la Toma de Decisiones en las Organizaciones
Facultad de Ingeniería, UNAM | Ciudad de México, 2026

Pipeline de tres modelos sobre el dataset maestro de colonias de la CDMX:
  1. K-Means          → segmentación por perfil urbano
  2. Regresión Lineal → línea base de consumo esperado
  3. Isolation Forest → detección de anomalías por cluster
  4. Diagnóstico      → etiqueta operativa por colonia

Salida: data/resultados_finales_IA.csv
================================================================================
"""

import os
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # Sin pantalla — compatible con servidor y CI
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
# CONFIGURACIÓN DE RUTAS
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
GRAFICAS_DIR = BASE_DIR / "graficas_reporte"
DATA_PATH    = DATA_DIR / "dataset_maestro_colonia_final.csv"
OUTPUT_PATH  = DATA_DIR / "resultados_finales_IA.csv"

DATA_DIR.mkdir(exist_ok=True)
GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)

# Consumo per cápita promedio de la CDMX: 366 L/hab/día (fuente: SACMEX 2019).
# Se usa para traducir metros cúbicos recuperables a población equivalente abastecida,
# una métrica que le da contexto social al impacto hídrico del sistema.
LITROS_PER_CAPITA_DIA = 366.0
M3_PER_CAPITA_ANUAL   = (LITROS_PER_CAPITA_DIA * 365) / 1000   # → 133.59 m³/hab/año


# ──────────────────────────────────────────────────────────────────────────────
# PASO 0 — CARGA Y ESCALADO
# ──────────────────────────────────────────────────────────────────────────────

def cargar_y_preparar():
    """
    Lee el dataset maestro generado por etl_pipeline.py y escala las features
    que usarán K-Means e Isolation Forest.

    Se escala con StandardScaler porque K-Means es sensible a la magnitud:
    sin escalar, consumo_per_capita (rango ~0-5000) dominaría la distancia
    euclidiana y las otras tres variables quedarían prácticamente ignoradas.

    Devuelve:
        df_model  — DataFrame limpio (sin NaN en features del modelo)
        X_scaled  — matriz numpy escalada, misma longitud que df_model
    """
    if not DATA_PATH.exists():
        print(f"[ERROR] No se encontró {DATA_PATH}. Ejecuta etl_pipeline.py primero.")
        return None, None

    df = pd.read_csv(DATA_PATH)

    features = ['consumo_per_capita', 'densidad_poblacional', 'uso_suelo_num', 'idsm']
    df_model = df.dropna(subset=features).copy()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[features])

    print(f"[CARGA] {len(df_model)} colonias listas para el pipeline.")
    return df_model, X_scaled


# ──────────────────────────────────────────────────────────────────────────────
# PASO 1 — SEGMENTACIÓN (K-MEANS)
# ──────────────────────────────────────────────────────────────────────────────

def ejecutar_clustering(df, X_scaled):
    """
    Agrupa las colonias en 4 perfiles urbanos usando K-Means.

    K=4 se eligió por interpretabilidad operativa, no solo por la curva del codo.
    Los 4 clusters corresponden a perfiles reales de la CDMX:
      · Residencial de bajo consumo
      · Residencial / mixto de consumo medio
      · Comercial / servicios de alto consumo
      · Industrial o atípico (outlier legítimo, ej. Industrial Vallejo)

    Un cluster de tamaño 1 es un resultado válido: significa que el modelo
    aisló un outlier real, no que el algoritmo falló.

    Parámetros fijos para reproducibilidad: random_state=42, n_init=10.
    """
    print("\n[K-MEANS] Iniciando segmentación por perfil urbano...")

    # Método del codo: calcula inercia para K=1..10 y guarda la gráfica.
    # La línea roja marca el K elegido para que el jurado vea la justificación.
    inercias = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inercias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), inercias, marker='o', color='#06b6d4', linewidth=2)
    ax.axvline(x=4, color='red', linestyle='--', linewidth=1.5, label='K elegido = 4')
    ax.set_xlabel('Número de clusters (K)')
    ax.set_ylabel('Inercia')
    ax.set_title('Método del Codo — Selección de K')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "metodo_codo.png", dpi=150)
    plt.close(fig)

    # Modelo final con K=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster_perfil'] = kmeans.fit_predict(X_scaled)

    print(f"[K-MEANS] Clusters generados: {kmeans.n_clusters}")
    print(df['cluster_perfil'].value_counts().to_string())

    return df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 2 — REGRESIÓN LINEAL (LÍNEA BASE DE CONSUMO ESPERADO)
# ──────────────────────────────────────────────────────────────────────────────

def estimar_consumo_base(df):
    """
    Entrena una Regresión Lineal Múltiple para estimar cuánto debería consumir
    cada colonia dados sus atributos demográficos y de uso de suelo.

    Variable dependiente : consumo_total (m³ facturados por la colonia en 2019)
    Variables independientes: población, densidad, superficie, IDSM, uso de suelo

    El R² esperado es moderado o bajo — eso no es un fallo del modelo.
    El consumo hídrico urbano depende de actividad económica, comercio informal
    y hábitos culturales que no están en los datos disponibles. Lo importante
    no es predecir con precisión sino tener una línea base estadística que
    permita detectar desviaciones (exceso_consumo) que el Isolation Forest
    confirmará o descartará como anomalías.

    Genera:
        consumo_esperado — predicción del modelo (m³)
        exceso_consumo   — diferencia real vs esperado (positivo = más de lo normal)
    """
    print("\n[REGRESIÓN] Entrenando modelo de consumo esperado...")

    X = df[['pob', 'densidad_poblacional', 'superficie_km2_calculada', 'idsm', 'uso_suelo_num']]
    y = df['consumo_total']

    modelo = LinearRegression()
    modelo.fit(X, y)

    df['consumo_esperado'] = modelo.predict(X)
    df['exceso_consumo']   = df['consumo_total'] - df['consumo_esperado']

    r2  = r2_score(y, df['consumo_esperado'])
    mae = mean_absolute_error(y, df['consumo_esperado'])

    print(f"  R²  : {r2:.4f}  (moderado/bajo esperado — ver nota técnica)")
    print(f"  MAE : {mae:,.2f} m³")
    print("\n  Coeficientes:")
    for var, coef in zip(X.columns, modelo.coef_):
        print(f"    {var:<35} {coef:+.4f}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 3 — DETECCIÓN DE ANOMALÍAS (ISOLATION FOREST POR CLUSTER)
# ──────────────────────────────────────────────────────────────────────────────

def detectar_anomalias(df, X_scaled):
    """
    Aplica Isolation Forest de forma independiente dentro de cada cluster.

    Por qué por cluster y no sobre todo el dataset:
    Si se aplicara globalmente, las colonias residenciales de bajo consumo
    serían marcadas como normales en comparación con las industriales, aunque
    sean atípicas dentro de su propio perfil. Aplicarlo por cluster garantiza
    que cada colonia se compara contra su grupo de similares.

    contamination=0.15 → se espera que ~15% de las colonias en cada cluster
    sean anómalas. Es un criterio conservador: SEGIAGUA estima pérdidas del
    30-40% en la red, pero preferimos subdetectar que sobredetectar para no
    saturar al equipo de campo con falsas alarmas.

    Genera:
        es_anomalia    — 1=normal, -1=anómala (convención de scikit-learn)
        anomalia_score — score continuo; más negativo = más anómala
    """
    print("\n[ISOLATION FOREST] Detectando anomalías por cluster...")

    df['es_anomalia']    = 1     # valor por defecto: normal
    df['anomalia_score'] = 0.0

    for cluster_id in sorted(df['cluster_perfil'].unique()):
        mask      = df['cluster_perfil'] == cluster_id
        X_cluster = X_scaled[mask]

        iso = IsolationForest(contamination=0.15, random_state=42)
        df.loc[mask, 'es_anomalia']    = iso.fit_predict(X_cluster)
        df.loc[mask, 'anomalia_score'] = iso.decision_function(X_cluster)

        n_anomalas = (df.loc[mask, 'es_anomalia'] == -1).sum()
        print(f"  Cluster {cluster_id}: {n_anomalas} anómalas de {mask.sum()} colonias")

    total = (df['es_anomalia'] == -1).sum()
    print(f"  Total anomalías detectadas: {total}")

    # Gráfica 1: Mapa analítico — consumo per cápita vs reportes ciudadanos.
    # Los ejes se recortan al percentil 95 para que los outliers extremos
    # no aplasten la vista del resto de las colonias.
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=df['consumo_per_capita'], y=df['total_reportes'],
        hue=df['es_anomalia'], palette={1: '#0A84FF', -1: '#FF2D55'},
        alpha=0.75, s=80, ax=ax,
    )
    x_lim = df['consumo_per_capita'].quantile(0.95)
    y_lim = df['total_reportes'].quantile(0.95)
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)

    # Etiqueta las 6 colonias anómalas más relevantes dentro del rango visible
    top_anomalas = df[
        (df['es_anomalia'] == -1) &
        (df['consumo_per_capita'] <= x_lim) &
        (df['total_reportes'] <= y_lim)
    ].nlargest(6, 'consumo_per_capita')
    for _, row in top_anomalas.iterrows():
        ax.annotate(
            row['colonia'],
            xy=(row['consumo_per_capita'], row['total_reportes']),
            xytext=(6, 4), textcoords='offset points',
            fontsize=7.5, color='#8B0000',
        )

    n_fuera = ((df['consumo_per_capita'] > x_lim) | (df['total_reportes'] > y_lim)).sum()
    if n_fuera:
        ax.annotate(
            f'* {n_fuera} punto(s) fuera del rango visible (outliers extremos)',
            xy=(0.01, 0.01), xycoords='axes fraction',
            fontsize=8, color='gray', style='italic',
        )

    ax.set_title('Mapa Analítico de Anomalías — HydroTrace AI', fontsize=13, fontweight='bold')
    ax.set_xlabel('Consumo Per Cápita (m³/hab)')
    ax.set_ylabel('Total de Reportes Ciudadanos')
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "mapa_anomalias_analitico.png", dpi=150)
    plt.close(fig)

    # Gráfica 2: Consumo per cápita vs densidad poblacional por tipo de colonia
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=df['densidad_poblacional'], y=df['consumo_per_capita'],
        hue=df['es_anomalia'], palette={1: '#0A84FF', -1: '#FF2D55'}, ax=ax2,
    )
    ax2.set_xlim(0, df['densidad_poblacional'].quantile(0.95))
    ax2.set_title('Consumo Per Cápita vs Densidad Poblacional')
    ax2.set_xlabel('Densidad Poblacional (hab/km²)')
    ax2.set_ylabel('Consumo Per Cápita (m³/hab)')
    fig2.tight_layout()
    fig2.savefig(GRAFICAS_DIR / "scatter_consumo_densidad.png", dpi=150)
    plt.close(fig2)

    return df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 4 — UMBRALES DINÁMICOS (solo sobre las colonias anómalas)
# ──────────────────────────────────────────────────────────────────────────────

def calibrar_umbrales(df_anomalas: pd.DataFrame) -> dict:
    """
    Calcula los umbrales de diagnóstico ÚNICAMENTE sobre las colonias que
    Isolation Forest marcó como anómalas (es_anomalia == -1).

    Por qué no sobre el dataset completo:
    Calcular percentiles sobre todas las colonias mezcla normales con anómalas
    y eleva artificialmente los umbrales, reduciendo la sensibilidad del
    diagnóstico. Al filtrar solo las anómalas, los percentiles reflejan la
    distribución real del subconjunto de interés.

    Por qué percentiles y no valores fijos:
    Si el dataset se actualiza a un año distinto, un valor fijo como "15 reportes"
    puede no tener ningún sentido. Los percentiles se recalibran solos.

    Umbrales:
        alto_reporte    — p75 de reportes entre anómalas → umbral CRÍTICO
        bajo_reporte    — p10 de reportes entre anómalas → umbral HUACHICOL
        alta_falta_agua — p75 de reportes_falta_agua     → umbral DEFICIENCIA
    """
    umbrales = {
        'alto_reporte'   : np.percentile(df_anomalas['total_reportes'], 75),
        'bajo_reporte'   : np.percentile(df_anomalas['total_reportes'], 10),
        'alta_falta_agua': np.percentile(df_anomalas['reportes_falta_agua'], 75),
    }
    print(f"\n[UMBRALES] Calculados sobre {len(df_anomalas)} colonias anómalas:")
    print(f"  CRÍTICO     — reportes >= {umbrales['alto_reporte']:.0f}  (p75)")
    print(f"  HUACHICOL   — reportes <= {umbrales['bajo_reporte']:.0f}  (p10)")
    print(f"  DEFICIENCIA — falta_agua >= {umbrales['alta_falta_agua']:.0f}  (p75)")
    return umbrales


# ──────────────────────────────────────────────────────────────────────────────
# PASO 5 — DIAGNÓSTICO INTEGRADO
# ──────────────────────────────────────────────────────────────────────────────

def clasificar_riesgo(row, umbrales):
    """
    Asigna una etiqueta operativa a cada colonia combinando tres señales:
      · es_anomalia   — ¿el Isolation Forest la marcó como atípica?
      · exceso_consumo — ¿consume más o menos de lo que predice la regresión?
      · reportes       — ¿cuántas quejas ciudadanas tiene?

    El orden de las condiciones importa: se evalúan de mayor a menor gravedad
    y se asigna la primera que se cumple.

    CRÍTICO          → anomalía + exceso + muchos reportes (fuga con evidencia ciudadana)
    SOSPECHOSO H     → anomalía + exceso + pocos reportes  (extracción silenciosa)
    DEFICIENCIA      → consumo bajo + muchas quejas de falta de agua
    SOSPECHOSO IA    → anomalía + exceso, reportes intermedios (señal de modelo sin validación ciudadana)
    NORMAL           → todo lo demás
    """
    exceso = row['exceso_consumo']

    if row['es_anomalia'] == -1 and exceso > 0 and row['total_reportes'] >= umbrales['alto_reporte']:
        return "CRÍTICO (Posible Fuga de Red)"

    if row['es_anomalia'] == -1 and exceso > 0 and row['total_reportes'] <= umbrales['bajo_reporte']:
        return "SOSPECHOSO (Posible Huachicol)"

    if exceso < 0 and row['reportes_falta_agua'] >= umbrales['alta_falta_agua']:
        return "DEFICIENCIA (Posible Baja Presión o Desabasto)"

    if row['es_anomalia'] == -1 and exceso > 0:
        return "SOSPECHOSO (Exceso Detectado por IA)"

    return "NORMAL"


# ──────────────────────────────────────────────────────────────────────────────
# ORQUESTADOR PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def ejecutar_pipeline():
    """
    Ejecuta los 5 pasos en orden y exporta resultados_finales_IA.csv.

    El orden no es arbitrario:
      Clustering primero → define los grupos de comparación para Isolation Forest.
      Regresión segundo  → genera exceso_consumo que el diagnóstico necesita.
      Isolation Forest   → requiere los clusters del paso 1.
      Umbrales al final  → se calculan sobre las anómalas ya identificadas.
    """
    df_final, X_s = cargar_y_preparar()
    if df_final is None:
        return

    df_final = ejecutar_clustering(df_final, X_s)
    df_final = estimar_consumo_base(df_final)
    df_final = detectar_anomalias(df_final, X_s)

    # Los umbrales se calculan DESPUÉS de detectar anomalías y SOLO sobre ellas
    anomalas = df_final[df_final['es_anomalia'] == -1]
    if anomalas.empty:
        print("[ERROR] No se detectaron anomalías. Verifica contamination en IsolationForest.")
        return
    umbrales = calibrar_umbrales(anomalas)

    print("\n[DIAGNÓSTICO] Clasificando colonias...")
    df_final['diagnostico_final'] = df_final.apply(
        lambda r: clasificar_riesgo(r, umbrales), axis=1
    ).astype(str).str.strip()

    # Métrica de impacto social: ¿cuántas personas podrían ser abastecidas
    # con el volumen en exceso recuperable al 20% de capacidad de reparación?
    # Fórmula: (m³ exceso × 0.20) / 133.59 m³/hab/año
    df_final['poblacion_equivalente'] = (
        df_final['exceso_consumo'].clip(lower=0) * 0.20 / M3_PER_CAPITA_ANUAL
    ).round(0).astype(int)

    print("\n[VALIDACIÓN] Distribución de diagnósticos:")
    print(df_final['diagnostico_final'].value_counts().to_string())
    print(f"\n  Población equivalente total recuperable: "
          f"{df_final['poblacion_equivalente'].sum():,.0f} hab/año")
    

    # Ordenar por severidad (CRÍTICO primero) y dentro de cada nivel
    # por consumo total descendente — así el CSV ya viene listo para el dashboard.
    orden_severidad = {
        "CRÍTICO (Posible Fuga de Red)"                  : 0,
        "SOSPECHOSO (Posible Huachicol)"                 : 1,
        "SOSPECHOSO (Exceso Detectado por IA)"           : 1,
        "DEFICIENCIA (Posible Baja Presión o Desabasto)" : 2,
        "NORMAL"                                         : 3,
    }
    df_final['_orden'] = df_final['diagnostico_final'].map(orden_severidad)
    df_final = df_final.sort_values(
        ['_orden', 'consumo_total'], ascending=[True, False]
    ).drop(columns='_orden')

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Pipeline completado → {OUTPUT_PATH}")
    print(f"     {len(df_final)} colonias | {(df_final['es_anomalia'] == -1).sum()} anomalías detectadas")

    generar_graficas_analiticas(df_final)


# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICAS ANALÍTICAS
# Se generan al final del pipeline con el DataFrame ya completo (diagnóstico
# incluido) y se guardan en graficas_reporte/ para el dashboard admin.
# ──────────────────────────────────────────────────────────────────────────────

def generar_graficas_analiticas(df: pd.DataFrame) -> None:
    """
    Genera 5 gráficas analíticas y las guarda en graficas_reporte/.

    1. heatmap_correlaciones.png  — correlación entre variables numéricas clave
    2. distribucion_diagnosticos.png — conteo de colonias por diagnóstico
    3. exceso_por_alcaldia.png    — exceso de consumo total agrupado por alcaldía
    4. boxplot_consumo_cluster.png — distribución de consumo per cápita por cluster
    5. top10_anomalias.png        — ranking de las 10 colonias más anómalas
    """
    print("\n[GRÁFICAS] Generando visualizaciones analíticas...")

    # ── 1. HEATMAP DE CORRELACIONES ──────────────────────────────────────────
    # Muestra qué tan relacionadas están las variables del modelo entre sí.
    # Correlaciones altas entre independientes (multicolinealidad) pueden
    # inflar los coeficientes de la regresión — útil para la defensa.
    cols_corr = [
        'consumo_per_capita', 'densidad_poblacional', 'idsm',
        'exceso_consumo', 'total_reportes', 'reportes_falta_agua',
        'consumo_total', 'pob',
    ]
    cols_corr = [c for c in cols_corr if c in df.columns]
    corr = df[cols_corr].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 8},
    )
    ax.set_title("Mapa de Correlaciones — Variables del Modelo", fontsize=13, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "heatmap_correlaciones.png", dpi=150)
    plt.close(fig)
    print("  ✓ heatmap_correlaciones.png")

    # ── 2. DISTRIBUCIÓN DE DIAGNÓSTICOS ──────────────────────────────────────
    # Cuántas colonias cayeron en cada categoría de riesgo.
    # Es la primera gráfica que debería ver el equipo de campo.
    colores_diag = {
        "CRÍTICO (Posible Fuga de Red)"                  : "#FF2D55",
        "SOSPECHOSO (Posible Huachicol)"                 : "#FF9F0A",
        "SOSPECHOSO (Exceso Detectado por IA)"           : "#FF6B35",
        "DEFICIENCIA (Posible Baja Presión o Desabasto)" : "#30D158",
        "NORMAL"                                         : "#0A84FF",
    }
    conteos = df['diagnostico_final'].value_counts()
    colores = [colores_diag.get(d, "#98989D") for d in conteos.index]
    etiquetas = [d.split("(")[0].strip() for d in conteos.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(etiquetas, conteos.values, color=colores, edgecolor='white')
    ax.bar_label(bars, padding=4, fontsize=10, fontweight='bold')
    ax.set_xlabel("Número de colonias")
    ax.set_title("Distribución de Diagnósticos por Colonia", fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.set_xlim(0, conteos.max() * 1.15)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "distribucion_diagnosticos.png", dpi=150)
    plt.close(fig)
    print("  ✓ distribucion_diagnosticos.png")

    # ── 3. EXCESO DE CONSUMO POR ALCALDÍA ────────────────────────────────────
    # Muestra qué alcaldías concentran más volumen de agua no contabilizada.
    # Útil para que SEGIAGUA priorice zonas de inspección a nivel macro.
    exceso_alc = (
        df[df['exceso_consumo'] > 0]
        .groupby('alcaldia')['exceso_consumo']
        .sum()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(exceso_alc.index, exceso_alc.values / 1000, color="#FF9F0A", edgecolor='white')
    ax.bar_label(bars, fmt='%.0f k', padding=4, fontsize=8)
    ax.set_xlabel("Exceso de Consumo (miles de m³)")
    ax.set_title("Exceso de Consumo Total por Alcaldía\n(solo colonias con exceso positivo)",
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "exceso_por_alcaldia.png", dpi=150)
    plt.close(fig)
    print("  ✓ exceso_por_alcaldia.png")

    # ── 4. BOXPLOT CONSUMO PER CÁPITA POR CLUSTER ────────────────────────────
    # Muestra la distribución de consumo dentro de cada cluster.
    # Valida que los 4 grupos tienen rangos de consumo distintos —
    # si se solaparan mucho, K=4 no estaría segmentando bien.
    fig, ax = plt.subplots(figsize=(9, 5))
    grupos = [
        df[df['cluster_perfil'] == k]['consumo_per_capita'].dropna()
        for k in sorted(df['cluster_perfil'].unique())
    ]
    bp = ax.boxplot(grupos, patch_artist=True, notch=False)
    colores_box = ["#0A84FF", "#30D158", "#FF9F0A", "#FF2D55"]
    for patch, color in zip(bp['boxes'], colores_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f"Cluster {k}" for k in sorted(df['cluster_perfil'].unique())])
    ax.set_ylabel("Consumo Per Cápita (m³/hab)")
    ax.set_title("Distribución de Consumo Per Cápita por Cluster", fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    # Recortar eje Y al percentil 95 para que los outliers extremos no aplasten la vista
    ax.set_ylim(0, df['consumo_per_capita'].quantile(0.95))
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "boxplot_consumo_cluster.png", dpi=150)
    plt.close(fig)
    print("  ✓ boxplot_consumo_cluster.png")

    # ── 5. TOP 10 COLONIAS MÁS ANÓMALAS ─────────────────────────────────────
    # Ranking de las colonias con el score de anomalía más negativo.
    # Score más negativo = más anómala según Isolation Forest.
    # Esta gráfica es el argumento más directo para el equipo de campo:
    # "aquí están las 10 colonias que más urge inspeccionar".
    top10 = (
        df[df['es_anomalia'] == -1]
        .nsmallest(10, 'anomalia_score')[['colonia', 'anomalia_score', 'diagnostico_final']]
    )
    colores_top = [colores_diag.get(d, "#98989D") for d in top10['diagnostico_final']]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        top10['colonia'].str.title(),
        top10['anomalia_score'].abs(),   # valor absoluto para que la barra sea positiva
        color=colores_top, edgecolor='white',
    )
    ax.bar_label(bars, fmt='%.3f', padding=4, fontsize=8)
    ax.set_xlabel("Score de Anomalía (valor absoluto — mayor = más anómala)")
    ax.set_title("Top 10 Colonias Más Anómalas — HydroTrace AI",
                 fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "top10_anomalias.png", dpi=150)
    plt.close(fig)
    print("  ✓ top10_anomalias.png")

    print(f"[GRÁFICAS] 5 gráficas guardadas en {GRAFICAS_DIR}/")


if __name__ == "__main__":
    ejecutar_pipeline()