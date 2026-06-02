"""
================================================================================

EQUIPO: PUMASCRIPT SOLUTIONS

PROYECTO: HydroTrace AI 
MÓDULO   : modelos_ml.py / Módulo de Machine Learning

Autores : 

- Ileana Lee / Project Manager, Lead Data Scientist & UX/UI
- Irving Morales / QA Data Tester & Data Scientist

Materia : Ciencia de Datos en la Toma de Decisiones en las Organizaciones
GRUPO: 04
Facultad de Ingeniería, UNAM | Ciudad de México, 2026

Pipeline de tres modelos sobre el dataset maestro de colonias de la CDMX:
  1. K-Means          → segmentación en 4 perfiles urbanos
  2. Regresión Lineal → línea base de consumo esperado por colonia
  3. Isolation Forest → detección de anomalías dentro de cada cluster
  4. Diagnóstico      → etiqueta operativa final por colonia

Salida: data/resultados_finales_IA.csv

COBERTURA TEMPORAL:
    El dataset de entrada cubre el primer semestre de 2019 (bimestres 1-3).
    Todos los KPIs derivados (exceso_consumo, poblacion_equivalente,
    impacto financiero) reflejan ese período.
    Para comparar contra benchmarks anuales usar consumo_per_capita_anualizado
    en lugar de consumo_per_capita.

================================================================================

CORRECCIONES APLICADAS RESPECTO A LA VERSIÓN ANTERIOR:

  [HC-01] Desajuste de escala temporal:
          poblacion_equivalente ahora usa consumo_anualizado en lugar de
          consumo_total semestral al comparar contra M3_PER_CAPITA_ANUAL.
          Se añade nota en constante y en cálculo final.

  [HC-02] Data leakage en Isolation Forest:
          Cada instancia de IsolationForest ahora recibe features re-escaladas
          con la estadística de su propio cluster (StandardScaler local),
          en lugar de un slice del escalado global. El escalado global solo
          se usa para K-Means (correcto) y Regresión Lineal.

  [HC-03] Rama inalcanzable en clasificar_riesgo():
          Se añade guardia explícita cuando bajo_reporte >= alto_reporte
          (umbrales degenerados por pocos datos). En ese caso la segunda
          condición usa el valor medio como punto de corte de emergencia.

  [HC-04] total_reportes = 0 indistinguible de dato ausente:
          clasificar_riesgo() ahora consulta tiene_reportes_2019 para
          diferenciar "nadie reportó" de "no hubo match en el join".
          Las colonias sin datos de reportes no se clasifican como HUACHICOL
          sino como SOSPECHOSO (Exceso Detectado por IA).

  [MD-03] Gráficas en bloque try/except independiente:
          Un fallo de matplotlib no cancela un pipeline que completó
          correctamente el entrenamiento y exportó el CSV.
"""

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ==============================================================================
# SECCIÓN 0: CONFIGURACIÓN DE RUTAS Y CONSTANTES DEL MODELO
# ==============================================================================

BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
GRAFICAS_DIR = BASE_DIR / "graficas_reporte"
DATA_PATH    = DATA_DIR / "dataset_maestro_colonia_final.csv"   # Entrada: producida por etl_pipeline.py
OUTPUT_PATH  = DATA_DIR / "resultados_finales_IA.csv"           # Salida: consumida por main.py

DATA_DIR.mkdir(exist_ok=True)
GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark de consumo per cápita diario de la CDMX (fuente: SACMEX 2019).
# Se usa para calcular cuántos habitantes podrían abastecerse con el exceso
# de consumo anual recuperable de cada colonia anómala.
LITROS_PER_CAPITA_DIA = 366.0
M3_PER_CAPITA_ANUAL   = (LITROS_PER_CAPITA_DIA * 365) / 1000   # → 133.59 m³/hab/año

# Features separadas en constantes para que main.py pueda importarlas
# sin necesidad de reimplementar la lógica de selección de variables.
FEATURES_CLUSTER   = ["consumo_per_capita", "densidad_poblacional", "uso_suelo_num", "idsm"]
FEATURES_REGRESION = ["pob", "densidad_poblacional", "superficie_km2_calculada",
                      "idsm", "uso_suelo_num"]

# Isolation Forest usa exceso_consumo además del perfil urbano.
# Esta feature solo existe DESPUÉS de que estimar_consumo_base() corra — por
# eso Isolation Forest se ejecuta en el paso 3, no antes.
FEATURES_ISO       = ["consumo_per_capita", "densidad_poblacional", "uso_suelo_num",
                      "idsm", "exceso_consumo"]


# ==============================================================================
# SECCIÓN 1: CARGA Y PREPARACIÓN DEL DATASET
# ==============================================================================

def cargar_y_preparar() -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """
    Lee el dataset maestro generado por etl_pipeline.py y escala las features
    que usará K-Means con un StandardScaler global.

    NOTA DE ESCALADO — por qué global aquí y local en Isolation Forest:
        K-Means necesita que todos los puntos estén en la misma referencia de
        escala para comparar distancias entre clusters correctamente.
        Isolation Forest en cambio compara colonias dentro de su propio cluster,
        por lo que requiere un escalado local para no heredar el sesgo de escala
        global entre grupos de distintas magnitudes.

    Devuelve:
        df_model  — DataFrame limpio, sin NaN en las features del modelo
        X_scaled  — matriz numpy escalada globalmente (solo para K-Means)
    """
    if not DATA_PATH.exists():
        print(f"[ERROR] No se encontró {DATA_PATH}. Ejecuta etl_pipeline.py primero.")
        return None, None

    df = pd.read_csv(DATA_PATH)

    # Verificar columnas nuevas del ETL mejorado — sin ellas algunos pasos degradan
    _advertir_columnas_faltantes(df)

    df_model = df.dropna(subset=FEATURES_CLUSTER).copy()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[FEATURES_CLUSTER])

    print(f"[CARGA] {len(df_model)} colonias listas para el pipeline.")
    return df_model, X_scaled


def _advertir_columnas_faltantes(df: pd.DataFrame) -> None:
    """
    Verifica que el CSV contiene las columnas producidas por etl_pipeline.py
    mejorado. Un CSV desactualizado no rompe el pipeline, pero degrada los
    resultados de escala temporal y la distinción de reportes ausentes vs cero.
    """
    columnas_nuevas = {
        "consumo_anualizado"            : "[HC-01] Ejecuta etl_pipeline.py actualizado para comparar contra benchmarks anuales.",
        "consumo_per_capita_anualizado" : "[HC-01] Necesario para poblacion_equivalente sin sesgo semestral.",
        "tiene_reportes_2019"           : "[HC-04] Sin esta columna no se puede distinguir '0 reportes reales' de 'dato ausente'.",
        "bimestres_cubiertos"           : "Informativo — número de bimestres disponibles por colonia.",
    }
    for col, msg in columnas_nuevas.items():
        if col not in df.columns:
            print(f"  ⚠ Columna '{col}' no encontrada en el CSV. {msg}")


# ==============================================================================
# SECCIÓN 2: SEGMENTACIÓN POR PERFIL URBANO (K-MEANS)
# ==============================================================================

def ejecutar_clustering(df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    """
    Agrupa las colonias en 4 perfiles urbanos usando K-Means con escalado global.

    K=4 se eligió por interpretabilidad operativa, no solo por la curva del codo.
    Los clusters corresponden a perfiles reales de la CDMX:
      · Residencial de bajo consumo
      · Residencial / mixto de consumo medio
      · Comercial / servicios de alto consumo
      · Industrial o atípico (outlier legítimo, ej. Industrial Vallejo)

    Un cluster de tamaño 1 es un resultado válido: el modelo aisló un outlier
    real. El contamination de Isolation Forest se ajusta dinámicamente para
    manejar estos casos sin que sklearn arroje error.

    Genera el archivo graficas_reporte/metodo_codo.png como evidencia visual
    de la elección de K.
    """
    print("\n[K-MEANS] Iniciando segmentación por perfil urbano...")

    # Método del codo — calcula inercia para K=1..10 y grafica el punto de inflexión
    inercias = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inercias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, 11), inercias, marker="o", color="#06b6d4", linewidth=2)
    ax.axvline(x=4, color="red", linestyle="--", linewidth=1.5, label="K elegido = 4")
    ax.set_xlabel("Número de clusters (K)")
    ax.set_ylabel("Inercia")
    ax.set_title("Método del Codo — Selección de K")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "metodo_codo.png", dpi=150)
    plt.close(fig)

    # Modelo final con los hiperparámetros definidos
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster_perfil"] = kmeans.fit_predict(X_scaled)

    print(f"[K-MEANS] Clusters generados: {kmeans.n_clusters}")
    conteos = df["cluster_perfil"].value_counts().sort_index()
    print(conteos.to_string())

    # Advertir clusters pequeños — el contamination de Isolation Forest
    # se ajustará dinámicamente en detectar_anomalias() para no fallar
    clusters_pequenos = conteos[conteos < 10]
    if not clusters_pequenos.empty:
        print(
            f"  ⚠ Clusters con menos de 10 colonias: "
            f"{clusters_pequenos.to_dict()} — contamination se ajustará dinámicamente."
        )

    return df


# ==============================================================================
# SECCIÓN 3: LÍNEA BASE DE CONSUMO ESPERADO (REGRESIÓN LINEAL)
# ==============================================================================

def estimar_consumo_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entrena una Regresión Lineal Múltiple para estimar cuánto debería consumir
    cada colonia dado su perfil demográfico y uso de suelo.

    Variable dependiente:   consumo_total (m³ facturados — primer semestre 2019)
    Variables predictoras:  población, densidad, superficie, IDSM, uso de suelo

    Por qué R² bajo no es un fallo:
        El consumo hídrico urbano depende de actividad económica informal,
        hábitos culturales y estado de la red — variables no disponibles en
        datos públicos. La regresión no busca alta precisión predictiva, sino
        una línea base estadística para que exceso_consumo capture las
        desviaciones que Isolation Forest confirmará o descartará.

    NOTA DE ESCALA [HC-01]:
        exceso_consumo queda en escala semestral (igual que consumo_total).
        Para métricas anuales (poblacion_equivalente, impacto financiero)
        se multiplica por 2 en el orquestador — ver paso HC-01.

    Genera:
        consumo_esperado — predicción de la regresión (m³ semestral)
        exceso_consumo   — diferencia real vs esperado (positivo = consume de más)
    """
    print("\n[REGRESIÓN] Entrenando modelo de consumo esperado...")

    # Ajuste defensivo: si alguna feature falta en el CSV, se omite sin romper el pipeline
    features_disponibles = [f for f in FEATURES_REGRESION if f in df.columns]
    features_faltantes   = [f for f in FEATURES_REGRESION if f not in df.columns]
    if features_faltantes:
        print(f"  ⚠ Features no disponibles para regresión: {features_faltantes} — se omiten.")

    X = df[features_disponibles]
    y = df["consumo_total"]

    modelo = LinearRegression()
    modelo.fit(X, y)

    df["consumo_esperado"] = modelo.predict(X)
    df["exceso_consumo"]   = df["consumo_total"] - df["consumo_esperado"]

    r2  = r2_score(y, df["consumo_esperado"])
    mae = mean_absolute_error(y, df["consumo_esperado"])

    print(f"  R²  : {r2:.4f}  (moderado/bajo esperado — ver nota técnica)")
    print(f"  MAE : {mae:,.2f} m³ (escala semestral)")
    print("\n  Coeficientes:")
    for var, coef in zip(features_disponibles, modelo.coef_):
        print(f"    {var:<35} {coef:+.4f}")

    return df


# ==============================================================================
# SECCIÓN 4: DETECCIÓN DE ANOMALÍAS POR CLUSTER (ISOLATION FOREST)
# ==============================================================================

def detectar_anomalias(df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    """
    Aplica Isolation Forest de forma independiente dentro de cada cluster.

    Por qué por cluster y no sobre el dataset completo:
        Si se aplicara globalmente, colonias residenciales de bajo consumo
        quedarían clasificadas como normales en comparación con las industriales,
        aunque sean atípicas dentro de su propio perfil. Al aplicarlo por cluster,
        cada colonia se compara exclusivamente contra sus similares.

    [HC-02] ESCALADO LOCAL POR CLUSTER:
        Cada cluster re-escala sus features con un StandardScaler propio antes
        de entrenar Isolation Forest. Heredar el escalado global introduciría
        data leakage: la distribución de un cluster afectaría los scores
        de otro cluster a través de la media/std global.

    [FIX-ISO-01] CONTAMINATION DINÁMICO:
        Un valor fijo de contamination=0.15 falla con clusters de 1 o 4 colonias.
        Fórmula: max(1/n, min(0.15, 0.49))
        Garantiza al menos 1 anomalía por cluster y nunca supera el 49%.

    [FIX-ISO-02] exceso_consumo COMO FEATURE:
        La versión anterior solo usaba el perfil urbano (FEATURES_CLUSTER).
        Incluir exceso_consumo — la señal directa de desviación estadística
        respecto a la regresión — mejora significativamente la detección.

    Genera:
        es_anomalia    — 1=normal, -1=anómala (convención de scikit-learn)
        anomalia_score — score continuo; más negativo = más anómala
    """
    print("\n[ISOLATION FOREST] Detectando anomalías por cluster (escalado local + exceso_consumo)...")

    # Verificar que exceso_consumo ya existe — requiere que la regresión haya corrido
    features_iso  = [f for f in FEATURES_ISO if f in df.columns]
    faltantes_iso = [f for f in FEATURES_ISO if f not in df.columns]
    if faltantes_iso:
        print(f"  ⚠ Features no disponibles para Isolation Forest: {faltantes_iso} — se omiten.")

    # Inicializar como normal — solo se sobreescribe si el cluster los marca como anómalos
    df["es_anomalia"]    = 1
    df["anomalia_score"] = 0.0

    for cluster_id in sorted(df["cluster_perfil"].unique()):
        mask = df["cluster_perfil"] == cluster_id
        idx  = df.index[mask]
        n    = mask.sum()

        X_cluster_raw = df.loc[idx, features_iso].values

        # [HC-02] Escalado local — estadística exclusiva del cluster
        scaler_local     = StandardScaler()
        X_cluster_scaled = scaler_local.fit_transform(X_cluster_raw)

        # [FIX-ISO-01] Un cluster singleton no puede tener contamination calculado
        # con la fórmula estándar — se marca directamente como anómalo
        if n == 1:
            df.loc[idx, "es_anomalia"]    = -1
            df.loc[idx, "anomalia_score"] = -1.0
            print(f"  Cluster {cluster_id}: 1 anómala de 1 colonia (singleton — marcada directamente)")
            continue

        # max(1/n) garantiza al menos 1 anomalía; 0.49 es el tope que sklearn acepta
        contam = float(max(1 / n, min(0.15, 0.49)))

        iso    = IsolationForest(contamination=contam, random_state=42)
        preds  = iso.fit_predict(X_cluster_scaled)
        scores = iso.decision_function(X_cluster_scaled)

        df.loc[idx, "es_anomalia"]    = preds
        df.loc[idx, "anomalia_score"] = scores

        n_anomalas = (preds == -1).sum()
        print(f"  Cluster {cluster_id}: {n_anomalas} anómalas de {n} colonias "
              f"(contamination={contam:.2f}, escalado local aplicado)")

    total = (df["es_anomalia"] == -1).sum()
    print(f"  Total anomalías detectadas: {total}")

    # Gráfica 1: mapa analítico — consumo per cápita vs reportes ciudadanos
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=df["consumo_per_capita"], y=df["total_reportes"],
        hue=df["es_anomalia"], palette={1: "#0A84FF", -1: "#FF2D55"},
        alpha=0.75, s=80, ax=ax,
    )
    # Limitar ejes al p95 para que los outliers extremos no compriman la nube principal
    x_lim = df["consumo_per_capita"].quantile(0.95)
    y_lim = df["total_reportes"].quantile(0.95)
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)

    # Etiquetar las 6 colonias anómalas de mayor consumo dentro del rango visible
    top_anomalas = df[
        (df["es_anomalia"] == -1) &
        (df["consumo_per_capita"] <= x_lim) &
        (df["total_reportes"] <= y_lim)
    ].nlargest(6, "consumo_per_capita")
    for _, row in top_anomalas.iterrows():
        ax.annotate(
            row["colonia"],
            xy=(row["consumo_per_capita"], row["total_reportes"]),
            xytext=(6, 4), textcoords="offset points",
            fontsize=7.5, color="#8B0000",
        )

    n_fuera = ((df["consumo_per_capita"] > x_lim) | (df["total_reportes"] > y_lim)).sum()
    if n_fuera:
        ax.annotate(
            f"* {n_fuera} punto(s) fuera del rango visible (outliers extremos)",
            xy=(0.01, 0.01), xycoords="axes fraction",
            fontsize=8, color="gray", style="italic",
        )

    ax.set_title("Mapa Analítico de Anomalías — HydroTrace AI", fontsize=13, fontweight="bold")
    ax.set_xlabel("Consumo Per Cápita (m³/hab — semestral)")
    ax.set_ylabel("Total de Reportes Ciudadanos 2019")
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "mapa_anomalias_analitico.png", dpi=150)
    plt.close(fig)

    # Gráfica 2: consumo per cápita vs densidad poblacional por estado de anomalía
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=df["densidad_poblacional"], y=df["consumo_per_capita"],
        hue=df["es_anomalia"], palette={1: "#0A84FF", -1: "#FF2D55"}, ax=ax2,
    )
    ax2.set_xlim(0, df["densidad_poblacional"].quantile(0.95))
    ax2.set_title("Consumo Per Cápita vs Densidad Poblacional")
    ax2.set_xlabel("Densidad Poblacional (hab/km²)")
    ax2.set_ylabel("Consumo Per Cápita (m³/hab — semestral)")
    fig2.tight_layout()
    fig2.savefig(GRAFICAS_DIR / "scatter_consumo_densidad.png", dpi=150)
    plt.close(fig2)

    return df


# ==============================================================================
# SECCIÓN 5: CALIBRACIÓN DE UMBRALES DINÁMICOS
# ==============================================================================

def calibrar_umbrales(df_anomalas: pd.DataFrame) -> dict:
    """
    Calcula los umbrales de diagnóstico SOLO sobre colonias anómalas (es_anomalia == -1).

    Por qué sobre el subconjunto anómalo y no el dataset completo:
        Incluir colonias normales elevaría artificialmente los percentiles y
        reduciría la sensibilidad del diagnóstico. Los percentiles sobre el
        subconjunto de interés reflejan la distribución real de las colonias
        que el sistema debe clasificar.

    Por qué percentiles y no valores fijos:
        Si el dataset se actualiza a otro año o ciudad, un umbral fijo
        como "15 reportes" puede carecer de sentido. Los percentiles se
        recalibran automáticamente con cada ejecución.

    [HC-03] Guardia de umbrales degenerados:
        Con muy pocas colonias anómalas, el p10 y p75 pueden converger o
        invertirse. Si bajo_reporte >= alto_reporte, se usa la mediana como
        punto de corte de emergencia para evitar que HUACHICOL sea inalcanzable.

    Umbrales resultantes:
        alto_reporte    → p75 de reportes en anómalas (umbral CRÍTICO)
        bajo_reporte    → p10 de reportes en anómalas (umbral HUACHICOL)
        alta_falta_agua → p75 de reportes_falta_agua  (umbral DEFICIENCIA)
    """
    alto_reporte    = float(np.percentile(df_anomalas["total_reportes"], 75))
    bajo_reporte    = float(np.percentile(df_anomalas["total_reportes"], 10))
    alta_falta_agua = float(np.percentile(df_anomalas["reportes_falta_agua"], 75))

    # [HC-03] Umbrales degenerados — ocurre con muy pocas colonias anómalas
    if bajo_reporte >= alto_reporte:
        mediana_reportes = float(df_anomalas["total_reportes"].median())
        print(
            f"  ⚠ [HC-03] Umbrales degenerados: bajo_reporte ({bajo_reporte:.0f}) >= "
            f"alto_reporte ({alto_reporte:.0f}). "
            f"Posiblemente hay muy pocas colonias anómalas ({len(df_anomalas)}). "
            f"Usando mediana ({mediana_reportes:.0f}) como punto de corte de emergencia."
        )
        alto_reporte = mediana_reportes
        bajo_reporte = mediana_reportes * 0.5

    umbrales = {
        "alto_reporte"   : alto_reporte,
        "bajo_reporte"   : bajo_reporte,
        "alta_falta_agua": alta_falta_agua,
    }

    print(f"\n[UMBRALES] Calculados sobre {len(df_anomalas)} colonias anómalas:")
    print(f"  CRÍTICO     — reportes >= {umbrales['alto_reporte']:.0f}  (p75)")
    print(f"  HUACHICOL   — reportes <= {umbrales['bajo_reporte']:.0f}  (p10)")
    print(f"  DEFICIENCIA — falta_agua >= {umbrales['alta_falta_agua']:.0f}  (p75)")

    return umbrales


# ==============================================================================
# SECCIÓN 6: DIAGNÓSTICO INTEGRADO — ETIQUETA OPERATIVA POR COLONIA
# ==============================================================================

def clasificar_riesgo(row: pd.Series, umbrales: dict) -> str:
    """
    Asigna una etiqueta operativa combinando tres señales independientes:
      1. es_anomalia        — ¿Isolation Forest la marcó como atípica?
      2. exceso_consumo     — ¿consume más o menos de lo que predice la regresión?
      3. reportes           — ¿cuántas quejas ciudadanas tiene y de qué tipo?
      4. tiene_reportes_2019— ¿participó del sistema de reportes SEGUIAGUA?

    El orden de evaluación importa — se asigna el primer diagnóstico que aplica,
    de mayor a menor gravedad operativa.

    [HC-04] DISTINCIÓN DATO AUSENTE vs CERO REAL:
        Si tiene_reportes_2019 == False, el campo total_reportes = 0 es por
        ausencia de datos en el join, no evidencia real de bajo reporte.
        Una colonia sin datos ciudadanos con exceso detectado se clasifica
        como SOSPECHOSO (Exceso IA), no como HUACHICOL.
        Asumir True por defecto sería peligroso: clasificaría colonias sin
        cobertura del sistema como extractores fraudulentos.

    Categorías de salida:
        CRÍTICO          → anomalía + exceso + muchos reportes (fuga con evidencia ciudadana)
        SOSPECHOSO H     → anomalía + exceso + reportes reales bajos (extracción silenciosa)
        DEFICIENCIA      → consumo bajo + muchas quejas de falta de agua (baja presión)
        SOSPECHOSO IA    → anomalía + exceso, sin datos ciudadanos confirmados
        NORMAL           → ninguna condición anterior se cumple
    """
    exceso = row["exceso_consumo"]

    # [HC-04] Default False: si la columna no existe, no asumir datos ciudadanos válidos
    tiene_datos_ciudadanos = bool(row.get("tiene_reportes_2019", False))

    # Anomalía con exceso Y evidencia ciudadana alta → fuga de red con corroboración
    if row["es_anomalia"] == -1 and exceso > 0 and row["total_reportes"] >= umbrales["alto_reporte"]:
        return "CRÍTICO (Posible Fuga de Red)"

    # Anomalía con exceso, datos ciudadanos confirmados Y reportes bajos → extracción ilícita
    if (row["es_anomalia"] == -1 and exceso > 0
            and tiene_datos_ciudadanos
            and row["total_reportes"] <= umbrales["bajo_reporte"]):
        return "SOSPECHOSO (Posible Huachicol)"

    # Sin anomalía pero consumo bajo Y muchas quejas de falta de agua → desabasto o baja presión
    if exceso < 0 and row["reportes_falta_agua"] >= umbrales["alta_falta_agua"]:
        return "DEFICIENCIA (Posible Baja Presión o Desabasto)"

    # Anomalía con exceso pero sin suficiente señal ciudadana para clasificar con certeza
    if row["es_anomalia"] == -1 and exceso > 0:
        return "SOSPECHOSO (Exceso Detectado por IA)"

    return "NORMAL"


# ==============================================================================
# SECCIÓN 7: ORQUESTADOR PRINCIPAL
# ==============================================================================

def ejecutar_pipeline() -> None:
    """
    Ejecuta los 5 pasos del pipeline en el orden correcto y exporta el CSV final.

    El orden no es arbitrario:
      Paso 1 — Clustering primero: define los grupos de comparación para Isolation Forest.
      Paso 2 — Regresión segundo: genera exceso_consumo, que el diagnóstico necesita
               y que Isolation Forest usa como feature en el paso 3.
      Paso 3 — Isolation Forest: requiere los clusters del paso 1 y exceso_consumo del paso 2.
      Paso 4 — Umbrales: se calculan sobre las anómalas ya identificadas en el paso 3.
      Paso 5 — Diagnóstico: combina señal del modelo con señal ciudadana.

    [MD-03] Las gráficas analíticas se ejecutan en un bloque try/except independiente.
    Un fallo de matplotlib no cancela la exportación del CSV ya completada.
    """
    df_final, X_s = cargar_y_preparar()
    if df_final is None:
        return

    df_final = ejecutar_clustering(df_final, X_s)
    df_final = estimar_consumo_base(df_final)
    df_final = detectar_anomalias(df_final, X_s)

    # Los umbrales se calculan DESPUÉS de detectar anomalías y exclusivamente sobre ellas
    anomalas = df_final[df_final["es_anomalia"] == -1]
    if anomalas.empty:
        print("[ERROR] No se detectaron anomalías. Verifica contamination en IsolationForest.")
        return
    umbrales = calibrar_umbrales(anomalas)

    print("\n[DIAGNÓSTICO] Clasificando colonias...")
    df_final["diagnostico_final"] = df_final.apply(
        lambda r: clasificar_riesgo(r, umbrales), axis=1
    ).astype(str).str.strip()

    # ── Métrica de impacto social [HC-01] ────────────────────────────────────
    # poblacion_equivalente = habitantes que podrían abastecerse un año con el
    # 20% del exceso de consumo anualizado recuperable de la colonia.
    # El 20% representa una estimación conservadora de reducción operativa factible.
    # Se usa consumo_anualizado (×2 del semestral) para estar en la misma escala
    # que M3_PER_CAPITA_ANUAL. Si la columna no existe, se aproxima aquí.
    if "consumo_anualizado" in df_final.columns:
        exceso_base = (
            df_final["consumo_anualizado"] - (df_final["consumo_esperado"] * 2)
        ).clip(lower=0)
        nota_escala = "anualizado (×2)"
    else:
        exceso_base = df_final["exceso_consumo"].clip(lower=0) * 2
        nota_escala = "semestral escalado ×2 (instala ETL mejorado para exactitud)"

    df_final["poblacion_equivalente"] = (
        exceso_base * 0.20 / M3_PER_CAPITA_ANUAL
    ).round(0).astype(int)

    print("\n[VALIDACIÓN] Distribución de diagnósticos:")
    print(df_final["diagnostico_final"].value_counts().to_string())
    print(
        f"\n  Población equivalente total recuperable: "
        f"{df_final['poblacion_equivalente'].sum():,.0f} hab/año "
        f"({nota_escala})"
    )

    # Ordenar por severidad operativa descendente, y dentro del mismo nivel por consumo
    orden_severidad = {
        "CRÍTICO (Posible Fuga de Red)"                  : 0,
        "SOSPECHOSO (Posible Huachicol)"                 : 1,
        "SOSPECHOSO (Exceso Detectado por IA)"           : 1,
        "DEFICIENCIA (Posible Baja Presión o Desabasto)" : 2,
        "NORMAL"                                         : 3,
    }
    df_final["_orden"] = df_final["diagnostico_final"].map(orden_severidad).fillna(4)
    df_final = df_final.sort_values(
        ["_orden", "consumo_total"], ascending=[True, False]
    ).drop(columns="_orden")

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[OK] Pipeline completado → {OUTPUT_PATH}")
    print(f"     {len(df_final)} colonias | {(df_final['es_anomalia'] == -1).sum()} anomalías detectadas")

    # [MD-03] Bloque independiente — un fallo aquí no cancela la exportación del CSV
    try:
        generar_graficas_analiticas(df_final)
    except Exception as exc:
        print(f"\n[ADVERTENCIA] Gráficas no generadas: {exc}")
        print("  El CSV de resultados fue exportado correctamente.")


# ==============================================================================
# SECCIÓN 8: GENERACIÓN DE GRÁFICAS ANALÍTICAS
# ==============================================================================

def generar_graficas_analiticas(df: pd.DataFrame) -> None:
    """
    Genera 5 gráficas analíticas y las guarda en graficas_reporte/.
    Se ejecutan después de la exportación del CSV para no bloquear el pipeline.

    1. heatmap_correlaciones.png      — correlación entre variables numéricas clave
    2. distribucion_diagnosticos.png  — conteo de colonias por diagnóstico
    3. exceso_por_alcaldia.png        — exceso de consumo total agrupado por alcaldía
    4. boxplot_consumo_cluster.png    — distribución de consumo per cápita por cluster
    5. top10_anomalias.png            — ranking de las 10 colonias más anómalas
    """
    print("\n[GRÁFICAS] Generando visualizaciones analíticas...")

    # ── 1. HEATMAP DE CORRELACIONES ──────────────────────────────────────────
    cols_corr = [
        "consumo_per_capita", "densidad_poblacional", "idsm",
        "exceso_consumo", "total_reportes", "reportes_falta_agua",
        "consumo_total", "pob",
    ]
    cols_corr = [c for c in cols_corr if c in df.columns]
    corr = df[cols_corr].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Mapa de Correlaciones — Variables del Modelo", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "heatmap_correlaciones.png", dpi=150)
    plt.close(fig)
    print("  ✓ heatmap_correlaciones.png")

    # ── 2. DISTRIBUCIÓN DE DIAGNÓSTICOS ──────────────────────────────────────
    colores_diag = {
        "CRÍTICO (Posible Fuga de Red)"                  : "#FF2D55",
        "SOSPECHOSO (Posible Huachicol)"                 : "#FF9F0A",
        "SOSPECHOSO (Exceso Detectado por IA)"           : "#FF6B35",
        "DEFICIENCIA (Posible Baja Presión o Desabasto)" : "#30D158",
        "NORMAL"                                         : "#0A84FF",
    }
    conteos   = df["diagnostico_final"].value_counts()
    colores   = [colores_diag.get(d, "#98989D") for d in conteos.index]
    etiquetas = [d.split("(")[0].strip() for d in conteos.index]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(etiquetas, conteos.values, color=colores, edgecolor="white")
    ax.bar_label(bars, padding=4, fontsize=10, fontweight="bold")
    ax.set_xlabel("Número de colonias")
    ax.set_title("Distribución de Diagnósticos por Colonia", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, conteos.max() * 1.15)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "distribucion_diagnosticos.png", dpi=150)
    plt.close(fig)
    print("  ✓ distribucion_diagnosticos.png")

    # ── 3. EXCESO DE CONSUMO POR ALCALDÍA ────────────────────────────────────
    col_exceso   = "consumo_anualizado" if "consumo_anualizado" in df.columns else "exceso_consumo"
    label_exceso = "Exceso Anualizado (miles de m³)" if col_exceso == "consumo_anualizado" \
                   else "Exceso de Consumo (miles de m³)"

    # Solo colonias con exceso positivo — las de deficiencia no aportan al indicador de sobreconsumo
    exceso_alc = (
        df[df["exceso_consumo"] > 0]
        .groupby("alcaldia")["exceso_consumo"]
        .sum()
        .sort_values(ascending=True)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(exceso_alc.index, exceso_alc.values / 1000, color="#FF9F0A", edgecolor="white")
    ax.bar_label(bars, fmt="%.0f k", padding=4, fontsize=8)
    ax.set_xlabel(label_exceso)
    ax.set_title(
        "Exceso de Consumo Total por Alcaldía\n(solo colonias con exceso positivo)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "exceso_por_alcaldia.png", dpi=150)
    plt.close(fig)
    print("  ✓ exceso_por_alcaldia.png")

    # ── 4. BOXPLOT CONSUMO PER CÁPITA POR CLUSTER ────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    grupos = [
        df[df["cluster_perfil"] == k]["consumo_per_capita"].dropna()
        for k in sorted(df["cluster_perfil"].unique())
    ]
    bp = ax.boxplot(grupos, patch_artist=True, notch=False)
    colores_box = ["#0A84FF", "#30D158", "#FF9F0A", "#FF2D55"]
    for patch, color in zip(bp["boxes"], colores_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([f"Cluster {k}" for k in sorted(df["cluster_perfil"].unique())])
    ax.set_ylabel("Consumo Per Cápita (m³/hab — semestral)")
    ax.set_title("Distribución de Consumo Per Cápita por Cluster", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    # El límite superior al p95 evita que outliers extremos aplasten la distribución principal
    ax.set_ylim(0, df["consumo_per_capita"].quantile(0.95))
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "boxplot_consumo_cluster.png", dpi=150)
    plt.close(fig)
    print("  ✓ boxplot_consumo_cluster.png")

    # ── 5. TOP 10 COLONIAS MÁS ANÓMALAS ─────────────────────────────────────
    # nsmallest sobre anomalia_score porque valores más negativos = más anómalos (convención sklearn)
    top10 = (
        df[df["es_anomalia"] == -1]
        .nsmallest(10, "anomalia_score")[["colonia", "anomalia_score", "diagnostico_final"]]
    )
    colores_top = [colores_diag.get(d, "#98989D") for d in top10["diagnostico_final"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        top10["colonia"].str.title(),
        top10["anomalia_score"].abs(),
        color=colores_top, edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    ax.set_xlabel("Score de Anomalía (valor absoluto — mayor = más anómala)")
    ax.set_title("Top 10 Colonias Más Anómalas — HydroTrace AI", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "top10_anomalias.png", dpi=150)
    plt.close(fig)
    print("  ✓ top10_anomalias.png")

    print(f"[GRÁFICAS] 5 gráficas guardadas en {GRAFICAS_DIR}/")


if __name__ == "__main__":
    ejecutar_pipeline()