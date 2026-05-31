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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
# Convierte m³ recuperables a población equivalente abastecida por un AÑO COMPLETO.
# IMPORTANTE [HC-01]: usar siempre con exceso_consumo anualizado, no semestral.
LITROS_PER_CAPITA_DIA = 366.0
M3_PER_CAPITA_ANUAL   = (LITROS_PER_CAPITA_DIA * 365) / 1000   # → 133.59 m³/hab/año

# Features del modelo de clustering y Regresión.
# Separadas en constantes para que main.py pueda leerlas sin reimplementarlas.
FEATURES_CLUSTER    = ["consumo_per_capita", "densidad_poblacional", "uso_suelo_num", "idsm"]
FEATURES_REGRESION  = ["pob", "densidad_poblacional", "superficie_km2_calculada",
                        "idsm", "uso_suelo_num"]

# Features para Isolation Forest: incluye exceso_consumo (desviación vs línea base).
# Se define aquí pero solo se usa DESPUÉS de que estimar_consumo_base() genera
# la columna exceso_consumo. Isolation Forest corre después de la regresión.
FEATURES_ISO        = ["consumo_per_capita", "densidad_poblacional", "uso_suelo_num",
                        "idsm", "exceso_consumo"]


# ──────────────────────────────────────────────────────────────────────────────
# PASO 0 — CARGA Y ESCALADO
# ──────────────────────────────────────────────────────────────────────────────

def cargar_y_preparar() -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """
    Lee el dataset maestro generado por etl_pipeline.py y escala las features
    que usará K-Means.

    NOTA DE ESCALADO:
        El StandardScaler global se usa exclusivamente para K-Means y Regresión
        Lineal, donde comparar magnitudes entre grupos distintos es correcto.
        Para Isolation Forest se aplica un re-escalado local por cluster
        (ver detectar_anomalias), ya que el objetivo es detectar anomalías
        dentro de cada grupo, no entre grupos.

    Devuelve:
        df_model  — DataFrame limpio (sin NaN en features del modelo)
        X_scaled  — matriz numpy escalada globalmente (para K-Means)
    """
    if not DATA_PATH.exists():
        print(f"[ERROR] No se encontró {DATA_PATH}. Ejecuta etl_pipeline.py primero.")
        return None, None

    df = pd.read_csv(DATA_PATH)

    # Validar que las columnas nuevas del ETL mejorado existen
    _advertir_columnas_faltantes(df)

    df_model = df.dropna(subset=FEATURES_CLUSTER).copy()

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_model[FEATURES_CLUSTER])

    print(f"[CARGA] {len(df_model)} colonias listas para el pipeline.")
    return df_model, X_scaled


def _advertir_columnas_faltantes(df: pd.DataFrame) -> None:
    """
    Verifica que el CSV contiene las columnas producidas por etl_pipeline.py
    mejorado. Emite advertencias para columnas nuevas opcionales ausentes.
    Las columnas críticas del modelo base no se verifican aquí — fallarán
    naturalmente en dropna() o en la regresión si no existen.
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


# ──────────────────────────────────────────────────────────────────────────────
# PASO 1 — SEGMENTACIÓN (K-MEANS)
# ──────────────────────────────────────────────────────────────────────────────

def ejecutar_clustering(df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
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
    El escalado usado es el global (correcto para K-Means: se comparan
    colonias entre sí usando las mismas referencias de escala).
    """
    print("\n[K-MEANS] Iniciando segmentación por perfil urbano...")

    # Método del codo: calcula inercia para K=1..10 y guarda la gráfica.
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

    # Modelo final con K=4
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["cluster_perfil"] = kmeans.fit_predict(X_scaled)

    print(f"[K-MEANS] Clusters generados: {kmeans.n_clusters}")
    conteos = df["cluster_perfil"].value_counts().sort_index()
    print(conteos.to_string())

    # Advertir clusters pequeños — contamination se ajusta dinámicamente en detectar_anomalias
    clusters_pequenos = conteos[conteos < 10]
    if not clusters_pequenos.empty:
        print(
            f"  ⚠ Clusters con menos de 10 colonias: "
            f"{clusters_pequenos.to_dict()} — contamination se ajustará dinámicamente."
        )

    return df


# ──────────────────────────────────────────────────────────────────────────────
# PASO 2 — REGRESIÓN LINEAL (LÍNEA BASE DE CONSUMO ESPERADO)
# ──────────────────────────────────────────────────────────────────────────────

def estimar_consumo_base(df: pd.DataFrame) -> pd.DataFrame:
    """
    Entrena una Regresión Lineal Múltiple para estimar cuánto debería consumir
    cada colonia dados sus atributos demográficos y de uso de suelo.

    Variable dependiente : consumo_total (m³ facturados — primer semestre 2019)
    Variables independientes: población, densidad, superficie, IDSM, uso de suelo

    El R² esperado es moderado o bajo — eso no es un fallo del modelo.
    El consumo hídrico urbano depende de actividad económica, comercio informal
    y hábitos culturales que no están en los datos disponibles. Lo importante
    no es predecir con precisión sino tener una línea base estadística que
    permita detectar desviaciones (exceso_consumo) que el Isolation Forest
    confirmará o descartará como anomalías.

    NOTA: exceso_consumo se calcula en la misma escala que consumo_total
    (semestral). Para impacto financiero o comparación anual usar
    exceso_consumo_anualizado (calculado en el paso 5).

    Genera:
        consumo_esperado       — predicción del modelo (m³ semestral)
        exceso_consumo         — diferencia real vs esperado (semestral)
    """
    print("\n[REGRESIÓN] Entrenando modelo de consumo esperado...")

    # Verificar que todas las features de regresión existen
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


# ──────────────────────────────────────────────────────────────────────────────
# PASO 3 — DETECCIÓN DE ANOMALÍAS (ISOLATION FOREST POR CLUSTER)
# ──────────────────────────────────────────────────────────────────────────────

def detectar_anomalias(df: pd.DataFrame, X_scaled: np.ndarray) -> pd.DataFrame:
    """
    Aplica Isolation Forest de forma independiente dentro de cada cluster.

    Por qué por cluster y no sobre todo el dataset:
        Si se aplicara globalmente, las colonias residenciales de bajo consumo
        serían marcadas como normales en comparación con las industriales, aunque
        sean atípicas dentro de su propio perfil. Aplicarlo por cluster garantiza
        que cada colonia se compara contra su grupo de similares.

    [HC-02] ESCALADO LOCAL POR CLUSTER:
        Cada cluster re-escala sus propias features con un StandardScaler local
        antes de pasárselas a Isolation Forest. K-Means sigue usando el escalado
        global (correcto — necesita que todos los puntos estén en la misma
        referencia de escala para comparar distancias entre clusters).

    [FIX-ISO-01] CONTAMINATION DINÁMICO:
        contamination=0.15 fijo rompe con clusters de 1 o 4 colonias (donde
        0.15×1=0.15 se redondea a 0 anomalías, o siempre fuerza exactamente 1).
        Ahora se calcula por cluster: mínimo 1 anomalía garantizada, máximo 15%.
        Fórmula: max(1/n, min(0.15, (n-1)/n))

    [FIX-ISO-02] FEATURES ENRIQUECIDAS CON exceso_consumo:
        La versión anterior usaba solo FEATURES_CLUSTER (perfil urbano).
        Ahora Isolation Forest también recibe exceso_consumo — la desviación
        real vs la línea base de la regresión — que es la señal más directa
        de comportamiento anómalo. Esto mejora significativamente la calidad
        de detección porque el modelo ya sabe cuáles colonias se desvían
        estadísticamente de lo esperado dado su perfil.

    Genera:
        es_anomalia    — 1=normal, -1=anómala (convención de scikit-learn)
        anomalia_score — score continuo; más negativo = más anómala
    """
    print("\n[ISOLATION FOREST] Detectando anomalías por cluster (escalado local + exceso_consumo)...")

    # Verificar que exceso_consumo existe (requiere que estimar_consumo_base() ya corrió)
    features_iso = [f for f in FEATURES_ISO if f in df.columns]
    faltantes_iso = [f for f in FEATURES_ISO if f not in df.columns]
    if faltantes_iso:
        print(f"  ⚠ Features no disponibles para Isolation Forest: {faltantes_iso} — se omiten.")

    df["es_anomalia"]    = 1      # valor por defecto: normal
    df["anomalia_score"] = 0.0

    for cluster_id in sorted(df["cluster_perfil"].unique()):
        mask  = df["cluster_perfil"] == cluster_id
        idx   = df.index[mask]
        n     = mask.sum()

        X_cluster_raw = df.loc[idx, features_iso].values

        # [HC-02] Re-escalar con estadística local del cluster
        scaler_local     = StandardScaler()
        X_cluster_scaled = scaler_local.fit_transform(X_cluster_raw)

        # [FIX-ISO-01] Contamination dinámico — cluster singleton se marca directo
        # porque sklearn no acepta contamination > 0.5 ni = 1.0.
        if n == 1:
            df.loc[idx, "es_anomalia"]    = -1
            df.loc[idx, "anomalia_score"] = -1.0
            print(f"  Cluster {cluster_id}: 1 anómala de 1 colonia (singleton — marcada directamente)")
            continue

        # max(1/n) garantiza al menos 1 anomalía; min(0.15) tope del 15%; nunca > 0.5
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

    # ── Gráfica 1: Mapa analítico ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=df["consumo_per_capita"], y=df["total_reportes"],
        hue=df["es_anomalia"], palette={1: "#0A84FF", -1: "#FF2D55"},
        alpha=0.75, s=80, ax=ax,
    )
    x_lim = df["consumo_per_capita"].quantile(0.95)
    y_lim = df["total_reportes"].quantile(0.95)
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)

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

    # ── Gráfica 2: Consumo per cápita vs densidad ────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────────────
# PASO 4 — UMBRALES DINÁMICOS (solo sobre las colonias anómalas)
# ──────────────────────────────────────────────────────────────────────────────

def calibrar_umbrales(df_anomalas: pd.DataFrame) -> dict:
    """
    Calcula los umbrales de diagnóstico ÚNICAMENTE sobre las colonias que
    Isolation Forest marcó como anómalas (es_anomalia == -1).

    Por qué no sobre el dataset completo:
        Calcular percentiles sobre todas las colonias mezcla normales con anómalas
        y eleva artificialmente los umbrales, reduciendo la sensibilidad.
        Al filtrar solo las anómalas, los percentiles reflejan la distribución
        real del subconjunto de interés.

    Por qué percentiles y no valores fijos:
        Si el dataset se actualiza a un año distinto, un valor fijo como
        "15 reportes" puede no tener ningún sentido. Los percentiles se
        recalibran solos.

    [HC-03] Guardia de umbrales degenerados:
        Si bajo_reporte >= alto_reporte (puede ocurrir con muy pocos datos
        anómalos donde p10 y p75 convergen), se usa la mediana como punto
        de corte de emergencia para evitar que la condición HUACHICOL quede
        inalcanzable. Se emite advertencia explícita.

    Umbrales:
        alto_reporte    — p75 de reportes entre anómalas → umbral CRÍTICO
        bajo_reporte    — p10 de reportes entre anómalas → umbral HUACHICOL
        alta_falta_agua — p75 de reportes_falta_agua     → umbral DEFICIENCIA
    """
    alto_reporte    = float(np.percentile(df_anomalas["total_reportes"], 75))
    bajo_reporte    = float(np.percentile(df_anomalas["total_reportes"], 10))
    alta_falta_agua = float(np.percentile(df_anomalas["reportes_falta_agua"], 75))

    # [HC-03] Detectar y corregir umbrales degenerados
    if bajo_reporte >= alto_reporte:
        mediana_reportes = float(df_anomalas["total_reportes"].median())
        print(
            f"  ⚠ [HC-03] Umbrales degenerados: bajo_reporte ({bajo_reporte:.0f}) >= "
            f"alto_reporte ({alto_reporte:.0f}). "
            f"Posiblemente hay muy pocas colonias anómalas ({len(df_anomalas)}). "
            f"Usando mediana ({mediana_reportes:.0f}) como punto de corte de emergencia."
        )
        alto_reporte = mediana_reportes
        bajo_reporte = mediana_reportes * 0.5  # 50% de la mediana como umbral bajo

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


# ──────────────────────────────────────────────────────────────────────────────
# PASO 5 — DIAGNÓSTICO INTEGRADO
# ──────────────────────────────────────────────────────────────────────────────

def clasificar_riesgo(row: pd.Series, umbrales: dict) -> str:
    """
    Asigna una etiqueta operativa a cada colonia combinando tres señales:
      · es_anomalia        — ¿el Isolation Forest la marcó como atípica?
      · exceso_consumo     — ¿consume más o menos de lo que predice la regresión?
      · reportes           — ¿cuántas quejas ciudadanas tiene?
      · tiene_reportes_2019— ¿la colonia tuvo presencia en el sistema de reportes?

    El orden de las condiciones importa: se evalúan de mayor a menor gravedad
    y se asigna la primera que se cumple.

    [HC-04] DISTINCIÓN DE DATO AUSENTE vs CERO REAL:
        Si tiene_reportes_2019 == False (o la columna no existe), la colonia
        no participó del sistema de reportes SEGUIAGUA en 2019 — su valor
        de total_reportes es 0 por ausencia de datos, no porque genuinamente
        nadie reportó nada. En ese caso NO se clasifica como HUACHICOL
        (que requiere evidencia positiva de bajo reporte) sino como
        SOSPECHOSO (Exceso Detectado por IA).

    CRÍTICO          → anomalía + exceso + muchos reportes (fuga con evidencia ciudadana)
    SOSPECHOSO H     → anomalía + exceso + reportes reales bajos (extracción silenciosa)
    DEFICIENCIA      → consumo bajo + muchas quejas de falta de agua
    SOSPECHOSO IA    → anomalía + exceso, sin datos ciudadanos o reportes intermedios
    NORMAL           → todo lo demás
    """
    exceso = row["exceso_consumo"]

    # [HC-04] La clasificación HUACHICOL requiere que la colonia sí aparezca
    # en el sistema de reportes — de lo contrario el 0 es dato ausente.
    # Default False: si la columna no existe, NO asumir datos ciudadanos válidos.
    # Asumir True sería peligroso: clasificaría colonias sin datos como HUACHICOL.
    tiene_datos_ciudadanos = bool(row.get("tiene_reportes_2019", False))

    if row["es_anomalia"] == -1 and exceso > 0 and row["total_reportes"] >= umbrales["alto_reporte"]:
        return "CRÍTICO (Posible Fuga de Red)"

    if (row["es_anomalia"] == -1 and exceso > 0
            and tiene_datos_ciudadanos
            and row["total_reportes"] <= umbrales["bajo_reporte"]):
        return "SOSPECHOSO (Posible Huachicol)"

    if exceso < 0 and row["reportes_falta_agua"] >= umbrales["alta_falta_agua"]:
        return "DEFICIENCIA (Posible Baja Presión o Desabasto)"

    if row["es_anomalia"] == -1 and exceso > 0:
        return "SOSPECHOSO (Exceso Detectado por IA)"

    return "NORMAL"


# ──────────────────────────────────────────────────────────────────────────────
# ORQUESTADOR PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def ejecutar_pipeline() -> None:
    """
    Ejecuta los 5 pasos en orden y exporta resultados_finales_IA.csv.

    El orden no es arbitrario:
      Clustering primero → define los grupos de comparación para Isolation Forest.
      Regresión segundo  → genera exceso_consumo que el diagnóstico necesita.
      Isolation Forest   → requiere los clusters del paso 1.
      Umbrales al final  → se calculan sobre las anómalas ya identificadas.

    [MD-03] Las gráficas analíticas se generan en un bloque try/except
    independiente. Un fallo de matplotlib (ej. dependencias de sistema)
    no cancela un pipeline que completó correctamente el entrenamiento
    y exportó el CSV.
    """
    df_final, X_s = cargar_y_preparar()
    if df_final is None:
        return

    df_final = ejecutar_clustering(df_final, X_s)
    df_final = estimar_consumo_base(df_final)
    df_final = detectar_anomalias(df_final, X_s)

    # Los umbrales se calculan DESPUÉS de detectar anomalías y SOLO sobre ellas
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
    # Usa consumo_anualizado si está disponible (ETL mejorado) para comparar
    # contra M3_PER_CAPITA_ANUAL en la misma escala temporal.
    # Fallback: consumo_total semestral × 2 calculado aquí si la columna no existe.
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

    # Ordenar por severidad (CRÍTICO primero) y dentro de cada nivel
    # por consumo total descendente
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

    # [MD-03] Gráficas en bloque independiente — un fallo aquí no cancela el pipeline
    try:
        generar_graficas_analiticas(df_final)
    except Exception as exc:
        print(f"\n[ADVERTENCIA] Gráficas no generadas: {exc}")
        print("  El CSV de resultados fue exportado correctamente.")


# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICAS ANALÍTICAS
# ──────────────────────────────────────────────────────────────────────────────

def generar_graficas_analiticas(df: pd.DataFrame) -> None:
    """
    Genera 5 gráficas analíticas y las guarda en graficas_reporte/.

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
    col_exceso = "consumo_anualizado" if "consumo_anualizado" in df.columns else "exceso_consumo"
    label_exceso = "Exceso Anualizado (miles de m³)" if col_exceso == "consumo_anualizado" \
                   else "Exceso de Consumo (miles de m³)"

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
    ax.set_ylim(0, df["consumo_per_capita"].quantile(0.95))
    fig.tight_layout()
    fig.savefig(GRAFICAS_DIR / "boxplot_consumo_cluster.png", dpi=150)
    plt.close(fig)
    print("  ✓ boxplot_consumo_cluster.png")

    # ── 5. TOP 10 COLONIAS MÁS ANÓMALAS ─────────────────────────────────────
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