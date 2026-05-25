"""
================================================================================
HydroTrace AI — Backend Flask
main.py

Autores : Ana, Christian, Ileana, Irving (integración)
Materia : Ciencia de Datos en la Toma de Decisiones en las Organizaciones
Facultad de Ingeniería, UNAM | Ciudad de México, 2026

Responsabilidades de este módulo:
  · Cargar y cachear resultados_finales_IA.csv (salida de modelos_ml.py)
  · Calcular métricas financieras e impacto social desde el CSV
  · Preparar datos geoespaciales para Leaflet.js
  · Exponer dashboards y endpoints JSON para el frontend

Rutas disponibles:
  /                         → landing page
  /dashboard-clientes       → panel público (SEGIAGUA)
  /dashboard-admin          → panel interno (equipo HydroTrace)
  /api/mapa-datos           → puntos GeoJSON para Leaflet
  /api/impacto-economico    → métricas financieras completas
  /api/modelo-negocio       → plan financiero SaaS B2G
  /api/resumen              → KPIs generales
  /api/etl-status           → estado del CSV y última actualización
  /ejecutar-entrenamiento   → dispara modelos_ml.py y refresca el caché
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from flask import (Flask, flash, jsonify, redirect,
                   render_template, request, send_from_directory,
                   session, url_for)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from database_models import db, User, Lead, get_cdmx_time

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hydrotrace")

BASE_DIR       = Path(os.path.abspath(os.path.dirname(__file__)))
FRONTEND_DIR   = BASE_DIR.parent / "Frontend"
DB_DIR         = BASE_DIR.parent / "Backend_app" / "database"
GRAFICAS_DIR   = BASE_DIR / "graficas_reporte"
CSV_RESULTADOS = BASE_DIR / "data" / "resultados_finales_IA.csv"

DB_DIR.mkdir(parents=True, exist_ok=True)
GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR),
    static_folder=str(FRONTEND_DIR / "assets"),
)
app.config["SECRET_KEY"]                     = "pumascript_ultra_secret_2026"
app.config["SQLALCHEMY_DATABASE_URI"]        = f"sqlite:///{DB_DIR / 'hydrotrace.db'}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["JSON_ENSURE_ASCII"]              = False

db.init_app(app)

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL MODELO FINANCIERO
# Fuentes documentadas para la defensa.
# ──────────────────────────────────────────────────────────────────────────────

# Tarifa SACMEX para uso comercial/industrial (Gaceta Oficial CDMX, enero 2024)
TARIFA_M3_PESOS: float = 18.40

# Capacidad de recuperación física estimada del volumen excedente identificado.
# El sistema detecta zonas — cuánto se recupera depende de la intervención de campo.
CAPACIDAD_RECUPERACION: float = 0.20

# Consumo per cápita promedio CDMX: 366 L/hab/día (SACMEX 2019)
# Convierte m³ recuperables en personas que podrían ser abastecidas por un año.
M3_PER_CAPITA_ANUAL: float = (366 * 365) / 1000   # → 133.59 m³/hab/año

# Plan de negocio SaaS B2G — valores del documento de proyecto.
# Son constantes del plan, no se calculan desde el CSV.
SAAS_MENSUAL_MXN:  float = 85_000.0
SAAS_ANUAL_MXN:    float = 900_000.0
OPEX_MENSUAL_MXN:  float = 24_750.0
CAPEX_MXN:         float = 445_000.0

# ──────────────────────────────────────────────────────────────────────────────
# CARGA Y CACHÉ DEL DATASET ML
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _cargar_resultados_cached() -> pd.DataFrame:
    """
    Lee el CSV una sola vez y lo guarda en memoria.
    lru_cache(maxsize=1) evita releer disco en cada request HTTP —
    con 160 filas no importa el tamaño, pero sí el tiempo de I/O acumulado
    en producción. Para recargar tras un nuevo entrenamiento,
    llama invalidar_cache_resultados().
    """
    if not CSV_RESULTADOS.exists():
        raise FileNotFoundError(
            f"No se encontró {CSV_RESULTADOS}. "
            "Ejecuta modelos_ml.py o visita /ejecutar-entrenamiento."
        )
    df = pd.read_csv(CSV_RESULTADOS, encoding='utf-8')
    log.info("Dataset cargado: %d colonias, %d columnas.", *df.shape)
    return df


def cargar_resultados() -> pd.DataFrame | None:
    """Punto de entrada seguro — devuelve None si el CSV no existe."""
    try:
        return _cargar_resultados_cached()
    except FileNotFoundError as exc:
        log.error(str(exc))
        return None


def invalidar_cache_resultados() -> None:
    """Limpia el caché para forzar recarga tras un nuevo entrenamiento."""
    _cargar_resultados_cached.cache_clear()
    log.info("Caché del dataset invalidado.")


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO FINANCIERO
# Calcula el impacto económico del Agua No Contabilizada detectada por la IA.
# ──────────────────────────────────────────────────────────────────────────────

def calcular_impacto_economico(df: pd.DataFrame) -> dict:
    """
    Traduce los m³ de exceso detectados a pesos MXN y a personas abastecibles.

    Solo considera colonias con exceso_consumo > 0 (las que consumen más de lo
    que el modelo estima como normal). Las colonias en DEFICIENCIA tienen exceso
    negativo y no generan costo de pérdida — su problema es desabasto, no exceso.

    Métricas calculadas:
      costo_perdida_total  — exceso total × tarifa SACMEX ($18.40/m³)
      roi_proyectado       — costo_total × 20% (capacidad de recuperación física)
      m3_en_riesgo         — suma de excesos positivos en m³
      poblacion_equivalente— m³ recuperables / 133.59 m³/hab/año
      colonias_con_exceso  — cuántas colonias tienen exceso > 0
      desglose_por_colonia — lista ordenada para la tabla del dashboard
    """
    df_exceso = df[df["exceso_consumo"] > 0].copy()
    df_exceso["costo_perdida"] = df_exceso["exceso_consumo"] * TARIFA_M3_PESOS

    costo_total      = float(df_exceso["costo_perdida"].sum())
    roi_proyectado   = costo_total * CAPACIDAD_RECUPERACION
    m3_exceso_total  = float(df_exceso["exceso_consumo"].sum())

    # Personas que podrían ser abastecidas un año con el volumen recuperable
    poblacion_eq = int(m3_exceso_total * CAPACIDAD_RECUPERACION / M3_PER_CAPITA_ANUAL)

    desglose = (
        df_exceso[[
            "alcaldia", "colonia", "exceso_consumo",
            "costo_perdida", "diagnostico_final",
        ]]
        .sort_values("costo_perdida", ascending=False)
        .to_dict(orient="records")
    )

    log.info(
        "Impacto: $%,.0f MXN en riesgo | ROI: $%,.0f MXN | ~%,d hab/año recuperables",
        costo_total, roi_proyectado, poblacion_eq,
    )

    return {
        "costo_perdida_total"   : round(costo_total, 2),
        "roi_proyectado"        : round(roi_proyectado, 2),
        "m3_en_riesgo"          : round(m3_exceso_total, 2),
        "poblacion_equivalente" : poblacion_eq,
        "colonias_con_exceso"   : len(df_exceso),
        "desglose_por_colonia"  : desglose,
    }


def calcular_modelo_negocio() -> dict:
    """
    Devuelve el plan financiero SaaS B2G del proyecto.

    Estos valores son constantes del plan de negocio documentado —
    no dependen del CSV porque representan la estructura de costos
    e ingresos de HydroTrace AI como empresa, no del dataset de 2019.

    Se exponen como endpoint para que el dashboard los consuma dinámicamente
    en lugar de estar hardcodeados en el HTML.
    """
    utilidad_mensual = SAAS_MENSUAL_MXN - OPEX_MENSUAL_MXN
    payback_meses    = CAPEX_MXN / utilidad_mensual
    margen_operativo = (utilidad_mensual / SAAS_MENSUAL_MXN) * 100

    # Ahorro anual por suscripción anual vs mensual
    ahorro_anual_pct = round(
        (1 - SAAS_ANUAL_MXN / (SAAS_MENSUAL_MXN * 12)) * 100, 1
    )

    return {
        "suscripcion": {
            "mensual_mxn"     : SAAS_MENSUAL_MXN,
            "anual_mxn"       : SAAS_ANUAL_MXN,
            "ahorro_anual_pct": ahorro_anual_pct,
            "recomendada"     : "anual",
        },
        "costos": {
            "capex_mxn"           : CAPEX_MXN,
            "opex_mensual_mxn"    : OPEX_MENSUAL_MXN,
            "utilidad_mensual_mxn": round(utilidad_mensual, 2),
            "margen_operativo_pct": round(margen_operativo, 1),
        },
        "rentabilidad": {
            "payback_meses"      : round(payback_meses, 2),
            "payback_descripcion": f"El sistema recupera la inversión inicial en "
                                   f"{payback_meses:.1f} meses con un cliente activo.",
            "flujo_anual_mxn"    : round(utilidad_mensual * 12, 2),
        },
        "escalabilidad": {
            "clientes_2_utilidad_mxn": round(utilidad_mensual * 2, 2),
            "nota": "El OPEX es mayormente fijo — cada cliente adicional "
                    "incrementa utilidad sin elevar costos proporcionalmente.",
        },
    }


def calcular_metricas_resumen(df: pd.DataFrame) -> dict:
    """KPIs para las tarjetas superiores del dashboard."""
    impacto   = calcular_impacto_economico(df)
    confianza = calcular_confianza_ml(df)

    return {
        "total_colonias"       : len(df),
        "colonias_anomalas"    : int((df["es_anomalia"] == -1).sum()),
        "m3_en_riesgo"         : f"{impacto['m3_en_riesgo']:,.0f}",
        "dinero_en_riesgo"     : f"${impacto['costo_perdida_total']:,.2f}",
        "roi_proyectado"       : f"${impacto['roi_proyectado']:,.2f}",
        "poblacion_equivalente": f"{impacto['poblacion_equivalente']:,}",
        "tasa_confianza"       : f"{confianza}%",
    }


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO DE VISUALIZACIÓN GEOESPACIAL
# Prepara los datos para Leaflet.js en el dashboard.
# ──────────────────────────────────────────────────────────────────────────────

# Colores por nivel de riesgo — se usan en los marcadores del mapa y en los badges.
# Editar aquí afecta automáticamente mapa, popups y dashboard.
_COLORES_RIESGO: dict[str, str] = {
    "CRITICO"    : "#FF2D55",   # Rojo neón   — intervención inmediata
    "CRÍTICO"    : "#FF2D55",
    "SOSPECHOSO" : "#FF9F0A",   # Naranja neón — inspección prioritaria
    "DEFICIENCIA": "#30D158",   # Verde neón   — revisión de red de distribución
    "NORMAL"     : "#0A84FF",   # Azul neón    — sin acción requerida
}


def color_por_diagnostico(diagnostico: str) -> str:
    """
    Busca el color por prefijo para tolerar el texto extra del diagnóstico
    (ej. 'CRÍTICO (Posible Fuga de Red)' → '#FF2D55').
    """
    d = diagnostico.upper()
    for clave, color in _COLORES_RIESGO.items():
        if clave in d:
            return color
    return "#98989D"   # Gris — diagnóstico desconocido o vacío


def preparar_datos_mapa(df: pd.DataFrame) -> list[dict]:
    """
    Convierte el DataFrame en una lista de puntos para Leaflet.js.

    Descarta filas sin coordenadas y colonias NORMAL con exceso negativo
    (no representan ningún riesgo operativo para el mapa).

    Cada punto incluye popup_html listo para bindPopup() en el frontend:
        L.circleMarker([p.lat, p.lon], { color: p.color, radius: 8 })
         .bindPopup(p.popup_html)
         .addTo(map);
    """
    df_geo = df.dropna(subset=["latitud", "longitud"]).copy()
    puntos: list[dict] = []

    for _, row in df_geo.iterrows():
        exceso_m3   = float(row["exceso_consumo"])
        diag        = str(row["diagnostico_final"]).upper()

        # Excluir colonias normales con consumo por debajo del esperado —
        # no representan un riesgo activo y saturarían el mapa de puntos azules.
        if exceso_m3 <= 0 and "DEFICIENCIA" not in diag and "SOSPECHOSO" not in diag and "TICO" not in diag:
            continue

        exceso_pesos      = exceso_m3 * TARIFA_M3_PESOS
        diagnostico_texto = str(row["diagnostico_final"])
        color             = color_por_diagnostico(diagnostico_texto)
        label_corto       = diagnostico_texto.split("(")[0].strip()

        popup_html = (
            f"<div style='font-family: sans-serif; min-width: 200px;'>"
            f"<h4 style='margin: 0 0 8px 0; color: #1a2338; "
            f"border-bottom: 2px solid {color}; padding-bottom: 4px;'>"
            f"{str(row['colonia']).title()}</h4>"
            f"<p style='margin: 0; font-size: 12px; line-height: 1.8;'>"
            f"<b>Alcaldía:</b> {str(row['alcaldia']).title()}<br>"
            f"<b>Nivel:</b> <span style='color:{color}; font-weight:bold;'>{label_corto}</span><br>"
            f"<b>Exceso:</b> {exceso_m3:,.2f} m³<br>"
            f"<b>Impacto:</b> ${exceso_pesos:,.2f} MXN<br>"
            f"<b>Reportes ciudadanos:</b> {int(row['total_reportes'])}<br>"
            f"<b>Score anomalía:</b> {float(row['anomalia_score']):.4f}"
            f"</p></div>"
        )

        puntos.append({
            "lat"        : float(row["latitud"]),
            "lon"        : float(row["longitud"]),
            "colonia"    : str(row["colonia"]),
            "alcaldia"   : str(row["alcaldia"]),
            "diagnostico": diagnostico_texto,
            "color"      : color,
            "score"      : round(float(row["anomalia_score"]), 4),
            "exceso_m3"  : round(exceso_m3, 2),
            "exceso_pesos": round(exceso_pesos, 2),
            "reportes"   : int(row["total_reportes"]) if pd.notna(row["total_reportes"]) else 0,
            "popup_html" : popup_html,
        })

    log.info("Mapa preparado: %d puntos.", len(puntos))
    return puntos


# ──────────────────────────────────────────────────────────────────────────────
# MÓDULO DE VALIDACIÓN ML
# ──────────────────────────────────────────────────────────────────────────────

def calcular_confianza_ml(df: pd.DataFrame) -> float:
    """
    Tasa de confianza del modelo: porcentaje de colonias anómalas que tienen
    respaldo en reportes ciudadanos (por encima de la mediana del dataset).

    Interpretación: si el 70% de las colonias que la IA marcó como anómalas
    también tienen más quejas que el promedio, hay coherencia entre la señal
    estadística y la señal ciudadana — el modelo no está detectando ruido.
    """
    mediana      = df["total_reportes"].median()
    anomalas     = df[df["es_anomalia"] == -1]
    con_respaldo = anomalas[anomalas["total_reportes"] > mediana]
    confianza    = (len(con_respaldo) / len(anomalas) * 100) if len(anomalas) > 0 else 0.0
    log.info("Confianza ML: %.2f%%", confianza)
    return round(confianza, 2)


def grafica_metodo_codo(X_scaled, max_k: int = 10) -> None:
    """Genera y guarda la gráfica del método del codo (para el dashboard admin)."""
    distortions = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        distortions.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, max_k + 1), distortions, "bx-", color="#06b6d4", linewidth=2)
    ax.axvline(x=4, color="red", linestyle="--", linewidth=1.5, label="K elegido = 4")
    ax.set_xlabel("Número de Clusters (k)")
    ax.set_ylabel("Inercia")
    ax.set_title("Método del Codo — Selección de K")
    ax.legend()
    path = GRAFICAS_DIR / "metodo_codo_justificacion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Gráfica del codo guardada: %s", path)


def visualizar_clusters_2d(X_scaled, clusters) -> None:
    """Proyecta los 4 clusters en 2D con PCA y guarda la imagen."""
    pca        = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=components[:, 0], y=components[:, 1],
        hue=clusters, palette="viridis", s=100, ax=ax,
    )
    ax.set_title("Segmentación de Colonias — Proyección PCA 2D")
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.legend(title="Cluster")
    path = GRAFICAS_DIR / "visualizacion_clusters_pca.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Mapa de clusters PCA guardado: %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# RUTAS FLASK
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/contacto", methods=["POST"])
def contacto():
    """
    Recibe el formulario de contacto del index.html,
    guarda el lead y regresa al index mostrando el mensaje de éxito.
    """
    db.session.add(Lead(
        nombre  = request.form.get("nombre", "").strip() or "—",
        email   = request.form.get("email",  "").strip() or "—",
        empresa = request.form.get("empresa","").strip() or "—",
        interes = request.form.get("interes","demo"),
        mensaje = request.form.get("mensaje","").strip() or "—",
    ))
    db.session.commit()
    log.info("Nuevo lead desde index: %s", request.form.get("nombre"))
    return redirect(url_for("home", success=1))


@app.route("/ejecutar-entrenamiento")
def disparar_ml():
    """
    Dispara el pipeline de ML manualmente desde el dashboard admin.
    Invalida el caché para que la siguiente request cargue los resultados frescos.
    """
    try:
        import modelos_ml
        modelos_ml.ejecutar_pipeline()
        invalidar_cache_resultados()
        flash("Modelos entrenados y resultados actualizados.", "success")
        log.info("Pipeline ML ejecutado desde /ejecutar-entrenamiento.")
    except Exception as exc:
        flash(f"Error al entrenar modelos: {exc}", "error")
        log.error("Fallo en /ejecutar-entrenamiento: %s", exc)
    return redirect(url_for("dashboard_admin"))


@app.route("/dashboard-admin")
def dashboard_admin():
    """
    Dashboard interno del equipo HydroTrace.
    Muestra métricas ML, tabla completa de colonias, leads del CRM y mapa.
    """
    df = cargar_resultados()

    try:
        leads       = Lead.query.order_by(Lead.fecha_registro.desc()).all()
        total_leads = len(leads)
    except Exception:
        leads       = []
        total_leads = 0

    # Cobertura: colonias con datos completos del Top 10 por alcaldía (máx 160)
    colonias_en_dataset = len(df) if df is not None else 0
    cobertura_pct       = f"{colonias_en_dataset / 160 * 100:.1f}%" if colonias_en_dataset > 0 else "—"

    stats = {
        "total_colonias" : colonias_en_dataset,
        "total_leads"    : total_leads,
        "cobertura"      : cobertura_pct,
        "poblacion_total": f"{df['pob'].sum():,.0f}" if df is not None and 'pob' in df.columns else "—",
    }

    if df is None:
        flash("No se encontró el dataset. Ejecuta el entrenamiento primero.", "warning")
        return render_template(
            "dashboard_admin.html",
            stats=stats,
            kpis={"dinero_riesgo": "Sin datos", "ahorro_proyectado": "Sin datos", "m3_perdidos": "—"},
            tabla_completa=[],
            leads=leads,
            metricas={"colonias_anomalas": 0, "confianza_ml": "0%"},
            mapa_json="[]",
            per_capita_json="[]",
            top_exceso_json="[]",
        )

    impacto  = calcular_impacto_economico(df)
    metricas = calcular_metricas_resumen(df)
    metricas["confianza_ml"] = f"{calcular_confianza_ml(df)}%"

    kpis = {
        "dinero_riesgo"    : f"${impacto['costo_perdida_total']:,.2f}",
        "ahorro_proyectado": f"${impacto['roi_proyectado']:,.2f}",
        "m3_perdidos"      : f"{impacto['m3_en_riesgo']:,.0f} m³",
        "poblacion_eq"     : f"{impacto['poblacion_equivalente']:,} hab/año",
    }

    cols_tabla = ["alcaldia", "colonia", "diagnostico_final",
                  "exceso_consumo", "anomalia_score", "total_reportes"]
    if "consumo_per_capita" in df.columns:
        cols_tabla.append("consumo_per_capita")
    tabla = df[cols_tabla].to_dict(orient="records")

    mapa_json = json.dumps(preparar_datos_mapa(df), ensure_ascii=False)

    # Top 10 por consumo per cápita — para el bar chart del dashboard
    per_capita_json = "[]"
    if "consumo_per_capita" in df.columns:
        top_pc = (
            df[["colonia", "consumo_per_capita", "diagnostico_final"]]
            .dropna(subset=["consumo_per_capita"])
            .sort_values("consumo_per_capita", ascending=False)
            .head(10)
        )
        per_capita_json = json.dumps(top_pc.to_dict(orient="records"), ensure_ascii=False)

    # Top 10 por exceso absoluto — para el chart de exceso
    top_exceso_json = "[]"
    if "exceso_consumo" in df.columns:
        top_exc = (
            df[["colonia", "exceso_consumo", "diagnostico_final"]]
            .dropna(subset=["exceso_consumo"])
            .sort_values("exceso_consumo", ascending=False)
            .head(10)
        )
        top_exceso_json = json.dumps(top_exc.to_dict(orient="records"), ensure_ascii=False)

    return render_template(
        "dashboard_admin.html",
        stats=stats,
        kpis=kpis,
        tabla_completa=tabla,
        leads=leads,
        metricas=metricas,
        mapa_json=mapa_json,
        per_capita_json=per_capita_json,
        top_exceso_json=top_exceso_json,
    )

@app.route("/dashboard-clientes")
def dashboard_clientes():
    df = cargar_resultados()

    if df is None:
        return render_template(
            "dashboard_clientes.html",
            ranking=[], dinero_riesgo="Sin datos", roi_proyectado="Sin datos",
            m3_riesgo="—", alertas_criticas=0, tasa_confianza=0,
            mapa_json="[]", total_colonias=0, costo_perdida_total=0,
            poblacion_equivalente=0, alcaldias_cubiertas=0,
            clusters_info=[],
        )

    impacto        = calcular_impacto_economico(df)
    tasa_confianza = calcular_confianza_ml(df)
    alertas_criticas = int(
        df["diagnostico_final"].str.contains("TICO", case=False, na=False).sum()
    )
    alcaldias_cubiertas = df["alcaldia"].nunique()

    # Clusters para el widget del sidebar
    nombres_cluster = {0: "Residencial Premium", 1: "Mixto Urbano", 2: "Industrial", 3: "Vulnerable"}
    colores_cluster = {0: "cluster-purple", 1: "cluster-sky", 2: "cluster-amber", 3: "cluster-rose"}
    conteos = df["cluster_perfil"].value_counts().to_dict() if "cluster_perfil" in df.columns else {}
    clusters_info = [
        {"nombre": nombre, "color": colores_cluster[k], "conteo": conteos.get(k, 0)}
        for k, nombre in nombres_cluster.items()
    ]
    
    clusters_json = json.dumps(clusters_info)

    ranking = df[df["diagnostico_final"] != "NORMAL"][[
        "alcaldia", "colonia", "diagnostico_final",
        "exceso_consumo", "total_reportes",
    ]].to_dict(orient="records")

    mapa_json = json.dumps(preparar_datos_mapa(df), ensure_ascii=False)

    return render_template(
        "dashboard_clientes.html",
        ranking              = ranking,
        dinero_riesgo        = f"${impacto['costo_perdida_total']:,.2f}",
        roi_proyectado       = f"${impacto['roi_proyectado']:,.2f}",
        m3_riesgo            = f"{impacto['m3_en_riesgo']:,.0f}",
        alertas_criticas     = alertas_criticas,
        tasa_confianza       = tasa_confianza,
        mapa_json            = mapa_json,
        total_colonias       = len(df),
        costo_perdida_total  = impacto["costo_perdida_total"],
        poblacion_equivalente= impacto["poblacion_equivalente"],
        alcaldias_cubiertas  = alcaldias_cubiertas,
        clusters_info        = clusters_info,
        clusters_json        = clusters_json,
    )
    
# ──────────────────────────────────────────────────────────────────────────────
# ENDPOINTS JSON
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/api/reporte-campo", methods=["POST"])
def api_reporte_campo():
    """Recibe el reporte de inspección de campo de la cuadrilla SEGIAGUA."""
    data = request.get_json()
    lead = Lead(
        nombre  = f"Campo: {data.get('colonia', '—')}",
        email   = "campo@segiagua.cdmx",
        empresa = data.get("alcaldia"),
        interes = data.get("hallazgo"),
        mensaje = f"[{data.get('fecha')}] Diagnóstico IA: {data.get('diagnostico')} | "
                  f"Hallazgo: {data.get('label')} | Notas: {data.get('notas', '—')}",
        status  = "Reporte Campo",
    )
    db.session.add(lead)
    db.session.commit()
    return jsonify({"ok": True, "id": lead.id})

@app.route("/api/inspecciones-calendario")
def api_inspecciones_calendario():
    """
    Genera el calendario de inspecciones programadas automáticamente
    basado en el ranking real del modelo (colonias no-NORMAL),
    ordenadas por prioridad (CRÍTICO → SOSPECHOSO → DEFICIENCIA).
    """
    df = cargar_resultados()
    if df is None:
        return jsonify({}), 503

    COLOR_MAP = {
        "CRÍTICO":     "#FF2D55",
        "SOSPECHOSO":  "#FF9F0A",
        "DEFICIENCIA": "#30D158",
        "NORMAL":      "#0A84FF",
    }

    # Solo colonias que requieren inspección
    df_insp = df[df["diagnostico_final"] != "NORMAL"].copy()

    # Normalizar etiqueta corta del diagnóstico
    def tipo_corto(diag):
        d = str(diag).upper()
        if "TICO" in d:     return "CRÍTICO"
        if "SOSPECHOSO" in d: return "SOSPECHOSO"
        if "DEFICIENCIA" in d: return "DEFICIENCIA"
        return "NORMAL"

    df_insp["tipo"] = df_insp["diagnostico_final"].apply(tipo_corto)

    # Ordenar: CRÍTICO primero, luego SOSPECHOSO, luego DEFICIENCIA
    orden = {"CRÍTICO": 0, "SOSPECHOSO": 1, "DEFICIENCIA": 2}
    df_insp["_orden"] = df_insp["tipo"].map(orden)
    df_insp = df_insp.sort_values("_orden")

    # Distribuir inspecciones: 2 por día hábil, empezando desde hoy
    import datetime
    calendario = {}
    fecha = datetime.date.today()
    colonias = df_insp[["colonia", "alcaldia", "tipo"]].to_dict(orient="records")

    i = 0
    while i < len(colonias):
        # Saltar fines de semana
        if fecha.weekday() < 5:
            lote = colonias[i:i+2]
            key = fecha.strftime("%Y-%m-%d")
            calendario[key] = [
                {
                    "colonia":  r["colonia"].title(),
                    "alcaldia": r["alcaldia"].title(),
                    "tipo":     r["tipo"],
                    "color":    COLOR_MAP.get(r["tipo"], "#98989D"),
                }
                for r in lote
            ]
            i += 2
        fecha += datetime.timedelta(days=1)

    return jsonify(calendario)

@app.route("/api/mapa-datos")
def api_mapa_datos():
    """Puntos georreferenciados para Leaflet.js. Alternativa a leer mapa_json en el template."""
    df = cargar_resultados()
    if df is None:
        return jsonify({"error": "Dataset no disponible"}), 503
    return jsonify(preparar_datos_mapa(df))


@app.route("/api/impacto-economico")
def api_impacto_economico():
    """
    Desglose financiero completo del Agua No Contabilizada detectada.
    Incluye m³ en riesgo, costo en MXN, ROI y población equivalente recuperable.
    """
    df = cargar_resultados()
    if df is None:
        return jsonify({"error": "Dataset no disponible"}), 503
    return jsonify(calcular_impacto_economico(df))


@app.route("/api/modelo-negocio")
def api_modelo_negocio():
    """
    Plan financiero SaaS B2G de HydroTrace AI.
    Incluye estructura de precios, CAPEX, OPEX, utilidad y payback.
    Estos valores son constantes del plan de negocio — no dependen del CSV.
    """
    return jsonify(calcular_modelo_negocio())


@app.route("/api/resumen")
def api_resumen():
    """KPIs generales del modelo: conteos, métricas financieras y confianza ML."""
    df = cargar_resultados()
    if df is None:
        return jsonify({"error": "Dataset no disponible"}), 503

    impacto  = calcular_impacto_economico(df)
    metricas = calcular_metricas_resumen(df)

    conteos = df["diagnostico_final"].str.upper().apply(
        lambda x: "CRITICO"    if "TICO"       in x else
                  "SOSPECHOSO" if "SOSPECHOSO"  in x else
                  "DEFICIENCIA" if "DEFICIENCIA" in x else "NORMAL"
    ).value_counts().to_dict()

    return jsonify({
        "total_colonias"      : len(df),
        "colonias_anomalas"   : int((df["es_anomalia"] == -1).sum()),
        "kpis"                : impacto,
        "metricas"            : metricas,
        "conteos_diagnostico" : conteos,
        "modelo_negocio"      : calcular_modelo_negocio(),
    })


@app.route("/api/etl-status")
def api_etl_status():
    """Estado del pipeline: si el CSV existe, cuándo fue generado y su tamaño."""
    import datetime
    existe    = CSV_RESULTADOS.exists()
    timestamp = None
    shape     = None
    if existe:
        timestamp = datetime.datetime.fromtimestamp(
            CSV_RESULTADOS.stat().st_mtime
        ).strftime("%d %b %Y, %H:%M:%S")
        df = cargar_resultados()
        if df is not None:
            shape = {"filas": len(df), "columnas": len(df.columns)}
    return jsonify({
        "csv_existe": existe,
        "csv_path"  : str(CSV_RESULTADOS),
        "ultima_mod": timestamp,
        "shape"     : shape,
    })


@app.route("/mapa-interactivo")
def mapa_interactivo():
    """Sirve el mapa standalone de Christian desde Frontend/assets/mapa_cdmx.html."""
    try:
        return send_from_directory(str(FRONTEND_DIR / "assets"), "mapa_cdmx.html")
    except Exception:
        return "Mapa en construcción — pon mapa_cdmx.html en Frontend/assets/", 404


@app.route("/graficas/<path:filename>")
def servir_graficas(filename):
    """Sirve las gráficas generadas por modelos_ml.py al dashboard admin."""
    return send_from_directory(GRAFICAS_DIR, filename)


@app.route("/actualizar-estatus/<int:lead_id>", methods=["POST"])
def actualizar_estatus(lead_id: int):
    lead        = Lead.query.get_or_404(lead_id)
    lead.status = request.form.get("nuevo_estatus")
    db.session.commit()
    return redirect(url_for("dashboard_admin"))


@app.route("/api/leads")
def api_leads():
    """Lista de leads del CRM ordenados por fecha de registro."""
    try:
        leads = Lead.query.order_by(Lead.fecha_registro.desc()).all()
        return jsonify([{
            "id"            : l.id,
            "nombre"        : l.nombre,
            "email"         : l.email,
            "empresa"       : l.empresa,
            "interes"       : l.interes,
            "mensaje"       : l.mensaje,
            "status"        : l.status,
            "fecha_registro": l.fecha_registro.strftime("%d %b %Y, %H:%M") if l.fecha_registro else "—",
        } for l in leads])
    except Exception:
        return jsonify([])


@app.route("/actualizar-crm", methods=["POST"])
def actualizar_crm():
    """Registra un nuevo lead desde el formulario de contacto del dashboard."""
    data = request.get_json()
    lead = Lead(
        nombre  = data.get("nombre"),
        email   = data.get("email"),
        empresa = data.get("inst"),
        interes = data.get("interes", "—"),
        mensaje = data.get("notas"),
    )
    db.session.add(lead)
    db.session.commit()
    return jsonify({"id": lead.id, "ok": True})



@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ──────────────────────────────────────────────────────────────────────────────
# ARRANQUE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        log.info("Base de datos inicializada.")
    app.run(debug=True, port=5000)