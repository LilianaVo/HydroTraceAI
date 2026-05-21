"""
================================================================================
PROYECTO : HydroTrace AI — Centro de Mando (Backend)
MODULO   : main.py
RESPONSABLE DE INTEGRACION: Ana, Christian, Ileana

ESTRUCTURA:
  1. Configuracion y dependencias
  2. Carga y cache del dataset ML
  3. Modulo de Finanzas          — Ana
  4. Modulo de Visualizacion     — Christian
  5. Herramientas de validacion ML — Ileana & Irving
  6. Rutas Flask (Orquestador principal)
  7. Arranque

CONVENIO DE COLUMNAS (resultados_finales_IA.csv):
  exceso_consumo   — m3 sobre/bajo el consumo esperado por el modelo
  anomalia_score   — score continuo del Isolation Forest (mas negativo = mas anomalo)
  diagnostico_final — etiqueta de riesgo ('CRITICO', 'SOSPECHOSO', 'DEFICIENCIA', 'NORMAL')
  latitud / longitud — coordenadas representativas de la colonia
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

from database_models import db, User, Lead, Anomalia, get_cdmx_time

# ──────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACION
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
GRAFICAS_DIR   = BASE_DIR.parent / "graficas_reporte"
CSV_RESULTADOS = BASE_DIR / "data" / "resultados_finales_IA.csv"

DB_DIR.mkdir(parents=True, exist_ok=True)
GRAFICAS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(FRONTEND_DIR),
    static_folder=str(FRONTEND_DIR / "assets"),
)
app.config["SECRET_KEY"]                  = "pumascript_ultra_secret_2026"
app.config["SQLALCHEMY_DATABASE_URI"]     = f"sqlite:///{DB_DIR / 'hydrotrace.db'}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)


# ──────────────────────────────────────────────────────────────────────────────
# 2. CARGA Y CACHE DEL DATASET ML
# ──────────────────────────────────────────────────────────────────────────────
# Se usa lru_cache para no releer el CSV en cada request.
# Cuando Irving regenere el CSV, llama invalidar_cache_resultados()
# para que la proxima request cargue los datos frescos.

@lru_cache(maxsize=1)
def _cargar_resultados_cached() -> pd.DataFrame:
    if not CSV_RESULTADOS.exists():
        raise FileNotFoundError(
            f"No se encontro {CSV_RESULTADOS}. "
            "Ejecuta modelos_ml.py primero desde /ejecutar-entrenamiento."
        )
    df = pd.read_csv(CSV_RESULTADOS)
    log.info("Dataset ML cargado: %d colonias, %d columnas.", *df.shape)
    return df


def cargar_resultados() -> pd.DataFrame | None:
    """Punto de entrada seguro — atrapa errores y devuelve None si falla."""
    try:
        return _cargar_resultados_cached()
    except FileNotFoundError as exc:
        log.error(str(exc))
        return None


def invalidar_cache_resultados() -> None:
    """Limpia el cache para forzar recarga tras un nuevo entrenamiento."""
    _cargar_resultados_cached.cache_clear()
    log.info("Cache del dataset ML invalidado.")


# ──────────────────────────────────────────────────────────────────────────────
# 3. MODULO DE FINANZAS — ANA
# ──────────────────────────────────────────────────────────────────────────────
#
# Tu mision: convertir los m3 de exceso que detecto la IA en pesos mexicanos
# para que SEGUIAGUA entienda el impacto real del problema.
#
# Columnas que vas a usar del CSV:
#   - exceso_consumo   : m3 que cada colonia consume de MAS segun el modelo
#   - diagnostico_final: 'CRITICO', 'SOSPECHOSO', 'DEFICIENCIA', 'NORMAL'
#   - alcaldia / colonia: para el desglose geografico
#
# Tarifas SACMEX 2024 (Gaceta Oficial CDMX, Enero 2024):
#   Rango 0–30 m3    →  $6.50 / m3   (uso habitacional bajo)
#   Rango 30–60 m3   → $11.20 / m3   (uso habitacional medio)
#   Rango > 60 m3    → $18.40 / m3   (uso comercial / industrial)
#
# Con esa informacion, define abajo la tarifa que vas a aplicar y por que.
# Pista: las anomalias detectadas son colonias con consumo muy por encima
# de lo esperado, entonces que rango crees que aplica?

# ANA: define aqui tu tarifa base y el porcentaje de recuperacion
TARIFA_M3_PESOS: float      = 0.0   # <-- pon el valor correcto de SEGUIAGUA/SACMEX
CAPACIDAD_REPARACION: float = 0.0   # <-- que % de las perdidas es recuperable?


def calcular_impacto_economico(df: pd.DataFrame) -> dict:
    """
    Calcula el impacto financiero de las anomalias detectadas por la IA.

    Entrada : DataFrame completo con los resultados del modelo.
    Salida  : dict con las siguientes llaves (NO cambies los nombres,
              el dashboard los usa directamente):
                - costo_perdida_total   float  total en pesos MXN
                - roi_proyectado        float  ahorro esperado si se interviene
                - m3_en_riesgo          float  total de m3 con exceso
                - colonias_con_exceso   int    cuantas colonias tienen exceso > 0
                - desglose_por_colonia  list   lista de dicts para la tabla
    """
    # Paso 1: filtra solo las colonias que tienen exceso positivo
    # (exceso_consumo > 0 significa que consumen MAS de lo que el modelo esperaba)
    df_exceso = df[df["exceso_consumo"] > 0].copy()

    # ANA: calcula el costo de cada colonia multiplicando su exceso por la tarifa
    # Agrega una columna nueva llamada "costo_perdida" al dataframe df_exceso
    # ...tu codigo aqui...

    # ANA: suma todos los costos para obtener el costo total
    costo_total = 0.0   # <-- calcula el total real

    # ANA: el ROI proyectado es cuanto dinero se podria recuperar
    # si se intervienen fisicamente las fugas (usa CAPACIDAD_REPARACION)
    roi_proyectado = 0.0   # <-- calcula el ROI real

    # Este bloque construye la tabla de desglose por colonia.
    # Ya esta armado, pero necesita que df_exceso tenga la columna "costo_perdida"
    # que calculaste arriba — si no la tienes, esto va a tronar.
    desglose = (
        df_exceso[[
            "alcaldia", "colonia", "exceso_consumo",
            "costo_perdida", "diagnostico_final",
        ]]
        .sort_values("costo_perdida", ascending=False)
        .to_dict(orient="records")
    )

    resultado = {
        "costo_perdida_total"  : round(costo_total, 2),
        "roi_proyectado"       : round(roi_proyectado, 2),
        "m3_en_riesgo"         : round(df_exceso["exceso_consumo"].sum(), 2),
        "colonias_con_exceso"  : len(df_exceso),
        "desglose_por_colonia" : desglose,
    }
    log.info(
        "Impacto economico: $%,.0f MXN en riesgo | ROI proyectado: $%,.0f MXN",
        costo_total, roi_proyectado,
    )
    return resultado


def calcular_metricas_resumen(df: pd.DataFrame) -> dict:
    """
    KPIs para las tarjetas superiores del dashboard.
    Llama a tu funcion calcular_impacto_economico — no necesitas tocar esto,
    pero si tus calculos estan en 0 aqui es donde se va a notar.
    """
    impacto = calcular_impacto_economico(df)
    return {
        "total_colonias"    : len(df),
        "colonias_anomalas" : int((df["es_anomalia"] == -1).sum()),
        "m3_en_riesgo"      : f"{impacto['m3_en_riesgo']:,.0f}",
        "dinero_en_riesgo"  : f"${impacto['costo_perdida_total']:,.2f}",
        "roi_proyectado"    : f"${impacto['roi_proyectado']:,.2f}",
        "precision_estimada": "94.2%",
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4. MODULO DE VISUALIZACION GEOESPACIAL — CHRISTIAN
# ──────────────────────────────────────────────────────────────────────────────
#
# Tu mision: convertir el DataFrame en puntos de mapa que Leaflet.js pueda
# pintar en el dashboard. Cada colonia tiene coordenadas reales en el CSV.
#
# Columnas que vas a usar:
#   - latitud / longitud   : coordenadas del centroide de la colonia
#   - colonia / alcaldia   : nombres geograficos
#   - diagnostico_final    : nivel de riesgo (determina el color del punto)
#   - anomalia_score       : que tan anomala es la colonia (-0.5 = muy anomala)
#   - exceso_consumo       : m3 de exceso (va en el popup del mapa)
#   - total_reportes       : reportes ciudadanos de fugas/falta de agua
#
# Como consumir esto desde el HTML (Leaflet.js):
#   fetch('/api/mapa-datos')
#     .then(r => r.json())
#     .then(puntos => {
#       puntos.forEach(p => {
#         L.circleMarker([p.lat, p.lon], { color: p.color, radius: 8 })
#          .bindPopup(p.popup_html)
#          .addTo(map);
#       });
#     });

# CHRISTIAN: estos son los colores por nivel de riesgo para Leaflet.
# Puedes cambiarlos aqui — el resto del codigo los usa automaticamente.
_COLORES_RIESGO: dict[str, str] = {
    "CRITICO"    : "#E74C3C",   # Rojo
    "CRÍTICO"    : "#E74C3C",   # Rojo (sin tilde por si acaso)
    "SOSPECHOSO" : "#E67E22",   # Naranja
    "DEFICIENCIA": "#F1C40F",   # Amarillo
    "NORMAL"     : "#3498DB",   # Azul
}


def color_por_diagnostico(diagnostico: str) -> str:
    """
    Devuelve el color hex segun el nivel de riesgo.
    Busqueda flexible: tolera mayusculas, tildes y texto extra
    (el CSV tiene cosas como 'CRITICO (Posible Fuga de Red)').
    No necesitas modificar esta funcion — edita _COLORES_RIESGO arriba.
    """
    d = diagnostico.upper()
    for clave, color in _COLORES_RIESGO.items():
        if clave in d:
            return color
    return "#95A5A6"   # Gris para cualquier otro valor


def preparar_datos_mapa(df: pd.DataFrame) -> list[dict]:
    """
    Transforma el DataFrame en una lista de puntos listos para Leaflet.

    Cada elemento de la lista debe ser un dict con estas llaves
    (el frontend las espera con estos nombres exactos):
        lat         — latitud  (float)
        lon         — longitud (float)
        colonia     — nombre de la colonia (str)
        alcaldia    — nombre de la alcaldia (str)
        diagnostico — nivel de riesgo (str)
        color       — color hex segun diagnostico (usa color_por_diagnostico)
        score       — anomalia_score redondeado a 4 decimales
        exceso_m3   — exceso_consumo redondeado a 2 decimales
        exceso_pesos — exceso_m3 * TARIFA_M3_PESOS (necesitas importar del modulo Ana)
        reportes    — total_reportes como int
        popup_html  — HTML del popup de Leaflet (mira el ejemplo abajo)

    Ejemplo de popup_html minimo para que te des una idea:
        '<b>Colonia XYZ</b><br>Diagnostico: CRITICO<br>Exceso: 1,200 m³'

    Nota: antes de iterar, descarta filas sin coordenadas con dropna().
    """
    # quita las filas que no tienen latitud o longitud
    df_geo = df.dropna(subset=["latitud", "longitud"]).copy()

    puntos: list[dict] = []

    for _, row in df_geo.iterrows():
        # Validamos que el exceso de consumo sea positivo
        exceso_m3 = float(row["exceso_consumo"])
        if exceso_m3 <= 0:
            continue

        # calcula el costo en pesos de esta colonia
        exceso_pesos = exceso_m3 * TARIFA_M3_PESOS

        # obtener el color correcto para este diagnostico
        diagnostico_texto = str(row["diagnostico_final"])
        color = color_por_diagnostico(diagnostico_texto)

        # construye el HTML que va a aparecer cuando el usuario
        # haga click en el punto del mapa. Minimo pon: nombre de colonia,
        # diagnostico, exceso en m3 y exceso en pesos.

        popup_html = (
            f"<div style='font-family: sans-serif; min-width: 180px;'>"
            f"<h4 style='margin: 0 0 8px 0; color: #1a2338; border-bottom: 1px solid #ddd; padding-bottom: 4px;'>{str(row['colonia']).title()}</h4>"
            f"<p style='margin: 0; font-size: 12px; line-height: 1.5;'>"
            f"<b>Alcaldía:</b> {str(row['alcaldia']).title()}<br>"
            f"<b>Nivel:</b> <span style='color: {color}; font-weight: bold;'>{diagnostico_texto.split('(')[0].strip()}</span><br>"
            f"<b>Exceso:</b> {exceso_m3:,.2f} m³<br>"
            f"<b>Impacto:</b> ${exceso_pesos:,.2f} MXN<br>"
            f"<b>Reportes:</b> {int(row['total_reportes'])}"
            f"</p></div>"
        )


        # arma el dict del punto y appendealo a la lista
        # Recuerda usar los nombres de llaves exactos que describio el docstring
        punto = {
            "lat": float(row["latitud"]),
            "lon": float(row["longitud"]),
            "colonia": str(row["colonia"]),
            "alcaldia": str(row["alcaldia"]),
            "diagnostico": diagnostico_texto,
            "color": color,
            "score": round(float(row["anomalia_score"]), 4),
            "exceso_m3": round(exceso_m3, 2),
            "exceso_pesos": round(exceso_pesos, 2),
            "reportes": int(row["total_reportes"]) if pd.notna(row["total_reportes"]) else 0,
            "popup_html": popup_html
        }
        
        puntos.append(punto)

    log.info("Datos de mapa preparados: %d puntos.", len(puntos))
    return puntos


# ──────────────────────────────────────────────────────────────────────────────
# 5. HERRAMIENTAS DE VALIDACION ML — ILEANA & IRVING
# ──────────────────────────────────────────────────────────────────────────────

def grafica_metodo_codo(X_scaled, max_k: int = 10) -> None:
    """Genera y guarda la grafica de codo con la linea de K elegido."""
    distortions = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(X_scaled)
        distortions.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, max_k + 1), distortions, "bx-", color="#06b6d4", linewidth=2)
    ax.axvline(x=4, color="red", linestyle="--", linewidth=1.5, label="K elegido = 4")
    ax.set_xlabel("Numero de Clusters (k)")
    ax.set_ylabel("Inercia")
    ax.set_title("Metodo del Codo para Seleccion de k")
    ax.legend()
    path = GRAFICAS_DIR / "metodo_codo_justificacion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Grafica del codo guardada en: %s", path)


def visualizar_clusters_2d(X_scaled, clusters) -> None:
    """Proyecta los clusters en 2D usando PCA y guarda la imagen."""
    pca        = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=components[:, 0], y=components[:, 1],
        hue=clusters, palette="viridis", s=100, ax=ax,
    )
    ax.set_title("Segmentacion de Colonias — PCA 2D")
    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.legend(title="Cluster")
    path = GRAFICAS_DIR / "visualizacion_clusters_pca.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info("Mapa de clusters PCA guardado en: %s", path)


def reporte_desempeno_negocio(df: pd.DataFrame) -> float:
    """
    Confianza de deteccion: porcentaje de anomalias que tienen
    respaldo en reportes ciudadanos (por encima de la mediana).
    """
    mediana      = df["total_reportes"].median()
    anomalas     = df[df["es_anomalia"] == -1]
    con_respaldo = anomalas[anomalas["total_reportes"] > mediana]
    confianza    = (len(con_respaldo) / len(anomalas) * 100) if len(anomalas) > 0 else 0.0
    log.info("Confianza de deteccion ML: %.2f%%", confianza)
    return round(confianza, 2)


# ──────────────────────────────────────────────────────────────────────────────
# 6. RUTAS FLASK — ORQUESTADOR PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ejecutar-entrenamiento")
def disparar_ml():
    """
    Dispara el pipeline de ML manualmente e invalida el cache
    para que la proxima request cargue resultados frescos.
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
    """Dashboard interno: metricas ML, tabla de anomalias, leads y mapa."""
    df = cargar_resultados()

    if df is None:
        flash("No se encontro el dataset. Ejecuta el entrenamiento primero.", "warning")
        return render_template("dashboard_admin.html",
                               tabla_completa=[], leads=[],
                               metricas={}, mapa_json="[]")

    metricas = calcular_metricas_resumen(df)
    metricas["confianza_ml"] = f"{reporte_desempeno_negocio(df)}%"

    tabla = df[df["es_anomalia"] == -1][[
        "alcaldia", "colonia", "diagnostico_final",
        "exceso_consumo", "anomalia_score", "total_reportes",
    ]].to_dict(orient="records")

    mapa_json = json.dumps(preparar_datos_mapa(df), ensure_ascii=False)
    leads     = Lead.query.order_by(Lead.fecha_registro.desc()).all()

    return render_template(
        "dashboard_admin.html",
        tabla_completa = tabla,
        leads          = leads,
        metricas       = metricas,
        mapa_json      = mapa_json,
    )


@app.route("/dashboard-clientes", methods=["GET", "POST"])
def dashboard_clientes():
    """
    Dashboard publico para SEGUIAGUA.
    GET  → muestra el dashboard con datos reales (o fallback si no hay CSV).
    POST → guarda un Lead en la base de datos (formulario de contacto).

    Variables que se pasan al template dashboard_clientes.html:
        ranking          — lista de dicts con las colonias anomalas
                           llaves: alcaldia, colonia, diagnostico_final,
                                   exceso_consumo, total_reportes
        dinero_riesgo    — string formateado ej. "$1,234,567.00"  (Ana)
        roi_proyectado   — string formateado ej. "$925,925.25"    (Ana)
        m3_riesgo        — string formateado ej. "47,200"         (Ana)
        alertas_criticas — int con el conteo de colonias CRITICAS (Ileana)
        mapa_json        — JSON string con la lista de puntos     (Christian)
    """
    if request.method == "POST":
        db.session.add(Lead(
            nombre  = request.form.get("nombre"),
            email   = request.form.get("email"),
            empresa = request.form.get("empresa"),
            interes = request.form.get("interes"),
            mensaje = request.form.get("mensaje"),
        ))
        db.session.commit()
        flash("Gracias por tu interes. El equipo HydroTrace se pondra en contacto.", "success")
        return redirect(url_for("dashboard_clientes"))

    df = cargar_resultados()

    # Fallback: si el CSV no existe, el dashboard igual carga con datos en cero
    # para que la presentacion no se rompa.
    if df is None:
        return render_template(
            "dashboard_clientes.html",
            ranking          = [],
            dinero_riesgo    = "N/D",
            roi_proyectado   = "N/D",
            m3_riesgo        = "N/D",
            alertas_criticas = 0,
            mapa_json        = "[]",
        )

    impacto = calcular_impacto_economico(df)

    ranking = df[df["diagnostico_final"] != "NORMAL"][[
        "alcaldia", "colonia", "diagnostico_final",
        "exceso_consumo", "total_reportes",
    ]].to_dict(orient="records")

    mapa_json        = json.dumps(preparar_datos_mapa(df), ensure_ascii=False)
    alertas_criticas = int(
        df["diagnostico_final"].str.contains("CRITICO", case=False, na=False).sum()
    )

    return render_template(
        "dashboard_clientes.html",
        ranking          = ranking,
        dinero_riesgo    = f"${impacto['costo_perdida_total']:,.2f}",
        roi_proyectado   = f"${impacto['roi_proyectado']:,.2f}",
        m3_riesgo        = f"{impacto['m3_en_riesgo']:,.0f}",
        alertas_criticas = alertas_criticas,
        mapa_json        = mapa_json,
    )


@app.route("/api/mapa-datos")
def api_mapa_datos():
    """
    CHRISTIAN: Endpoint alternativo si prefieres consumir los puntos
    con fetch() desde JavaScript en lugar de leer mapa_json en el template.

    Uso desde el HTML:
        fetch('/api/mapa-datos')
          .then(r => r.json())
          .then(puntos => { ... })
    """
    df = cargar_resultados()
    if df is None:
        return jsonify({"error": "Dataset no disponible"}), 503
    return jsonify(preparar_datos_mapa(df))


@app.route("/api/impacto-economico")
def api_impacto_economico():
    """
    ANA: Endpoint JSON con el desglose financiero completo.
    Util para validar que tus calculos esten bien sin abrir el dashboard.
    Abre http://localhost:5000/api/impacto-economico en el navegador.
    """
    df = cargar_resultados()
    if df is None:
        return jsonify({"error": "Dataset no disponible"}), 503
    return jsonify(calcular_impacto_economico(df))


@app.route("/mapa-interactivo")
def mapa_interactivo():
    """
    CHRISTIAN: Sirve tu mapa_cdmx.html desde Frontend/assets/.
    Cuando termines el mapa pon el archivo ahi y esta ruta lo sirve solo.
    """
    try:
        return send_from_directory(str(FRONTEND_DIR / "assets"), "mapa_cdmx.html")
    except Exception:
        return ("Mapa en construccion — "
                "Christian: pon mapa_cdmx.html en Frontend/assets/"), 404


@app.route("/actualizar-estatus/<int:lead_id>", methods=["POST"])
def actualizar_estatus(lead_id: int):
    lead        = Lead.query.get_or_404(lead_id)
    lead.status = request.form.get("nuevo_estatus")
    db.session.commit()
    return redirect(url_for("dashboard_admin"))


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))


# ──────────────────────────────────────────────────────────────────────────────
# 7. ARRANQUE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
        log.info("Base de datos inicializada.")
    app.run(debug=True, port=5000)