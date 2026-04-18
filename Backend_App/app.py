# -*- coding: utf-8 -*-
"""
HydroTrace AI — Dashboard Streamlit
Autor: Proyecto Universitario CDMX
Descripción: Aplicación web interactiva para visualizar
             los resultados del análisis de fugas y
             extracción irregular de agua en la CDMX.

Ejecutar con:  streamlit run app.py
"""

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────
# CONFIGURACIÓN GENERAL DE LA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HydroTrace AI · CDMX",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Estilo CSS extra para una presentación más limpia
st.markdown(
    """
    <style>
        .block-container { padding-top: 2rem; }
        .metric-card { background:#0e3d6b; border-radius:10px; padding:1rem; }
        .riesgo-alto   { color:#ff4b4b; font-weight:bold; }
        .riesgo-medio  { color:#ffa500; font-weight:bold; }
        .riesgo-bajo   { color:#21c55d; font-weight:bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# CARGA DE DATOS (con caché para rendimiento)
# ─────────────────────────────────────────────
DATA_PATH = os.path.join("data", "resultados_anomalias.csv")

@st.cache_data
def cargar_datos(path: str) -> pd.DataFrame:
    """Carga el CSV de resultados y valida columnas mínimas."""
    df = pd.read_csv(path, encoding="utf-8")
    return df

# Verificar que el archivo exista
if not os.path.exists(DATA_PATH):
    st.error(
        f"⚠️ No se encontró `{DATA_PATH}`. "
        "Ejecuta primero `etl_pipeline.py` y luego `modelos_ml.py`."
    )
    st.stop()

df = cargar_datos(DATA_PATH)

# Mapeo de colores para nivel de riesgo
COLORES_RIESGO = {"Alto": "#ff4b4b", "Medio": "#ffa500", "Bajo": "#21c55d"}


# ─────────────────────────────────────────────
# HEADER PRINCIPAL
# ─────────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 8])
with col_logo:
    st.markdown("# 💧")
with col_titulo:
    st.title("HydroTrace AI")
    st.caption("Detector de Fugas y Extracción Irregular de Agua · 16 Alcaldías CDMX")

st.divider()


# ─────────────────────────────────────────────
# PESTAÑAS PRINCIPALES
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(
    ["🌐 Visión General", "🗺️ Mapa de Riesgo", "💰 Simulador ROI"]
)


# ════════════════════════════════════════════
# TAB 1 — VISIÓN GENERAL
# ════════════════════════════════════════════
with tab1:

    # ── Pitch comercial ───────────────────────
    st.subheader("El Problema del Agua en la CDMX")
    st.markdown(
        """
        > *"La Ciudad de México pierde hasta el **40 % de su agua potable** antes de
        > llegar a los hogares. Una parte se debe a infraestructura deteriorada, pero
        > una fracción significativa corresponde a **extracción clandestina** —
        > el llamado 'huachicol de agua'— que afecta de manera desigual a las
        > 16 alcaldías y pone en riesgo el suministro de millones de habitantes."*

        **HydroTrace AI** analiza datos históricos de consumo, densidad poblacional
        y reportes ciudadanos para identificar alcaldías con patrones anómalos,
        priorizar intervenciones y estimar el ahorro potencial de cada reparación.
        """
    )

    st.divider()

    # ── KPIs clave ────────────────────────────
    st.subheader("📊 Métricas Clave del Análisis")

    total_alcaldias   = len(df)
    alcaldias_alto    = (df["Riesgo"] == "Alto").sum()
    alcaldias_medio   = (df["Riesgo"] == "Medio").sum()
    alcaldias_anomala = df["es_anomalia"].sum() if "es_anomalia" in df.columns else 0

    consumo_promedio  = df["consumo_per_capita"].mean() if "consumo_per_capita" in df.columns else 0
    desviacion_prom   = df["desviacion_consumo"].mean() if "desviacion_consumo" in df.columns else 0

    # Fila de métricas principales
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        label="💧 Alcaldías Analizadas",
        value=total_alcaldias,
        help="Total de alcaldías de la CDMX en el modelo",
    )
    c2.metric(
        label="🔴 Riesgo ALTO",
        value=alcaldias_alto,
        delta=f"{alcaldias_alto/total_alcaldias*100:.0f}% del total",
        delta_color="inverse",
    )
    c3.metric(
        label="🟠 Riesgo MEDIO",
        value=alcaldias_medio,
        delta=f"{alcaldias_medio/total_alcaldias*100:.0f}% del total",
        delta_color="off",
    )
    c4.metric(
        label="⚠️ Anomalías (Isolation Forest)",
        value=int(alcaldias_anomala),
        help="Alcaldías con consumo estadísticamente atípico",
    )

    st.divider()

    # Fila secundaria de métricas de consumo
    c5, c6, c7 = st.columns(3)
    c5.metric(
        label="📈 Consumo Per Cápita Prom.",
        value=f"{consumo_promedio:,.2f}",
        help="Unidades dependen del CSV fuente (m³ o litros por habitante)",
    )
    c6.metric(
        label="📉 Desviación Promedio vs. Esperado",
        value=f"{desviacion_prom:+,.2f}",
        help="Diferencia entre consumo real y el teórico calculado por regresión",
        delta_color="inverse",
    )
    c7.metric(
        label="🌊 Pérdida Estimada CDMX",
        value="~40 %",
        help="Porcentaje de agua potable perdida antes de llegar a los hogares (SACMEX)",
    )

    # ── Tabla resumen ─────────────────────────
    st.subheader("📋 Resumen por Alcaldía")
    cols_tabla = [c for c in ["alcaldia", "Riesgo", "consumo_per_capita",
                               "desviacion_consumo", "cluster_nombre",
                               "score_anomalia"] if c in df.columns]
    st.dataframe(
        df[cols_tabla].sort_values("Riesgo").reset_index(drop=True),
        use_container_width=True,
        height=420,
    )


# ════════════════════════════════════════════
# TAB 2 — MAPA DE RIESGO
# ════════════════════════════════════════════
with tab2:

    st.subheader("🗺️ Top 10 Alcaldías con Mayor Anomalía Detectada")
    st.caption(
        "Ordenado por **score de Isolation Forest** (más negativo = más anómalo). "
        "El color indica el nivel de riesgo asignado."
    )

    # Top 10 por score de anomalía (más anómalas primero)
    if "score_anomalia" in df.columns:
        df_top10 = (
            df.nsmallest(10, "score_anomalia")
              .reset_index(drop=True)
        )
    else:
        # Fallback: ordenar por consumo_per_capita descendente
        df_top10 = df.nlargest(10, "consumo_per_capita").reset_index(drop=True)

    # Asignar color por nivel de riesgo
    df_top10["color"] = df_top10["Riesgo"].map(COLORES_RIESGO)

    # ── Gráfico de barras horizontal ─────────
    fig_barras = px.bar(
        df_top10,
        x="score_anomalia" if "score_anomalia" in df_top10.columns else "consumo_per_capita",
        y="alcaldia",
        color="Riesgo",
        color_discrete_map=COLORES_RIESGO,
        orientation="h",
        text="Riesgo",
        title="Score de Anomalía por Alcaldía (Isolation Forest)",
        labels={
            "score_anomalia"     : "Score de Anomalía",
            "consumo_per_capita" : "Consumo Per Cápita",
            "alcaldia"           : "Alcaldía",
        },
    )
    fig_barras.update_traces(textposition="outside")
    fig_barras.update_layout(
        yaxis={"categoryorder": "total ascending"},
        height=480,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
    )
    st.plotly_chart(fig_barras, use_container_width=True)

    # ── Gráfico de dispersión consumo vs densidad ──
    st.subheader("📍 Consumo Per Cápita vs. Densidad Poblacional")
    st.caption("El tamaño del punto representa el total de reportes ciudadanos.")

    if all(c in df.columns for c in ["consumo_per_capita", "densidad_poblacional"]):
        tam = df["total_reportes"] if "total_reportes" in df.columns else None

        fig_scatter = px.scatter(
            df,
            x="densidad_poblacional",
            y="consumo_per_capita",
            color="Riesgo",
            color_discrete_map=COLORES_RIESGO,
            size="total_reportes" if "total_reportes" in df.columns else None,
            size_max=40,
            text="alcaldia",
            hover_data=[c for c in ["cluster_nombre", "score_anomalia",
                                     "desviacion_consumo"] if c in df.columns],
            title="Distribución de Alcaldías: Consumo vs. Densidad",
            labels={
                "densidad_poblacional": "Densidad Poblacional (hab/km²)",
                "consumo_per_capita"  : "Consumo Per Cápita",
                "alcaldia"            : "Alcaldía",
            },
        )
        fig_scatter.update_traces(textposition="top center", textfont_size=9)
        fig_scatter.update_layout(
            height=500,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Distribución de clusters ──────────────
    if "cluster_nombre" in df.columns:
        st.subheader("🔵 Distribución de Clusters (K-Means)")
        conteo_clusters = df["cluster_nombre"].value_counts().reset_index()
        conteo_clusters.columns = ["Cluster", "Cantidad"]

        fig_pie = px.pie(
            conteo_clusters,
            names="Cluster",
            values="Cantidad",
            title="Agrupamiento de Alcaldías por K-Means",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.4,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#FAFAFA",
        )
        st.plotly_chart(fig_pie, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 — SIMULADOR ROI
# ════════════════════════════════════════════
with tab3:

    st.subheader("💰 Simulador de Retorno de Inversión (ROI)")
    st.markdown(
        """
        Estima el **ahorro económico y volumétrico** al reparar un número
        determinado de fugas o conexiones clandestinas detectadas por HydroTrace AI.
        Ajusta los parámetros con los sliders y el cálculo se actualiza en tiempo real.
        """
    )

    st.divider()

    # ── Parámetros del simulador ───────────────
    col_izq, col_der = st.columns([1, 1])

    with col_izq:
        fugas_reparar = st.slider(
            "🔧 Número de fugas / conexiones a reparar",
            min_value=1,
            max_value=500,
            value=50,
            step=5,
            help="Cada 'fuga' representa una conexión ilegal o daño en red primaria",
        )

        costo_reparacion_unitario = st.slider(
            "🔨 Costo promedio por reparación (MXN)",
            min_value=5_000,
            max_value=200_000,
            value=30_000,
            step=5_000,
            format="$%d",
            help="Costo unitario estimado de reparación / sellado de conexión",
        )

        litros_fuga_dia = st.slider(
            "💧 Litros perdidos por fuga al día",
            min_value=500,
            max_value=50_000,
            value=8_000,
            step=500,
            help="Volumen diario promedio que escapa por cada punto de fuga",
        )

    with col_der:
        precio_m3 = st.slider(
            "💲 Precio del m³ de agua (MXN)",
            min_value=5,
            max_value=100,
            value=20,
            step=1,
            help="Tarifa o costo de producción del m³ de agua potable",
        )

        horizon_anios = st.slider(
            "📅 Horizonte de evaluación (años)",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
        )

    st.divider()

    # ── Cálculos ROI ──────────────────────────
    # Ahorro volumétrico
    litros_recuperados_dia  = fugas_reparar * litros_fuga_dia
    m3_recuperados_dia      = litros_recuperados_dia / 1_000
    m3_recuperados_anio     = m3_recuperados_dia * 365
    m3_recuperados_total    = m3_recuperados_anio * horizon_anios

    # Ahorro económico
    ahorro_anio_mxn   = m3_recuperados_anio * precio_m3
    ahorro_total_mxn  = ahorro_anio_mxn * horizon_anios
    inversion_total   = fugas_reparar * costo_reparacion_unitario
    roi_neto          = ahorro_total_mxn - inversion_total

    # Tiempo de recuperación (payback)
    payback_anios = inversion_total / ahorro_anio_mxn if ahorro_anio_mxn > 0 else float("inf")

    # ── KPIs ROI ──────────────────────────────
    r1, r2, r3, r4 = st.columns(4)

    r1.metric(
        label="💧 Litros recuperados/día",
        value=f"{litros_recuperados_dia:,.0f} L",
    )
    r2.metric(
        label=f"💵 Ahorro anual estimado",
        value=f"${ahorro_anio_mxn/1_000_000:.2f} M",
        help="Millones de pesos MXN por año",
    )
    r3.metric(
        label=f"📈 Ahorro total ({horizon_anios} años)",
        value=f"${ahorro_total_mxn/1_000_000:.2f} M",
    )
    r4.metric(
        label="🏦 ROI Neto (ahorro − inversión)",
        value=f"${roi_neto/1_000_000:.2f} M",
        delta="Positivo ✅" if roi_neto > 0 else "Negativo ⚠️",
        delta_color="normal" if roi_neto > 0 else "inverse",
    )

    st.metric(
        label="⏱️ Tiempo de Recuperación de Inversión (Payback)",
        value=f"{payback_anios:.1f} años" if payback_anios < 100 else "No recuperable",
    )

    st.divider()

    # ── Gráfico de flujo acumulado ─────────────
    st.subheader("📈 Proyección de Ahorro Acumulado vs. Inversión")

    anios_eje     = list(range(0, horizon_anios + 1))
    ahorro_acum   = [min(i * ahorro_anio_mxn, ahorro_total_mxn) for i in anios_eje]
    inversion_acum = [inversion_total] * len(anios_eje)

    fig_roi = go.Figure()
    fig_roi.add_trace(go.Scatter(
        x=anios_eje, y=[v / 1_000_000 for v in ahorro_acum],
        mode="lines+markers",
        name="Ahorro Acumulado",
        line=dict(color="#21c55d", width=3),
        fill="tozeroy",
        fillcolor="rgba(33,197,93,0.15)",
    ))
    fig_roi.add_trace(go.Scatter(
        x=anios_eje, y=[v / 1_000_000 for v in inversion_acum],
        mode="lines",
        name="Inversión Total",
        line=dict(color="#ff4b4b", width=2, dash="dash"),
    ))
    fig_roi.update_layout(
        xaxis_title="Años",
        yaxis_title="Millones de MXN",
        legend=dict(orientation="h", y=1.1),
        height=380,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
    )
    st.plotly_chart(fig_roi, use_container_width=True)

    # Desglose numérico descargable
    desglose = pd.DataFrame({
        "Año"                : anios_eje,
        "Ahorro Acumulado MXN" : ahorro_acum,
        "Inversión MXN"      : inversion_acum,
        "ROI Neto MXN"       : [a - inversion_total for a in ahorro_acum],
    })
    with st.expander("📄 Ver desglose numérico del ROI"):
        st.dataframe(desglose.style.format({
            "Ahorro Acumulado MXN": "${:,.0f}",
            "Inversión MXN"       : "${:,.0f}",
            "ROI Neto MXN"        : "${:,.0f}",
        }), use_container_width=True)

    csv_roi = desglose.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Descargar proyección ROI como CSV",
        data=csv_roi,
        file_name="proyeccion_roi_hydrotrace.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "HydroTrace AI · Proyecto Universitario CDMX · "
    "Datos: SACMEX / INEGI 2019 · "
    "Modelos: K-Means, Isolation Forest, Regresión Lineal (scikit-learn)"
)