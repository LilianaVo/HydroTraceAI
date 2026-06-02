# -*- coding: utf-8 -*-
"""
================================================================================
EQUIPO: PUMASCRIPT SOLUTIONS

PROYECTO : HydroTrace AI
MODULO   : generar_metricas_informe.py
DESCRIPCION:
    Script standalone para extraer las metricas reales del modelo y
    producir un reporte plano listo para copiar al documento de
    investigacion (seccion 8.2).

    NO modifica ningun archivo del proyecto. Solo lee los CSVs y
    re-ejecuta los calculos estadisticos sobre ellos.

USO:
    Ejecuta:

        python generar_metricas_informe.py

    Salida: metricas_informe.txt
================================================================================
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

# ==============================================================================
# RUTAS
# ==============================================================================

BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
CSV_MAESTRO = DATA_DIR / "dataset_maestro_colonia_final.csv"
CSV_RESULT  = DATA_DIR / "resultados_finales_IA.csv"
OUT_TXT     = BASE_DIR / "metricas_informe.txt"

FEATURES_CLUSTER   = ["consumo_per_capita", "densidad_poblacional", "uso_suelo_num", "idsm"]
FEATURES_REGRESION = ["pob", "densidad_poblacional", "superficie_km2_calculada",
                      "idsm", "uso_suelo_num"]
N_CLUSTERS = 4


# ==============================================================================
# CALCULOS
# ==============================================================================

def calcular_regresion(df):
    features  = [f for f in FEATURES_REGRESION if f in df.columns]
    faltantes = [f for f in FEATURES_REGRESION if f not in df.columns]
    if faltantes:
        print("  AVISO features no disponibles para regresion: " + str(faltantes))

    X = df[features]
    y = df["consumo_total"]

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    coefs = {}
    for f, c in zip(features, modelo.coef_):
        coefs[f] = round(float(c), 4)

    return {
        "r2"                  : round(float(r2_score(y, y_pred)), 4),
        "mae"                 : round(float(mean_absolute_error(y, y_pred)), 2),
        "n_colonias"          : len(df),
        "features_usadas"     : features,
        "coeficientes"        : coefs,
        "intercepto"          : round(float(modelo.intercept_), 4),
    }


def calcular_silhouette(df):
    df_clean = df.dropna(subset=FEATURES_CLUSTER).copy()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[FEATURES_CLUSTER])

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(X_scaled, labels)

    conteo = {}
    for k, v in pd.Series(labels).value_counts().sort_index().items():
        conteo[int(k)] = int(v)

    return {
        "silhouette_score"   : round(float(score), 4),
        "n_clusters"         : N_CLUSTERS,
        "conteo_por_cluster" : conteo,
        "inercia"            : round(float(kmeans.inertia_), 2),
    }


def calcular_isolation_forest(df_result):
    total      = len(df_result)
    n_anomalas = int((df_result["es_anomalia"] == -1).sum())
    tasa       = round(n_anomalas / float(total) * 100, 2)

    dist = {}
    for k, v in df_result["diagnostico_final"].value_counts().items():
        dist[str(k)] = int(v)

    return {
        "total_colonias"            : total,
        "n_anomalas"                : n_anomalas,
        "tasa_anomalias_pct"        : tasa,
        "distribucion_diagnosticos" : dist,
    }


def calcular_confianza(df_result):
    if "total_reportes" not in df_result.columns:
        return {"error": "Columna total_reportes no encontrada."}

    mediana      = float(df_result["total_reportes"].median())
    anomalas     = df_result[df_result["es_anomalia"] == -1]
    con_respaldo = anomalas[anomalas["total_reportes"] > mediana]
    n_anomalas   = len(anomalas)

    confianza = round(len(con_respaldo) / float(n_anomalas) * 100, 2) if n_anomalas > 0 else 0.0

    return {
        "tasa_confianza_pct"    : confianza,
        "mediana_reportes"      : round(mediana, 2),
        "anomalas_con_respaldo" : len(con_respaldo),
        "anomalas_sin_respaldo" : n_anomalas - len(con_respaldo),
    }


def calcular_cobertura(df_result, df_maestro):
    alcaldias    = int(df_result["alcaldia"].nunique())
    colonias_geo = int(df_result.dropna(subset=["latitud", "longitud"]).shape[0])
    pct_geo      = round(colonias_geo / float(len(df_result)) * 100, 1)
    orig         = len(df_maestro) if df_maestro is not None else "N/A"

    return {
        "alcaldias_cubiertas"         : alcaldias,
        "total_alcaldias_cdmx"        : 16,
        "pct_alcaldias"               : round(alcaldias / 16.0 * 100, 1),
        "colonias_en_modelo"          : len(df_result),
        "colonias_con_coordenadas"    : colonias_geo,
        "pct_colonias_geolocalizadas" : pct_geo,
        "colonias_dataset_original"   : orig,
    }


def calcular_reproducibilidad(df_result):
    esperadas = [
        "alcaldia", "colonia", "consumo_total", "exceso_consumo",
        "es_anomalia", "anomalia_score", "cluster_perfil", "diagnostico_final",
    ]
    presentes = [c for c in esperadas if c in df_result.columns]
    ausentes  = [c for c in esperadas if c not in df_result.columns]

    nulos = {}
    for col in presentes:
        nulos[col] = int(df_result[col].isnull().sum())

    return {
        "columnas_esperadas"   : len(esperadas),
        "columnas_presentes"   : len(presentes),
        "columnas_ausentes"    : ausentes,
        "nulos_por_columna"    : nulos,
        "total_nulos_criticos" : sum(nulos.values()),
        "pipeline_reproducible": len(ausentes) == 0 and sum(nulos.values()) == 0,
    }


# ==============================================================================
# REPORTE DE TEXTO
# ==============================================================================

def generar_reporte(m):
    sep  = "=" * 72
    sep2 = "-" * 72
    lines = []

    def L(s=""):
        lines.append(s)

    L(sep)
    L("REPORTE DE METRICAS DEL MODELO -- HydroTrace AI")
    L("Seccion 8.2: Metricas minimas esperadas del modelo")
    L(sep)
    L()

    # 1. Regresion
    L("1. REGRESION LINEAL MULTIPLE")
    L(sep2)
    L("  R2 (Coeficiente de Determinacion)  : " + str(m["regresion"]["r2"]))
    L("  MAE (Error Absoluto Medio)          : " + "{:,.2f}".format(m["regresion"]["mae"]) + " m3 (semestral)")
    L("  Colonias usadas                     : " + str(m["regresion"]["n_colonias"]))
    L("  Features                            : " + ", ".join(m["regresion"]["features_usadas"]))
    L()
    L("  Coeficientes del modelo:")
    for var, coef in m["regresion"]["coeficientes"].items():
        L("    {:<40} {:+.4f}".format(var, coef))
    L("    {:<40} {:+.4f}".format("intercepto", m["regresion"]["intercepto"]))
    L()
    L("  NOTA R2: Un R2 moderado o bajo es esperado. El consumo hidrico")
    L("  depende de variables no publicas (actividad informal, habitos,")
    L("  estado de la red). La regresion genera la variable exceso_consumo")
    L("  como linea base para que Isolation Forest detecte desviaciones.")
    L()

    # 2. K-Means
    L("2. SEGMENTACION K-MEANS (K=4)")
    L(sep2)
    L("  Score de Silueta                    : " + str(m["kmeans"]["silhouette_score"]))
    L("  Inercia                             : " + "{:,.2f}".format(m["kmeans"]["inercia"]))
    L("  Numero de clusters                  : " + str(m["kmeans"]["n_clusters"]))
    L()
    L("  Distribucion por cluster:")
    for k, v in sorted(m["kmeans"]["conteo_por_cluster"].items()):
        L("    Cluster {}: {} colonias".format(k, v))
    L()
    L("  NOTA Silueta: Un valor entre 0.20 y 0.55 es aceptable para datos")
    L("  urbanos reales. El objetivo de K-Means no es separabilidad maxima")
    L("  sino grupos con interpretabilidad operativa (residencial, mixto,")
    L("  comercial, industrial) para mejorar los diagnosticos de IF.")
    L()

    # 3. Isolation Forest
    L("3. DETECCION DE ANOMALIAS (ISOLATION FOREST)")
    L(sep2)
    L("  Total colonias analizadas           : " + str(m["iso"]["total_colonias"]))
    L("  Colonias anomalas                   : " + str(m["iso"]["n_anomalas"]))
    L("  Tasa de anomalias                   : " + str(m["iso"]["tasa_anomalias_pct"]) + "%")
    L()
    L("  Distribucion de diagnosticos:")
    for diag, cnt in sorted(m["iso"]["distribucion_diagnosticos"].items(),
                            key=lambda x: x[1], reverse=True):
        L("    {:<52} {} colonias".format(diag, cnt))
    L()

    # 4. Confianza
    L("4. TASA DE CONFIANZA DE DETECCION")
    L(sep2)
    if "error" in m["confianza"]:
        L("  ERROR: " + m["confianza"]["error"])
    else:
        L("  Tasa de confianza                   : " + str(m["confianza"]["tasa_confianza_pct"]) + "%")
        L("  Mediana reportes ciudadanos          : " + str(m["confianza"]["mediana_reportes"]))
        L("  Anomalas CON respaldo ciudadano      : " + str(m["confianza"]["anomalas_con_respaldo"]))
        L("  Anomalas SIN respaldo ciudadano      : " + str(m["confianza"]["anomalas_sin_respaldo"]))
    L()
    L("  NOTA Confianza: No es una metrica estandar de ML, sino una")
    L("  validacion cruzada del proyecto. Mide coherencia entre la senal")
    L("  estadistica (Isolation Forest) y la senal ciudadana (SEGUIAGUA).")
    L("  Tasa alta = las anomalias del modelo coinciden con zonas donde")
    L("  los habitantes reportan problemas reales en la red hidrica.")
    L()

    # 5. Cobertura
    L("5. COBERTURA GEOESPACIAL")
    L(sep2)
    L("  Alcaldias cubiertas                 : {}/{} ({}%)".format(
        m["geo"]["alcaldias_cubiertas"],
        m["geo"]["total_alcaldias_cdmx"],
        m["geo"]["pct_alcaldias"]))
    L("  Colonias en el modelo               : " + str(m["geo"]["colonias_en_modelo"]))
    L("  Colonias con coordenadas validas    : {} ({}%)".format(
        m["geo"]["colonias_con_coordenadas"],
        m["geo"]["pct_colonias_geolocalizadas"]))
    L("  Colonias dataset original (ETL)     : " + str(m["geo"]["colonias_dataset_original"]))
    L()

    # 6. Reproducibilidad
    L("6. REPRODUCIBILIDAD DEL PIPELINE")
    L(sep2)
    L("  Columnas esperadas                  : " + str(m["repro"]["columnas_esperadas"]))
    L("  Columnas presentes                  : " + str(m["repro"]["columnas_presentes"]))
    ausentes_str = str(m["repro"]["columnas_ausentes"]) if m["repro"]["columnas_ausentes"] else "Ninguna"
    L("  Columnas ausentes                   : " + ausentes_str)
    L("  Nulos en columnas criticas          : " + str(m["repro"]["total_nulos_criticos"]))
    estado = "SI" if m["repro"]["pipeline_reproducible"] else "NO -- revisar columnas ausentes o nulos"
    L("  Pipeline reproducible               : " + estado)
    L()
    L(sep)
    L("FIN DEL REPORTE")
    L(sep)

    return "\n".join(lines)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("\n" + "=" * 72)
    print("HydroTrace AI -- Generador de Metricas para Informe")
    print("=" * 72)

    if not CSV_RESULT.exists():
        print("\n[ERROR] No se encontro: " + str(CSV_RESULT))
        print("  Ejecuta modelos_ml.py primero.")
        sys.exit(1)

    print("\n[OK] CSV resultados : " + str(CSV_RESULT))
    df_result = pd.read_csv(str(CSV_RESULT))
    print("     " + str(len(df_result)) + " colonias, " + str(len(df_result.columns)) + " columnas.")

    df_maestro = None
    if CSV_MAESTRO.exists():
        print("[OK] Dataset maestro: " + str(CSV_MAESTRO))
        df_maestro = pd.read_csv(str(CSV_MAESTRO))
        print("     " + str(len(df_maestro)) + " colonias originales.")
    else:
        print("[AVISO] Dataset maestro no encontrado. Algunas metricas usaran N/A.")

    fuente = df_maestro if df_maestro is not None else df_result

    print("\n[CALCULANDO] Regresion Lineal...")
    m_reg = calcular_regresion(fuente.dropna(subset=FEATURES_REGRESION + ["consumo_total"]))

    print("[CALCULANDO] Score de Silueta (K-Means)...")
    m_km = calcular_silhouette(fuente)

    print("[CALCULANDO] Isolation Forest...")
    m_iso = calcular_isolation_forest(df_result)

    print("[CALCULANDO] Tasa de confianza...")
    m_conf = calcular_confianza(df_result)

    print("[CALCULANDO] Cobertura geoespacial...")
    m_geo = calcular_cobertura(df_result, df_maestro)

    print("[CALCULANDO] Reproducibilidad...")
    m_repro = calcular_reproducibilidad(df_result)

    metricas = {
        "regresion": m_reg,
        "kmeans"   : m_km,
        "iso"      : m_iso,
        "confianza": m_conf,
        "geo"      : m_geo,
        "repro"    : m_repro,
    }

    reporte = generar_reporte(metricas)
    with open(str(OUT_TXT), "w") as f:
        f.write(reporte)
    print("[OK] TXT guardado en  : " + str(OUT_TXT))

    print("\n" + reporte)


if __name__ == "__main__":
    main()