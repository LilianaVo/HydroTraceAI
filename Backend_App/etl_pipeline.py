"""
================================================================================
PROYECTO : HydroTrace AI — Ciencia de Datos UNAM
MÓDULO   : etl_pipeline_colonia.py
DESCRIPCIÓN:
    Pipeline ETL a nivel COLONIA para la CDMX.
    Integra 5 fuentes heterogéneas, calcula superficie por colonia vía
    geometrías GeoJSON proyectadas al ITRF2008/LCC (EPSG:6372), imputa
    valores faltantes con medianas por alcaldía y exporta el Top-10 de
    mayor consumo por alcaldía listo para Isolation Forest y Regresión Lineal.

SALIDA  : dataset_maestro_colonia_final.csv
AUTOR   : HydroTrace AI — Data Engineering Team
================================================================================
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely.geometry import shape
from shapely.ops import transform

warnings.filterwarnings("ignore", category=FutureWarning)

# ──────────────────────────────────────────────────────────────────────────────
# 0. CONFIGURACIÓN DE RUTAS
# ──────────────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RUTAS: dict[str, Path] = {
    "consumo"    : DATA_DIR / "consumo_agua_historico_2019.csv",
    "ids_ut"     : DATA_DIR / "ids_ut.xlsx",
    "colonias"   : DATA_DIR / "coloniascdmx.csv",
    "reportes"   : DATA_DIR / "reportes_agua_hist.csv",
    "superficie" : DATA_DIR / "superficie_alcaldias.csv",
}

OUTPUT_PATH = DATA_DIR / "dataset_maestro_colonia_final.csv"

# Proyector universal: WGS84 → México ITRF2008 / LCC (EPSG:6372)
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:6372", always_xy=True)


# ──────────────────────────────────────────────────────────────────────────────
# 1. UTILIDADES DE NORMALIZACIÓN
# ──────────────────────────────────────────────────────────────────────────────

# Tabla de homologación explícita para alcaldías con nombres irregulares
_MAPEO_ALCALDIAS: dict[str, str] = {
    "CUAJIMALPA DE MORELOS" : "CUAJIMALPA",
    "LA MAGDALENA CONTRERAS": "MAGDALENA CONTRERAS",
    "GUSTAVO A MADERO"      : "GUSTAVO A. MADERO",
}


def normalizar(texto: object) -> object:
    """
    Normalización robusta para nombres de colonias y alcaldías:
      1. Quita acentos (NFD + strip combining chars).
      2. Convierte a mayúsculas y elimina espacios laterales.
      3. Aplica mapeo de homologación de alcaldías.

    Devuelve el valor original si no es cadena (NaN, etc.).
    """
    if not isinstance(texto, str):
        return texto
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.strip().upper()
    return _MAPEO_ALCALDIAS.get(texto, texto)


# ──────────────────────────────────────────────────────────────────────────────
# 2. CÁLCULO GEOGRÁFICO — SUPERFICIE POR COLONIA
# ──────────────────────────────────────────────────────────────────────────────

def _proyectar(geom):
    """Proyecta una geometría Shapely de EPSG:4326 a EPSG:6372."""
    return transform(lambda x, y: _TRANSFORMER.transform(x, y), geom)


def calcular_superficie_km2(geo_shape_str: str) -> float:
    """
    Recibe la cadena JSON del campo geo_shape (GeoJSON Polygon/MultiPolygon),
    proyecta al CRS métrico EPSG:6372 y devuelve el área en km².
    Retorna np.nan si el JSON es inválido o la geometría está vacía.
    """
    try:
        geom = shape(json.loads(geo_shape_str))
        if geom.is_empty:
            return np.nan
        return _proyectar(geom).area / 1_000_000  # m² → km²
    except Exception:
        return np.nan


def cargar_geometrias_colonias() -> pd.DataFrame:
    """
    Carga coloniascdmx.csv, normaliza nombres y calcula la superficie
    en km² para cada colonia usando su polígono oficial.

    Columnas de salida: alcaldia, colonia, superficie_km2_calculada
    """
    print("[ETL] Calculando superficies por colonia (EPSG:4326 → EPSG:6372)...")
    df = pd.read_csv(RUTAS["colonias"])

    # Normalización de claves de unión
    df["alcaldia"] = df["alcaldia"].apply(normalizar)
    df["colonia"]  = df["nombre"].apply(normalizar)

    # Cálculo vectorizado con apply (shapely no soporta operaciones nativas en arrays)
    df["superficie_km2_calculada"] = df["geo_shape"].apply(calcular_superficie_km2)

    invalidas = df["superficie_km2_calculada"].isna().sum()
    if invalidas:
        print(f"  ⚠ {invalidas} geometrías inválidas → se imputarán con mediana de alcaldía.")

    return df[["alcaldia", "colonia", "superficie_km2_calculada"]].drop_duplicates(
        subset=["alcaldia", "colonia"]
    )


# ──────────────────────────────────────────────────────────────────────────────
# 3. MAPEO DE USO DE SUELO (Highest-Weight Rule)
# ──────────────────────────────────────────────────────────────────────────────

_PESOS_SUELO: list[tuple[list[str], int]] = [
    (["INDUSTRIAL"],                          2),   # Consumo esperado alto → baja anomalía
    (["COMERCIAL", "CORPORATIVO", "SERVICIOS"], 1),  # Uso mixto
    # Habitacional / Rural / Reserva → 0 (alta sensibilidad a anomalías)
]


def mapear_uso_suelo(vocacion: object) -> int:
    """
    Asigna un peso numérico a la vocación territorial según la
    Highest-Weight Rule del proyecto HydroTrace AI.
    """
    if not isinstance(vocacion, str):
        return 0
    v = vocacion.upper()
    for palabras_clave, peso in _PESOS_SUELO:
        if any(kw in v for kw in palabras_clave):
            return peso
    return 0


def cargar_uso_suelo() -> pd.DataFrame:
    """
    Carga superficie_alcaldias.csv y genera las columnas de uso de suelo
    como atributo de alcaldía (se propagará a todas sus colonias en el merge).

    Columnas de salida: alcaldia, vocacion_principal, uso_suelo_num
    """
    print("[ETL] Procesando uso de suelo por alcaldía...")
    df = pd.read_csv(RUTAS["superficie"])
    df["alcaldia"]       = df["alcaldia"].apply(normalizar)
    df["uso_suelo_num"]  = df["vocacion_principal"].apply(mapear_uso_suelo)
    return df[["alcaldia", "vocacion_principal", "uso_suelo_num"]]


# ──────────────────────────────────────────────────────────────────────────────
# 4. CONSUMO TOTAL POR COLONIA
# ──────────────────────────────────────────────────────────────────────────────

def cargar_consumo() -> pd.DataFrame:
    """
    Agrega el consumo total anual 2019 por (alcaldia, colonia).
    También captura latitud/longitud representativa (primer registro).

    Columnas de salida: alcaldia, colonia, consumo_total,
                        consumo_total_dom, consumo_total_no_dom,
                        latitud, longitud
    """
    print("[ETL] Agregando consumo de agua 2019 por colonia...")
    df = pd.read_csv(RUTAS["consumo"])
    df["alcaldia"] = df["alcaldia"].apply(normalizar)
    df["colonia"]  = df["colonia"].apply(normalizar)

    # Coordenadas representativas: primer registro de la colonia
    coords = (
        df.sort_values("fecha_referencia")
          .groupby(["alcaldia", "colonia"])[["latitud", "longitud"]]
          .first()
          .reset_index()
    )

    agg = df.groupby(["alcaldia", "colonia"]).agg(
        consumo_total         = ("consumo_total",        "sum"),
        consumo_total_dom     = ("consumo_total_dom",    "sum"),
        consumo_total_no_dom  = ("consumo_total_no_dom", "sum"),
    ).reset_index()

    return agg.merge(coords, on=["alcaldia", "colonia"], how="left")


# ──────────────────────────────────────────────────────────────────────────────
# 5. POBLACIÓN POR COLONIA (ids_ut.xlsx)
# ──────────────────────────────────────────────────────────────────────────────

def cargar_poblacion() -> pd.DataFrame:
    """
    Lee la hoja 'base_ut_final' del archivo ids_ut.xlsx y extrae la
    población (pob) por Unidad Territorial, que corresponde a colonias.

    Columnas de salida: alcaldia, colonia, pob, idsm, e_idsm
    """
    print("[ETL] Cargando datos poblacionales (ids_ut)...")
    df = pd.read_excel(RUTAS["ids_ut"], sheet_name="base_ut_final")
    df = df.dropna(subset=["alcaldia", "nombre_ut"])      # elimina filas cabecera/vacías
    df["alcaldia"] = df["alcaldia"].apply(normalizar)
    df["colonia"]  = df["nombre_ut"].apply(normalizar)

    # Agregamos por si hay UTs duplicadas con el mismo nombre en la misma alcaldía
    agg = df.groupby(["alcaldia", "colonia"]).agg(
        pob    = ("pob",   "sum"),
        idsm   = ("idsm",  "mean"),
        e_idsm = ("e_idsm", "first"),
    ).reset_index()

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# 6. REPORTES CIUDADANOS (SEGUIAGUA) POR COLONIA
# ──────────────────────────────────────────────────────────────────────────────

def cargar_reportes() -> pd.DataFrame:
    """
    Filtra reportes de 'Fuga' o 'Falta de agua' y cuenta el total por colonia.
    Usa la columna colonia_datos_abiertos (ya normalizada/oficial) como clave.

    Columnas de salida: alcaldia, colonia, total_reportes,
                        reportes_fuga, reportes_falta_agua
    """
    print("[ETL] Procesando reportes ciudadanos SEGUIAGUA...")
    df = pd.read_csv(RUTAS["reportes"])
    df["alcaldia"] = df["alcaldia"].apply(normalizar)
    df["colonia"]  = df["colonia_datos_abiertos"].apply(normalizar)

    # Filtro de relevancia del proyecto
    mask = df["tipo_de_falla"].str.contains(r"Fuga|Falta", case=False, na=False)
    df = df[mask].copy()

    # Conteo desglosado por tipo de falla para mayor granularidad
    df["es_fuga"]       = df["tipo_de_falla"].str.contains("Fuga",  case=False, na=False)
    df["es_falta_agua"] = df["tipo_de_falla"].str.contains("Falta", case=False, na=False)

    agg = df.groupby(["alcaldia", "colonia"]).agg(
        total_reportes    = ("folio",         "count"),
        reportes_fuga     = ("es_fuga",        "sum"),
        reportes_falta_agua = ("es_falta_agua","sum"),
    ).reset_index()

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# 7. AUDITORÍA DE CALIDAD DE DATOS
# ──────────────────────────────────────────────────────────────────────────────

def auditar(df: pd.DataFrame, nombre: str, cols_criticas: list[str] | None = None) -> None:
    """
    Audita un DataFrame en busca de:
      - Valores nulos (advertencia).
      - Valores negativos en columnas numéricas (error crítico).
      - Ceros en columnas críticas (alerta de integración).
    Lanza ValueError si detecta negativos — indican corrupción de datos.
    """
    print(f"\n[AUDITORÍA] {nombre} — shape: {df.shape}")

    nulos = df.isnull().sum().sum()
    if nulos:
        print(f"  ⚠  {nulos} valores nulos detectados.")

    # Columnas donde valores negativos son geográficamente válidos (longitud occidental)
    _EXCLUIR_NEGATIVOS = {"longitud", "latitud", "idsm"}
    for col in df.select_dtypes(include="number").columns:
        if col in _EXCLUIR_NEGATIVOS:
            continue
        n_neg = (df[col] < 0).sum()
        if n_neg:
            raise ValueError(
                f"  ✗  ERROR CRÍTICO: '{col}' contiene {n_neg} valores negativos en '{nombre}'."
            )

    if cols_criticas:
        for col in cols_criticas:
            if col in df.columns:
                n_cero = (df[col] == 0).sum()
                if n_cero:
                    muestra = df.loc[df[col] == 0, "colonia"].head(5).tolist()
                    print(f"  !  ALERTA: '{col}' tiene {n_cero} ceros. Muestra: {muestra}")

    print(f"  ✓  Auditoría completada para '{nombre}'.")


# ──────────────────────────────────────────────────────────────────────────────
# 8. IMPUTACIÓN POR MEDIANA DE ALCALDÍA
# ──────────────────────────────────────────────────────────────────────────────

def imputar_con_mediana_alcaldia(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """
    Para cada columna en `columnas`, imputa los valores 0 o NaN de una colonia
    con la mediana de las colonias válidas de su misma alcaldía.
    Esto preserva la distribución real sin sesgar el modelo con ceros espurios.
    """
    df = df.copy()
    for col in columnas:
        if col not in df.columns:
            continue
        # Marcar como NaN los ceros para tratarlos uniformemente
        df[col] = df[col].replace(0, np.nan)
        medianas = df.groupby("alcaldia")[col].transform("median")
        df[col]  = df[col].fillna(medianas)
        # Si toda la alcaldía es NaN (caso extremo), imputar con mediana global
        df[col]  = df[col].fillna(df[col].median())
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 9. ORQUESTADOR PRINCIPAL — INTEGRACIÓN Y KPIs
# ──────────────────────────────────────────────────────────────────────────────

def integrar_dataset_maestro() -> pd.DataFrame:
    """
    Orquesta la integración de las 5 fuentes a nivel colonia:
      1. Base geográfica  (coloniascdmx  → superficie_km2_calculada)
      2. Uso de suelo     (superficie_alcaldias → vocacion, uso_suelo_num)
      3. Consumo          (consumo_agua_historico_2019)
      4. Población        (ids_ut)
      5. Reportes SEGUIAGUA  (reportes_agua_hist)

    Luego:
      - Imputa superficie y población faltantes con mediana de alcaldía.
      - Calcula KPIs de negocio (consumo_per_capita, densidad_poblacional).
      - Filtra Top-10 colonias de mayor consumo total por alcaldía.
    """
    print("\n" + "="*72)
    print("  HYDROTRACE AI — ETL PIPELINE COLONIA")
    print("="*72)

    # ── 9.1 Base geográfica ──────────────────────────────────────────────────
    df_geo  = cargar_geometrias_colonias()
    df_suelo = cargar_uso_suelo()
    df_base = df_geo.merge(df_suelo, on="alcaldia", how="left")

    # ── 9.2 Módulos de datos ─────────────────────────────────────────────────
    df_consumo  = cargar_consumo()
    df_pob      = cargar_poblacion()
    df_reportes = cargar_reportes()

    # ── 9.3 Merge progresivo (LEFT desde la base geográfica oficial) ─────────
    print("\n[ETL] Integrando fuentes (LEFT JOIN desde geometría oficial)...")

    df = df_base.merge(df_consumo,  on=["alcaldia", "colonia"], how="left")
    df = df.merge(df_pob,           on=["alcaldia", "colonia"], how="left")
    df = df.merge(df_reportes,      on=["alcaldia", "colonia"], how="left")

    print(f"  → Colonias en dataset integrado: {len(df)}")

    # ── 9.4 Imputación robusta con mediana de alcaldía ───────────────────────
    print("[ETL] Imputando valores faltantes con mediana de alcaldía...")
    df = imputar_con_mediana_alcaldia(
        df,
        columnas=["superficie_km2_calculada", "pob",
                  "consumo_total", "total_reportes"],
    )

    # Rellenar otros nulos numéricos con 0 (columnas secundarias no críticas)
    df = df.fillna({
        "consumo_total_dom"     : 0,
        "consumo_total_no_dom"  : 0,
        "reportes_fuga"         : 0,
        "reportes_falta_agua"   : 0,
        "uso_suelo_num"         : 0,
        "idsm"                  : 0,
        "e_idsm"                : "Sin dato",
    })

    # ── 9.5 Auditoría del dataset integrado ──────────────────────────────────
    auditar(
        df, "Dataset Maestro Integrado",
        cols_criticas=["consumo_total", "pob", "superficie_km2_calculada"],
    )

    # ── 9.6 KPIs de Negocio ──────────────────────────────────────────────────
    print("\n[ETL] Calculando KPIs de negocio...")

    df["consumo_per_capita"]    = df["consumo_total"]  / df["pob"]
    df["densidad_poblacional"]  = df["pob"]             / df["superficie_km2_calculada"]

    # Limpiar divisiones por cero o infinitos
    df = df.replace([np.inf, -np.inf], np.nan)
    df = imputar_con_mediana_alcaldia(
        df, columnas=["consumo_per_capita", "densidad_poblacional"]
    )

    # ── 9.7 Filtro Estratégico: Top-10 por alcaldía ──────────────────────────
    print("[ETL] Aplicando filtro Top-10 colonias por consumo por alcaldía...")
    df_top10 = (
        df.sort_values("consumo_total", ascending=False)
          .groupby("alcaldia", group_keys=False)
          .head(10)
          .reset_index(drop=True)
    )
    print(f"  → Colonias en dataset final: {len(df_top10)}")

    # ── 9.8 Orden de columnas para el modelo ────────────────────────────────
    columnas_modelo = [
        "alcaldia", "colonia",
        "superficie_km2_calculada",
        "vocacion_principal", "uso_suelo_num",
        "consumo_total", "consumo_total_dom", "consumo_total_no_dom",
        "pob", "pob_nbi",
        "consumo_per_capita", "densidad_poblacional",
        "total_reportes", "reportes_fuga", "reportes_falta_agua",
        "idsm", "e_idsm",
        "latitud", "longitud",
    ]
    # Solo incluir columnas que existan en el DataFrame final
    columnas_salida = [c for c in columnas_modelo if c in df_top10.columns]
    df_top10 = df_top10[columnas_salida]

    return df_top10


# ──────────────────────────────────────────────────────────────────────────────
# 10. PUNTO DE ENTRADA
# ──────────────────────────────────────────────────────────────────────────────

def ejecutar_pipeline() -> pd.DataFrame | None:
    """Punto de entrada del pipeline ETL. Orquesta todo y exporta el CSV."""
    try:
        df_final = integrar_dataset_maestro()

        # Exportar
        df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

        print("\n" + "="*72)
        print(f"  [ÉXITO] Dataset exportado en: {OUTPUT_PATH}")
        print(f"  Dimensiones finales: {df_final.shape}")
        print(f"  Colonias únicas    : {df_final['colonia'].nunique()}")
        print(f"  Alcaldías cubiertas: {df_final['alcaldia'].nunique()}")
        print("="*72)

        print("\n[MUESTRA] Primeras 5 filas del dataset maestro:")
        print(df_final.head().to_string())

        print("\n[ESTADÍSTICAS] Variables cuantitativas clave:")
        print(df_final[[
            "consumo_total", "pob",
            "consumo_per_capita", "densidad_poblacional",
            "superficie_km2_calculada", "total_reportes",
        ]].describe().round(2).to_string())

        return df_final

    except Exception as exc:
        print(f"\n[ERROR CRÍTICO] El pipeline falló: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    ejecutar_pipeline()