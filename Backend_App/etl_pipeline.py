"""
================================================================================
PROYECTO : HydroTrace AI — Ciencia de Datos UNAM
MÓDULO   : etl_pipeline.py
DESCRIPCIÓN:
    Pipeline ETL a nivel COLONIA para la CDMX.
    Integra 5 fuentes heterogéneas, calcula superficie por colonia vía
    geometrías GeoJSON proyectadas al ITRF2008/LCC (EPSG:6372), imputa
    valores faltantes con medianas por alcaldía y exporta el Top-10 de
    mayor consumo por alcaldía listo para Isolation Forest y Regresión Lineal.

COBERTURA TEMPORAL:
    · Consumo : Bimestres 1-3 de 2019 (enero–junio).
      Los bimestres 4-6 no están disponibles en datos abiertos de SACMEX.
      Todos los KPIs de consumo reflejan el primer semestre de 2019.
    · Reportes: Filtrados al año 2019 para mantener coherencia temporal
      con el dataset de consumo.

SALIDA   : dataset_maestro_colonia_final.csv
AUTORES  : HydroTrace AI — Data Engineering Team
================================================================================

HISTORIAL DE CAMBIOS:
  [FIX-F08] cargar_consumo: se documenta explícitamente que el dataset cubre
            solo el primer semestre (bimestres 1-3). Renombrado de etiquetas
            en docstrings y prints para no llamarlo "anual".
  [FIX-F09] cargar_reportes: se filtra por año == 2019 antes de agregar,
            eliminando reportes de 2018, 2020 y 2021 que contaminaban
            los conteos de corroboración ciudadana.
  [FIX-F12] mapear_uso_suelo: 'Habitacional/Industrial' baja de peso 2 a
            peso 1. Solo vocaciones de inicio 'Industrial/...' reciben peso 2,
            ya que son las zonas predominantemente industriales.
  [FIX-D05] pob_nbi eliminada de columnas_modelo (columna nunca producida
            por ninguna fuente del ETL; su presencia era silenciosa).

  [MEJ-01] cargar_consumo: validación de año en el CSV de consumo para
           detectar registros fuera de ANIO_REFERENCIA (coherencia futura).
  [MEJ-02] cargar_consumo: nueva columna consumo_per_capita_anualizado
           (consumo semestral × 2) para que modelos_ml.py pueda comparar
           contra benchmarks anuales sin sesgo de escala temporal.
  [MEJ-03] cargar_reportes: nueva columna tiene_reportes_2019 (bool) para
           distinguir "genuinamente 0 reportes" de "sin match en el join".
           Evita clasificar como HUACHICOL colonias con datos ausentes.
  [MEJ-04] imputar_con_mediana_alcaldia: advertencia explícita cuando una
           alcaldía tiene una sola colonia y se usa la mediana global como
           fallback en lugar de la mediana de alcaldía.
  [MEJ-05] cargar_consumo: validación de separador en vocacion_principal
           (asumido '/') para detectar formatos inesperados en superficie_alcaldias.csv.
  [MEJ-06] integrar_dataset_maestro: log de cobertura de join por fuente
           para detectar fugas de colonias entre merges.
"""

from __future__ import annotations

import json
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

# Año de referencia: debe coincidir con el año del dataset de consumo.
# Si en el futuro se actualiza el CSV, solo cambiar esta constante.
# Afecta: filtro de reportes, validación de consumo, prints de cobertura.
ANIO_REFERENCIA: int = 2019

# Factor de anualización del consumo semestral.
# El dataset cubre bimestres 1-3 (6 meses). Multiplicar por este factor
# produce una estimación anual comparada con benchmarks de consumo (p.ej. SACMEX).
# ADVERTENCIA: es una proyección lineal — no captura estacionalidad.
FACTOR_ANUALIZACION: float = 2.0


# ──────────────────────────────────────────────────────────────────────────────
# 1. UTILIDADES DE NORMALIZACIÓN
# ──────────────────────────────────────────────────────────────────────────────

# Tabla de homologación explícita para alcaldías con nombres irregulares
# entre fuentes (SACMEX vs INEGI vs datos abiertos CDMX).
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

    df["alcaldia"] = df["alcaldia"].apply(normalizar)
    df["colonia"]  = df["nombre"].apply(normalizar)

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

# [FIX-F12] Tabla de pesos revisada:
#   · Peso 2 → solo vocaciones cuya ACTIVIDAD PRINCIPAL es industrial
#              (formato "Industrial/..."). Zonas donde consumo alto es esperado.
#   · Peso 1 → vocaciones mixtas con componente industrial, comercial,
#              corporativo o de servicios como actividad SECUNDARIA.
#   · Peso 0 → vocaciones residenciales puras, rurales o de reserva ecológica.
#
# [MEJ-05] La lógica de split("/") asume que superficie_alcaldias.csv usa
#          la barra "/" como separador de vocaciones compuestas.
#          La función _validar_separador_vocaciones() verifica esto al cargar.

_PESOS_SUELO: list[tuple[list[str], int]] = [
    (["INDUSTRIAL"],                           2),  # Actividad principal industrial
    (["HABITACIONAL", "COMERCIAL",
      "CORPORATIVO", "SERVICIOS", "MIXTO"],    1),  # Mixto o habitacional con componente comercial
    # Habitacional puro / Rural / Reserva → 0
]


def mapear_uso_suelo(vocacion: object) -> int:
    """
    Asigna un peso numérico a la vocación territorial.

    Regla de prioridad (Highest-Weight First):
      1. Si el PRIMER componente de la vocación es 'INDUSTRIAL' → peso 2.
         Diferencia "Industrial/Comercial" (zona netamente industrial) de
         "Habitacional/Industrial" (zona residencial con industria periférica).
      2. Si la vocación contiene cualquier componente no residencial → peso 1.
      3. Default → peso 0.

    Asume separador "/" entre componentes de vocación compuesta.
    """
    if not isinstance(vocacion, str):
        return 0
    primer_termino = vocacion.split("/")[0].strip().upper()
    if primer_termino == "INDUSTRIAL":
        return 2
    v = vocacion.upper()
    if any(kw in v for kw in ["INDUSTRIAL", "COMERCIAL", "CORPORATIVO", "SERVICIOS"]):
        return 1
    return 0


def _validar_separador_vocaciones(df: pd.DataFrame) -> None:
    """
    [MEJ-05] Verifica que las vocaciones en superficie_alcaldias.csv usen '/'
    como separador. Emite advertencia si detecta filas con separadores distintos
    (' y ', ',') que podrían hacer que mapear_uso_suelo() las clasifique mal.
    """
    if "vocacion_principal" not in df.columns:
        return
    separadores_alternativos = [" Y ", " AND ", " & "]
    for sep in separadores_alternativos:
        afectadas = df["vocacion_principal"].str.upper().str.contains(sep, na=False).sum()
        if afectadas:
            print(
                f"  ⚠ [MEJ-05] {afectadas} vocaciones usan '{sep.strip()}' como separador "
                f"en lugar de '/'. Ejemplo: "
                f"{df.loc[df['vocacion_principal'].str.upper().str.contains(sep, na=False), 'vocacion_principal'].iloc[0]!r}. "
                f"Revisar superficie_alcaldias.csv — podrían clasificarse con peso incorrecto."
            )


def cargar_uso_suelo() -> pd.DataFrame:
    """
    Carga superficie_alcaldias.csv y genera las columnas de uso de suelo
    como atributo de alcaldía (se propagará a todas sus colonias en el merge).

    Columnas de salida: alcaldia, vocacion_principal, uso_suelo_num
    """
    print("[ETL] Procesando uso de suelo por alcaldía...")
    df = pd.read_csv(RUTAS["superficie"])
    df["alcaldia"]      = df["alcaldia"].apply(normalizar)

    _validar_separador_vocaciones(df)  # [MEJ-05]

    df["uso_suelo_num"] = df["vocacion_principal"].apply(mapear_uso_suelo)
    return df[["alcaldia", "vocacion_principal", "uso_suelo_num"]]


# ──────────────────────────────────────────────────────────────────────────────
# 4. CONSUMO TOTAL POR COLONIA — PRIMER SEMESTRE 2019
# ──────────────────────────────────────────────────────────────────────────────

def cargar_consumo() -> pd.DataFrame:
    """
    Agrega el consumo total del primer semestre 2019 (bimestres 1, 2 y 3)
    por (alcaldia, colonia). También captura latitud/longitud representativa.

    NOTA DE COBERTURA [FIX-F08]:
        El dataset público de SACMEX disponible en datos.cdmx.gob.mx
        contiene únicamente los bimestres 1, 2 y 3 de 2019 (enero–junio).
        Los bimestres 4-6 no están disponibles como datos abiertos.
        Por tanto, 'consumo_total' representa el volumen facturado en los
        primeros 6 meses del año, no el total anual.
        Esto se propaga a todos los KPIs derivados (consumo_per_capita,
        exceso_consumo, impacto económico).

    [MEJ-01] Validación de año:
        Se verifica que los registros sean del ANIO_REFERENCIA. Si el CSV
        es actualizado en el futuro con datos de otro año, el pipeline lo
        detecta en lugar de procesarlos silenciosamente.

    [MEJ-02] consumo_per_capita_anualizado:
        Columna adicional = consumo_total × FACTOR_ANUALIZACION (×2).
        Permite comparar contra benchmarks anuales (ej. 133.59 m³/hab/año
        de SACMEX) sin sesgo por cobertura semestral.
        Es una proyección lineal — no captura estacionalidad hídrica.

    Columnas de salida: alcaldia, colonia, consumo_total,
                        consumo_total_dom, consumo_total_no_dom,
                        consumo_anualizado, bimestres_cubiertos,
                        latitud, longitud
    """
    print(f"[ETL] Agregando consumo de agua {ANIO_REFERENCIA} por colonia...")
    df = pd.read_csv(RUTAS["consumo"])
    df["alcaldia"] = df["alcaldia"].apply(normalizar)
    df["colonia"]  = df["colonia"].apply(normalizar)

    # [MEJ-01] Validación de año en el CSV de consumo
    if "fecha_referencia" in df.columns:
        df["fecha_referencia"] = pd.to_datetime(df["fecha_referencia"], errors="coerce")
        n_otros = (df["fecha_referencia"].dt.year != ANIO_REFERENCIA).sum()
        n_nulos = df["fecha_referencia"].isna().sum()
        if n_otros:
            print(
                f"  ⚠ [MEJ-01] {n_otros} registros tienen año distinto a {ANIO_REFERENCIA} "
                f"en consumo_agua_historico_2019.csv. Se procesan igual — verificar fuente."
            )
        if n_nulos:
            print(f"  ⚠ [MEJ-01] {n_nulos} fechas no parseables en consumo — se ignoran en validación.")

    bimestres = sorted(df["bimestre"].unique()) if "bimestre" in df.columns else []
    if bimestres:
        print(
            f"  ℹ Bimestres disponibles: {bimestres} "
            f"(cobertura: {len(bimestres)} de 6 bimestres anuales — primer semestre)"
        )

    # Coordenadas representativas: primer registro cronológico de la colonia
    sort_col = "fecha_referencia" if "fecha_referencia" in df.columns else df.columns[0]
    coords = (
        df.sort_values(sort_col)
          .groupby(["alcaldia", "colonia"])[["latitud", "longitud"]]
          .first()
          .reset_index()
    )

    agg_dict = {
        "consumo_total"        : ("consumo_total",        "sum"),
        "consumo_total_dom"    : ("consumo_total_dom",    "sum"),
        "consumo_total_no_dom" : ("consumo_total_no_dom", "sum"),
    }
    if "bimestre" in df.columns:
        agg_dict["bimestres_cubiertos"] = ("bimestre", "nunique")

    agg = df.groupby(["alcaldia", "colonia"]).agg(**agg_dict).reset_index()

    # [MEJ-02] Consumo anualizado como proyección lineal del semestral
    agg["consumo_anualizado"] = (agg["consumo_total"] * FACTOR_ANUALIZACION).round(2)

    return agg.merge(coords, on=["alcaldia", "colonia"], how="left")


# ──────────────────────────────────────────────────────────────────────────────
# 5. POBLACIÓN POR COLONIA (ids_ut.xlsx)
# ──────────────────────────────────────────────────────────────────────────────

def cargar_poblacion() -> pd.DataFrame:
    """
    Lee la hoja 'base_ut_final' del archivo ids_ut.xlsx y extrae la
    población (pob) por Unidad Territorial, que corresponde a colonias.

    NOTA: El ids_ut tiene 37 filas con encoding corrupto ('?' en lugar de
    caracteres especiales como ñ o á). Estas filas no coincidirán con el
    dataset de consumo y su población quedará imputada con la mediana de
    su alcaldía en el paso 9.4. Es una limitación conocida del archivo fuente.

    Columnas de salida: alcaldia, colonia, pob, idsm, e_idsm
    """
    print("[ETL] Cargando datos poblacionales (ids_ut)...")
    df = pd.read_excel(RUTAS["ids_ut"], sheet_name="base_ut_final")
    df = df.dropna(subset=["alcaldia", "nombre_ut"])
    df["alcaldia"] = df["alcaldia"].apply(normalizar)
    df["colonia"]  = df["nombre_ut"].apply(normalizar)

    agg = df.groupby(["alcaldia", "colonia"]).agg(
        pob    = ("pob",    "sum"),
        idsm   = ("idsm",   "mean"),
        e_idsm = ("e_idsm", "first"),
    ).reset_index()

    return agg


# ──────────────────────────────────────────────────────────────────────────────
# 6. REPORTES CIUDADANOS (SEGUIAGUA) POR COLONIA — AÑO 2019
# ──────────────────────────────────────────────────────────────────────────────

def cargar_reportes() -> pd.DataFrame:
    """
    Filtra reportes de 'Fuga' o 'Falta de agua' del año 2019 y los agrega
    por colonia.

    [FIX-F09] FILTRO TEMPORAL:
        El dataset original contiene reportes de 2018 a 2021 (~254,730 registros).
        Se filtra por año == ANIO_REFERENCIA (2019) para mantener coherencia
        temporal con el dataset de consumo.

    [MEJ-03] INDICADOR DE PRESENCIA EN JOIN (tiene_reportes_2019):
        Las colonias que sí aparecen en este resultado tienen reportes reales
        en 2019 (aunque su conteo sea 0 después del filtro de tipo de falla).
        Las colonias que NO aparecen recibirán total_reportes = 0 por el
        left join en el orquestador, pero ahí ese 0 significa "sin datos",
        no "nadie reportó nada".
        La columna 'tiene_reportes_2019' = True en este resultado permite que
        el orquestador marque correctamente cuáles colonias tuvieron presencia
        en el sistema de reportes y cuáles simplemente no hicieron match.

    Columnas de salida: alcaldia, colonia, total_reportes,
                        reportes_fuga, reportes_falta_agua,
                        tiene_reportes_2019
    """
    print(f"[ETL] Procesando reportes ciudadanos SEGUIAGUA (solo {ANIO_REFERENCIA})...")
    df = pd.read_csv(RUTAS["reportes"])

    # [FIX-F09] Filtrar solo el año de referencia
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    n_total     = len(df)
    df          = df[df["fecha"].dt.year == ANIO_REFERENCIA].copy()
    n_filtrados = len(df)
    print(
        f"  ℹ Reportes {ANIO_REFERENCIA}: {n_filtrados:,} de {n_total:,} totales "
        f"({n_filtrados / n_total * 100:.1f}% del dataset)"
    )

    df["alcaldia"] = df["alcaldia"].apply(normalizar)
    df["colonia"]  = df["colonia_datos_abiertos"].apply(normalizar)

    mask = df["tipo_de_falla"].str.contains(r"Fuga|Falta", case=False, na=False)
    df   = df[mask].copy()

    df["es_fuga"]       = df["tipo_de_falla"].str.contains("Fuga",  case=False, na=False)
    df["es_falta_agua"] = df["tipo_de_falla"].str.contains("Falta", case=False, na=False)

    agg = df.groupby(["alcaldia", "colonia"]).agg(
        total_reportes      = ("folio",         "count"),
        reportes_fuga       = ("es_fuga",        "sum"),
        reportes_falta_agua = ("es_falta_agua",  "sum"),
    ).reset_index()

    # [MEJ-03] Marcar colonias con presencia confirmada en el sistema de reportes 2019
    agg["tiene_reportes_2019"] = True

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

    # Columnas donde valores negativos son geográficamente válidos
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
    Para cada columna en `columnas`, imputa los valores NaN de una colonia
    con la mediana de las colonias válidas de su misma alcaldía.

    NOTA: Solo imputa NaN. Los ceros legítimos (como 0 reportes de fuga en
    una colonia que genuinamente no tuvo fugas) NO se reemplazan.
    Las columnas que deben imputar ceros se convierten a NaN explícitamente
    antes de llamar a esta función cuando el cero no es semánticamente válido
    (ej: superficie, población).

    [MEJ-04] Advertencia de alcaldías con una sola colonia:
        Si una alcaldía tiene solo 1 colonia en el dataset, su mediana es
        el valor de esa misma colonia. Si ese valor es NaN, la mediana también
        lo es y se usa el fallback global. Se emite advertencia para trazabilidad.
    """
    df = df.copy()
    for col in columnas:
        if col not in df.columns:
            continue

        # [MEJ-04] Detectar alcaldías con una sola colonia (mediana = valor único)
        tamanio_alcaldia = df.groupby("alcaldia")[col].transform("count")
        alcaldias_singleton = df.loc[
            (tamanio_alcaldia == 1) & df[col].isna(), "alcaldia"
        ].unique()
        if len(alcaldias_singleton):
            print(
                f"  ℹ [MEJ-04] Columna '{col}': {len(alcaldias_singleton)} alcaldía(s) "
                f"con 1 sola colonia y valor NaN → se usará mediana global como fallback. "
                f"Alcaldías: {list(alcaldias_singleton)}"
            )

        medianas = df.groupby("alcaldia")[col].transform("median")
        df[col]  = df[col].fillna(medianas)
        # Fallback global si toda la alcaldía es NaN
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
      3. Consumo          (consumo_agua_historico_2019 — primer semestre)
      4. Población        (ids_ut)
      5. Reportes SEGUIAGUA  (reportes_agua_hist — filtrado a 2019)

    Luego:
      - Imputa superficie y población faltantes con mediana de alcaldía.
      - Calcula KPIs (consumo_per_capita, consumo_per_capita_anualizado,
        densidad_poblacional).
      - Filtra Top-10 colonias de mayor consumo total por alcaldía.

    SOBRE EL FILTRO TOP-10:
        Con ~1,494 colonias únicas en el dataset y recursos universitarios
        limitados, se seleccionan las 10 colonias de mayor consumo total
        por alcaldía (≤160 colonias en total). Estas representan los focos
        de mayor volumen hídrico. Colonias de bajo consumo absoluto se
        excluyen — limitación documentada del alcance del proyecto.
    """
    print("\n" + "="*72)
    print("  HYDROTRACE AI — ETL PIPELINE COLONIA")
    print(f"  Año de referencia: {ANIO_REFERENCIA} | Cobertura: Primer Semestre (Bim. 1-3)")
    print("="*72)

    # ── 9.1 Base geográfica ──────────────────────────────────────────────────
    df_geo   = cargar_geometrias_colonias()
    df_suelo = cargar_uso_suelo()
    df_base  = df_geo.merge(df_suelo, on="alcaldia", how="left")
    print(f"\n  Base geográfica: {len(df_base)} colonias únicas en coloniascdmx.csv")

    # ── 9.2 Módulos de datos ─────────────────────────────────────────────────
    df_consumo  = cargar_consumo()
    df_pob      = cargar_poblacion()
    df_reportes = cargar_reportes()

    # ── 9.3 Merge progresivo con log de cobertura [MEJ-06] ──────────────────
    print("\n[ETL] Integrando fuentes (LEFT JOIN desde geometría oficial)...")

    df = df_base.merge(df_consumo,  on=["alcaldia", "colonia"], how="left")
    n_sin_consumo = df["consumo_total"].isna().sum()
    if n_sin_consumo:
        print(f"  ⚠ [MEJ-06] {n_sin_consumo} colonias sin match en consumo "
              f"({n_sin_consumo / len(df) * 100:.1f}%) → se imputarán.")

    df = df.merge(df_pob, on=["alcaldia", "colonia"], how="left")
    n_sin_pob = df["pob"].isna().sum()
    if n_sin_pob:
        print(f"  ⚠ [MEJ-06] {n_sin_pob} colonias sin match en población "
              f"({n_sin_pob / len(df) * 100:.1f}%) → se imputarán.")

    df = df.merge(df_reportes, on=["alcaldia", "colonia"], how="left")
    n_sin_reportes = df["tiene_reportes_2019"].isna().sum()
    if n_sin_reportes:
        print(f"  ℹ [MEJ-06] {n_sin_reportes} colonias sin presencia en reportes 2019 "
              f"({n_sin_reportes / len(df) * 100:.1f}%) → total_reportes = 0 (dato ausente, no cero real).")

    # Materializar indicador de presencia antes del fillna general [MEJ-03]
    df["tiene_reportes_2019"] = df["tiene_reportes_2019"].fillna(False)

    print(f"  → Colonias en dataset integrado: {len(df)}")

    # ── 9.4 Imputación robusta con mediana de alcaldía ───────────────────────
    # Para superficie y población: cero no es un valor válido (ninguna colonia
    # tiene 0 km² ni 0 habitantes), así que se reemplazan antes de imputar.
    # Para reportes: cero SÍ es válido (colonia sin reportes ese año), por lo
    # que solo se imputan NaN (colonias que no aparecen en el join de reportes).
    # consumo_total: tratado igual que superficie — cero no es consumo válido.
    print("[ETL] Imputando valores faltantes con mediana de alcaldía...")

    df["superficie_km2_calculada"] = df["superficie_km2_calculada"].replace(0, np.nan)
    df["pob"]                      = df["pob"].replace(0, np.nan)
    df["consumo_total"]            = df["consumo_total"].replace(0, np.nan)

    df = imputar_con_mediana_alcaldia(
        df,
        columnas=["superficie_km2_calculada", "pob", "consumo_total"],
    )

    # Columnas secundarias: NaN → 0 (colonias sin presencia en esa fuente)
    df = df.fillna({
        "consumo_total_dom"     : 0,
        "consumo_total_no_dom"  : 0,
        "consumo_anualizado"    : 0,
        "total_reportes"        : 0,
        "reportes_fuga"         : 0,
        "reportes_falta_agua"   : 0,
        "uso_suelo_num"         : 0,
        "idsm"                  : 0,
        "e_idsm"                : "Sin dato",
        "bimestres_cubiertos"   : 0,
    })

    # ── 9.5 Auditoría del dataset integrado ──────────────────────────────────
    auditar(
        df, "Dataset Maestro Integrado",
        cols_criticas=["consumo_total", "pob", "superficie_km2_calculada"],
    )

    # ── 9.6 KPIs de Negocio ──────────────────────────────────────────────────
    print("\n[ETL] Calculando KPIs de negocio...")

    # consumo_per_capita: basado en consumo semestral real
    df["consumo_per_capita"]           = df["consumo_total"]      / df["pob"]
    df["densidad_poblacional"]         = df["pob"]                / df["superficie_km2_calculada"]

    # [MEJ-02] consumo_per_capita_anualizado: proyección lineal para benchmarks anuales
    # Fórmula: (consumo_semestral / pob) * FACTOR_ANUALIZACION
    # ADVERTENCIA: proyección lineal — no captura estacionalidad hídrica.
    # Usar para comparar contra M3_PER_CAPITA_ANUAL en modelos_ml.py.
    df["consumo_per_capita_anualizado"] = (
        (df["consumo_anualizado"] / df["pob"]).round(4)
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df = imputar_con_mediana_alcaldia(
        df, columnas=["consumo_per_capita", "consumo_per_capita_anualizado",
                      "densidad_poblacional"]
    )

    # ── 9.7 Filtro Estratégico: Top-10 por alcaldía ──────────────────────────
    print("[ETL] Aplicando filtro Top-10 colonias por consumo por alcaldía...")
    df_top10 = (
        df.sort_values("consumo_total", ascending=False)
          .groupby("alcaldia", group_keys=False)
          .head(10)
          .reset_index(drop=True)
    )
    print(
        f"  → Colonias en dataset final: {len(df_top10)} "
        f"({df_top10['alcaldia'].nunique()} alcaldías × ≤10 colonias)"
    )

    # ── 9.8 Orden de columnas para el modelo ────────────────────────────────
    # [FIX-D05] pob_nbi eliminada: no la produce ninguna fuente del ETL.
    # [MEJ-02]  consumo_anualizado y consumo_per_capita_anualizado añadidos.
    # [MEJ-03]  tiene_reportes_2019 añadido para distinguir ausencia de dato vs cero.
    columnas_modelo = [
        "alcaldia", "colonia",
        "superficie_km2_calculada",
        "vocacion_principal", "uso_suelo_num",
        # Consumo semestral real (lo que el modelo debe usar internamente)
        "consumo_total", "consumo_total_dom", "consumo_total_no_dom",
        "bimestres_cubiertos",
        # Consumo anualizado (proyección ×2 para benchmarks externos)
        "consumo_anualizado",
        # Población y KPIs
        "pob",
        "consumo_per_capita",
        "consumo_per_capita_anualizado",
        "densidad_poblacional",
        # Reportes y calidad de señal ciudadana
        "total_reportes", "reportes_fuga", "reportes_falta_agua",
        "tiene_reportes_2019",
        # Índice socioeconómico
        "idsm", "e_idsm",
        # Coordenadas
        "latitud", "longitud",
    ]
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

        df_final.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

        print("\n" + "="*72)
        print(f"  [ÉXITO] Dataset exportado en: {OUTPUT_PATH}")
        print(f"  Dimensiones finales    : {df_final.shape}")
        print(f"  Colonias únicas        : {df_final['colonia'].nunique()}")
        print(f"  Alcaldías cubiertas    : {df_final['alcaldia'].nunique()}")
        print(f"  Cobertura temporal     : Primer semestre {ANIO_REFERENCIA} (bimestres 1-3)")
        print(f"  Columnas exportadas    : {list(df_final.columns)}")
        print("="*72)

        print("\n[MUESTRA] Primeras 5 filas del dataset maestro:")
        print(df_final.head().to_string())

        print("\n[ESTADÍSTICAS] Variables cuantitativas clave:")
        cols_stats = [
            "consumo_total", "consumo_anualizado", "pob",
            "consumo_per_capita", "consumo_per_capita_anualizado",
            "densidad_poblacional", "superficie_km2_calculada",
            "total_reportes",
        ]
        cols_stats = [c for c in cols_stats if c in df_final.columns]
        print(df_final[cols_stats].describe().round(2).to_string())

        # Resumen de cobertura de señal ciudadana [MEJ-03]
        n_con_reportes = df_final["tiene_reportes_2019"].sum() if "tiene_reportes_2019" in df_final.columns else "N/A"
        print(f"\n[COBERTURA] Colonias con presencia confirmada en reportes 2019: "
              f"{n_con_reportes} de {len(df_final)}")

        return df_final

    except Exception as exc:
        print(f"\n[ERROR CRÍTICO] El pipeline falló: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    ejecutar_pipeline()