# HydroTrace AI

**PumaScript Solutions — Facultad de Ingeniería, UNAM | Ciudad de México, 2026**  
Materia: Ciencia de Datos en la Toma de Decisiones en las Organizaciones — Grupo 04

---

## Descripcion

HydroTrace AI es un sistema de detección de anomalías de consumo hídrico para la Ciudad de México. Integra datos abiertos de SACMEX, los procesa a nivel colonia y aplica un pipeline de tres modelos de Machine Learning para identificar zonas con consumo irregular, posible huachicol de agua o fugas no reportadas.

El sistema genera métricas financieras, mapas interactivos y un panel de administración con CRM integrado, orientado a ser adoptado como servicio SaaS B2G por organismos como la Agencia del Agua de la CDMX (SEGIAGUA).

---

## Equipo

| Integrante | Rol |
|---|---|
| Lee Obando Ileana Veronica | Project Manager · Lead Data Scientist · UX/UI |
| Garcia Lopez Bolivar | Data Engineer |
| Morales Esteban Irving | QA Data Tester · Data Scientist |
| Ramirez Monzon Ana Cristina | Business Analyst · Financial Lead |
| Romero Pizano Christian Gustavo | Data Visualization Specialist |

---

## Arquitectura del Sistema

```
Frontend/
├── index.html                  # Landing page publica con formulario de contacto
├── dashboard_clientes.html     # Panel publico (orientado a SEGIAGUA)
└── dashboard_admin.html        # Panel interno con CRM y control de modelos

Backend_app/
├── main.py                     # Punto de entrada Flask — orquesta rutas y APIs
├── database_models.py          # Modelos ORM (User, Lead) via Flask-SQLAlchemy
├── etl_pipeline.py             # Pipeline ETL a nivel colonia
├── modelos_ml.py               # Pipeline de Machine Learning (3 modelos)
└── data/
    ├── consumo_agua_historico_2019.csv
    ├── ids_ut.xlsx
    ├── coloniascdmx.csv
    ├── reportes_agua_hist.csv
    ├── superficie_alcaldias.csv
    ├── dataset_maestro_colonia_final.csv   # Salida del ETL
    └── resultados_finales_IA.csv           # Salida del pipeline ML
```

---

## Pipeline de Datos

### 1. ETL (`etl_pipeline.py`)

Integra cinco fuentes heterogeneas a nivel colonia:

- Consumo historico SACMEX (bimestres 1–3 de 2019, enero–junio)
- IDs de unidad territorial del gobierno de CDMX
- Catalogo de colonias de la CDMX
- Reportes ciudadanos de fallas de agua (filtrados a 2019)
- Superficie de alcaldias con geometrias GeoJSON

Calcula areas reales por colonia usando la proyeccion oficial de Mexico **ITRF2008/LCC (EPSG:6372)**, imputa valores faltantes con medianas por alcaldia y exporta el Top-10 de mayor consumo por alcaldia para el paso de ML.

**Salida:** `dataset_maestro_colonia_final.csv`

### 2. Machine Learning (`modelos_ml.py`)

Pipeline de tres modelos encadenados:

| Modelo | Proposito |
|---|---|
| K-Means (k=4) | Segmentacion en perfiles urbanos homogeneos |
| Regresion Lineal | Linea base de consumo esperado por colonia |
| Isolation Forest | Deteccion de anomalias dentro de cada cluster |

El diagnostico final etiqueta cada colonia como:

- `HUACHICOL` — exceso de consumo + reportes ciudadanos confirmados
- `SOSPECHOSO (Exceso IA)` — anomalia detectada sin reportes previos
- `FUGA` — exceso de consumo + reportes de fallas de infraestructura
- `NORMAL` — consumo dentro del rango esperado del cluster

**Salida:** `resultados_finales_IA.csv`

### 3. Backend API (`main.py`)

Flask expone los resultados como endpoints JSON consumidos por los dashboards:

| Ruta | Descripcion |
|---|---|
| `/` | Landing page con formulario de contacto |
| `/dashboard-clientes` | Panel publico |
| `/dashboard-admin` | Panel interno (requiere autenticacion) |
| `/api/mapa-datos` | Puntos GeoJSON para Leaflet.js |
| `/api/impacto-economico` | Metricas financieras completas |
| `/api/modelo-negocio` | Plan SaaS B2G |
| `/api/resumen` | KPIs generales |
| `/api/leads` | Lista del CRM |
| `/ejecutar-entrenamiento` | Dispara `modelos_ml.py` y refresca cache |

El CSV se carga una sola vez en memoria con `lru_cache` para minimizar I/O entre requests.

---

## Modelo de Negocio

HydroTrace AI se plantea como servicio **SaaS B2G** (Software as a Service orientado a gobierno):

| Concepto | Valor |
|---|---|
| Suscripcion mensual | $85,000 MXN |
| Suscripcion anual | $900,000 MXN |
| OPEX mensual | $24,750 MXN |
| CAPEX inicial | $445,000 MXN |

Las metricas financieras se calculan usando la tarifa SACMEX vigente para uso comercial/industrial: **$18.40 MXN/m³** (Gaceta Oficial CDMX, enero 2024), con una capacidad de recuperacion estimada del 20% del volumen excedente identificado.

---

## Instalacion

### Requisitos

Python 3.10 o superior.

### Dependencias

```
pip install flask flask-sqlalchemy pandas numpy scikit-learn \
            matplotlib seaborn pyproj shapely werkzeug
```

O con el archivo de requirements:

```
pip install -r requirements.txt
```

> **Nota:** El archivo `requirements.txt` incluido en el repositorio puede estar incompleto respecto a las dependencias reales del proyecto. Las librerias listadas arriba son las efectivamente usadas segun el codigo fuente. Se recomienda instalarlas manualmente o usar un entorno virtual.

### Ejecucion

```bash
# 1. Ejecutar el ETL para generar el dataset maestro
python etl_pipeline.py

# 2. Entrenar los modelos y generar resultados_finales_IA.csv
python modelos_ml.py

# 3. Iniciar el servidor Flask
python main.py
```

El servidor estara disponible en `http://localhost:5000`.

Para el primer arranque, `main.py` crea automaticamente la base de datos SQLite y el usuario administrador por defecto.

---

## Cobertura Temporal

Los datos de consumo cubren el **primer semestre de 2019** (bimestres 1–3: enero–junio). Los bimestres 4–6 no estan disponibles en los datos abiertos de SACMEX. Todos los KPIs de consumo, anomalias y metricas financieras reflejan exclusivamente ese periodo.

Para comparaciones contra benchmarks anuales, el ETL genera la columna `consumo_per_capita_anualizado` (consumo semestral × 2).

---

## Stack Tecnologico

**Backend**  
Flask · Flask-SQLAlchemy · SQLite · Python 3.10+

**Machine Learning**  
scikit-learn (K-Means, Isolation Forest, Linear Regression) · pandas · numpy · matplotlib · seaborn

**Geoprocesamiento**  
pyproj · shapely · Leaflet.js · leaflet-heat

**Frontend**  
HTML5 · Tailwind CSS · Chart.js 4.4.1 · Leaflet.js 1.9.4

---

## Fuentes de Datos

- SACMEX — Consumo de agua historico 2019 (datos abiertos)
- Gobierno CDMX — Catalogo de colonias y unidades territoriales
- SACMEX 2019 — Consumo per capita promedio CDMX: 366 L/hab/dia
- Gaceta Oficial CDMX, enero 2024 — Tarifa comercial/industrial: $18.40 MXN/m³
- GeoJSON de limites de colonias de la CDMX (EPSG:4326)

---

## Creditos

Proyecto academico desarrollado para la materia **Ciencia de Datos en la Toma de Decisiones en las Organizaciones**, Grupo 04.  
Facultad de Ingenieria, Universidad Nacional Autonoma de Mexico — 2026.

**PumaScript Solutions**