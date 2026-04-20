# 💧 HydroTrace AI

**Optimización hídrica con IA predictiva para la CDMX**

HydroTrace AI es una solución desarrollada por **PumaScript Solutions (Facultad de Ingeniería, UNAM)** enfocada en combatir la crisis del agua en la Ciudad de México.

El sistema utiliza **ciencia de datos e inteligencia artificial** para detectar:

* Fugas en la red hidráulica
* Extracciones irregulares
* Patrones anómalos de consumo

Todo esto en tiempo (casi) real, con el objetivo de reducir el **Agua No Contabilizada (ANC)** y mejorar la eficiencia operativa.

---

## Proyecto enfocado en:

**Data Science & Machine Learning**

* Python
* Pandas
* NumPy
* Scikit-Learn

**Modelos implementados**

* Isolation Forest → detección de anomalías
* K-Means → segmentación de usuarios
* Regresión Lineal → estimación de consumo teórico

**Backend**

* Flask

**Frontend**

* HTML5
* Tailwind CSS
* JavaScript (Vanilla)
* Lucide Icons

---

## 📁 Estructura del Proyecto

El proyecto sigue una arquitectura modular que separa claramente el procesamiento de datos, los modelos y la interfaz web:

```
HydroTraceAI/
├── data/                       # Archivos fuente (CSV) y dataset maestro
├── Frontend/                   # Plantillas HTML (Flask)
│   ├── index.html              # Landing page
│   ├── dashboard_clientes.html # Vista ejecutiva (organismo regulador)
│   └── dashboard_admin.html    # Consola técnica (PumaScript)
├── Frontend/assets/            # Recursos estáticos (CSS, JS, imágenes)
├── etl_pipeline.py             # ETL: limpieza y unificación de datos
├── modelos_ml.py               # Modelos de IA: entrenamiento y predicción
├── main.py                     # Servidor Flask + API
└── README.md                   # Documentación del proyecto
```

---

## ⚙️ Ejecución local

Para ejecutar correctamente el proyecto, sigue el flujo completo de datos:

### 1. Preparar el entorno

Asegúrate de tener Python instalado y luego instala las dependencias desde el archivo requirements.txt:

```bash
pip install -r requirements.txt

```

---

### 2. Procesar datos (ETL)

Genera el dataset limpio y unificado:

```bash
py etl_pipeline.py
```

Esto creará el archivo:

```
dataset_maestro.csv
```

---

### 3. Ejecutar modelos de Machine Learning

Procesa el dataset y genera las predicciones:

```bash
py modelos_ml.py
```

Salida:

```
resultados_anomalias.csv
```

---

### 4. Levantar el servidor

Inicia la aplicación web:

```bash
py main.py
```

---

### 5. Visualización

---

## Objetivo del proyecto

HydroTrace AI busca demostrar cómo la inteligencia artificial puede aplicarse a problemas urbanos reales, generando herramientas que apoyen la toma de decisiones en infraestructura crítica.

---

## Equipo

Desarrollado por **PumaScript Solutions**
Facultad de Ingeniería, UNAM — 2026

---
