💧 HydroTrace AI

Optimización hídrica impulsada por IA Predictiva para la CDMX

Bienvenido al repositorio oficial de HydroTrace AI, una solución desarrollada por PumaScript Solutions (Facultad de Ingeniería, UNAM). Nuestro objetivo es combatir la crisis del agua y el "Agua No Contabilizada" (ANC) mediante inteligencia de datos, detectando fugas y extracciones irregulares en tiempo real.

🚀 Stack Tecnológico

Data Science & ML: Python, Pandas, Scikit-Learn, NumPy

Modelos: Isolation Forest (Anomalías), K-Means (Segmentación), Regresión Lineal (Consumo Teórico)

Backend: Flask

Frontend: HTML5, Tailwind CSS, JavaScript (Vanilla), Lucide Icons

📁 Estructura del Proyecto

El proyecto está diseñado de forma modular para separar el procesamiento de datos de la interfaz web:

HydroTraceAI/
├── data/                       # Archivos CSV fuente y el dataset maestro generado
├── Frontend/                   # Plantillas HTML para el servidor Flask
│   ├── index.html              # Landing Page
│   ├── dashboard_clientes.html # Vista para el organismo regulador (ROI)
│   └── dashboard_admin.html    # Consola técnica de PumaScript
├── Frontend/assets/            # Recursos estáticos (CSS, JS, imágenes)
├── etl_pipeline.py             # Script 1: Limpieza y unificación de datos
├── modelos_ml.py               # Script 2: Entrenamiento y predicción de IA
├── main.py                     # Script 3: Servidor web y API con Flask
└── README.md                   # Este archivo


⚙️ ¿Cómo ejecutar el proyecto localmente?

Para que los dashboards funcionen correctamente, es necesario seguir el flujo de datos exacto: desde la limpieza hasta el despliegue del servidor.

1. Preparar el entorno

Asegúrate de tener Python instalado y ejecuta el siguiente comando para instalar las dependencias necesarias:

pip install flask pandas scikit-learn numpy


2. Procesar los datos (ETL)

Toma los archivos fuente de la carpeta /data y genera el dataset_maestro.csv limpio y unificado:

python etl_pipeline.py


3. Entrenar y ejecutar los Modelos (Machine Learning)

Lee el dataset maestro, aplica los algoritmos de detección y genera el archivo resultados_anomalias.csv (que alimentará la web):

python modelos_ml.py


4. Levantar el Servidor Web

Inicia la aplicación Flask para visualizar la plataforma:

python main.py


5. Visualizar

Abre tu navegador web y visita:

🌍 Landing Page: http://127.0.0.1:5000/

📊 Dashboard Ejecutivo: http://127.0.0.1:5000/dashboard-clientes

💻 Admin Console: http://127.0.0.1:5000/dashboard-admin

Desarrollado con 🖤 por el equipo de PumaScript Solutions - UNAM (2026).