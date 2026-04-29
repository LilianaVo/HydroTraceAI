"""
PROYECTO: HydroTrace AI - Centro de Mando (Backend)
UBICACIÓN: /backend/main.py
RESPONSABLE DE INTEGRACIÓN: Ileana Lee

ESTRUCTURA:
1. Configuración y Dependencias
2. Herramientas de Validación ML (Apoyo para Irving)
3. Lógica de Negocio y Finanzas (Ana)
4. Lógica de Visualización Geoespacial (Christian)
5. Seguridad y Sesiones
6. Rutas de Integración y Dashboards
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Importamos la base de datos y modelos
from database_models import db, User, Lead, Anomalia, get_cdmx_time

# ─────────────────────────────────────────────────────────────────────────────
# 1. CONFIGURACIÓN DEL SISTEMA
# ─────────────────────────────────────────────────────────────────────────────

base_dir = os.path.abspath(os.path.dirname(__file__))
frontend_dir = os.path.abspath(os.path.join(base_dir, '..', 'Frontend'))

app = Flask(__name__, 
            template_folder=frontend_dir, 
            static_folder=os.path.join(frontend_dir, 'assets'))

app.config['SECRET_KEY'] = 'pumascript_ultra_secret_2026'

db_dir = os.path.abspath(os.path.join(base_dir, "..", "Backend_app", "database"))
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(db_dir, "hydrotrace.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# ─────────────────────────────────────────────────────────────────────────────
# 2. HERRAMIENTAS DE VALIDACIÓN ML ILEANA & IRVING
# ─────────────────────────────────────────────────────────────────────────────

def grafica_metodo_codo(X_scaled, max_k=10):
    """Genera la gráfica de codo para justificar k clusters."""
    output_dir = os.path.abspath(os.path.join(base_dir, '..', 'graficas_reporte'))
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    distortions = []
    K = range(1, max_k + 1)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, n_init=10, random_state=42)
        kmeanModel.fit(X_scaled)
        distortions.append(kmeanModel.inertia_)
    
    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, 'bx-', color='#06b6d4', linewidth=2)
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('Inercia (Distorsión)')
    plt.title('Método del Codo para Selección de k')
    
    path = os.path.join(output_dir, 'metodo_codo_justificacion.png')
    plt.savefig(path)
    plt.close()
    print(f"[ML_SUPPORT] Gráfica del codo generada en: {path}")

def visualizar_clusters_2d(X_scaled, clusters):
    """Proyecta los clusters en 2D usando PCA."""
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=clusters, palette='viridis', s=100)
    plt.title('Visualización de Segmentación de Alcaldías (PCA)')
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(title='Cluster')
    
    output_dir = os.path.abspath(os.path.join(base_dir, '..', 'graficas_reporte'))
    path = os.path.join(output_dir, 'visualizacion_clusters_pca.png')
    plt.savefig(path)
    plt.close()
    print(f"[ML_SUPPORT] Mapa de clusters PCA generado en: {path}")

def reporte_desempeno_negocio(df):
    """Compara anomalías detectadas vs reportes históricos."""
    match = df[(df['es_anomalia'] == -1) & (df['total_reportes'] > df['total_reportes'].median())]
    total_anomalias = len(df[df['es_anomalia'] == -1])
    accuracy_negocio = (len(match) / total_anomalias) * 100 if total_anomalias > 0 else 0
    
    print(f"\n[KPI NEGOCIO] Confianza de Detección: {accuracy_negocio:.2f}%")
    return accuracy_negocio

# ─────────────────────────────────────────────────────────────────────────────
# 3. LÓGICA DE NEGOCIO Y FINANZAS (ANA)
# ─────────────────────────────────────────────────────────────────────────────

def calcular_impacto_economico_ana(m3_desviados):
    """Ana: Implementar tarifas de SACMEX aquí."""
    # TODO: Investigar costo por m3 y retornar total
    return 0 

def simular_ahorro_prescriptivo(m3_totales, capacidad_reparacion):
    """Ana: Implementar lógica de ROI aquí."""
    # TODO: (m3_totales * tarifa) * capacidad_reparacion
    return 0

# Demás funciones...


# ─────────────────────────────────────────────────────────────────────────────
# 4. LÓGICA DE VISUALIZACIÓN GEOESPACIAL (CHRISTIAN)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/mapa-interactivo')
def mapa_interactivo():
    """Sirve el mapa que Christian debe generar."""
    try:
        return send_from_directory(os.path.join(frontend_dir, 'assets'), 'mapa_cdmx.html')
    except Exception:
        return "Mapa en construcción por Christian..."

# Demás funciones....

# ─────────────────────────────────────────────────────────────────────────────
# 5. RUTAS DE INTEGRACIÓN Y DASHBOARDS (ILEANA)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ejecutar-entrenamiento')
def disparar_ml():
    """
    Ruta para que Irving pruebe su modelo.
    Importa el script de modelos_ml y lo ejecuta.
    """
    try:
        import modelos_ml
        modelos_ml.ejecutar_pipeline_ml()
        flash('Modelos entrenados y resultados actualizados.', 'success')
    except Exception as e:
        flash(f'Error al entrenar modelos: {e}', 'error')
    return redirect(url_for('dashboard_admin'))

@app.route('/dashboard-admin')
def dashboard_admin():
    # 1. Traemos las anomalías ordenadas por fecha (lo más reciente arriba)
    anomalias = Anomalia.query.order_by(Anomalia.fecha_analisis.desc()).all()
    
    # 2. Traemos los leads ordenados por fecha de registro (lo más nuevo arriba)
    leads = Lead.query.order_by(Lead.fecha_registro.desc()).all()
    
    # 3. Ajustamos las métricas (OJO: corregí 'precision_ia' por 'precision_estimada' 
    #    para que coincida con tu dashboard_admin.html)
    metricas = {
        "total_registros": len(anomalias),
        "anomalias_totales": len([a for a in anomalias if a.es_anomalia]),
        "leads_pendientes": len([l for l in leads if l.status == 'Nuevo']),
        "precision_estimada": "94.2%" # Aquí corregí el nombre de la llave
    }
    
    return render_template('dashboard_admin.html', 
                           tabla_completa=anomalias, 
                           leads=leads, 
                           metricas=metricas)

@app.route('/dashboard-clientes', methods=['GET', 'POST']) # Agregamos POST
def dashboard_clientes():
    # Si vienes del formulario (POST), guardamos el "Lead"
    if request.method == 'POST':
        nuevo_lead = Lead(
            nombre=request.form.get('nombre'),
            email=request.form.get('email'),
            empresa=request.form.get('empresa'),
            interes=request.form.get('interes'),
            mensaje=request.form.get('mensaje')
        )
        db.session.add(nuevo_lead)
        db.session.commit()
        return redirect(url_for('dashboard_clientes'))
    anomalias = Anomalia.query.all()
    m3_totales = sum([a.desviacion_consumo for a in anomalias])
    
    dinero_en_riesgo = calcular_impacto_economico_ana(m3_totales)
    
    return render_template('dashboard_clientes.html', 
                           ranking=anomalias,
                           dinero_riesgo=f"{dinero_en_riesgo:,.2f}",
                           anc_porcentaje=40.2,
                           alertas_criticas=len([a for a in anomalias if a.riesgo == 'Alto']))

@app.route('/actualizar-estatus/<int:lead_id>', methods=['POST'])
def actualizar_estatus(lead_id):
    lead = Lead.query.get_or_404(lead_id)
    lead.status = request.form.get('nuevo_estatus')
    db.session.commit()
    return redirect(url_for('dashboard_admin'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)