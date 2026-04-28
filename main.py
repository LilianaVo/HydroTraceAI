"""
PROYECTO: HydroTrace AI - Ciencia de Datos UNAM
RESPONSABLE DE INTEGRACIÓN: Ileana Lee
RESPONSABLES DE LÓGICA: 
- Ana Cristina (Business Analyst): Finanzas y ROI
- Christian Gustavo (Viz Specialist): Mapa Geoespacial e Interacción
"""

import os
from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from database_models import db, User, Lead, Anomalia, get_cdmx_time

app = Flask(__name__, 
            template_folder='Frontend', 
            static_folder='Frontend/assets')

app.config['SECRET_KEY'] = 'pumascript_ultra_secret_2026'
db_dir = os.path.join(os.getcwd(), "Backend_app", "database")
if not os.path.exists(db_dir): os.makedirs(db_dir)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(db_dir, "hydrotrace.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

# =============================================================================
# TAREAS DE ANA: LÓGICA FINANCIERA (Unidad 6)
# =============================================================================

def calcular_impacto_economico_ana(m3_desviados):
    """
    TAREA 1: Valor monetario del agua.
    TODO: Investigar tarifas de SACMEX y retornar ahorro en pesos.
    """
    return 0 

def simular_ahorro_intervencion(m3_totales, porcentaje_exito):
    """
    TAREA 2: Modelo Prescriptivo.
    TODO: Implementar fórmula de ROI.
    """
    return 0

# =============================================================================
# TAREAS DE CHRISTIAN: VISUALIZACIÓN GEOESPACIAL (Unidad 3)
# =============================================================================

def generar_capas_mapa_christian(anomalias):
    """
    TAREA 1: Procesamiento de Coordenadas.
    Objetivo: Christian debes crear un script (usando Folium o Plotly) que genere
    un archivo 'mapa_cdmx.html' basado en el nivel de riesgo de cada alcaldía.
    """
    # TODO: Christian debes mapear las alcaldías con sus coordenadas Lat/Lon
    # y asignar colores según el riesgo (Rojo=Alto, Naranja=Medio, Verde=Bajo).
    pass

@app.route('/mapa-interactivo')
def mapa_interactivo():
    """
    TAREA 2: Servir el mapa dinámico.
    Christian debe asegurar que el archivo 'mapa_cdmx.html' se guarde en la carpeta 
    'Frontend/assets/' para que esta ruta lo pueda mostrar en el iframe del dashboard.
    """
    return send_from_directory('Frontend/assets', 'mapa_cdmx.html')

# =============================================================================
# SEGURIDAD Y SESIONES (ILEANA)
# =============================================================================

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# =============================================================================
# RUTAS DE LA APLICACIÓN
# =============================================================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password_hash, request.form.get('password')):
            session['logged_in'] = True
            session['username'] = user.username
            return redirect(url_for('dashboard_admin'))
        flash('Credenciales incorrectas.', 'error')
    return render_template('login.html')

@app.route('/dashboard-admin')
@login_required
def dashboard_admin():
    anomalias = Anomalia.query.all()
    leads = Lead.query.all()
    
    metricas = {
        "total_registros": len(anomalias),
        "anomalias_totales": len([a for a in anomalias if a.es_anomalia]),
        "leads_pendientes": len([l for l in leads if l.status == 'Nuevo'])
    }
    
    return render_template('dashboard_admin.html', 
                           tabla_completa=anomalias, 
                           leads=leads, 
                           metricas=metricas)

@app.route('/dashboard-clientes')
def dashboard_clientes():
    anomalias = Anomalia.query.all()
    m3_totales = sum([a.desviacion_consumo for a in anomalias])
    
    # Datos que vienen de la lógica de Ana
    dinero_en_riesgo = calcular_impacto_economico_ana(m3_totales)
    
    return render_template('dashboard_clientes.html', 
                           ranking=anomalias,
                           dinero_riesgo=f"{dinero_en_riesgo:,.2f}",
                           anc_porcentaje=40.2, 
                           alertas_criticas=len([a for a in anomalias if a.riesgo == 'Alto']))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('home'))

# =============================================================================
# INICIALIZACIÓN
# =============================================================================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username='admin').first():
            db.session.add(User(
                username='admin', 
                password_hash=generate_password_hash('pumascript2026')
            ))
            db.session.commit()
            
    app.run(debug=True, port=5000)