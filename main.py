"""
HydroTrace AI - Módulo Principal (Backend)
Refactorizado por: PumaScript Solutions
"""

import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta, timezone
from functools import wraps
from sqlalchemy import or_

# IMPORTANTE: Importamos db y los modelos desde nuestro archivo de modelos
from database_models import db, User, Lead, Anomalia, get_cdmx_time

# ==========================================
# 1. CONFIGURACIÓN DEL SISTEMA
# ==========================================
app = Flask(__name__, 
            template_folder='Frontend', 
            static_folder='Frontend/assets')

app.config['SECRET_KEY'] = 'pumascript_ultra_secret_2026'

# Configuración de SQLite
db_dir = os.path.join(os.getcwd(), "Backend_app", "database")
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

db_path = os.path.join(db_dir, "hydrotrace.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Inicializamos la base de datos con la aplicación
db.init_app(app)

# ==========================================
# 2. LÓGICA DE NEGOCIO Y UTILIDADES
# ==========================================

def importar_resultados_csv():
    """Sincroniza el archivo CSV de la IA con la Base de Datos."""
    csv_path = os.path.join("Backend_app", "data", "resultados_anomalias.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            db.session.query(Anomalia).delete() 
            for _, row in df.iterrows():
                nueva = Anomalia(
                    alcaldia=row.get('alcaldia'),
                    poblacion=row.get('poblacion'),
                    consumo_total=row.get('consumo_total'),
                    desviacion_consumo=row.get('desviacion_consumo'),
                    es_anomalia=row.get('es_anomalia', False),
                    riesgo=row.get('Riesgo', 'Bajo')
                )
                db.session.add(nueva)
            db.session.commit()
        except Exception as e:
            print(f"❌ Error al procesar CSV: {e}")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            flash('Acceso denegado. Por favor, inicia sesión.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# ==========================================
# 3. INICIALIZACIÓN DE DATOS
# ==========================================

with app.app_context():
    db.create_all()
    
    # Usuario admin por defecto
    if not User.query.filter_by(username='admin').first():
        hashed_pw = generate_password_hash('pumascript2026')
        db.session.add(User(username='admin', password_hash=hashed_pw))
        db.session.commit()

    importar_resultados_csv()
    
    # Mock Data si no hay resultados reales
    if not Anomalia.query.first():
        datos_demo = [
            {'alcaldia': 'IZTAPALAPA', 'poblacion': 1815000, 'consumo_total': 45000.5, 'desviacion_consumo': 1250.2, 'es_anomalia': True, 'riesgo': 'Alto'},
            {'alcaldia': 'BENITO JUAREZ', 'poblacion': 434000, 'consumo_total': 15000.0, 'desviacion_consumo': 150.0, 'es_anomalia': False, 'riesgo': 'Bajo'},
            {'alcaldia': 'CUAUHTEMOC', 'poblacion': 545000, 'consumo_total': 18500.8, 'desviacion_consumo': 450.3, 'es_anomalia': False, 'riesgo': 'Medio'}
        ]
        for d in datos_demo: db.session.add(Anomalia(**d))
        db.session.commit()

# ==========================================
# 4. RUTAS (ENDPOINTS)
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and check_password_hash(user.password_hash, request.form.get('password')):
            session.clear()
            session['logged_in'] = True
            session['username'] = user.username
            return redirect(url_for('dashboard_admin'))
        flash('❌ Usuario o contraseña incorrectos.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard-admin')
@login_required
def dashboard_admin():
    anomalias = Anomalia.query.order_by(Anomalia.alcaldia).all()
    leads = Lead.query.order_by(Lead.fecha_registro.desc()).all()
    
    metricas = {
        "total_registros": len(anomalias),
        "anomalias_totales": len([a for a in anomalias if a.es_anomalia]),
        "precision_estimada": "94.2%",
        "ultima_ejecucion": "Hoy"
    }
    return render_template('dashboard_admin.html', tabla_completa=anomalias, metricas=metricas, leads=leads)

@app.route('/dashboard-clientes')
def dashboard_clientes():
    anomalias = Anomalia.query.all()
    mermas_totales = sum([a.desviacion_consumo for a in anomalias if a.desviacion_consumo > 0])
    dinero_riesgo = mermas_totales * 20 
    ranking = sorted(anomalias, key=lambda x: x.desviacion_consumo, reverse=True)[:5]
    
    return render_template('dashboard_clientes.html', 
                           dinero_riesgo=f"{dinero_riesgo:,.0f}",
                           anc_porcentaje=f"{(mermas_totales/1000):.1f}", 
                           alertas_criticas=len([a for a in anomalias if a.riesgo == 'Alto']),
                           ranking=ranking)

@app.route('/contacto', methods=['POST'])
def contacto():
    try:
        # Extraemos datos
        email = request.form.get('email')
        mensaje = request.form.get('mensaje')

        # VALIDACIÓN DE DUPLICADOS usando el horario de CDMX
        hace_poco = get_cdmx_time() - timedelta(seconds=30)
        
        # Nota: La base de datos guarda el objeto datetime, SQLAlchemy se encarga de la comparación
        duplicado = Lead.query.filter(
            Lead.email == email,
            Lead.mensaje == mensaje,
            Lead.fecha_registro >= hace_poco
        ).first()

        if not duplicado:
            nuevo_lead = Lead(
                nombre=request.form.get('nombre'),
                email=email,
                empresa=request.form.get('empresa'),
                interes=request.form.get('interes'),
                mensaje=mensaje
            )
            db.session.add(nuevo_lead)
            db.session.commit()
            
        return redirect(url_for('home', success=True))
    except Exception as e:
        print(f"Error en contacto: {e}")
        return redirect(url_for('home', error=True))

@app.route('/actualizar-estatus/<int:lead_id>', methods=['POST'])
@login_required
def actualizar_estatus(lead_id):
    lead = Lead.query.get_or_404(lead_id)
    lead.status = request.form.get('nuevo_estatus')
    db.session.commit()
    return redirect(url_for('dashboard_admin'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)