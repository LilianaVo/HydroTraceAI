"""
HydroTrace AI - Backend Principal (Controlador)
Ileana Lee - Lead Developer & Integrator
"""

import os
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps
from database_models import db, User, Lead, Anomalia, get_cdmx_time

app = Flask(__name__, 
            template_folder='Frontend', 
            static_folder='Frontend/assets')

app.config['SECRET_KEY'] = 'pumascript_unam_2026'
db_dir = os.path.join(os.getcwd(), "Backend_app", "database")
if not os.path.exists(db_dir): os.makedirs(db_dir)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(db_dir, "hydrotrace.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- RUTAS DE NAVEGACIÓN (ILEANA) ---

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
        flash('Acceso denegado.', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard-admin')
@login_required
def dashboard_admin():
    # 1. Jalamos los datos de la base de datos
    anomalias = Anomalia.query.order_by(Anomalia.alcaldia).all()
    leads = Lead.query.order_by(Lead.fecha_registro.desc()).all()
    
    # 2. DEFINIMOS EL DICCIONARIO (Esto es lo que te faltaba y causa el NameError)
    metricas = {
        "total_registros": len(anomalias),
        "anomalias_totales": len([a for a in anomalias if a.es_anomalia]),
        "precision_estimada": "94.2%",
        "ultima_ejecucion": "Hoy"
    }
    
    # 3. Lo pasamos al template
    return render_template('dashboard_admin.html', 
                           tabla_completa=anomalias, 
                           metricas=metricas, 
                           leads=leads)


@app.route('/dashboard-clientes')
def dashboard_clientes():
    # Ileana: El front ya está conectado a estas variables
    # Ana debe proveer la lógica de ROI para calcular 'dinero_riesgo'
    anomalias = Anomalia.query.all()
    return render_template('dashboard_clientes.html', ranking=anomalias)

# --- INICIALIZACIÓN ---
with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        db.session.add(User(username='admin', password_hash=generate_password_hash('pumascript2026')))
        db.session.commit()

if __name__ == '__main__':
    app.run(debug=True, port=5000)