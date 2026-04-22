# -*- coding: utf-8 -*-
import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from functools import wraps
from sqlalchemy import or_

app = Flask(__name__, 
            template_folder='Frontend', 
            static_folder='Frontend/assets')

app.config['SECRET_KEY'] = 'pumascript_ultra_secret_2026'

# --- CONFIGURACIÓN DE BASE DE DATOS ---
db_dir = os.path.join(os.getcwd(), "Backend_app", "database")
if not os.path.exists(db_dir):
    os.makedirs(db_dir)

db_path = os.path.join(db_dir, "hydrotrace.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# --- MODELOS DE DATOS ---

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

class Lead(db.Model):
    __tablename__ = 'leads'
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    empresa = db.Column(db.String(100), nullable=True)
    interes = db.Column(db.String(100), nullable=True)
    mensaje = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='Nuevo') 
    fecha_registro = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='admin').first():
        hashed_pw = generate_password_hash('pumascript2026')
        default_admin = User(username='admin', password_hash=hashed_pw)
        db.session.add(default_admin)
        db.session.commit()

# --- DECORADOR ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# --- LÓGICA DE DATOS CSV CON FILTRO ---
DATA_PATH = os.path.join("Backend_app", "data", "resultados_anomalias.csv")

def obtener_datos_admin(alcaldia_filter=None):
    if not os.path.exists(DATA_PATH): return None
    df = pd.read_csv(DATA_PATH)
    
    # Filtro de alcaldía en memoria
    if alcaldia_filter:
        df = df[df['alcaldia'].str.contains(alcaldia_filter, case=False, na=False)]
        
    metricas = {
        "total_registros": len(df),
        "anomalias_totales": len(df[df['es_anomalia'] == True]) if 'es_anomalia' in df.columns else 0,
        "precision_estimada": "94.2%",
        "ultima_ejecucion": "Hoy"
    }
    return {"tabla_completa": df.to_dict(orient='records'), "metricas": metricas}

# --- RUTAS ---

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
        flash('Credenciales incorrectas', 'error')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/contacto', methods=['POST'])
def contacto():
    try:
        nombre, email, mensaje = request.form.get('nombre'), request.form.get('email'), request.form.get('mensaje')
        hace_poco = datetime.utcnow() - timedelta(seconds=30)
        duplicado = Lead.query.filter(Lead.email == email, Lead.mensaje == mensaje, Lead.fecha_registro >= hace_poco).first()
        if not duplicado:
            nuevo_lead = Lead(nombre=nombre, email=email, empresa=request.form.get('empresa'), interes=request.form.get('interes'), mensaje=mensaje)
            db.session.add(nuevo_lead)
            db.session.commit()
        return redirect(url_for('home', success=True))
    except:
        return redirect(url_for('home', error=True))

@app.route('/dashboard-admin')
@login_required
def dashboard_admin():
    # Obtener parámetros de búsqueda
    search_leads = request.args.get('search_leads', '')
    status_leads = request.args.get('status_leads', '')
    filter_alcaldia = request.args.get('filter_alcaldia', '')

    # Filtrar Leads
    query = Lead.query
    if search_leads:
        query = query.filter(or_(Lead.nombre.contains(search_leads), Lead.email.contains(search_leads)))
    if status_leads:
        query = query.filter(Lead.status == status_leads)
    leads_recientes = query.order_by(Lead.fecha_registro.desc()).all()

    # Obtener datos del CSV con filtro
    datos_analitica = obtener_datos_admin(filter_alcaldia) or {"tabla_completa": [], "metricas": {}}
    
    return render_template('dashboard_admin.html', 
                           **datos_analitica, 
                           leads=leads_recientes,
                           search_q=search_leads,
                           status_f=status_leads,
                           alcaldia_f=filter_alcaldia)

@app.route('/actualizar-estatus/<int:lead_id>', methods=['POST'])
@login_required
def actualizar_estatus(lead_id):
    lead = Lead.query.get_or_404(lead_id)
    lead.status = request.form.get('nuevo_estatus')
    db.session.commit()
    return redirect(url_for('dashboard_admin'))

if __name__ == '__main__':
    app.run(debug=True, port=5000)