from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta, timezone

# Inicializamos la instancia de SQLAlchemy
db = SQLAlchemy()

def get_cdmx_time():
    """Retorna la hora actual en Ciudad de México (UTC-6)."""
    return datetime.now(timezone(timedelta(hours=-6)))

class User(db.Model):
    """Administradores con acceso al panel de control."""
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

class Lead(db.Model):
    """Prospectos capturados desde el formulario de contacto."""
    __tablename__ = 'leads'
    id = db.Column(db.Integer, primary_key=True)
    nombre = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    empresa = db.Column(db.String(100), nullable=True)
    interes = db.Column(db.String(100), nullable=True)
    mensaje = db.Column(db.Text, nullable=True)
    status = db.Column(db.String(20), default='Nuevo') 
    # Usamos la función de CDMX para el valor por defecto
    fecha_registro = db.Column(db.DateTime, default=get_cdmx_time)

class Anomalia(db.Model):
    """Resultados finales del procesamiento de Machine Learning."""
    __tablename__ = 'anomalias'
    id = db.Column(db.Integer, primary_key=True)
    alcaldia = db.Column(db.String(50), nullable=False)
    poblacion = db.Column(db.Integer)
    consumo_total = db.Column(db.Float)
    desviacion_consumo = db.Column(db.Float)
    es_anomalia = db.Column(db.Boolean, default=False)
    riesgo = db.Column(db.String(20)) # 'Alto', 'Medio', 'Bajo'
    fecha_analisis = db.Column(db.DateTime, default=get_cdmx_time)