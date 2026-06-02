"""
================================================================================
EQUIPO: PUMASCRIPT SOLUTIONS

PROYECTO : HydroTrace AI
MÓDULO   : database_models.py
AUTOR  : Lee Obando Ileana Verónica (PM/Lead Data Scientist)
Materia : Ciencia de Datos en la Toma de Decisiones en las Organizaciones
GRUPO: 04
Facultad de Ingeniería, UNAM | Ciudad de México, 2026

DESCRIPCIÓN:
    Define los modelos ORM de la base de datos usando Flask-SQLAlchemy.
    Contiene dos entidades: User (administradores del panel) y Lead
    (prospectos capturados desde el formulario de contacto público).

    Este módulo es importado por main.py para inicializar la base de datos
    y registrar los modelos antes de crear las tablas con db.create_all().

================================================================================
"""

from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta, timezone


# ==============================================================================
# SECCIÓN 1: INICIALIZACIÓN DE LA INSTANCIA ORM
# ==============================================================================

# La instancia se crea sin app para permitir el patrón Application Factory.
# main.py la vincula a la app con db.init_app(app) antes de crear las tablas.
db = SQLAlchemy()


# ==============================================================================
# SECCIÓN 2: UTILIDADES DE TIEMPO
# ==============================================================================

def get_cdmx_time():
    """
    Retorna la hora actual en Ciudad de México (UTC-6).

    Se usa como default dinámico en Lead.fecha_registro para que cada nuevo
    registro tome la hora de CDMX al momento de la inserción, no la hora UTC
    del servidor (que puede diferir según el entorno de despliegue).
    """
    return datetime.now(timezone(timedelta(hours=-6)))


# ==============================================================================
# SECCIÓN 3: MODELOS DE BASE DE DATOS
# ==============================================================================

class User(db.Model):
    """
    Administradores con acceso al panel de control de HydroTrace AI.

    Solo los usuarios registrados en esta tabla pueden autenticarse
    en las rutas protegidas del dashboard de administrador en main.py.
    Las contraseñas se almacenan como hash — nunca en texto plano.
    """
    __tablename__ = 'users'

    id            = db.Column(db.Integer,     primary_key=True)
    username      = db.Column(db.String(50),  unique=True, nullable=False)
    # Hash generado con werkzeug.security.generate_password_hash en main.py
    password_hash = db.Column(db.String(128), nullable=False)


class Lead(db.Model):
    """
    Prospectos capturados desde el formulario de contacto público del dashboard.

    Cada fila representa una empresa o persona que expresó interés en el
    sistema HydroTrace AI. El campo 'status' permite al administrador
    llevar seguimiento del ciclo de ventas desde el panel de control.

    Flujo típico de un lead:
        Formulario público → status='Nuevo' → revisión admin → 'Contactado' / 'Cerrado'
    """
    __tablename__ = 'leads'

    id             = db.Column(db.Integer,     primary_key=True)
    nombre         = db.Column(db.String(100), nullable=False)
    email          = db.Column(db.String(100), nullable=False)
    empresa        = db.Column(db.String(100), nullable=True)
    telefono       = db.Column(db.String(30),  nullable=True)
    interes        = db.Column(db.String(100), nullable=True)
    mensaje        = db.Column(db.Text,        nullable=True)

    # Estado del lead en el pipeline de ventas — gestionado desde el panel de admin
    status         = db.Column(db.String(20),  default='Nuevo')

    # Se asigna automáticamente al insertar — no requiere valor en el formulario
    fecha_registro = db.Column(db.DateTime,    default=get_cdmx_time)