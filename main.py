# -*- coding: utf-8 -*-
import os
import pandas as pd
from flask import Flask, render_template

# Inicializamos Flask con tu estructura de carpetas exacta
# template_folder: Donde están tus HTML
# static_folder: Donde están tus CSS/Imágenes
app = Flask(__name__, 
            template_folder='Frontend', 
            static_folder='Frontend/assets')

# Ajustamos la ruta según tu estructura: Backend_app -> data -> resultados_anomalias.csv
DATA_PATH = os.path.join("Backend_app", "data", "resultados_anomalias.csv")

def obtener_datos_admin():
    """Lee el CSV para la vista técnica de PumaScript."""
    if not os.path.exists(DATA_PATH): 
        return None
    
    df = pd.read_csv(DATA_PATH)
    metricas = {
        "total_registros": len(df),
        "anomalias_totales": len(df[df['es_anomalia'] == True]),
        "precision_estimada": "94.2%",
        "ultima_ejecucion": "Hoy"
    }
    return {"tabla_completa": df.to_dict(orient='records'), "metricas": metricas}

def obtener_datos_clientes():
    """Calcula los KPIs financieros para el dashboard del cliente."""
    if not os.path.exists(DATA_PATH): 
        return None
    
    df = pd.read_csv(DATA_PATH)
    # Lógica: Desviación * Población * $20 MXN por m3
    mermas = (df[df['desviacion_consumo'] > 0]['desviacion_consumo'] * df['poblacion']).sum()
    
    # Calculamos el porcentaje de Agua No Contabilizada (ANC)
    anc_val = (df['desviacion_consumo'].abs().sum() / df['consumo_total'].sum()) * 100
    
    return {
        "dinero_riesgo": f"{mermas * 20:,.0f}",
        "anc_porcentaje": f"{anc_val:.1f}",
        "alertas_criticas": len(df[df['Riesgo'] == 'Alto']),
        "ranking": df.sort_values(by='desviacion_consumo', ascending=False).head(5).to_dict(orient='records')
    }

# --- DEFINICIÓN DE RUTAS (ENDPOINT) ---

@app.route('/')
def home():
    """Página de inicio"""
    return render_template('index.html')

@app.route('/dashboard-clientes')
def dashboard_clientes():
    """Ruta para el Portal del Cliente"""
    stats = obtener_datos_clientes() or {}
    return render_template('dashboard_clientes.html', **stats)

@app.route('/dashboard-admin')
def dashboard_admin():
    """Ruta para el Panel de PumaScript (Admin)"""
    datos = obtener_datos_admin() or {"tabla_completa": [], "metricas": {}}
    return render_template('dashboard_admin.html', **datos)

if __name__ == '__main__':
    # El puerto 5000 es el estándar, debug=True ayuda a ver errores en tiempo real
    print("🚀 HydroTrace AI arrancando en http://127.0.0.1:5000")
    app.run(debug=True, port=5000)