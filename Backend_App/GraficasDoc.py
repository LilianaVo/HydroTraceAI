import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
import os

"""
PROYECTO: HydroTrace AI - Ciencia de Datos UNAM
DESCRIPCIÓN: Generación unificada de gráficas diagnósticas para el documento final.
             Este script produce el análisis de estrés hídrico y eficiencia de red.
"""

# --- CONFIGURACIÓN DE ENTORNO ---
output_folder = 'graficas_reporte'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Estilo global de las gráficas
sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#fbfbfb", "grid.color": "#e0e0e0"})

# =============================================================================
# SECCIÓN 1: DIAGNÓSTICO DE ESTRÉS HÍDRICO (TENDENCIA CUTZAMALA)
# =============================================================================
def generar_grafica_tendencia():
    print("[1/2] Generando gráfica de tendencia histórica...")
    
    años = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
    niveles = [82.29, 74.12, 46.75, 53.66, 45.94, 34.77, 56.19, 76.72]

    fig, ax = plt.subplots(figsize=(11, 6))
    line_color = "#102C57" 
    fill_color = "#A8D8EA" 
    critical_color = "#D21F3C" # El detalle que faltaba para resaltar la crisis

    # Dibujo de área y línea
    plt.fill_between(años, niveles, color=fill_color, alpha=0.5, label='Curva de Almacenamiento')
    plt.plot(años, niveles, color=line_color, marker='o', linewidth=3.5, markersize=8, 
             markerfacecolor="white", markeredgewidth=2)

    # RESALTADO DE PUNTO CRÍTICO (El detallito de 2024)
    plt.plot(2024, 34.77, marker='o', markersize=12, color=critical_color, label='Punto Crítico (Día Cero)')
    plt.annotate('Crisis Hídrica (34.7%)', 
                 xy=(2024, 34.77), xytext=(2022.5, 25),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
                 fontsize=10, fontweight='bold', color=critical_color)

    # Formateo técnico de ejes
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    plt.xticks(años, fontsize=11, color='#333333')
    
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
    plt.yticks(range(0, 101, 10), fontsize=11, color='#333333')
    plt.ylim(0, 105)

    # Etiquetas de ejes
    plt.xlabel('Año de Corte (Periodo Crítico Marzo/Abril)', fontsize=12.5, fontweight='600', labelpad=12)
    plt.ylabel('Nivel de Capacidad Operativa (%)', fontsize=12.5, fontweight='600', labelpad=12)
    
    plt.suptitle('DIAGNÓSTICO DE ESTRÉS HÍDRICO CDMX: SISTEMA CUTZAMALA', 
                 fontsize=15, fontweight='bold', x=0.515, y=0.98)
    plt.title('Tendencia histórica de almacenamiento (2019-2026)', fontsize=12, color='#555555', pad=15)

    sns.despine(left=True, bottom=True)
    plt.legend(loc='upper right', frameon=True, facecolor='white', fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 

    path = os.path.join(output_folder, 'HydroTraceAI_Tendencia_Cutzamala.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Guardada: {path}")

# =============================================================================
# SECCIÓN 2: EFICIENCIA DE LA RED (AGUA NO CONTABILIZADA)
# =============================================================================
def generar_grafica_eficiencia():
    print("[2/2] Generando gráfica de eficiencia de red (Dona)...")
    
    labels = ['Consumo Facturado\n(Uso Efectivo)', 'Agua No Contabilizada\n(Fugas y Robo)']
    sizes = [60, 40]
    
    billed_color = "#A8D8EA"   
    anc_color = "#D21F3C"     
    colors = [billed_color, anc_color]
    explode = (0.05, 0) 

    fig, ax = plt.subplots(figsize=(8, 8))

    # Gráfica de dona
    wedges, texts, autotexts = ax.pie(
        sizes, 
        labels=labels, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors, 
        pctdistance=0.82, 
        explode=explode,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    # Estilo de etiquetas externas
    for text in texts:
        text.set_fontsize(12)
        text.set_fontweight('bold')
        text.set_color('#333333')

    # Estilo de porcentajes internos
    for autotext in autotexts:
        autotext.set_fontsize(14)
        autotext.set_fontweight('bold')
        autotext.set_color('#102C57') 

    # Círculo central para efecto dona
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    # Títulos
    plt.suptitle('EFICIENCIA DE LA RED DE DISTRIBUCIÓN CDMX', fontsize=16, fontweight='bold', y=0.98)
    plt.title('Pérdidas estimadas por fugas y extracción irregular (ANC)', fontsize=12, color='#555555', pad=10)

    plt.tight_layout()

    path = os.path.join(output_folder, 'HydroTraceAI_Eficiencia_Red_Dona.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> Guardada: {path}")

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    print("--- INICIANDO GENERACIÓN DE RECURSOS VISUALES PARA DOCUMENTO ---")
    generar_grafica_tendencia()
    generar_grafica_eficiencia()
    print("\n¡Listo! Tus gráficas ya tienen todos los detalles restaurados.")
    print("Revisa la carpeta:", output_folder)