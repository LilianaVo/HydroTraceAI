import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as ticker
import os

# Datos extraídos de los reportes PDF de SACMEX/CONAGUA (2019-2026)
años = [2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
niveles = [82.29, 74.12, 46.75, 53.66, 45.94, 34.77, 56.19, 76.72]


sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#fbfbfb", "grid.color": "#e0e0e0"})

fig, ax = plt.subplots(figsize=(11, 6))

line_color = "#102C57" 
fill_color = "#A8D8EA" 
critical_color = "#D21F3C" 

# --- CREACIÓN DE LA GRÁFICA ---
plt.fill_between(años, niveles, color=fill_color, alpha=0.5, label='Curva de Almacenamiento')
plt.plot(años, niveles, color=line_color, marker='o', linewidth=3.5, markersize=8, markerfacecolor="white", markeredgewidth=2)

# --- DETALLES DE IMPACTO ---
# 2. Resaltar la crisis de 2024 con anotación de alto nivel
critical_year = 2024
critical_value = 34.77
plt.annotate(
    f'COLAPSO HISTÓRICO\n{critical_value}% de capacidad',
    xy=(critical_year, critical_value),
    xytext=(critical_year - 0.7, critical_value - 18),
    arrowprops=dict(facecolor=critical_color, edgecolor=critical_color, shrink=0.05, width=1.5, headwidth=7, connectionstyle="arc3,rad=-0.1"),
    fontsize=10.5, fontweight='bold', color=critical_color,
    horizontalalignment='center', backgroundcolor='#fbfbfb'
)

# 3. Formateo de ejes profesionales
# Fijar años como enteros (eliminar .0)
ax.xaxis.set_major_locator(ticker.FixedLocator(años))
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
plt.xticks(fontsize=11, color='#333333')

# Añadir símbolo de % al eje Y
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))
plt.yticks(range(0, 101, 10), fontsize=11, color='#333333')
plt.ylim(0, 105) # Un poco de espacio extra arriba

# Etiquetas de ejes claras y directas
plt.xlabel('Año de Corte (Periodo Crítico Marzo/Abril)', fontsize=12.5, fontweight='600', labelpad=12)
plt.ylabel('Nivel de Capacidad Operativa (%)', fontsize=12.5, fontweight='600', labelpad=12)

plt.suptitle('DIAGNÓSTICO DE ESTRÉS HÍDRICO CDMX: SISTEMA CUTZAMALA', fontsize=15, fontweight='bold', x=0.515, y=0.98)
plt.title('Tendencia histórica de almacenamiento (2019-2026)', fontsize=12, color='#555555', pad=15)

sns.despine(left=True, bottom=True)

plt.tight_layout(rect=[0, 0, 1, 0.96]) 

# --- GENERACIÓN AUTOMÁTICA DE IMAGEN ---
file_name = 'HydroTraceAI_Tendencia_Cutzamala_2019_2026.png'

# --- GESTIÓN DE DIRECTORIOS PROFESIONAL ---
output_folder = 'graficas'

# Si la carpeta no existe, el script la crea automáticamente
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f" Carpeta '{output_folder}' creada exitosamente.")

file_name = 'HydroTraceAI_Tendencia_Cutzamala_2019_2026.png'

output_path = os.path.join(output_folder, file_name)

plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

print(f"\n¡Éxito, jefa! La gráfica profesional se guardó en: {output_path}")