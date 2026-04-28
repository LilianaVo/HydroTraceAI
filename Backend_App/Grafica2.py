import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- DATOS OFICIALES (SACMEX / AUDITORÍA CDMX) ---
labels = ['Consumo Facturado\n(Uso Efectivo)', 'Agua No Contabilizada\n(Fugas y Robo)']
sizes = [60, 40]

output_folder = 'graficas'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

billed_color = "#A8D8EA"   
anc_color = "#D21F3C"     
colors = [billed_color, anc_color]
explode = (0.05, 0) 

fig, ax = plt.subplots(figsize=(8, 8))

# Crear la gráfica de dona
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

for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')
    text.set_color('#333333')

for autotext in autotexts:
    autotext.set_fontsize(14)
    autotext.set_fontweight('bold')
    autotext.set_color('#102C57') 

# Convertirlo en DONA (Círculo blanco central)
centre_circle = plt.Circle((0,0), 0.70, fc='white')
fig.gca().add_artist(centre_circle)

plt.suptitle(
    'EFICIENCIA DE LA RED DE DISTRIBUCIÓN CDMX',
    fontsize=16,
    fontweight='bold',
    y=0.98  # más arriba = más espacio
)

plt.title(
    'Distribución del Caudal: Agua Facturada vs. No Contabilizada (ANC)',
    fontsize=12,
    color='#555555',
    pad=30  
)

# Fuente de datos en la parte inferior
plt.text(0, -1.3, 'Fuente: Elaboración propia con datos consolidados de SACMEX e IPDP (2022-2026).', 
         ha='center', fontsize=10, color='gray', style='italic')

output_path = os.path.join(output_folder, 'HydroTraceAI_Distribucion_ANC.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

print(f"\n¡Éxito! Gráfica ejecutiva guardada en: {output_path}")
plt.show()