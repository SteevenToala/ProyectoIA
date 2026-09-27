"""
=============================================================================
FASE 5: EVALUACIÓN Y COMPARACIÓN DE RESULTADOS
=============================================================================
¿Qué hace este script?
1. Compara las 3 etapas del proyecto con los datos de prueba (vinos que
   el modelo nunca vio durante el entrenamiento):
   - Modelo 1: PRISM con reglas rígidas (discretización clásica).
   - Modelo 2: Lógica Difusa con los cortes iniciales por cuantiles.
   - Modelo 3: Lógica Difusa con límites y pesos OPTIMIZADOS por el AG.
2. Muestra la tabla comparativa de aciertos.
3. Genera un gráfico sencillo 'grafico_comparativa.png' para la sustentación.
=============================================================================
"""

import os
import sys
import json
import importlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Rutas automáticas
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(DIRECTORIO_ACTUAL, ".."))

# Importamos las funciones de la Fase 3
sys.path.append(os.path.join(DIRECTORIO_RAIZ, "Fase_3_Logica_Difusa"))
mod_difusa = importlib.import_module("3_logica_difusa")
evaluar_vino_difuso = mod_difusa.evaluar_vino_difuso
evaluar_dataset = mod_difusa.evaluar_dataset

print("\n" + "="*60)
print("     FASE 5: EVALUACIÓN Y COMPARACIÓN DE RESULTADOS")
print("="*60)

# 1. Cargar datos de prueba y entrenamiento desde Fase 1
ruta_train = os.path.join(DIRECTORIO_RAIZ, "Fase_1_Preparacion", "datos_train.csv")
ruta_test = os.path.join(DIRECTORIO_RAIZ, "Fase_1_Preparacion", "datos_test.csv")
ruta_cortes = os.path.join(DIRECTORIO_RAIZ, "Fase_1_Preparacion", "cortes_iniciales.json")
ruta_reglas = os.path.join(DIRECTORIO_RAIZ, "Fase_2_PRISM", "reglas_descubiertas.json")
ruta_modelo = os.path.join(DIRECTORIO_RAIZ, "Fase_4_Algoritmo_Genetico", "modelo_optimizado.json")

df_train = pd.read_csv(ruta_train)
df_test = pd.read_csv(ruta_test)

with open(ruta_cortes) as f:
    cortes_iniciales = json.load(f)
with open(ruta_reglas) as f:
    reglas = json.load(f)
with open(ruta_modelo) as f:
    modelo_opt = json.load(f)

cortes_opt = modelo_opt['cortes_optimizados']
pesos_opt = modelo_opt['pesos_optimizados']

# 2. EVALUACIÓN 1: Reglas Rígidas de PRISM (Discreto)
def evaluar_prism_rigido(df):
    aciertos = 0
    for _, fila in df.iterrows():
        prediccion = None
        for r in reglas:
            coincide = True
            for var, etiqueta in r['condiciones'].items():
                if fila[var + '_disc'] != etiqueta:
                    coincide = False
                    break
            if coincide:
                prediccion = r['consecuente']
                break
        if prediccion is None:
            prediccion = 'MEDIA'
            
        if prediccion == fila['calidad']:
            aciertos += 1
    return round((aciertos / len(df)) * 100, 2)

acc_prism_train = evaluar_prism_rigido(df_train)
acc_prism_test = evaluar_prism_rigido(df_test)

# 3. EVALUACIÓN 2: Lógica Difusa Inicial (Cuantiles)
pesos_iniciales = [r['peso'] for r in reglas]
acc_difuso_train = evaluar_dataset(df_train, cortes_iniciales, reglas, pesos_iniciales)
acc_difuso_test = evaluar_dataset(df_test, cortes_iniciales, reglas, pesos_iniciales)

# 4. EVALUACIÓN 3: Lógica Difusa Optimizada con Algoritmo Genético
acc_opt_train = evaluar_dataset(df_train, cortes_opt, reglas, pesos_opt)
acc_opt_test = evaluar_dataset(df_test, cortes_opt, reglas, pesos_opt)

# 5. Mostrar Tabla de Resultados
print("\n" + "="*72)
print("TABLA COMPARATIVA DE RESULTADOS (TRAIN vs TEST)")
print("="*72)
print(f"{'Enfoque / Modelo':<38} | {'Train (Acierto)':<15} | {'Test (Acierto)':<15}")
print("-" * 72)
print(f"{'1. PRISM (Reglas Rígidas)':<38} | {acc_prism_train:>13.2f} % | {acc_prism_test:>13.2f} %")
print(f"{'2. Lógica Difusa Inicial (Cuantiles)':<38} | {acc_difuso_train:>13.2f} % | {acc_difuso_test:>13.2f} %")
print(f"{'3. Lógica Difusa + Algoritmo Genético':<38} | {acc_opt_train:>13.2f} % | {acc_opt_test:>13.2f} %")
print("="*72)

# 6. Crear un gráfico comparativo sencillo en Fase_5_Evaluacion/
modelos = ['1. PRISM Rígido', '2. Difuso Cuantiles', '3. Difuso + AG']
aciertos_train = [acc_prism_train, acc_difuso_train, acc_opt_train]
aciertos_test = [acc_prism_test, acc_difuso_test, acc_opt_test]

x = np.arange(len(modelos))
ancho = 0.35

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Gráfico de barras comparativo
ax1.bar(x - ancho/2, aciertos_train, ancho, label='Entrenamiento (Train)', color='#4A90E2')
ax1.bar(x + ancho/2, aciertos_test, ancho, label='Prueba (Test)', color='#50E3C2')
ax1.set_ylabel('Exactitud (% de aciertos)')
ax1.set_title('Comparativa de Modelos en Wine Quality', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(modelos)
ax1.set_ylim(30, 80)
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

for i in range(len(modelos)):
    ax1.text(x[i] - ancho/2, aciertos_train[i] + 0.8, f"{aciertos_train[i]:.1f}%", ha='center', fontsize=9)
    ax1.text(x[i] + ancho/2, aciertos_test[i] + 0.8, f"{aciertos_test[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

# Gráfico de la evolución del Algoritmo Genético
historial = modelo_opt['historial']
ax2.plot(range(len(historial)), historial, marker='o', color='#E74C3C', linewidth=2)
ax2.set_xlabel('Generación')
ax2.set_ylabel('Acierto en Entrenamiento (%)')
ax2.set_title('Convergencia del Algoritmo Genético', fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
ruta_grafico = os.path.join(DIRECTORIO_ACTUAL, "grafico_comparativa.png")
plt.savefig(ruta_grafico, dpi=150)
plt.close()

# 7. Gráfico de Matrices de Confusión Comparativas
from sklearn.metrics import confusion_matrix

def obtener_predicciones(df, cortes, pesos):
    return [evaluar_vino_difuso(fila, cortes, reglas, pesos)[0] for _, fila in df.iterrows()]

def obtener_predicciones_prism(df):
    preds = []
    for _, fila in df.iterrows():
        p = None
        for r in reglas:
            if all(fila[v + '_disc'] == e for v, e in r['condiciones'].items()):
                p = r['consecuente']
                break
        preds.append(p if p else 'MEDIA')
    return preds

y_test_real = df_test['calidad'].values
preds_prism = obtener_predicciones_prism(df_test)
preds_difuso = obtener_predicciones(df_test, cortes_iniciales, pesos_iniciales)
preds_opt = obtener_predicciones(df_test, cortes_opt, pesos_opt)

clases = ['ALTA', 'MEDIA', 'BAJA']
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

modelos_cm = [
    ("1. PRISM (Reglas Rígidas)", preds_prism),
    ("2. Difuso Inicial (Cuantiles)", preds_difuso),
    ("3. Difuso Optimizado (AG)", preds_opt)
]

for ax, (titulo, p_arr) in zip(axes, modelos_cm):
    cm = confusion_matrix(y_test_real, p_arr, labels=clases)
    ax.imshow(cm, cmap=plt.cm.Blues, interpolation='nearest')
    ax.set_title(titulo, fontweight='bold', fontsize=11)
    ax.set_xticks(range(len(clases)))
    ax.set_yticks(range(len(clases)))
    ax.set_xticklabels(clases)
    ax.set_yticklabels(clases)
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Clase Real')
    
    thresh = cm.max() / 2.0
    for i in range(len(clases)):
        for j in range(len(clases)):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight='bold')

plt.tight_layout()
ruta_cm = os.path.join(DIRECTORIO_ACTUAL, "grafico_matrices_confusion.png")
plt.savefig(ruta_cm, dpi=150)
plt.close()

# 8. Gráfico de Funciones de Pertenencia (Inicial vs Optimizado por el AG)
pertenencia_bajo = mod_difusa.pertenencia_bajo
pertenencia_medio = mod_difusa.pertenencia_medio
pertenencia_alto = mod_difusa.pertenencia_alto

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
nombres_vars = {
    'alcohol': 'Alcohol (% vol.)',
    'volatile acidity': 'Acidez Volátil (g/dm³)',
    'sulphates': 'Sulfatos (g/dm³)',
    'pH': 'pH'
}

for ax, (var, titulo) in zip(axes.flatten(), nombres_vars.items()):
    min_v = cortes_iniciales[var]['min']
    max_v = cortes_iniciales[var]['max']
    x_vals = np.linspace(min_v, max_v, 300)
    
    # Cortes iniciales
    c_ini = cortes_iniciales[var]
    u_b_ini = [pertenencia_bajo(x, c_ini['m1'], c_ini['m2']) for x in x_vals]
    u_m_ini = [pertenencia_medio(x, c_ini['m1'], c_ini['m2'], c_ini['m3']) for x in x_vals]
    u_a_ini = [pertenencia_alto(x, c_ini['m2'], c_ini['m3']) for x in x_vals]
    
    # Cortes optimizados
    c_opt = cortes_opt[var]
    u_b_opt = [pertenencia_bajo(x, c_opt['m1'], c_opt['m2']) for x in x_vals]
    u_m_opt = [pertenencia_medio(x, c_opt['m1'], c_opt['m2'], c_opt['m3']) for x in x_vals]
    u_a_opt = [pertenencia_alto(x, c_opt['m2'], c_opt['m3']) for x in x_vals]
    
    # Curvas optimizadas (líneas continuas)
    ax.plot(x_vals, u_b_opt, label='Bajo (Optimizado)', color='#1f77b4', linewidth=2.2)
    ax.plot(x_vals, u_m_opt, label='Medio (Optimizado)', color='#2ca02c', linewidth=2.2)
    ax.plot(x_vals, u_a_opt, label='Alto (Optimizado)', color='#d62728', linewidth=2.2)
    
    # Curvas iniciales (líneas punteadas)
    ax.plot(x_vals, u_b_ini, ':', color='#1f77b4', alpha=0.45, label='Bajo (Inicial)')
    ax.plot(x_vals, u_m_ini, ':', color='#2ca02c', alpha=0.45, label='Medio (Inicial)')
    ax.plot(x_vals, u_a_ini, ':', color='#d62728', alpha=0.45, label='Alto (Inicial)')
    
    ax.set_title(titulo, fontweight='bold', fontsize=11)
    ax.set_ylabel('Grado de Pertenencia (μ)')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle='--', alpha=0.4)
    if var == 'alcohol':
        ax.legend(loc='center right', fontsize=8)

plt.suptitle('Comparativa de Funciones Difusas: Inicial (Cuantiles) vs Optimizado (AG)', fontweight='bold', fontsize=13)
plt.tight_layout()
ruta_mf = os.path.join(DIRECTORIO_ACTUAL, "grafico_funciones_pertenencia.png")
plt.savefig(ruta_mf, dpi=150)
plt.close()

print("\nGráficos generados en Fase_5_Evaluacion/:")
print("  -> grafico_comparativa.png")
print("  -> grafico_matrices_confusion.png")
print("  -> grafico_funciones_pertenencia.png")
print("--- FASE 5 COMPLETADA CON ÉXITO ---\n")

