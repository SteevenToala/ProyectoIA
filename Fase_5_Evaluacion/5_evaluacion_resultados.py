# Proyecto Inteligencia Artificial
# Fase 5: Evaluacion y comparativa entre PRISM rigido, Difuso inicial y Difuso optimizado con AG

import os
import sys
import json
import importlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

carpeta_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_proyecto = os.path.abspath(os.path.join(carpeta_actual, ".."))

# Importamos las funciones de la Fase 3
sys.path.append(os.path.join(carpeta_proyecto, "Fase_3_Logica_Difusa"))
modulo_difusa = importlib.import_module("3_logica_difusa")
evaluar_vino_completo = modulo_difusa.evaluar_vino_completo
calcular_exactitud_dataset = modulo_difusa.calcular_exactitud_dataset
calcular_pertenencia_baja = modulo_difusa.calcular_pertenencia_baja
calcular_pertenencia_media = modulo_difusa.calcular_pertenencia_media
calcular_pertenencia_alta = modulo_difusa.calcular_pertenencia_alta

# 1. Cargamos los datos de prueba y entrenamiento
ruta_train = os.path.join(carpeta_proyecto, "Fase_1_Preparacion", "datos_train.csv")
ruta_test = os.path.join(carpeta_proyecto, "Fase_1_Preparacion", "datos_test.csv")
ruta_cortes_base = os.path.join(carpeta_proyecto, "Fase_1_Preparacion", "cortes_iniciales.json")
ruta_reglas_prism = os.path.join(carpeta_proyecto, "Fase_2_PRISM", "reglas_descubiertas.json")
ruta_modelo_opt = os.path.join(carpeta_proyecto, "Fase_4_Algoritmo_Genetico", "modelo_optimizado.json")

datos_entrenamiento = pd.read_csv(ruta_train)
datos_prueba = pd.read_csv(ruta_test)

with open(ruta_cortes_base) as archivo_cortes:
    cortes_iniciales = json.load(archivo_cortes)
with open(ruta_reglas_prism) as archivo_reglas:
    lista_reglas = json.load(archivo_reglas)
with open(ruta_modelo_opt) as archivo_modelo:
    modelo_optimizado = json.load(archivo_modelo)

cortes_optimizados = modelo_optimizado['cortes_optimizados']
pesos_optimizados = modelo_optimizado['pesos_optimizados']
pesos_iniciales = [regla['peso'] for regla in lista_reglas]


# 2. Evaluacion del Modelo 1: PRISM con reglas rigidas discretas
def evaluar_prism_reglas_rigidas(df_vinos):
    aciertos = 0
    for _, fila in df_vinos.iterrows():
        prediccion = None
        for regla in lista_reglas:
            cumple_todas = True
            for variable_quimica, etiqueta in regla['condiciones'].items():
                columna_discreta = variable_quimica + '_discreto'
                if columna_discreta == 'acidez_volatil_discreto':
                    columna_discreta = 'acidez_discreto'
                if fila[columna_discreta] != etiqueta:
                    cumple_todas = False
                    break
            if cumple_todas:
                prediccion = regla['consecuente']
                break
        if prediccion is None:
            prediccion = 'MEDIA'
            
        if prediccion == fila['calidad']:
            aciertos += 1
            
    return round((aciertos / len(df_vinos)) * 100.0, 2)

exactitud_prism_train = evaluar_prism_reglas_rigidas(datos_entrenamiento)
exactitud_prism_test = evaluar_prism_reglas_rigidas(datos_prueba)

# 3. Evaluacion del Modelo 2: Logica Difusa Inicial (Cortes por cuantiles)
exactitud_difuso_train = calcular_exactitud_dataset(
    datos_entrenamiento, cortes_iniciales, lista_reglas, pesos_iniciales
)
exactitud_difuso_test = calcular_exactitud_dataset(
    datos_prueba, cortes_iniciales, lista_reglas, pesos_iniciales
)

# 4. Evaluacion del Modelo 3: Logica Difusa Optimizada con Algoritmo Genetico
exactitud_opt_train = calcular_exactitud_dataset(
    datos_entrenamiento, cortes_optimizados, lista_reglas, pesos_optimizados
)
exactitud_opt_test = calcular_exactitud_dataset(
    datos_prueba, cortes_optimizados, lista_reglas, pesos_optimizados
)

# 5. Generacion de graficos comparativos


# --- Grafico 1: Barras comparativas Train vs Test y curva del AG ---
nombres_modelos = ['PRISM Rigido', 'Difuso Cuantiles', 'Difuso + AG']
valores_train = [exactitud_prism_train, exactitud_difuso_train, exactitud_opt_train]
valores_test = [exactitud_prism_test, exactitud_difuso_test, exactitud_opt_test]

posiciones_x = np.arange(len(nombres_modelos))
ancho_barra = 0.35

fig, (ax_barras, ax_convergencia) = plt.subplots(1, 2, figsize=(12, 5))

# Barras comparativas
ax_barras.bar(posiciones_x - ancho_barra/2, valores_train, ancho_barra, label='Entrenamiento (Train)', color='#4A90E2')
ax_barras.bar(posiciones_x + ancho_barra/2, valores_test, ancho_barra, label='Prueba (Test)', color='#50E3C2')
ax_barras.set_ylabel('Exactitud (%)')
ax_barras.set_title('Comparativa de Modelos en Wine Quality')
ax_barras.set_xticks(posiciones_x)
ax_barras.set_xticklabels(nombres_modelos)
ax_barras.set_ylim(25, 80)
ax_barras.grid(True, linestyle='--', alpha=0.5)
ax_barras.legend()

for i in range(len(nombres_modelos)):
    ax_barras.text(posiciones_x[i] - ancho_barra/2, valores_train[i] + 0.8, f"{valores_train[i]:.1f}%", ha='center', fontsize=9)
    ax_barras.text(posiciones_x[i] + ancho_barra/2, valores_test[i] + 0.8, f"{valores_test[i]:.1f}%", ha='center', fontsize=9, fontweight='bold')

# Curva de convergencia del Algoritmo Genetico
historial_generaciones = modelo_optimizado['historial']
ax_convergencia.plot(range(len(historial_generaciones)), historial_generaciones, marker='o', color='#E74C3C', linewidth=2)
ax_convergencia.set_xlabel('Generacion')
ax_convergencia.set_ylabel('Exactitud en Train (%)')
ax_convergencia.set_title('Convergencia del Algoritmo Genetico')
ax_convergencia.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
ruta_grafico_comparativo = os.path.join(carpeta_actual, "grafico_comparativa.png")
plt.savefig(ruta_grafico_comparativo, dpi=130)
plt.close()


# --- Grafico 2: Matrices de Confusion ---
def predecir_con_difuso(df, cortes, pesos):
    return [evaluar_vino_completo(fila, cortes, lista_reglas, pesos)[0] for _, fila in df.iterrows()]

def predecir_con_prism(df):
    predicciones = []
    for _, fila in df.iterrows():
        p = None
        for regla in lista_reglas:
            cumple = True
            for var, val in regla['condiciones'].items():
                col = var + '_discreto'
                if col == 'acidez_volatil_discreto':
                    col = 'acidez_discreto'
                if fila[col] != val:
                    cumple = False
                    break
            if cumple:
                p = regla['consecuente']
                break
        predicciones.append(p if p else 'MEDIA')
    return predicciones

calidades_reales_test = datos_prueba['calidad'].values
predicciones_prism = predecir_con_prism(datos_prueba)
predicciones_difuso_inicial = predecir_con_difuso(datos_prueba, cortes_iniciales, pesos_iniciales)
predicciones_difuso_optimizado = predecir_con_difuso(datos_prueba, cortes_optimizados, pesos_optimizados)

clases_calidad = ['ALTA', 'MEDIA', 'BAJA']
fig, ejes_matriz = plt.subplots(1, 3, figsize=(14, 4.2))

modelos_matrices = [
    ("PRISM Rigido", predicciones_prism),
    ("Difuso Cuantiles", predicciones_difuso_inicial),
    ("Difuso + AG", predicciones_difuso_optimizado)
]

for eje, (titulo, predicciones_modelo) in zip(ejes_matriz, modelos_matrices):
    matriz = confusion_matrix(calidades_reales_test, predicciones_modelo, labels=clases_calidad)
    eje.imshow(matriz, cmap=plt.cm.Blues, interpolation='nearest')
    eje.set_title(titulo, fontsize=11, fontweight='bold')
    eje.set_xticks(range(len(clases_calidad)))
    eje.set_yticks(range(len(clases_calidad)))
    eje.set_xticklabels(clases_calidad)
    eje.set_yticklabels(clases_calidad)
    eje.set_xlabel('Predicho')
    eje.set_ylabel('Real')
    
    umbral = matriz.max() / 2.0
    for fila in range(len(clases_calidad)):
        for col in range(len(clases_calidad)):
            eje.text(col, fila, format(matriz[fila, col], 'd'),
                     ha="center", va="center",
                     color="white" if matriz[fila, col] > umbral else "black",
                     fontweight='bold')

plt.tight_layout()
ruta_grafico_matrices = os.path.join(carpeta_actual, "grafico_matrices_confusion.png")
plt.savefig(ruta_grafico_matrices, dpi=130)
plt.close()


# --- Grafico 3: Comparativa de Funciones de Pertenencia ---
fig, ejes_curvas = plt.subplots(2, 2, figsize=(12, 7.5))
nombres_variables_mostrar = {
    'alcohol': 'Alcohol (% vol.)',
    'acidez_volatil': 'Acidez Volatil (g/dm3)',
    'sulfatos': 'Sulfatos (g/dm3)',
    'ph': 'pH'
}

for eje, (var, titulo) in zip(ejes_curvas.flatten(), nombres_variables_mostrar.items()):
    min_quimico = cortes_iniciales[var]['minimo']
    max_quimico = cortes_iniciales[var]['maximo']
    puntos_x = np.linspace(min_quimico, max_quimico, 300)
    
    # Curvas con cortes iniciales
    c_ini = cortes_iniciales[var]
    curva_baja_ini = [calcular_pertenencia_baja(x, c_ini['corte_bajo'], c_ini['punto_medio']) for x in puntos_x]
    curva_media_ini = [calcular_pertenencia_media(x, c_ini['corte_bajo'], c_ini['punto_medio'], c_ini['corte_alto']) for x in puntos_x]
    curva_alta_ini = [calcular_pertenencia_alta(x, c_ini['punto_medio'], c_ini['corte_alto']) for x in puntos_x]
    
    # Curvas con cortes optimizados por el AG
    c_opt = cortes_optimizados[var]
    curva_baja_opt = [calcular_pertenencia_baja(x, c_opt['corte_bajo'], c_opt['punto_medio']) for x in puntos_x]
    curva_media_opt = [calcular_pertenencia_media(x, c_opt['corte_bajo'], c_opt['punto_medio'], c_opt['corte_alto']) for x in puntos_x]
    curva_alta_opt = [calcular_pertenencia_alta(x, c_opt['punto_medio'], c_opt['corte_alto']) for x in puntos_x]
    
    # Graficar curvas optimizadas (linea continua)
    eje.plot(puntos_x, curva_baja_opt, label='Bajo (Opt)', color='#1f77b4', linewidth=2)
    eje.plot(puntos_x, curva_media_opt, label='Medio (Opt)', color='#2ca02c', linewidth=2)
    eje.plot(puntos_x, curva_alta_opt, label='Alto (Opt)', color='#d62728', linewidth=2)
    
    # Graficar curvas iniciales (linea punteada)
    eje.plot(puntos_x, curva_baja_ini, ':', color='#1f77b4', alpha=0.4, label='Bajo (Ini)')
    eje.plot(puntos_x, curva_media_ini, ':', color='#2ca02c', alpha=0.4, label='Medio (Ini)')
    eje.plot(puntos_x, curva_alta_ini, ':', color='#d62728', alpha=0.4, label='Alto (Ini)')
    
    eje.set_title(titulo, fontsize=10, fontweight='bold')
    eje.set_ylabel('Pertenencia')
    eje.set_ylim(-0.05, 1.05)
    eje.grid(True, linestyle='--', alpha=0.4)
    if var == 'alcohol':
        eje.legend(loc='center right', fontsize=8)

plt.suptitle('Funciones de Pertenencia: Inicial vs Optimizado con Algoritmo Genetico', fontsize=12, fontweight='bold')
plt.tight_layout()

ruta_grafico_funciones = os.path.join(carpeta_actual, "grafico_funciones_pertenencia.png")
plt.savefig(ruta_grafico_funciones, dpi=130)
plt.close()

print("\nTabla comparativa de resultados:")
print("-" * 65)
print(f"{'Enfoque / Modelo':<35} | {'Train Acc':<12} | {'Test Acc':<12}")
print("-" * 65)
print(f"{'1. PRISM (reglas rigidas)':<35} | {exactitud_prism_train:>9.2f}% | {exactitud_prism_test:>9.2f}%")
print(f"{'2. Logica Difusa (cuantiles)':<35} | {exactitud_difuso_train:>9.2f}% | {exactitud_difuso_test:>9.2f}%")
print(f"{'3. Logica Difusa + Alg. Genetico':<35} | {exactitud_opt_train:>9.2f}% | {exactitud_opt_test:>9.2f}%")
print("-" * 65)
