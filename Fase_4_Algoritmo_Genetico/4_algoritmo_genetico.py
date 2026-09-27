# Proyecto Inteligencia Artificial
# Fase 4: Algoritmo Genetico para optimizar cortes y pesos de reglas en Wine Quality

import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

carpeta_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_proyecto = os.path.abspath(os.path.join(carpeta_actual, ".."))

print("=== Fase 4: Algoritmo Genetico para Optimizar el Modelo del Vino ===")

# 1. Cargamos los datos de entrenamiento, cortes base y reglas PRISM
ruta_datos_train = os.path.join(carpeta_proyecto, "Fase_1_Preparacion", "datos_train.csv")
ruta_cortes_base = os.path.join(carpeta_proyecto, "Fase_1_Preparacion", "cortes_iniciales.json")
ruta_reglas_prism = os.path.join(carpeta_proyecto, "Fase_2_PRISM", "reglas_descubiertas.json")

datos_entrenamiento = pd.read_csv(ruta_datos_train)
with open(ruta_cortes_base) as archivo_cortes:
    cortes_base = json.load(archivo_cortes)
with open(ruta_reglas_prism) as archivo_reglas:
    lista_reglas = json.load(archivo_reglas)

lista_variables_quimicas = ['alcohol', 'acidez_volatil', 'sulfatos', 'ph']
total_reglas = len(lista_reglas)
total_vinos_entrenamiento = len(datos_entrenamiento)

# Limites minimos y maximos de cada variable
limites_quimicos_extremos = {
    var: (cortes_base[var]['minimo'], cortes_base[var]['maximo'])
    for var in lista_variables_quimicas
}

valores_alcohol = datos_entrenamiento['alcohol'].values
valores_acidez = datos_entrenamiento['acidez_volatil'].values
valores_sulfatos = datos_entrenamiento['sulfatos'].values
valores_ph = datos_entrenamiento['ph'].values
calidades_reales = datos_entrenamiento['calidad'].values


# 2. Funcion para decodificar el cromosoma (vector continuo de 67 numeros)
# en los cortes quimicos de las 4 variables y los 55 pesos de las reglas
def decodificar_cromosoma(cromosoma):
    cortes_decodificados = {}
    indice = 0
    
    for variable in lista_variables_quimicas:
        minimo_quimico, maximo_quimico = limites_quimicos_extremos[variable]
        
        # Tomamos los 3 cortes de esta variable y los ordenamos de menor a mayor
        corte_uno, corte_dos, corte_tres = sorted(cromosoma[indice : indice + 3])
        
        # Validamos que esten dentro de los limites permitidos
        corte_uno = max(minimo_quimico, min(corte_uno, maximo_quimico - 0.03))
        corte_dos = max(corte_uno + 0.01, min(corte_dos, maximo_quimico - 0.02))
        corte_tres = max(corte_dos + 0.01, min(corte_tres, maximo_quimico - 0.01))
        
        cortes_decodificados[variable] = {
            'minimo': minimo_quimico,
            'corte_bajo': corte_uno,
            'punto_medio': corte_dos,
            'corte_alto': corte_tres,
            'maximo': maximo_quimico
        }
        indice += 3
        
    pesos_decodificados = [
        max(0.0, min(peso, 2.0))
        for peso in cromosoma[indice : indice + total_reglas]
    ]
    
    return cortes_decodificados, pesos_decodificados


# Evaluacion rapida vectorizada de funciones de pertenencia para todos los vinos
def calcular_pertenencias_vectorizadas(valores_columna, corte_bajo, corte_medio, corte_alto):
    # Grado bajo (hombro izquierdo)
    grado_bajo = np.ones_like(valores_columna)
    rango_bajo = (valores_columna > corte_bajo) & (valores_columna < corte_medio)
    grado_bajo[rango_bajo] = (corte_medio - valores_columna[rango_bajo]) / (corte_medio - corte_bajo)
    grado_bajo[valores_columna >= corte_medio] = 0.0
    
    # Grado medio (triangular)
    grado_medio = np.zeros_like(valores_columna)
    subida = (valores_columna > corte_bajo) & (valores_columna <= corte_medio)
    grado_medio[subida] = (valores_columna[subida] - corte_bajo) / (corte_medio - corte_bajo)
    bajada = (valores_columna > corte_medio) & (valores_columna < corte_alto)
    grado_medio[bajada] = (corte_alto - valores_columna[bajada]) / (corte_alto - corte_medio)
    
    # Grado alto (hombro derecho)
    grado_alto = np.zeros_like(valores_columna)
    rango_alto = (valores_columna > corte_medio) & (valores_columna < corte_alto)
    grado_alto[rango_alto] = (valores_columna[rango_alto] - corte_medio) / (corte_alto - corte_medio)
    grado_alto[valores_columna >= corte_alto] = 1.0
    
    return {'bajo': grado_bajo, 'medio': grado_medio, 'alto': grado_alto}


# 3. Funcion de Aptitud (Fitness): Porcentaje de aciertos en entrenamiento (%)
def calcular_aptitud_individuo(cromosoma):
    cortes_evaluar, pesos_evaluar = decodificar_cromosoma(cromosoma)
    
    pertenencias = {
        'alcohol': calcular_pertenencias_vectorizadas(
            valores_alcohol,
            cortes_evaluar['alcohol']['corte_bajo'],
            cortes_evaluar['alcohol']['punto_medio'],
            cortes_evaluar['alcohol']['corte_alto']
        ),
        'acidez_volatil': calcular_pertenencias_vectorizadas(
            valores_acidez,
            cortes_evaluar['acidez_volatil']['corte_bajo'],
            cortes_evaluar['acidez_volatil']['punto_medio'],
            cortes_evaluar['acidez_volatil']['corte_alto']
        ),
        'sulfatos': calcular_pertenencias_vectorizadas(
            valores_sulfatos,
            cortes_evaluar['sulfatos']['corte_bajo'],
            cortes_evaluar['sulfatos']['punto_medio'],
            cortes_evaluar['sulfatos']['corte_alto']
        ),
        'ph': calcular_pertenencias_vectorizadas(
            valores_ph,
            cortes_evaluar['ph']['corte_bajo'],
            cortes_evaluar['ph']['punto_medio'],
            cortes_evaluar['ph']['corte_alto']
        )
    }
    
    puntajes_acumulados_baja = np.zeros(total_vinos_entrenamiento)
    puntajes_acumulados_media = np.zeros(total_vinos_entrenamiento)
    puntajes_acumulados_alta = np.zeros(total_vinos_entrenamiento)
    
    for indice_regla, regla in enumerate(lista_reglas):
        peso = pesos_evaluar[indice_regla]
        if peso <= 0.01:
            continue
            
        disparo_regla = np.ones(total_vinos_entrenamiento)
        for variable_quimica, etiqueta in regla['condiciones'].items():
            disparo_regla = np.minimum(disparo_regla, pertenencias[variable_quimica][etiqueta])
            
        impacto_ponderado = peso * disparo_regla
        
        if regla['consecuente'] == 'BAJA':
            puntajes_acumulados_baja += impacto_ponderado
        elif regla['consecuente'] == 'MEDIA':
            puntajes_acumulados_media += impacto_ponderado
        else:
            puntajes_acumulados_alta += impacto_ponderado
            
    # Defuzzificacion por maxima pertenencia para cada vino
    predicciones = []
    for i in range(total_vinos_entrenamiento):
        puntaje_b = puntajes_acumulados_baja[i]
        puntaje_m = puntajes_acumulados_media[i]
        puntaje_a = puntajes_acumulados_alta[i]
        
        if puntaje_a > puntaje_m and puntaje_a > puntaje_b:
            predicciones.append('ALTA')
        elif puntaje_b > puntaje_m and puntaje_b >= puntaje_a:
            predicciones.append('BAJA')
        else:
            predicciones.append('MEDIA')
            
    vinos_acertados = np.sum(np.array(predicciones) == calidades_reales)
    exactitud = (vinos_acertados / total_vinos_entrenamiento) * 100.0
    return round(exactitud, 2)


# 4. Creacion del cromosoma semilla inicial
cromosoma_semilla = []
for variable in lista_variables_quimicas:
    cromosoma_semilla.extend([
        cortes_base[variable]['corte_bajo'],
        cortes_base[variable]['punto_medio'],
        cortes_base[variable]['corte_alto']
    ])
cromosoma_semilla.extend([regla['peso'] for regla in lista_reglas])

def generar_individuo_aleatorio():
    individuo = list(cromosoma_semilla)
    for i in range(len(individuo)):
        if i < 12:
            # Perturbacion inicial de cortes
            individuo[i] += random.gauss(0, 0.05 * abs(individuo[i]))
        else:
            # Perturbacion inicial de pesos
            individuo[i] = max(0.1, min(individuo[i] + random.gauss(0, 0.2), 1.5))
    return individuo


# 5. Hiperparametros del Algoritmo Genetico
random.seed(42)
np.random.seed(42)

tamano_poblacion = 30
numero_generaciones = 25
probabilidad_cruce = 0.85
probabilidad_mutacion = 0.20
cantidad_elitismo = 1
tamano_torneo = 2

# Seleccion por torneo binario
def seleccion_por_torneo(poblacion, lista_aptitudes, k=tamano_torneo):
    participantes = random.sample(range(len(poblacion)), k)
    mejor_indice = participantes[0]
    for indice in participantes[1:]:
        if lista_aptitudes[indice] > lista_aptitudes[mejor_indice]:
            mejor_indice = indice
    return poblacion[mejor_indice]

# Cruce intermedio (Intermediate Crossover)
def cruzar_cromosomas_intermedio(padre_uno, padre_dos):
    factor_aleatorio = random.random()
    hijo = []
    for gen_uno, gen_dos in zip(padre_uno, padre_dos):
        gen_hijo = gen_uno + factor_aleatorio * (gen_dos - gen_uno)
        hijo.append(gen_hijo)
    return hijo

# Mutacion gaussiana adaptativa
def mutar_cromosoma_gaussiano(cromosoma):
    cromosoma_mutado = list(cromosoma)
    for indice_gen in range(len(cromosoma_mutado)):
        if random.random() < probabilidad_mutacion:
            if indice_gen < 12:
                # Mutacion en cortes
                cromosoma_mutado[indice_gen] += random.gauss(0, 0.04 * abs(cromosoma_mutado[indice_gen]))
            else:
                # Mutacion en pesos
                cromosoma_mutado[indice_gen] = max(0.05, min(cromosoma_mutado[indice_gen] + random.gauss(0, 0.15), 2.0))
    return cromosoma_mutado


# 6. Bucle Evolutivo
print(f"Poblacion: {tamano_poblacion} | Generaciones: {numero_generaciones}")
print(f"Probabilidad Cruce: {probabilidad_cruce} | Probabilidad Mutacion: {probabilidad_mutacion}")

aptitud_inicial = calcular_aptitud_individuo(cromosoma_semilla)
print(f"Aptitud inicial con cortes por cuantiles: {aptitud_inicial}%\n")

poblacion_actual = [list(cromosoma_semilla)]
for _ in range(tamano_poblacion - 1):
    poblacion_actual.append(generar_individuo_aleatorio())

mejor_cromosoma_global = list(cromosoma_semilla)
mejor_aptitud_global = aptitud_inicial
historial_convergencia = [aptitud_inicial]

for generacion in range(1, numero_generaciones + 1):
    aptitudes_poblacion = [calcular_aptitud_individuo(ind) for ind in poblacion_actual]
    
    indice_mejor_gen = np.argmax(aptitudes_poblacion)
    if aptitudes_poblacion[indice_mejor_gen] > mejor_aptitud_global:
        mejor_aptitud_global = aptitudes_poblacion[indice_mejor_gen]
        mejor_cromosoma_global = list(poblacion_actual[indice_mejor_gen])
        
    historial_convergencia.append(mejor_aptitud_global)
    
    if generacion % 5 == 0 or generacion == numero_generaciones:
        print(f"  Generacion {generacion:02d}/{numero_generaciones} -> Mejor Exactitud en Train: {mejor_aptitud_global:.2f}%")
        
    # Elitismo: preservamos al mejor individuo
    nueva_poblacion = [list(mejor_cromosoma_global)]
    
    while len(nueva_poblacion) < tamano_poblacion:
        padre_uno = seleccion_por_torneo(poblacion_actual, aptitudes_poblacion)
        padre_dos = seleccion_por_torneo(poblacion_actual, aptitudes_poblacion)
        
        if random.random() < probabilidad_cruce:
            hijo = cruzar_cromosomas_intermedio(padre_uno, padre_dos)
        else:
            hijo = list(padre_uno)
            
        hijo = mutar_cromosoma_gaussiano(hijo)
        nueva_poblacion.append(hijo)
        
    poblacion_actual = nueva_poblacion

print(f"\nOptimizacion finalizada. Mejor exactitud alcanzada: {mejor_aptitud_global:.2f}%")

# Guardamos el modelo optimizado
cortes_optimizados, pesos_optimizados = decodificar_cromosoma(mejor_cromosoma_global)

resultado_modelo = {
    'aptitud_inicial': aptitud_inicial,
    'aptitud_optimizada': mejor_aptitud_global,
    'cortes_optimizados': cortes_optimizados,
    'pesos_optimizados': [round(p, 4) for p in pesos_optimizados],
    'historial': historial_convergencia
}

ruta_guardar_modelo = os.path.join(carpeta_actual, "modelo_optimizado.json")
with open(ruta_guardar_modelo, "w") as archivo_json:
    json.dump(resultado_modelo, archivo_json, indent=4)

# Grafico de convergencia
plt.figure(figsize=(7, 4))
plt.plot(range(len(historial_convergencia)), historial_convergencia, marker='o', color='#2b5c8f', linewidth=2)
plt.title('Convergencia del Algoritmo Genetico')
plt.xlabel('Generacion')
plt.ylabel('Exactitud en Train (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

ruta_grafico_convergencia = os.path.join(carpeta_actual, "grafico_convergencia_ga.png")
plt.savefig(ruta_grafico_convergencia, dpi=120)
plt.close()

print("\nArchivos generados en Fase_4_Algoritmo_Genetico:")
print("  - modelo_optimizado.json")
print("  - grafico_convergencia_ga.png")
