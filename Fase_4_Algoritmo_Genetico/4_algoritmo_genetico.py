"""
=============================================================================
FASE 4: ALGORITMO GENÉTICO (OPTIMIZACIÓN EVOLUTIVA)
=============================================================================
¿Qué hace este script?
1. El Algoritmo Genético busca la MEJOR configuración para clasificar los vinos.
2. ¿Qué optimiza?
   - Modifica los límites de:
       Alcohol bajo, medio, alto
       Acidez baja, media, alta
       Sulfatos bajo, medio, alto
       pH bajo, medio, alto
   - Modifica el peso de cada regla (para darle más fuerza a las reglas buenas
     y menos fuerza a las dudosas).
3. ¿Cómo funciona? (Evolución pura en bucles sencillos):
   - Población: conjunto de diferentes combinaciones de límites y pesos.
   - Fitness (Aptitud): porcentaje de vinos bien clasificados en entrenamiento.
   - Selección por Torneo: compiten dos soluciones y gana la mejor.
   - Cruce: combina los parámetros de dos soluciones exitosas.
   - Mutación: pequeños cambios aleatorios para explorar nuevas ideas.
   - Elitismo: el mejor de cada generación nunca se pierde.
4. Guarda el mejor modelo en 'Fase_4_Algoritmo_Genetico/modelo_optimizado.json'.
=============================================================================
"""

import os
import json
import random
import numpy as np
import pandas as pd

# Rutas automáticas
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(DIRECTORIO_ACTUAL, ".."))

print("\n" + "="*60)
print("     FASE 4: ALGORITMO GENÉTICO (OPTIMIZACIÓN)")
print("="*60)

# 1. Cargar datos de entrenamiento, cortes iniciales y reglas
ruta_train = os.path.join(DIRECTORIO_RAIZ, "Fase_1_Preparacion", "datos_train.csv")
ruta_cortes = os.path.join(DIRECTORIO_RAIZ, "Fase_1_Preparacion", "cortes_iniciales.json")
ruta_reglas = os.path.join(DIRECTORIO_RAIZ, "Fase_2_PRISM", "reglas_descubiertas.json")

df_train = pd.read_csv(ruta_train)
with open(ruta_cortes) as f:
    cortes_base = json.load(f)
with open(ruta_reglas) as f:
    reglas = json.load(f)

variables = ['alcohol', 'volatile acidity', 'sulphates', 'pH']
n_reglas = len(reglas)

limites_extremos = {v: (cortes_base[v]['min'], cortes_base[v]['max']) for v in variables}
X_vals = {v: df_train[v].values for v in variables}
y_reales = df_train['calidad'].values
n_muestras = len(df_train)

# =============================================================================
# FUNCIONES RÁPIDAS DEL ALGORITMO GENÉTICO
# =============================================================================

def decodificar(cromosoma):
    """Convierte la lista de números del cromosoma en cortes y pesos."""
    cortes = {}
    idx = 0
    for v in variables:
        min_v, max_v = limites_extremos[v]
        c1, c2, c3 = sorted(cromosoma[idx:idx+3])
        c1 = max(min_v, min(c1, max_v - 0.03))
        c2 = max(c1 + 0.01, min(c2, max_v - 0.02))
        c3 = max(c2 + 0.01, min(c3, max_v - 0.01))
        cortes[v] = {'min': min_v, 'm1': c1, 'm2': c2, 'm3': c3, 'max': max_v}
        idx += 3
        
    pesos = [max(0.0, min(w, 2.0)) for w in cromosoma[idx:idx+n_reglas]]
    return cortes, pesos

def calcular_pertenencia_rapida(x_arr, m1, m2, m3):
    """Calcula vectorizadamente bajo, medio, alto para agilizar la evaluación."""
    u_bajo = np.ones_like(x_arr)
    m1_m2 = (x_arr > m1) & (x_arr < m2)
    u_bajo[m1_m2] = (m2 - x_arr[m1_m2]) / (m2 - m1 + 1e-9)
    u_bajo[x_arr >= m2] = 0.0
    
    u_medio = np.zeros_like(x_arr)
    m1_med = (x_arr > m1) & (x_arr <= m2)
    u_medio[m1_med] = (x_arr[m1_med] - m1) / (m2 - m1 + 1e-9)
    med_m3 = (x_arr > m2) & (x_arr < m3)
    u_medio[med_m3] = (m3 - x_arr[med_m3]) / (m3 - m2 + 1e-9)
    
    u_alto = np.zeros_like(x_arr)
    med_alto = (x_arr > m2) & (x_arr < m3)
    u_alto[med_alto] = (x_arr[med_alto] - m2) / (m3 - m2 + 1e-9)
    u_alto[x_arr >= m3] = 1.0
    
    return {'bajo': u_bajo, 'medio': u_medio, 'alto': u_alto}

def calcular_fitness(cromosoma):
    """Calcula el porcentaje de aciertos en entrenamiento para un cromosoma."""
    cortes, pesos = decodificar(cromosoma)
    
    grados = {}
    for v in variables:
        c = cortes[v]
        grados[v] = calcular_pertenencia_rapida(X_vals[v], c['m1'], c['m2'], c['m3'])
        
    scores = {'BAJA': np.zeros(n_muestras), 'MEDIA': np.zeros(n_muestras), 'ALTA': np.zeros(n_muestras)}
    
    for i, regla in enumerate(reglas):
        w = pesos[i]
        if w <= 0.01:
            continue
            
        disparo = np.ones(n_muestras)
        for var, etiqueta in regla['condiciones'].items():
            disparo = np.minimum(disparo, grados[var][etiqueta])
            
        scores[regla['consecuente']] += w * disparo
        
    predicciones = []
    s_baja = scores['BAJA']
    s_media = scores['MEDIA']
    s_alta = scores['ALTA']
    
    for i in range(n_muestras):
        b, m, a = s_baja[i], s_media[i], s_alta[i]
        if a > m and a > b:
            predicciones.append('ALTA')
        elif b > m and b >= a:
            predicciones.append('BAJA')
        else:
            predicciones.append('MEDIA')
            
    aciertos = np.sum(np.array(predicciones) == y_reales)
    return round((aciertos / n_muestras) * 100, 2)


# =============================================================================
# OPERADORES GENÉTICOS
# =============================================================================

cromosoma_semilla = []
for v in variables:
    cromosoma_semilla.extend([cortes_base[v]['m1'], cortes_base[v]['m2'], cortes_base[v]['m3']])
cromosoma_semilla.extend([r['peso'] for r in reglas])
longitud_cromosoma = len(cromosoma_semilla)

def crear_individuo():
    ind = list(cromosoma_semilla)
    for i in range(len(ind)):
        if i < 12:
            ind[i] += random.gauss(0, 0.05 * abs(ind[i]))
        else:
            ind[i] = max(0.1, min(ind[i] + random.gauss(0, 0.2), 1.5))
    return ind

# =============================================================================
# HIPERPARÁMETROS DEL ALGORITMO GENÉTICO (SEGÚN DIAPOSITIVAS)
# =============================================================================

random.seed(42)
np.random.seed(42)

TAM_POBLACION = 30              # N: Tamaño de la población
GENERACIONES = 25               # G: Criterio de parada (Generations)
METODO_SELECCION = "tournament" # Opciones: "tournament" o "roulette"
TOURNAMENT_SIZE = 2             # Tamaño del torneo si se usa Tournament
PROBABILIDAD_CRUCE = 0.85       # Pc: Crossover rate (85%)
PROBABILIDAD_MUTACION = 0.20    # Pm: Mutation rate (20%)
ELITE_COUNT = 1                 # Elite count: mejores individuos conservados

# 1. SELECCIÓN POR TORNEO (Slide: "Selection: Tournament")
def seleccion_torneo(poblacion, fitnesses, k=TOURNAMENT_SIZE):
    participantes = random.sample(range(len(poblacion)), k)
    mejor_idx = participantes[0]
    for idx in participantes[1:]:
        if fitnesses[idx] > fitnesses[mejor_idx]:
            mejor_idx = idx
    return poblacion[mejor_idx]

# 2. SELECCIÓN POR RULETA (Slide: "Selection: Roulette")
def seleccion_ruleta(poblacion, fitnesses):
    total_fit = sum(fitnesses)
    tiro = random.uniform(0, total_fit)
    acumulado = 0.0
    for ind, fit in zip(poblacion, fitnesses):
        acumulado += fit
        if acumulado >= tiro:
            return ind
    return poblacion[-1]

def seleccionar_padre(poblacion, fitnesses):
    if METODO_SELECCION == "roulette":
        return seleccion_ruleta(poblacion, fitnesses)
    else:
        return seleccion_torneo(poblacion, fitnesses)

# 3. CRUCE INTERMEDIO (Slide: "Crossover: Intermediate")
# Fórmula exacta de la diapositiva: HIJO = PADRE1 + rnd * (PADRE2 - PADRE1)
def cruzar_intermediate(padre1, padre2):
    hijo = []
    for g1, g2 in zip(padre1, padre2):
        rnd = random.uniform(0.3, 0.7)
        hijo.append(g1 + rnd * (g2 - g1))
    return hijo

# 4. MUTACIÓN GAUSSIANA (Slide: "Mutation: Gaussian")
# Agrega un número aleatorio gaussiano con media 0
def mutar_gaussian(individuo, prob_mutacion):
    for i in range(len(individuo)):
        if random.random() < prob_mutacion:
            if i < 12:
                individuo[i] += random.gauss(0, 0.03 * abs(individuo[i]))
            else:
                individuo[i] = max(0.0, min(individuo[i] + random.gauss(0, 0.15), 2.0))
    return individuo

print(f"Configuración del Algoritmo Genético (Toolbox):")
print(f"  • Tamaño de población (N):      {TAM_POBLACION} individuos")
print(f"  • Criterio de parada:           {GENERACIONES} generaciones")
print(f"  • Método de Selección:          {METODO_SELECCION.upper()}")
print(f"  • Tipo de Crossover:            INTERMEDIATE (Cruce Intermedio)")
print(f"  • Tipo de Mutación:             GAUSSIAN (Gaussiana media 0)")
print(f"  • Probabilidad de Cruce (Pc):   {PROBABILIDAD_CRUCE * 100:.0f} %")
print(f"  • Probabilidad de Mutación (Pm):{PROBABILIDAD_MUTACION * 100:.0f} %")
print(f"  • Elite Count:                  {ELITE_COUNT} individuo(s)")
print(f"  • Number of Variables:          {longitud_cromosoma} parámetros en theta")

poblacion = [cromosoma_semilla]
for _ in range(TAM_POBLACION - 1):
    poblacion.append(crear_individuo())

fitnesses = [calcular_fitness(ind) for ind in poblacion]
mejor_fitness_inicial = max(fitnesses)
print(f"\nAcierto inicial antes de optimizar (Cuantiles): {mejor_fitness_inicial:.2f} %")
print("\nComenzando evolución...")

mejor_global = poblacion[fitnesses.index(mejor_fitness_inicial)]
mejor_fitness_global = mejor_fitness_inicial
historial = [mejor_fitness_global]

for gen in range(1, GENERACIONES + 1):
    nueva_poblacion = []
    
    # REPRODUCTION: Elite Count (pasan intactos los mejores)
    indices_ordenados = np.argsort(fitnesses)[::-1]
    for e in range(ELITE_COUNT):
        nueva_poblacion.append(list(poblacion[indices_ordenados[e]]))
    
    # REPRODUCCIÓN HASTA COMPLETAR N INDIVIDUOS
    while len(nueva_poblacion) < TAM_POBLACION:
        # Selección
        padre1 = seleccionar_padre(poblacion, fitnesses)
        padre2 = seleccionar_padre(poblacion, fitnesses)
        
        # Crossover (Intermediate)
        if random.random() < PROBABILIDAD_CRUCE:
            hijo = cruzar_intermediate(padre1, padre2)
        else:
            hijo = list(padre1)
            
        # Mutation (Gaussian)
        hijo = mutar_gaussian(hijo, PROBABILIDAD_MUTACION)
        
        nueva_poblacion.append(hijo)

        
    poblacion = nueva_poblacion
    fitnesses = [calcular_fitness(ind) for ind in poblacion]
    
    max_gen = max(fitnesses)
    if max_gen > mejor_fitness_global:
        mejor_fitness_global = max_gen
        mejor_global = list(poblacion[fitnesses.index(max_gen)])
        
    historial.append(mejor_fitness_global)
    
    if gen % 5 == 0 or gen == GENERACIONES:
        print(f"  Generación {gen:02d} | Mejor acierto en entrenamiento: {mejor_fitness_global:.2f} %")

cortes_optimizados, pesos_optimizados = decodificar(mejor_global)

resultado_final = {
    'cortes_optimizados': cortes_optimizados,
    'pesos_optimizados': [round(w, 4) for w in pesos_optimizados],
    'fitness_inicial': mejor_fitness_inicial,
    'fitness_final': mejor_fitness_global,
    'historial': historial
}

ruta_guardar = os.path.join(DIRECTORIO_ACTUAL, "modelo_optimizado.json")
with open(ruta_guardar, "w") as f:
    json.dump(resultado_final, f, indent=4)

# Guardar gráfico de convergencia del AG
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4.5))
plt.plot(range(len(historial)), historial, marker='o', color='#E74C3C', linewidth=2.2, markersize=5)
plt.title('Convergencia del Algoritmo Genético', fontsize=12, fontweight='bold')
plt.xlabel('Generación')
plt.ylabel('Acierto en Entrenamiento (%)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
ruta_grafico_ga = os.path.join(DIRECTORIO_ACTUAL, "grafico_convergencia_ga.png")
plt.savefig(ruta_grafico_ga, dpi=150)
plt.close()

print("\n" + "="*70)
print(f"OPTIMIZACIÓN COMPLETADA:")
print(f"  • Acierto inicial (cuantiles): {mejor_fitness_inicial:.2f} %")
print(f"  • Acierto final optimizado:   {mejor_fitness_global:.2f} % (Mejora: +{mejor_fitness_global - mejor_fitness_inicial:.2f} %)")
print("="*70)

print("\nNuevos límites químicos optimizados por el AG:")
for v in variables:
    c = cortes_optimizados[v]
    print(f"  • {v:17s} -> m1 (Bajo/Medio): {c['m1']:.2f} | m2 (Centro): {c['m2']:.2f} | m3 (Medio/Alto): {c['m3']:.2f}")

print("\nArchivos generados en Fase_4_Algoritmo_Genetico/:")
print("  -> modelo_optimizado.json")
print("  -> grafico_convergencia_ga.png")
print("--- FASE 4 COMPLETADA CON ÉXITO ---\n")

