"""
=============================================================================
FASE 3: SISTEMA DE INFERENCIA DIFUSA TIPO MAMDANI
=============================================================================
Implementación canónica en las 4 etapas del modelo de Ebrahim Mamdani (1975):

  1. FUZZIFICACIÓN:
     Convierte las entradas numéricas reales (crisp) en grados de pertenencia
     lingüísticos mu en [0, 1] mediante funciones de pertenencia (hombro y triangular).

  2. INFERENCIA (Implicación Mamdani):
     Evalúa las reglas PRISM con la T-Norma MIN (operador Y / AND) para obtener
     la fuerza de disparo alpha de cada regla y aplica el peso w.

  3. AGREGACIÓN:
     Combina las salidas de todas las reglas activadas por cada categoría
     consecuente (BAJA, MEDIA, ALTA) en una distribución acumulada.

  4. DEFUZZIFICACIÓN:
     Convierte la salida difusa agregada en un veredicto nítido (crisp):
     - Método del Centroide (Center of Gravity / CoG)
     - Método de Máxima Pertenencia (Clase Ganadora con porcentaje de certidumbre)
=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np

# Rutas automáticas
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(DIRECTORIO_ACTUAL, ".."))

# =============================================================================
# FUNCIONES DE PERTENENCIA (MEMBERSHIP FUNCTIONS)
# =============================================================================

def pertenencia_bajo(x, m1, m2):
    """
    Función de hombro izquierdo:
    - Vale 1.0 si x <= m1
    - Cae suavemente en rampa de 1.0 a 0.0 entre m1 y m2
    - Vale 0.0 si x >= m2
    """
    if x <= m1:
        return 1.0
    elif m1 < x < m2:
        return (m2 - x) / (m2 - m1 + 1e-9)
    else:
        return 0.0

def pertenencia_medio(x, m1, m2, m3):
    """
    Función triangular:
    - Sube de 0.0 a 1.0 entre m1 y m2
    - Alcanza su máximo (1.0) en m2
    - Cae de 1.0 a 0.0 entre m2 y m3
    - Fuera de ese rango vale 0.0
    """
    if m1 < x <= m2:
        return (x - m1) / (m2 - m1 + 1e-9)
    elif m2 < x < m3:
        return (m3 - x) / (m3 - m2 + 1e-9)
    else:
        return 0.0

def pertenencia_alto(x, m2, m3):
    """
    Función de hombro derecho:
    - Vale 0.0 si x <= m2
    - Sube suavemente en rampa de 0.0 a 1.0 entre m2 y m3
    - Vale 1.0 si x >= m3
    """
    if x <= m2:
        return 0.0
    elif m2 < x < m3:
        return (x - m2) / (m3 - m2 + 1e-9)
    else:
        return 1.0

def calcular_grados(valor, limites_var):
    """Calcula los tres grados (bajo, medio, alto) para una variable continua."""
    m1 = limites_var['m1']
    m2 = limites_var['m2']
    m3 = limites_var['m3']
    
    return {
        'bajo': round(pertenencia_bajo(valor, m1, m2), 4),
        'medio': round(pertenencia_medio(valor, m1, m2, m3), 4),
        'alto': round(pertenencia_alto(valor, m2, m3), 4)
    }


# =============================================================================
# LAS 4 ETAPAS DEL SISTEMA DIFUSO MAMDANI
# =============================================================================

# -----------------------------------------------------------------------------
# ETAPA 1: FUZZIFICACIÓN (Fuzzification)
# -----------------------------------------------------------------------------
def fuzzificacion(fila_vino, cortes):
    """
    ETAPA 1: Convierte los valores numéricos nítidos (crisp) de entrada
    en grados de pertenencia difusa lingüísticos: Bajo, Medio y Alto.
    """
    grados = {}
    for var in ['alcohol', 'volatile acidity', 'sulphates', 'pH']:
        valor_real = float(fila_vino[var])
        grados[var] = calcular_grados(valor_real, cortes[var])
    return grados


# -----------------------------------------------------------------------------
# ETAPA 2: INFERENCIA / EVALUACIÓN DE REGLAS (Mamdani Implication)
# -----------------------------------------------------------------------------
def inferencia_mamdani(grados, reglas, pesos):
    """
    ETAPA 2: Evalúa el antecedente de cada regla mediante la T-Norma MIN de Mamdani:
        alpha_i = min(mu_1, mu_2, ...)
    Y pondera el consecuente multiplicando por el peso de la regla w_i.
    """
    reglas_disparadas = []
    
    for i, regla in enumerate(reglas):
        w = pesos[i]
        if w <= 0.01:
            continue
            
        # Operador MIN (conjunción difusa AND)
        fuerza_disparo = 1.0
        for var, etiqueta in regla['condiciones'].items():
            u = grados[var][etiqueta]
            if u < fuerza_disparo:
                fuerza_disparo = u
                
        # Implicación ponderada: fuerza de corte sobre el consecuente
        impacto_regla = fuerza_disparo * w
        reglas_disparadas.append({
            'regla_idx': i,
            'consecuente': regla['consecuente'],
            'fuerza_disparo': fuerza_disparo,
            'peso': w,
            'impacto': impacto_regla,
            'condiciones': regla['condiciones']
        })
        
    return reglas_disparadas


# -----------------------------------------------------------------------------
# ETAPA 3: AGREGACIÓN DIFUSA (Aggregation)
# -----------------------------------------------------------------------------
def agregacion_difusa(reglas_disparadas):
    """
    ETAPA 3: Agrega (combina) los impactos de todas las reglas activadas
    para cada una de las clases de salida: BAJA, MEDIA y ALTA.
    """
    puntajes_acumulados = {'BAJA': 0.0, 'MEDIA': 0.0, 'ALTA': 0.0}
    
    for disparo in reglas_disparadas:
        clase = disparo['consecuente']
        puntajes_acumulados[clase] += disparo['impacto']
        
    return puntajes_acumulados


# -----------------------------------------------------------------------------
# ETAPA 4: DEFUZZIFICACIÓN (Defuzzification)
# -----------------------------------------------------------------------------
# Centros enológicos de calidad para el cálculo del Centroide
CENTROS_CALIDAD = {'BAJA': 4.5, 'MEDIA': 6.0, 'ALTA': 7.5}

def defuzzificacion(puntajes_acumulados):
    """
    ETAPA 4: Convierte la distribución agregada en una decisión final nítida (crisp):
    1. Método de Máxima Pertenencia: Gana la clase con mayor puntaje acumulado.
    2. Método del Centroide (Center of Gravity / CoG):
           y* = sum(Score_c * Centro_c) / sum(Score_c)
    3. Certidumbre: Porcentaje relativo de la clase ganadora.
    """
    suma_puntajes = sum(puntajes_acumulados.values())
    
    # 1. Decisión por Máxima Pertenencia
    clase_ganadora = max(puntajes_acumulados, key=puntajes_acumulados.get)
    if puntajes_acumulados[clase_ganadora] == 0:
        clase_ganadora = 'MEDIA'
        
    # 2. Cálculo del Centroide Mamdani (CoG)
    if suma_puntajes > 0:
        centroide = sum(puntajes_acumulados[c] * CENTROS_CALIDAD[c] for c in ['BAJA', 'MEDIA', 'ALTA']) / suma_puntajes
        certidumbre = (puntajes_acumulados[clase_ganadora] / suma_puntajes) * 100.0
    else:
        centroide = 6.0
        certidumbre = 33.33
        
    return clase_ganadora, round(certidumbre, 1), round(centroide, 2)


# =============================================================================
# INTEGRACIÓN COMPLETA DEL MOTOR MAMDANI
# =============================================================================

def evaluar_vino_difuso(fila_vino, cortes, reglas, pesos=None):
    """
    Ejecuta el ciclo completo Mamdani en 4 pasos:
    1. Fuzzificación
    2. Inferencia (Mamdani min)
    3. Agregación
    4. Defuzzificación
    """
    if pesos is None:
        pesos = [r.get('peso', 1.0) for r in reglas]
        
    # 1. Fuzzificación
    grados = fuzzificacion(fila_vino, cortes)
    
    # 2. Inferencia Mamdani
    reglas_disparadas = inferencia_mamdani(grados, reglas, pesos)
    
    # 3. Agregación
    puntajes_acumulados = agregacion_difusa(reglas_disparadas)
    
    # 4. Defuzzificación
    clase_ganadora, certidumbre, centroide = defuzzificacion(puntajes_acumulados)
    
    return clase_ganadora, puntajes_acumulados, grados

def evaluar_dataset(df, cortes, reglas, pesos=None):
    """Evalúa un lote completo de vinos y calcula la exactitud global."""
    aciertos = 0
    total = len(df)
    
    for _, fila in df.iterrows():
        prediccion, _, _ = evaluar_vino_difuso(fila, cortes, reglas, pesos)
        if prediccion == fila['calidad']:
            aciertos += 1
            
    return round((aciertos / total) * 100, 2)


# =============================================================================
# EJECUCIÓN DIRECTA DEL ARCHIVO
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*65)
    print("     FASE 3: SISTEMA DE INFERENCIA DIFUSA TIPO MAMDANI")
    print("="*65)
    
    ruta_train = os.path.join(DIRECTORIO_RAIZ, "Fase_1_Preparacion", "datos_train.csv")
    ruta_test = os.path.join(DIRECTORIO_RAIZ, "Fase_1_Preparacion", "datos_test.csv")
    ruta_cortes = os.path.join(DIRECTORIO_RAIZ, "Fase_1_Preparacion", "cortes_iniciales.json")
    ruta_reglas = os.path.join(DIRECTORIO_RAIZ, "Fase_2_PRISM", "reglas_descubiertas.json")
    
    df_train = pd.read_csv(ruta_train)
    df_test = pd.read_csv(ruta_test)
    
    with open(ruta_cortes) as f:
        cortes_iniciales = json.load(f)
    with open(ruta_reglas) as f:
        reglas_prism = json.load(f)
        
    pesos_iniciales = [r['peso'] for r in reglas_prism]
    print(f"Total de reglas difusas cargadas: {len(reglas_prism)}")
    
    # DEMOSTRACIÓN DIDÁCTICA PASO A PASO EN UN VINO DE PRUEBA
    ejemplo = df_test.iloc[0]
    print("\n" + "-"*65)
    print("DEMOSTRACION DE LAS 4 ETAPAS MAMDANI EN UN VINO REAL:")
    print("-"*65)
    
    # PASO 1: FUZZIFICACIÓN
    grados_ejemplo = fuzzificacion(ejemplo, cortes_iniciales)
    print("1. FUZZIFICACION (Entradas Reales -> Grados Linguisticos):")
    for var, g in grados_ejemplo.items():
        print(f"   * {var:17s} ({ejemplo[var]}): Bajo={g['bajo']:.2f} | Medio={g['medio']:.2f} | Alto={g['alto']:.2f}")
        
    # PASO 2: INFERENCIA
    disparadas = inferencia_mamdani(grados_ejemplo, reglas_prism, pesos_iniciales)
    disparadas.sort(key=lambda d: d['impacto'], reverse=True)
    print("\n2. INFERENCIA MAMDANI (Evaluacion del operador MIN en reglas):")
    for d in disparadas[:3]:
        cond_str = " AND ".join([f"{k} es {v.upper()}" for k, v in d['condiciones'].items()])
        print(f"   * SI {cond_str} ENTONCES calidad = {d['consecuente']} (Fuerza={d['fuerza_disparo']:.2f}, Peso={d['peso']:.2f} -> Impacto={d['impacto']:.3f})")
        
    # PASO 3: AGREGACIÓN
    agregados = agregacion_difusa(disparadas)
    print("\n3. AGREGACION (Combinacion de impactos por clase de salida):")
    for c, score in agregados.items():
        print(f"   * Salida acumulada {c:5s}: {score:.3f} puntos")
        
    # PASO 4: DEFUZZIFICACIÓN
    ganadora, cert, centroide = defuzzificacion(agregados)
    print("\n4. DEFUZZIFICACION (Salida Difusa -> Salida Nitida Crisp):")
    print(f"   * Metodo de Maxima Pertenencia: [{ganadora}] ({cert}% de certidumbre)")
    print(f"   * Metodo del Centroide (CoG):    {centroide:.2f} / 10 puntos de calidad")
    print(f"   * Valor Real del vino:           [{ejemplo['calidad']}]")
    print("-"*65)
    
    # Evaluación global inicial
    acc_train = evaluar_dataset(df_train, cortes_iniciales, reglas_prism, pesos_iniciales)
    acc_test = evaluar_dataset(df_test, cortes_iniciales, reglas_prism, pesos_iniciales)
    
    print("\nResultados Globales del Sistema Mamdani Inicial (Cortes por cuantiles):")
    print(f"  Exactitud en Entrenamiento: {acc_train} %")
    print(f"  Exactitud en Prueba:        {acc_test} %")
    print("--- FASE 3 COMPLETADA CON ÉXITO ---\n")
