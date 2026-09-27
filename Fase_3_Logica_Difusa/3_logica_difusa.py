# Proyecto Inteligencia Artificial
# Fase 3: Sistema de Inferencia Difusa Mamdani para la Calidad del Vino
# Flujo Secuencial:
#   1. Cargar datos, cortes y reglas
#   2. Paso 1 - Fuzzificacion (funciones de pertenencia del vino)
#   3. Paso 2 - Inferencia Mamdani (operador MIN y multiplicacion por peso)
#   4. Paso 3 - Agregacion de puntajes por clase
#   5. Paso 4 - Defuzzificacion (Centroide CoG y Maxima Pertenencia)
#   6. Evaluacion del conjunto completo de vinos

import os
import json
import pandas as pd
import numpy as np

carpeta_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_proyecto = os.path.abspath(os.path.join(carpeta_actual, ".."))


# =============================================================================
# 1. CARGA DE ARCHIVOS DEL PROYECTO
# =============================================================================
ruta_datos_entrenamiento = os.path.join(carpeta_proyecto, "Fase_1_Preparacion", "datos_train.csv")
ruta_datos_prueba = os.path.join(carpeta_proyecto, "Fase_1_Preparacion", "datos_test.csv")
ruta_cortes_iniciales = os.path.join(carpeta_proyecto, "Fase_1_Preparacion", "cortes_iniciales.json")
ruta_reglas_prism = os.path.join(carpeta_proyecto, "Fase_2_PRISM", "reglas_descubiertas.json")

datos_entrenamiento = pd.read_csv(ruta_datos_entrenamiento)
datos_prueba = pd.read_csv(ruta_datos_prueba)

with open(ruta_cortes_iniciales) as archivo_cortes:
    cortes_quimicos = json.load(archivo_cortes)

with open(ruta_reglas_prism) as archivo_reglas:
    lista_reglas_prism = json.load(archivo_reglas)

pesos_reglas_iniciales = [regla['peso'] for regla in lista_reglas_prism]


# =============================================================================
# 2. PASO 1: FUNCIONES DE PERTENENCIA Y FUZZIFICACION
# =============================================================================

# Funcion de hombro izquierdo: asigna pertenencia al valor 'bajo'
def calcular_pertenencia_baja(valor_quimico, corte_bajo, corte_medio):
    if valor_quimico <= corte_bajo:
        return 1.0
    elif corte_bajo < valor_quimico < corte_medio:
        ancho_intervalo = corte_medio - corte_bajo
        if ancho_intervalo == 0:
            return 0.0
        return (corte_medio - valor_quimico) / ancho_intervalo
    else:
        return 0.0

# Funcion triangular: asigna pertenencia al valor 'medio'
def calcular_pertenencia_media(valor_quimico, corte_bajo, corte_medio, corte_alto):
    if corte_bajo < valor_quimico <= corte_medio:
        ancho_subida = corte_medio - corte_bajo
        if ancho_subida == 0:
            return 0.0
        return (valor_quimico - corte_bajo) / ancho_subida
    elif corte_medio < valor_quimico < corte_alto:
        ancho_bajada = corte_alto - corte_medio
        if ancho_bajada == 0:
            return 0.0
        return (corte_alto - valor_quimico) / ancho_bajada
    else:
        return 0.0

# Funcion de hombro derecho: asigna pertenencia al valor 'alto'
def calcular_pertenencia_alta(valor_quimico, corte_medio, corte_alto):
    if valor_quimico <= corte_medio:
        return 0.0
    elif corte_medio < valor_quimico < corte_alto:
        ancho_subida = corte_alto - corte_medio
        if ancho_subida == 0:
            return 0.0
        return (valor_quimico - corte_medio) / ancho_subida
    else:
        return 1.0

# Fuzzificacion completa de una muestra de vino (sus 4 variables quimicas)
def fuzzificar_vino(fila_vino, diccionario_cortes):
    # Calculo para el alcohol
    valor_alcohol = float(fila_vino['alcohol'])
    corte_alc_bajo = diccionario_cortes['alcohol']['corte_bajo']
    corte_alc_medio = diccionario_cortes['alcohol']['punto_medio']
    corte_alc_alto = diccionario_cortes['alcohol']['corte_alto']
    
    pertenencias_alcohol = {
        'bajo': round(calcular_pertenencia_baja(valor_alcohol, corte_alc_bajo, corte_alc_medio), 4),
        'medio': round(calcular_pertenencia_media(valor_alcohol, corte_alc_bajo, corte_alc_medio, corte_alc_alto), 4),
        'alto': round(calcular_pertenencia_alta(valor_alcohol, corte_alc_medio, corte_alc_alto), 4)
    }

    # Calculo para la acidez volatil
    valor_acidez = float(fila_vino['acidez_volatil'])
    corte_acid_bajo = diccionario_cortes['acidez_volatil']['corte_bajo']
    corte_acid_medio = diccionario_cortes['acidez_volatil']['punto_medio']
    corte_acid_alto = diccionario_cortes['acidez_volatil']['corte_alto']
    
    pertenencias_acidez = {
        'bajo': round(calcular_pertenencia_baja(valor_acidez, corte_acid_bajo, corte_acid_medio), 4),
        'medio': round(calcular_pertenencia_media(valor_acidez, corte_acid_bajo, corte_acid_medio, corte_acid_alto), 4),
        'alto': round(calcular_pertenencia_alta(valor_acidez, corte_acid_medio, corte_acid_alto), 4)
    }

    # Calculo para los sulfatos
    valor_sulfatos = float(fila_vino['sulfatos'])
    corte_sulf_bajo = diccionario_cortes['sulfatos']['corte_bajo']
    corte_sulf_medio = diccionario_cortes['sulfatos']['punto_medio']
    corte_sulf_alto = diccionario_cortes['sulfatos']['corte_alto']
    
    pertenencias_sulfatos = {
        'bajo': round(calcular_pertenencia_baja(valor_sulfatos, corte_sulf_bajo, corte_sulf_medio), 4),
        'medio': round(calcular_pertenencia_media(valor_sulfatos, corte_sulf_bajo, corte_sulf_medio, corte_sulf_alto), 4),
        'alto': round(calcular_pertenencia_alta(valor_sulfatos, corte_sulf_medio, corte_sulf_alto), 4)
    }

    # Calculo para el pH
    valor_ph = float(fila_vino['ph'])
    corte_ph_bajo = diccionario_cortes['ph']['corte_bajo']
    corte_ph_medio = diccionario_cortes['ph']['punto_medio']
    corte_ph_alto = diccionario_cortes['ph']['corte_alto']
    
    pertenencias_ph = {
        'bajo': round(calcular_pertenencia_baja(valor_ph, corte_ph_bajo, corte_ph_medio), 4),
        'medio': round(calcular_pertenencia_media(valor_ph, corte_ph_bajo, corte_ph_medio, corte_ph_alto), 4),
        'alto': round(calcular_pertenencia_alta(valor_ph, corte_ph_medio, corte_ph_alto), 4)
    }

    return {
        'alcohol': pertenencias_alcohol,
        'acidez_volatil': pertenencias_acidez,
        'sulfatos': pertenencias_sulfatos,
        'ph': pertenencias_ph
    }


# =============================================================================
# 3. PASO 2: INFERENCIA MAMDANI
# =============================================================================
def inferencia_mamdani_vino(grados_pertenencia_vino, lista_reglas, lista_pesos):
    reglas_disparadas = []
    
    for indice, regla in enumerate(lista_reglas):
        peso_de_la_regla = lista_pesos[indice]
        if peso_de_la_regla <= 0.01:
            continue
            
        # Operador MIN (T-Norma) para todas las condiciones del antecedente
        fuerza_de_disparo = 1.0
        for variable_quimica, etiqueta_requerida in regla['condiciones'].items():
            pertenencia_variable = grados_pertenencia_vino[variable_quimica][etiqueta_requerida]
            if pertenencia_variable < fuerza_de_disparo:
                fuerza_de_disparo = pertenencia_variable
                
        # Impacto de la regla en el consecuente
        impacto_en_calidad = fuerza_de_disparo * peso_de_la_regla
        
        reglas_disparadas.append({
            'indice_regla': indice,
            'calidad_consecuente': regla['consecuente'],
            'fuerza_disparo': fuerza_de_disparo,
            'peso': peso_de_la_regla,
            'impacto': impacto_en_calidad,
            'condiciones': regla['condiciones']
        })
        
    return reglas_disparadas


# =============================================================================
# 4. PASO 3: AGREGACION DIFUSA
# =============================================================================
def agregar_impactos_por_calidad(reglas_disparadas):
    puntajes_acumulados = {'BAJA': 0.0, 'MEDIA': 0.0, 'ALTA': 0.0}
    
    for disparo in reglas_disparadas:
        calidad = disparo['calidad_consecuente']
        puntajes_acumulados[calidad] += disparo['impacto']
        
    return puntajes_acumulados


# =============================================================================
# 5. PASO 4: DEFUZZIFICACION (CENTROIDE CoG Y MAXIMA PERTENENCIA)
# =============================================================================
CENTRO_ENOLOGICO_BAJA = 4.5
CENTRO_ENOLOGICO_MEDIA = 6.0
CENTRO_ENOLOGICO_ALTA = 7.5

def defuzzificar_calidad(puntajes_acumulados):
    suma_total_puntajes = sum(puntajes_acumulados.values())
    
    # Metodo 1: Maxima pertenencia (clase con mayor puntaje)
    calidad_ganadora = max(puntajes_acumulados, key=puntajes_acumulados.get)
    if puntajes_acumulados[calidad_ganadora] == 0:
        calidad_ganadora = 'MEDIA'
        
    # Metodo 2: Centroide de Gravedad (calificacion continua entre 0 y 10)
    if suma_total_puntajes > 0:
        calificacion_centroide = (
            puntajes_acumulados['BAJA'] * CENTRO_ENOLOGICO_BAJA +
            puntajes_acumulados['MEDIA'] * CENTRO_ENOLOGICO_MEDIA +
            puntajes_acumulados['ALTA'] * CENTRO_ENOLOGICO_ALTA
        ) / suma_total_puntajes
        porcentaje_certidumbre = (puntajes_acumulados[calidad_ganadora] / suma_total_puntajes) * 100.0
    else:
        calificacion_centroide = 6.0
        porcentaje_certidumbre = 33.33
        
    return calidad_ganadora, round(porcentaje_certidumbre, 1), round(calificacion_centroide, 2)


# =============================================================================
# 6. EVALUACION DE UN VINO Y DE TODO EL CONJUNTO DE DATOS
# =============================================================================

# Evalua un solo vino ejecutando los 4 pasos secuenciales de Mamdani
def evaluar_vino_completo(fila_vino, diccionario_cortes, lista_reglas, lista_pesos=None):
    if lista_pesos is None:
        lista_pesos = [r.get('peso', 1.0) for r in lista_reglas]
        
    # Paso 1: Fuzzificacion
    grados_vino = fuzzificar_vino(fila_vino, diccionario_cortes)
    
    # Paso 2: Inferencia
    disparos = inferencia_mamdani_vino(grados_vino, lista_reglas, lista_pesos)
    
    # Paso 3: Agregacion
    puntajes = agregar_impactos_por_calidad(disparos)
    
    # Paso 4: Defuzzificacion
    calidad_final, certidumbre, calificacion_continua = defuzzificar_calidad(puntajes)
    
    return calidad_final, puntajes, grados_vino


# Evalua un dataframe completo de vinos y calcula la exactitud (%)
def calcular_exactitud_dataset(df_vinos, diccionario_cortes, lista_reglas, lista_pesos=None):
    vinos_correctos = 0
    total_vinos = len(df_vinos)
    
    for _, fila_vino in df_vinos.iterrows():
        prediccion_calidad, _, _ = evaluar_vino_completo(
            fila_vino, diccionario_cortes, lista_reglas, lista_pesos
        )
        if prediccion_calidad == fila_vino['calidad']:
            vinos_correctos += 1
            
    porcentaje_exactitud = (vinos_correctos / total_vinos) * 100.0
    return round(porcentaje_exactitud, 2)


# Alias compatibles para otras fases si las necesitan
fuzzificacion = fuzzificar_vino
inferencia_mamdani = inferencia_mamdani_vino
agregacion_difusa = agregar_impactos_por_calidad
defuzzificacion = defuzzificar_calidad
evaluar_vino_difuso = evaluar_vino_completo
evaluar_dataset = calcular_exactitud_dataset
pertenencia_bajo = calcular_pertenencia_baja
pertenencia_medio = calcular_pertenencia_media
pertenencia_alto = calcular_pertenencia_alta


# =============================================================================
# 7. DEMOSTRACION SECUENCIAL EN UN VINO DE PRUEBA
# =============================================================================
if __name__ == "__main__":
    print("=== Fase 3: Sistema de Inferencia Difusa (Mamdani) ===")
    print("Total de reglas PRISM cargadas:", len(lista_reglas_prism))
    print(f"Vinos en entrenamiento: {len(datos_entrenamiento)} | Vinos en prueba: {len(datos_prueba)}")
    
    # Tomamos el primer vino del conjunto de prueba
    vino_de_prueba = datos_prueba.iloc[0]
    print(f"\n--- Probando Vino 1 (Calidad Real: {vino_de_prueba['calidad']}) ---")
    print(f"Propiedades quimicas:")
    print(f"  Alcohol:         {vino_de_prueba['alcohol']}%")
    print(f"  Acidez Volatil:  {vino_de_prueba['acidez_volatil']} g/dm3")
    print(f"  Sulfatos:        {vino_de_prueba['sulfatos']} g/dm3")
    print(f"  pH:              {vino_de_prueba['ph']}")
    
    # [Paso 1] Fuzzificacion
    grados_obtenidos = fuzzificar_vino(vino_de_prueba, cortes_quimicos)
    print("\n[Paso 1] Fuzzificacion:")
    for caracteristica, grados in grados_obtenidos.items():
        print(f"  {caracteristica:15s} -> Bajo={grados['bajo']:.2f}, Medio={grados['medio']:.2f}, Alto={grados['alto']:.2f}")
        
    # [Paso 2] Inferencia
    disparos_obtenidos = inferencia_mamdani_vino(grados_obtenidos, lista_reglas_prism, pesos_reglas_iniciales)
    disparos_obtenidos.sort(key=lambda d: d['impacto'], reverse=True)
    print("\n[Paso 2] Inferencia (Reglas que mas se activaron):")
    for disparo in disparos_obtenidos[:3]:
        partes_texto = []
        for variable, valor in disparo['condiciones'].items():
            partes_texto.append(variable + " = " + valor)
        texto_si = " Y ".join(partes_texto)
        
        calidad = disparo['calidad_consecuente']
        fuerza = round(disparo['fuerza_disparo'], 2)
        impacto = round(disparo['impacto'], 3)
        print(f"  SI {texto_si} -> Calidad = {calidad} (Fuerza = {fuerza}, Impacto = {impacto})")
        
    # [Paso 3] Agregacion
    puntajes_obtenidos = agregar_impactos_por_calidad(disparos_obtenidos)
    print("\n[Paso 3] Agregacion difusa por calidad:")
    for calidad, puntaje in puntajes_obtenidos.items():
        print(f"  Puntaje {calidad:5s}: {puntaje:.3f}")
        
    # [Paso 4] Defuzzificacion
    calidad_predicha, certidumbre, calificacion_continua = defuzzificar_calidad(puntajes_obtenidos)
    print("\n[Paso 4] Defuzzificacion:")
    print(f"  Clase predicha por Maxima Pertenencia: [{calidad_predicha}] ({certidumbre}% de certidumbre)")
    print(f"  Calificacion continua por Centroide CoG: {calificacion_continua:.2f} / 10")
    
    # Evaluacion global del dataset
    exactitud_entrenamiento = calcular_exactitud_dataset(
        datos_entrenamiento, cortes_quimicos, lista_reglas_prism, pesos_reglas_iniciales
    )
    exactitud_prueba = calcular_exactitud_dataset(
        datos_prueba, cortes_quimicos, lista_reglas_prism, pesos_reglas_iniciales
    )
    
    print("\nExactitud global preliminar con cortes por cuantiles:")
    print(f"  Entrenamiento: {exactitud_entrenamiento}%")
    print(f"  Prueba:        {exactitud_prueba}%")
