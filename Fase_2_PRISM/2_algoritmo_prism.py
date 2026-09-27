# Proyecto Inteligencia Artificial
# Fase 2: Algoritmo PRISM para clasificar la calidad del vino

import os
import json
import pandas as pd

carpeta_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_proyecto = os.path.abspath(os.path.join(carpeta_actual, ".."))

print("=== Fase 2: Algoritmo PRISM para el Vino ===")

# 1. Cargamos los datos de entrenamiento preparados en la Fase 1
ruta_datos_train = os.path.join(carpeta_proyecto, "Fase_1_Preparacion", "datos_train.csv")
datos_entrenamiento = pd.read_csv(ruta_datos_train)
total_vinos_entrenamiento = len(datos_entrenamiento)

print("Total de vinos de entrenamiento cargados:", total_vinos_entrenamiento)

# Caracteristicas discretizadas que usara PRISM
columnas_caracteristicas = ['alcohol_discreto', 'acidez_discreto', 'sulfatos_discreto', 'ph_discreto']

# Mapeo al nombre limpio de cada variable
nombre_variable_limpio = {
    'alcohol_discreto': 'alcohol',
    'acidez_discreto': 'acidez_volatil',
    'sulfatos_discreto': 'sulfatos',
    'ph_discreto': 'ph'
}

categorias_calidad = ['ALTA', 'MEDIA', 'BAJA']
etiquetas_posibles = ['bajo', 'medio', 'alto']

# Probabilidad previa P(Calidad) de cada clase para calcular el Lift
probabilidad_previa = {}
for calidad in categorias_calidad:
    conteo_vinos_clase = (datos_entrenamiento['calidad'] == calidad).sum()
    probabilidad_previa[calidad] = conteo_vinos_clase / total_vinos_entrenamiento


# 2. Funcion que busca la mejor condicion (por ejemplo: alcohol_discreto == 'alto')
# basada en la medida de CONFIANZA vista en clase
def buscar_mejor_condicion(condiciones_candidatas, datos_actuales, clase_objetivo):
    mejor_condicion = None
    mayor_confianza = -1.0
    mayor_cobertura = -1
    
    for caracteristica, etiqueta in condiciones_candidatas:
        # Filtramos los vinos que cumplen con esta condicion
        vinos_cumplen_condicion = datos_actuales[datos_actuales[caracteristica] == etiqueta]
        total_vinos_condicion = len(vinos_cumplen_condicion)
        
        if total_vinos_condicion == 0:
            continue
            
        # Contamos cuantos de ellos pertenecen a la clase objetivo
        vinos_acertados = (vinos_cumplen_condicion['calidad'] == clase_objetivo).sum()
        confianza = vinos_acertados / total_vinos_condicion
        
        # Criterio PRISM: mayor confianza, y en caso de empate mayor cantidad de vinos cubiertos
        if confianza > mayor_confianza:
            mayor_confianza = confianza
            mayor_cobertura = vinos_acertados
            mejor_condicion = (caracteristica, etiqueta)
        elif confianza == mayor_confianza and vinos_acertados > mayor_cobertura:
            mayor_cobertura = vinos_acertados
            mejor_condicion = (caracteristica, etiqueta)
            
    return mejor_condicion, mayor_confianza, mayor_cobertura


# 3. Funcion que aprende una sola regla para una clase de vino especifica
def aprender_una_regla_para_vino(clase_objetivo, datos_disponibles, lista_caracteristicas):
    condiciones_regla = {}
    caracteristicas_disponibles = list(lista_caracteristicas)
    datos_filtrados = datos_disponibles.copy()
    
    while caracteristicas_disponibles:
        # Creamos las combinaciones posibles de caracteristica = etiqueta
        candidatas = []
        for caracteristica in caracteristicas_disponibles:
            for etiqueta in etiquetas_posibles:
                candidatas.append((caracteristica, etiqueta))
                
        condicion_elegida, confianza, cobertura = buscar_mejor_condicion(
            candidatas, datos_filtrados, clase_objetivo
        )
        
        if condicion_elegida is None or cobertura < 3:
            break
            
        caracteristica_seleccionada, etiqueta_seleccionada = condicion_elegida
        variable_real = nombre_variable_limpio[caracteristica_seleccionada]
        condiciones_regla[variable_real] = etiqueta_seleccionada
        caracteristicas_disponibles.remove(caracteristica_seleccionada)
        
        # Filtramos los datos para evaluar la siguiente condicion con el operador AND
        datos_filtrados = datos_filtrados[datos_filtrados[caracteristica_seleccionada] == etiqueta_seleccionada]
        
        # Si la regla es pura (sin ejemplos negativos) o supera 65% de confianza, se detiene
        vinos_otra_clase = (datos_filtrados['calidad'] != clase_objetivo).sum()
        if vinos_otra_clase == 0 or confianza >= 0.65 or len(condiciones_regla) >= 3:
            break
            
    return condiciones_regla


# 4. Procedimiento principal de Recubrimiento Secuencial para vinos
def inducir_reglas_prism(lista_clases, lista_caracteristicas, datos_completos):
    lista_reglas_descubiertas = []
    
    for clase_actual in lista_clases:
        print(f"Induciendo reglas para calidad = {clase_actual}...")
        datos_restantes = datos_completos.copy()
        total_inicial_clase = (datos_restantes['calidad'] == clase_actual).sum()
        
        while True:
            conteo_restante = (datos_restantes['calidad'] == clase_actual).sum()
            if conteo_restante <= max(5, int(0.08 * total_inicial_clase)):
                break
                
            condiciones_obtenidas = aprender_una_regla_para_vino(
                clase_actual, datos_restantes, lista_caracteristicas
            )
            
            if not condiciones_obtenidas:
                break
                
            # Evaluamos la regla obtenida
            mascara_filtro = pd.Series(True, index=datos_restantes.index)
            for variable_real, etiqueta in condiciones_obtenidas.items():
                columna_discreta = variable_real + '_discreto'
                if columna_discreta == 'acidez_volatil_discreto':
                    columna_discreta = 'acidez_discreto'
                mascara_filtro = mascara_filtro & (datos_restantes[columna_discreta] == etiqueta)
                
            cobertura_clase = (mascara_filtro & (datos_restantes['calidad'] == clase_actual)).sum()
            total_que_cumplen_antecedente = mascara_filtro.sum()
            
            if cobertura_clase < 3:
                indices_eliminar = datos_restantes[datos_restantes['calidad'] == clase_actual].index
                if len(indices_eliminar) > 0:
                    datos_restantes = datos_restantes.drop(indices_eliminar[0])
                continue
                
            # Calculamos las 4 metricas formales vistas en clase
            confianza_regla = cobertura_clase / total_que_cumplen_antecedente
            soporte_regla = cobertura_clase / total_vinos_entrenamiento
            lift_regla = confianza_regla / probabilidad_previa[clase_actual]
            
            regla_informacion = {
                'condiciones': condiciones_obtenidas,
                'consecuente': clase_actual,
                'confianza': round(float(confianza_regla), 4),
                'soporte': round(float(soporte_regla), 4),
                'cobertura': int(cobertura_clase),
                'lift': round(float(lift_regla), 2),
                'peso': round(float(confianza_regla), 4)
            }
            lista_reglas_descubiertas.append(regla_informacion)
            
            # Principio de recubrimiento secuencial:
            # Quitamos los vinos que esta regla ya cubrio para que la siguiente regla aprenda con los que faltan
            vinos_ya_cubiertos = mascara_filtro & (datos_restantes['calidad'] == clase_actual)
            indices_cubiertos = datos_restantes[vinos_ya_cubiertos].index
            datos_restantes = datos_restantes.drop(indices_cubiertos)
            
    return lista_reglas_descubiertas


# 5. Ejecucion
reglas_descubiertas = inducir_reglas_prism(categorias_calidad, columnas_caracteristicas, datos_entrenamiento)
print("\nTotal de reglas PRISM descubiertas:", len(reglas_descubiertas))

print("\nPrimeras reglas descubiertas:")
for numero_regla, regla in enumerate(reglas_descubiertas[:5], 1):
    condiciones_texto = " AND ".join([f"{var}={val}" for var, val in regla['condiciones'].items()])
    print(f"  Regla {numero_regla}: SI {condiciones_texto} ENTONCES calidad={regla['consecuente']} (confianza: {regla['confianza']*100:.1f}%, cobertura: {regla['cobertura']}, lift: {regla['lift']})")

# Guardamos las reglas en archivo JSON
ruta_guardar_reglas = os.path.join(carpeta_actual, "reglas_descubiertas.json")
with open(ruta_guardar_reglas, "w") as archivo_json:
    json.dump(reglas_descubiertas, archivo_json, indent=4)

print("\nReglas guardadas exitosamente en:")
print("  - reglas_descubiertas.json")
