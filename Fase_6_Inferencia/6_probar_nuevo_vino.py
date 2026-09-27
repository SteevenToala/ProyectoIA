# Proyecto Inteligencia Artificial
# Fase 6: Inferencia y prueba de un nuevo vino con el modelo optimizado

import os
import sys
import json
import importlib

carpeta_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_proyecto = os.path.abspath(os.path.join(carpeta_actual, ".."))

# Importamos las funciones difusas de la Fase 3
sys.path.append(os.path.join(carpeta_proyecto, "Fase_3_Logica_Difusa"))
modulo_difusa = importlib.import_module("3_logica_difusa")
fuzzificar_vino = modulo_difusa.fuzzificar_vino
inferencia_mamdani_vino = modulo_difusa.inferencia_mamdani_vino
agregar_impactos_por_calidad = modulo_difusa.agregar_impactos_por_calidad
defuzzificar_calidad = modulo_difusa.defuzzificar_calidad

# Cargamos el modelo optimizado y las reglas PRISM
ruta_modelo_opt = os.path.join(carpeta_proyecto, "Fase_4_Algoritmo_Genetico", "modelo_optimizado.json")
ruta_reglas_prism = os.path.join(carpeta_proyecto, "Fase_2_PRISM", "reglas_descubiertas.json")

with open(ruta_modelo_opt) as archivo_modelo:
    modelo_optimizado = json.load(archivo_modelo)
with open(ruta_reglas_prism) as archivo_reglas:
    lista_reglas = json.load(archivo_reglas)

cortes_optimizados = modelo_optimizado['cortes_optimizados']
pesos_optimizados = modelo_optimizado['pesos_optimizados']


def predecir_calidad_de_un_vino(alcohol, acidez_volatil, sulfatos, ph, nombre_del_vino="Vino de Prueba"):
    fila_vino = {
        'alcohol': float(alcohol),
        'acidez_volatil': float(acidez_volatil),
        'sulfatos': float(sulfatos),
        'ph': float(ph)
    }
    
    # 1. Fuzzificacion
    grados_vino = fuzzificar_vino(fila_vino, cortes_optimizados)
    
    # 2. Inferencia Mamdani
    disparos = inferencia_mamdani_vino(grados_vino, lista_reglas, pesos_optimizados)
    
    # 3. Agregacion
    puntajes = agregar_impactos_por_calidad(disparos)
    
    # 4. Defuzzificacion
    calidad_predicha, porcentaje_certidumbre, calificacion_continua = defuzzificar_calidad(puntajes)
    
    print(f"\n{nombre_del_vino}:")
    print(f"  Quimica: Alcohol={alcohol}%, Acidez Volatil={acidez_volatil}, Sulfatos={sulfatos}, pH={ph}")
    print(f"  Prediccion: Calidad {calidad_predicha} (Certidumbre: {porcentaje_certidumbre:.1f}%, Nota: {calificacion_continua:.2f}/10)")
    
    return calidad_predicha


# Alias corto para pruebas
predecir_vino = predecir_calidad_de_un_vino

if __name__ == "__main__":
    # Probamos 3 vinos representativos
    predecir_calidad_de_un_vino(12.5, 0.35, 0.85, 3.28, "Vino de Alta Gama (Esperado: ALTA)")
    predecir_calidad_de_un_vino(10.2, 0.52, 0.62, 3.35, "Vino Estandar de Mesa (Esperado: MEDIA)")
    predecir_calidad_de_un_vino(9.1, 0.88, 0.42, 3.52, "Vino Defectuoso (Esperado: BAJA)")
    
    # Prueba manual interactiva
    print("Deseas ingresar los datos de un vino manualmente? (s/n): ", end="")
    try:
        respuesta = input().strip().lower()
        if respuesta == 's':
            alcohol_ingresado = float(input("Alcohol (% ej. 11.5): "))
            acidez_ingresada = float(input("Acidez volatil (g/dm3 ej. 0.45): "))
            sulfatos_ingresados = float(input("Sulfatos (g/dm3 ej. 0.70): "))
            ph_ingresado = float(input("pH (ej. 3.30): "))
            predecir_calidad_de_un_vino(alcohol_ingresado, acidez_ingresada, sulfatos_ingresados, ph_ingresado, "Vino Ingresado por Teclado")
    except Exception:
        pass
