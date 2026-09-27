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
    print(f"\n--- Evaluando: {nombre_del_vino} ---")
    
    fila_vino = {
        'alcohol': float(alcohol),
        'acidez_volatil': float(acidez_volatil),
        'sulfatos': float(sulfatos),
        'ph': float(ph)
    }
    
    print("Valores quimicos de entrada:")
    print(f"  Alcohol:        {fila_vino['alcohol']:.2f}% vol.")
    print(f"  Acidez Volatil: {fila_vino['acidez_volatil']:.3f} g/dm3")
    print(f"  Sulfatos:       {fila_vino['sulfatos']:.3f} g/dm3")
    print(f"  pH:             {fila_vino['ph']:.2f}")
    
    # Paso 1: Fuzzificacion
    grados_vino = fuzzificar_vino(fila_vino, cortes_optimizados)
    print("\n[Paso 1] Fuzzificacion:")
    for caracteristica, grados in grados_vino.items():
        print(f"  {caracteristica:15s} -> Bajo={grados['bajo']:.2f}, Medio={grados['medio']:.2f}, Alto={grados['alto']:.2f}")
        
    # Paso 2: Inferencia Mamdani
    disparos = inferencia_mamdani_vino(grados_vino, lista_reglas, pesos_optimizados)
    disparos.sort(key=lambda d: d['impacto'], reverse=True)
    
    print("\n[Paso 2] Inferencia (Reglas que mas aportaron):")
    for disparo in disparos[:3]:
        partes_texto = []
        for variable, valor in disparo['condiciones'].items():
            partes_texto.append(variable + " = " + valor)
        texto_si = " Y ".join(partes_texto)
        
        calidad = disparo['calidad_consecuente']
        fuerza = round(disparo['fuerza_disparo'], 2)
        impacto = round(disparo['impacto'], 3)
        print(f"  SI {texto_si} -> Calidad = {calidad} (Fuerza = {fuerza}, Impacto = {impacto})")
        
    # Paso 3: Agregacion difusa
    puntajes = agregar_impactos_por_calidad(disparos)
    suma_total_puntajes = sum(puntajes.values())
    
    print("\n[Paso 3] Agregacion de puntajes:")
    for calidad in ['BAJA', 'MEDIA', 'ALTA']:
        porcentaje_relativo = (puntajes[calidad] / suma_total_puntajes * 100.0) if suma_total_puntajes > 0 else 33.3
        print(f"  Puntaje {calidad:5s}: {puntajes[calidad]:.3f} ({porcentaje_relativo:.1f}%)")
        
    # Paso 4: Defuzzificacion
    calidad_predicha, porcentaje_certidumbre, calificacion_continua = defuzzificar_calidad(puntajes)
    print("\n[Paso 4] Defuzzificacion:")
    print(f"  Calidad Predicha por Maxima Pertenencia: [{calidad_predicha}]")
    print(f"  Porcentaje de Certidumbre:              {porcentaje_certidumbre:.1f}%")
    print(f"  Calificacion continua (Centroide CoG):  {calificacion_continua:.2f} / 10")
    print(f"=> Resultado Final: {calidad_predicha}\n")
    
    return calidad_predicha


# Alias corto para pruebas
predecir_vino = predecir_calidad_de_un_vino

if __name__ == "__main__":
    print("=== Fase 6: Inferencia de Calidad en Nuevos Vinos ===")
    
    # 3 Casos de prueba tipicos
    predecir_calidad_de_un_vino(alcohol=12.5, acidez_volatil=0.35, sulfatos=0.85, ph=3.28, nombre_del_vino="Vino Reserva A (Esperado: ALTA)")
    predecir_calidad_de_un_vino(alcohol=10.2, acidez_volatil=0.52, sulfatos=0.62, ph=3.35, nombre_del_vino="Vino de Mesa B (Esperado: MEDIA)")
    predecir_calidad_de_un_vino(alcohol=9.1, acidez_volatil=0.88, sulfatos=0.42, ph=3.52, nombre_del_vino="Vino Defectuoso C (Esperado: BAJA)")
    
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
