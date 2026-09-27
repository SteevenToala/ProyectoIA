"""
=============================================================================
FASE 6: PREDICCIÓN DE UN NUEVO VINO (MOTOR DIFUSO MAMDANI)
=============================================================================
Aplica el ciclo completo Mamdani en 4 pasos para evaluar cualquier vino:
  1. Fuzzificación:     Entradas químicas -> Grados lingüísticos (bajo/medio/alto)
  2. Inferencia:        Operador MIN en reglas PRISM ponderadas con peso w
  3. Agregación:        Combinación de impactos en BAJA, MEDIA y ALTA
  4. Defuzzificación:   Máxima Pertenencia + Centroide de Calidad (CoG)
=============================================================================
"""

import os
import sys
import json
import importlib

# Rutas automáticas
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(DIRECTORIO_ACTUAL, ".."))

# Importamos las 4 funciones del motor Mamdani de la Fase 3
sys.path.append(os.path.join(DIRECTORIO_RAIZ, "Fase_3_Logica_Difusa"))
mod_difusa = importlib.import_module("3_logica_difusa")
fuzzificacion = mod_difusa.fuzzificacion
inferencia_mamdani = mod_difusa.inferencia_mamdani
agregacion_difusa = mod_difusa.agregacion_difusa
defuzzificacion = mod_difusa.defuzzificacion

print("\n" + "="*65)
print("     FASE 6: EVALUACIÓN DE UN NUEVO VINO (SISTEMA MAMDANI)")
print("="*65)

# Cargar el modelo optimizado por el Algoritmo Genético
ruta_modelo = os.path.join(DIRECTORIO_RAIZ, "Fase_4_Algoritmo_Genetico", "modelo_optimizado.json")
ruta_reglas = os.path.join(DIRECTORIO_RAIZ, "Fase_2_PRISM", "reglas_descubiertas.json")

with open(ruta_modelo) as f:
    modelo = json.load(f)
with open(ruta_reglas) as f:
    reglas = json.load(f)

cortes = modelo['cortes_optimizados']
pesos = modelo['pesos_optimizados']

def predecir_vino(alcohol, acidez_volatil, sulfatos, ph, nombre="Vino de Prueba"):
    print("\n" + "-"*65)
    print(f"ANÁLISIS MAMDANI: {nombre.upper()}")
    print("-"*65)
    
    medidas = {
        'alcohol': float(alcohol),
        'volatile acidity': float(acidez_volatil),
        'sulphates': float(sulfatos),
        'pH': float(ph)
    }
    
    print("0. MEDIDAS QUIMICAS DE ENTRADA (CRISP INPUTS):")
    print(f"   * Alcohol:          {medidas['alcohol']:.2f} % vol.")
    print(f"   * Acidez Volatil:   {medidas['volatile acidity']:.3f} g/dm3")
    print(f"   * Sulfatos:         {medidas['sulphates']:.3f} g/dm3")
    print(f"   * pH:               {medidas['pH']:.2f}")
    
    # -------------------------------------------------------------------------
    # PASO 1: FUZZIFICACIÓN
    # -------------------------------------------------------------------------
    grados = fuzzificacion(medidas, cortes)
    print("\n1. FUZZIFICACION (Calculo de pertenencias linguisticas):")
    for var, g in grados.items():
        print(f"   * {var:17s} -> Bajo={g['bajo']:.2f} | Medio={g['medio']:.2f} | Alto={g['alto']:.2f}")
        
    # -------------------------------------------------------------------------
    # PASO 2: INFERENCIA (MAMDANI MIN)
    # -------------------------------------------------------------------------
    disparos = inferencia_mamdani(grados, reglas, pesos)
    disparos.sort(key=lambda d: d['impacto'], reverse=True)
    
    print("\n2. INFERENCIA MAMDANI (Reglas activadas con operador MIN):")
    for d in disparos[:4]:
        cond_str = " AND ".join([f"{k} es {v.upper()}" for k, v in d['condiciones'].items()])
        print(f"   * SI {cond_str} ENTONCES calidad = {d['consecuente']} (Fuerza={d['fuerza_disparo']:.2f}, Peso={d['peso']:.2f} -> Impacto={d['impacto']:.3f})")
        
    # -------------------------------------------------------------------------
    # PASO 3: AGREGACIÓN
    # -------------------------------------------------------------------------
    puntajes = agregacion_difusa(disparos)
    total_pts = sum(puntajes.values())
    print("\n3. AGREGACION (Suma difusa de impactos por categoria):")
    for cat in ['BAJA', 'MEDIA', 'ALTA']:
        pct = (puntajes[cat] / total_pts * 100) if total_pts > 0 else 33.3
        print(f"   * Salida acumulada {cat:5s}: {puntajes[cat]:.3f} puntos ({pct:.1f}% de certidumbre)")
        
    # -------------------------------------------------------------------------
    # PASO 4: DEFUZZIFICACIÓN
    # -------------------------------------------------------------------------
    calidad_ganadora, certidumbre, centroide = defuzzificacion(puntajes)
    print("\n4. DEFUZZIFICACION (Salida Difusa -> Decision Crisp):")
    print(f"   * Metodo de Maxima Pertenencia: [{calidad_ganadora}]")
    print(f"   * Certidumbre del veredicto:    {certidumbre:.1f} %")
    print(f"   * Metodo del Centroide (CoG):   {centroide:.2f} / 10 puntos enologicos")
    
    print(f"\n=> RESULTADO FINAL: CALIDAD DEL VINO = [{calidad_ganadora}]")
    print("-" * 65)
    return calidad_ganadora


# =============================================================================
# PRUEBAS DEMOSTRATIVAS Y MODO INTERACTIVO
# =============================================================================
if __name__ == "__main__":
    print("\n--- CASOS DE PRUEBA PREDEFINIDOS ---")
    
    # Caso 1: Vino de alta calidad
    predecir_vino(alcohol=12.5, acidez_volatil=0.35, sulfatos=0.85, ph=3.28, nombre="Vino Reserva A")
    
    # Caso 2: Vino de calidad media
    predecir_vino(alcohol=10.2, acidez_volatil=0.52, sulfatos=0.62, ph=3.35, nombre="Vino de Mesa B")
    
    # Caso 3: Vino de baja calidad
    predecir_vino(alcohol=9.1, acidez_volatil=0.88, sulfatos=0.42, ph=3.52, nombre="Vino Defectuoso C")
    
    print("\n¿Quieres ingresar los datos de un vino tú mismo? (s/n): ", end="")
    try:
        resp = input().strip().lower()
        if resp == 's':
            alc = float(input("Alcohol (% ej. 11.8): "))
            acid = float(input("Acidez volátil (g/dm³ ej. 0.42): "))
            sulf = float(input("Sulfatos (g/dm³ ej. 0.70): "))
            ph_val = float(input("pH (ej. 3.30): "))
            predecir_vino(alcohol=alc, acidez_volatil=acid, sulfatos=sulf, ph=ph_val, nombre="Tu Vino Personalizado")
    except Exception as e:
        print(f"Nota: No se ingresaron datos manuales o hubo error ({e}).")
        
    print("\n--- FASE 6 COMPLETADA CON ÉXITO ---\n")
