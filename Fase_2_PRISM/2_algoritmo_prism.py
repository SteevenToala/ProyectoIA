"""
=============================================================================
FASE 2: ALGORITMO PRISM (J. CENDROWSKA, 1987)
=============================================================================
Implementación fiel a la teoría vista en clase:
• Tipo de técnica: Reglas de clasificación (predicen la clase).
• Paradigma: Algoritmo de recubrimiento secuencial (Separate and Conquer).
• Medidas de evaluación empleadas:
    - Confianza: P(Clase | Antecedente) = Ejemplos(Ant y Cons) / Ejemplos(Ant)
    - Soporte:   P(Ant y Cons) = Ejemplos(Ant y Cons) / Total Ejemplos
    - Cobertura: Número absoluto de ejemplos que cumplen la regla
    - Lift:      Confianza / P(Clase) [Lift > 1 indica regla predictiva útil]

Estructura de procedimientos según las diapositivas:
  1. Recubrimiento_secuencial(Clases, Atributos, Ejemplos)
  2. AprenderUnaRegla(Clase, Ejemplos, Atributos)
  3. MejorRestriccion(Restricciones, Regla, Ejemplos, Clase)
=============================================================================
"""

import os
import json
import pandas as pd

# Rutas automáticas
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(DIRECTORIO_ACTUAL, ".."))

print("\n" + "="*65)
print("     FASE 2: ALGORITMO PRISM (RECUBRIMIENTO SECUENCIAL)")
print("="*65)

# 1. Cargar datos de entrenamiento desde la Fase 1
ruta_train = os.path.join(DIRECTORIO_RAIZ, "Fase_1_Preparacion", "datos_train.csv")
df_train = pd.read_csv(ruta_train)
N_TOTAL_EJEMPLOS = len(df_train)

print(f"Total de ejemplos de entrenamiento cargados: {N_TOTAL_EJEMPLOS}")

atributos_discretos = ['alcohol_disc', 'volatile acidity_disc', 'sulphates_disc', 'pH_disc']
mapa_nombres = {
    'alcohol_disc': 'alcohol',
    'volatile acidity_disc': 'volatile acidity',
    'sulphates_disc': 'sulphates',
    'pH_disc': 'pH'
}
clases = ['ALTA', 'MEDIA', 'BAJA']
valores_posibles = ['bajo', 'medio', 'alto']

# Probabilidad previa P(Clase) para el cálculo del Lift
prob_previa_clase = {c: (df_train['calidad'] == c).sum() / N_TOTAL_EJEMPLOS for c in clases}


# =============================================================================
# PROCEDIMIENTO 3: MejorRestriccion(Restricciones, Regla, Ejemplos, Clase)
# =============================================================================
def mejor_restriccion(restricciones, condiciones_actuales, datos_actuales, clase):
    """
    Evalúa cada restricción candidata {A = v} en base a la medida de CONFIANZA.
    En caso de empate en confianza, escoge la restricción de MAYOR COBERTURA.
    """
    mejor_restr = None
    mejor_confianza = -1.0
    mejor_cobertura = -1
    
    for atr, val in restricciones:
        # Evaluamos el subconjunto que cumple el antecedente acumulado + la nueva restricción
        filtro = (datos_actuales[atr] == val)
        subconjunto = datos_actuales[filtro]
        total_ant = len(subconjunto)
        
        if total_ant == 0:
            continue
            
        # Cobertura: ejemplos que cumplen antecedente y consecuente
        aciertos_clase = (subconjunto['calidad'] == clase).sum()
        confianza = aciertos_clase / total_ant
        
        # Criterio PRISM: mayor Confianza; desempate por Cobertura
        if confianza > mejor_confianza:
            mejor_confianza = confianza
            mejor_cobertura = aciertos_clase
            mejor_restr = (atr, val)
        elif abs(confianza - mejor_confianza) < 1e-6 and aciertos_clase > mejor_cobertura:
            mejor_cobertura = aciertos_clase
            mejor_restr = (atr, val)
            
    return mejor_restr, mejor_confianza, mejor_cobertura


# =============================================================================
# PROCEDIMIENTO 2: AprenderUnaRegla(Clase, Ejemplos, Atributos)
# =============================================================================
def aprender_una_regla(clase, datos_ejemplos, atributos_disponibles):
    """
    Construye una regla comenzando con antecedente vacío:
        SI --- ENTONCES calidad = Clase
    Mientras la regla cubra ejemplos negativos y queden atributos,
    añade la mejor restricción basada en la Confianza.
    """
    condiciones_regla = {}
    atributos_restantes = list(atributos_disponibles)
    datos_filtrados = datos_ejemplos.copy()
    
    # MIENTRAS (regla cubre algún ejemplo negativo AND Atributos != vacío)
    while atributos_restantes:
        # Generar conjunto de todas las restricciones candidatas {A = v}
        restricciones_candidatas = []
        for atr in atributos_restantes:
            for val in valores_posibles:
                restricciones_candidatas.append((atr, val))
                
        # Seleccionar la mejor restricción con el procedimiento MejorRestriccion
        restr_elegida, conf, cob = mejor_restriccion(
            restricciones_candidatas, condiciones_regla, datos_filtrados, clase
        )
        
        if restr_elegida is None or cob < 3:
            break
            
        atr_sel, val_sel = restr_elegida
        var_real = mapa_nombres[atr_sel]
        condiciones_regla[var_real] = val_sel
        atributos_restantes.remove(atr_sel)
        
        # Filtrar los ejemplos para el siguiente término conjuntivo (AND)
        datos_filtrados = datos_filtrados[datos_filtrados[atr_sel] == val_sel]
        
        # Condición de pureza: si no cubre ejemplos negativos (Confianza = 1.0) o alta pureza
        ejemplos_negativos = (datos_filtrados['calidad'] != clase).sum()
        if ejemplos_negativos == 0 or conf >= 0.65 or len(condiciones_regla) >= 3:
            break
            
    return condiciones_regla


# =============================================================================
# PROCEDIMIENTO 1: Recubrimiento_secuencial(Clases, Atributos, Ejemplos)
# =============================================================================
def recubrimiento_secuencial(lista_clases, atributos, df_ejemplos):
    """
    Procedimiento principal de PRISM:
    Para cada clase C:
        E = Ejemplos
        Mientras (E contenga ejemplos de la clase C):
            Regla = AprenderUnaRegla(C, E, Atributos)
            Reglas = Reglas + {Regla}
            E = E - {ejemplos cubiertos por Regla de clase C}
    """
    reglas_aprendidas = []
    
    for C in lista_clases:
        print(f"\n[PRISM] Induciendo reglas para la clase: CALIDAD = {C}...")
        E = df_ejemplos.copy()
        total_clase_inicial = (E['calidad'] == C).sum()
        
        # Mientras E contenga ejemplos de la clase C
        while True:
            ejemplos_clase_restantes = (E['calidad'] == C).sum()
            if ejemplos_clase_restantes <= max(5, int(0.08 * total_clase_inicial)):
                break
                
            condiciones = aprender_una_regla(C, E, atributos)
            if not condiciones:
                break
                
            # Evaluar la regla aprendida sobre los datos actuales
            mascara = pd.Series(True, index=E.index)
            for var_real, val in condiciones.items():
                col_disc = var_real + '_disc'
                mascara = mascara & (E[col_disc] == val)
                
            cobertura_clase = (mascara & (E['calidad'] == C)).sum()
            cumplen_antecedente = mascara.sum()
            
            if cobertura_clase < 3:
                # Retirar una instancia problemática para evitar bucle infinito
                idx_quitar = E[E['calidad'] == C].index
                if len(idx_quitar) > 0:
                    E = E.drop(idx_quitar[0])
                continue
                
            # =================================================================
            # CÁLCULO DE LAS MÉTRICAS FORMALES (Diapositivas)
            # =================================================================
            # Confianza: P(Clase | Antecedente) = aciertos / cumplen antecedente
            confianza = cobertura_clase / cumplen_antecedente
            
            # Soporte: P(Antecedente y Consecuente) = aciertos / total ejemplos
            soporte = cobertura_clase / N_TOTAL_EJEMPLOS
            
            # Cobertura: número absoluto de ejemplos que cumplen la regla
            cobertura = int(cobertura_clase)
            
            # Lift: Confianza / P(Clase) -> Lift > 1 indica correlación positiva útil
            lift = confianza / prob_previa_clase[C]
            
            regla_dict = {
                'condiciones': condiciones,
                'consecuente': C,
                'confianza': round(float(confianza), 4),
                'soporte': round(float(soporte), 4),
                'cobertura': cobertura,
                'lift': round(float(lift), 2),
                'peso': round(float(confianza), 4)  # Usado luego como peso en Lógica Difusa
            }
            reglas_aprendidas.append(regla_dict)
            
            # E = E - {ejemplos cubiertos por Regla de clase C} (Principio de recubrimiento)
            E = E[~(mascara & (E['calidad'] == C))]
            
    return reglas_aprendidas


# =============================================================================
# EJECUCIÓN DEL ALGORITMO
# =============================================================================
reglas_descubiertas = recubrimiento_secuencial(clases, atributos_discretos, df_train)

# 3. Mostrar las reglas con la nomenclatura formal vista en clase
print("\n" + "="*85)
print(f"REGLAS DE CLASIFICACIÓN INDUCIDAS POR PRISM (Total: {len(reglas_descubiertas)})")
print("="*85)

for i, r in enumerate(reglas_descubiertas[:10], 1):
    partes = [f"{var} es {val.upper()}" for var, val in r['condiciones'].items()]
    antecedente = " AND ".join(partes)
    consecuente = f"calidad = {r['consecuente']}"
    print(f"Regla {i:02d}: SI {antecedente} ENTONCES {consecuente}")
    print(f"         [Confianza: {r['confianza']*100:.1f}% | Soporte: {r['soporte']*100:.1f}% | Cobertura: {r['cobertura']} vinos | Lift: {r['lift']:.2f}]")

if len(reglas_descubiertas) > 10:
    print(f"\n... y {len(reglas_descubiertas) - 10} reglas adicionales guardadas en el archivo.")

# 4. Guardar reglas en archivo JSON
ruta_reglas = os.path.join(DIRECTORIO_ACTUAL, "reglas_descubiertas.json")
with open(ruta_reglas, "w") as f:
    json.dump(reglas_descubiertas, f, indent=4)

print("\nArchivo generado:")
print(f"  -> {ruta_reglas}")
print("--- FASE 2 COMPLETADA CON ÉXITO ---\n")
