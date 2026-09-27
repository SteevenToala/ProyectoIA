"""
=============================================================================
EJECUTOR GENERAL: CORRE TODAS LAS FASES EN SUS RESPECTIVAS CARPETAS
=============================================================================
Puedes ejecutar cada script individualmente:
  python Fase_1_Preparacion/1_preparacion_datos.py
  python Fase_2_PRISM/2_algoritmo_prism.py
  python Fase_3_Logica_Difusa/3_logica_difusa.py
  python Fase_4_Algoritmo_Genetico/4_algoritmo_genetico.py
  python Fase_5_Evaluacion/5_evaluacion_resultados.py
  python Fase_6_Inferencia/6_probar_nuevo_vino.py

O simplemente correr este archivo para ejecutar todo de principio a fin:
  python ejecutar_todo.py
=============================================================================
"""

import os
import sys
import subprocess

DIRECTORIO_RAIZ = os.path.dirname(os.path.abspath(__file__))

fases = [
    (os.path.join(DIRECTORIO_RAIZ, "Fase_1_Preparacion", "1_preparacion_datos.py"), "FASE 1: Preparación y limpieza de datos"),
    (os.path.join(DIRECTORIO_RAIZ, "Fase_2_PRISM", "2_algoritmo_prism.py"), "FASE 2: Inducción de reglas con PRISM"),
    (os.path.join(DIRECTORIO_RAIZ, "Fase_3_Logica_Difusa", "3_logica_difusa.py"), "FASE 3: Sistema de lógica difusa"),
    (os.path.join(DIRECTORIO_RAIZ, "Fase_4_Algoritmo_Genetico", "4_algoritmo_genetico.py"), "FASE 4: Algoritmo genético de optimización"),
    (os.path.join(DIRECTORIO_RAIZ, "Fase_5_Evaluacion", "5_evaluacion_resultados.py"), "FASE 5: Evaluación y comparación de resultados"),
]

print("\n" + "#"*70)
print("#   EJECUCIÓN DEL PIPELINE COMPLETO WINE QUALITY (FASES 1 A 5)")
print("#"*70)

for ruta_script, descripcion in fases:
    carpeta = os.path.dirname(ruta_script)
    archivo = os.path.basename(ruta_script)
    print(f"\n>>> Ejecutando {descripcion}...")
    resultado = subprocess.run([sys.executable, ruta_script], cwd=carpeta)
    if resultado.returncode != 0:
        print(f"Error al ejecutar {archivo}. Abortando.")
        sys.exit(1)

print("\n" + "#"*70)
print("#   ¡TODAS LAS FASES SE EJECUTARON CON ÉXITO!")
print("#   Para probar vinos nuevos ejecuta:")
print("#   python Fase_6_Inferencia/6_probar_nuevo_vino.py")
print("#"*70 + "\n")
