# Proyecto Inteligencia Artificial
# Ejecucion secuencial de las Fases 1 a 5

import os
import sys
import subprocess

carpeta_proyecto = os.path.dirname(os.path.abspath(__file__))

lista_fases = [
    (os.path.join(carpeta_proyecto, "Fase_1_Preparacion", "1_preparacion_datos.py"), "Fase 1: Preparacion de datos del vino"),
    (os.path.join(carpeta_proyecto, "Fase_2_PRISM", "2_algoritmo_prism.py"), "Fase 2: Algoritmo PRISM"),
    (os.path.join(carpeta_proyecto, "Fase_3_Logica_Difusa", "3_logica_difusa.py"), "Fase 3: Sistema de Inferencia Difusa"),
    (os.path.join(carpeta_proyecto, "Fase_4_Algoritmo_Genetico", "4_algoritmo_genetico.py"), "Fase 4: Optimizacion con Algoritmo Genetico"),
    (os.path.join(carpeta_proyecto, "Fase_5_Evaluacion", "5_evaluacion_resultados.py"), "Fase 5: Evaluacion y comparativa de resultados"),
]

for ruta_script, nombre_fase in lista_fases:
    carpeta_fase = os.path.dirname(ruta_script)
    print(f"\n--- {nombre_fase} ---", flush=True)
    resultado = subprocess.run([sys.executable, ruta_script], cwd=carpeta_fase)
    if resultado.returncode != 0:
        print(f"Error al ejecutar {nombre_fase}")
        sys.exit(1)
