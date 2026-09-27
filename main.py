"""
Punto de entrada simple para el proyecto.
Ejecuta el pipeline completo de las fases o abre la prueba de vinos.
"""
import os
import sys
import subprocess

DIRECTORIO_RAIZ = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--interactivo':
        ruta_fase6 = os.path.join(DIRECTORIO_RAIZ, "Fase_6_Inferencia", "6_probar_nuevo_vino.py")
        subprocess.run([sys.executable, ruta_fase6], cwd=os.path.dirname(ruta_fase6))
    else:
        ruta_ejecutar = os.path.join(DIRECTORIO_RAIZ, "ejecutar_todo.py")
        subprocess.run([sys.executable, ruta_ejecutar], cwd=DIRECTORIO_RAIZ)
