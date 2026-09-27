# Proyecto Inteligencia Artificial
# Lanzador principal

import os
import sys
import subprocess

dir_raiz = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--interactivo':
        ruta_fase6 = os.path.join(dir_raiz, "Fase_6_Inferencia", "6_probar_nuevo_vino.py")
        subprocess.run([sys.executable, ruta_fase6], cwd=os.path.dirname(ruta_fase6))
    else:
        ruta_ejecutar = os.path.join(dir_raiz, "ejecutar_todo.py")
        subprocess.run([sys.executable, ruta_ejecutar], cwd=dir_raiz)
