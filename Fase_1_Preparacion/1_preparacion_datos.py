"""
=============================================================================
FASE 1: PREPARACIÓN Y LIMPIEZA DE DATOS
=============================================================================
En este script:
1. Leemos el archivo CSV original de Wine Quality.
2. Seleccionamos las 4 variables químicas más importantes:
   - alcohol
   - volatile acidity (acidez volátil)
   - sulphates (sulfatos)
   - pH
3. Convertimos la calidad numérica (3 a 8) en 3 categorías:
   - BAJA  (calidad <= 5)
   - MEDIA (calidad == 6)
   - ALTA  (calidad >= 7)
4. Calculamos los cortes iniciales por cuantiles (terciles) para discretizar
   las variables en: 'bajo', 'medio' y 'alto'.
5. Dividimos en Entrenamiento (80%) y Prueba (20%).
6. Guardamos los datos listos para las siguientes fases.
=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np

# Rutas automáticas para funcionar desde cualquier directorio
DIRECTORIO_ACTUAL = os.path.dirname(os.path.abspath(__file__))
DIRECTORIO_RAIZ = os.path.abspath(os.path.join(DIRECTORIO_ACTUAL, ".."))

print("\n" + "="*60)
print("     FASE 1: PREPARACIÓN Y LIMPIEZA DE DATOS")
print("="*60)

# 1. Cargar el dataset original
ruta_csv = os.path.join(DIRECTORIO_RAIZ, "Dataset_Wine_Qhality", "winequality-red.csv")
df = pd.read_csv(ruta_csv, sep=";")
print(f"Total de vinos cargados desde el dataset: {len(df)}")

# 2. Seleccionar las 4 variables químicas clave
variables = ['alcohol', 'volatile acidity', 'sulphates', 'pH']
df_reducido = df[variables].copy()

# 3. Agrupar la calidad en BAJA, MEDIA y ALTA
def clasificar_calidad(nota):
    if nota <= 5:
        return 'BAJA'
    elif nota == 6:
        return 'MEDIA'
    else:
        return 'ALTA'

df_reducido['calidad'] = df['quality'].apply(clasificar_calidad)

print("\nDistribución de calidad en el dataset:")
print(df_reducido['calidad'].value_counts())

# 4. Discretizar en 'bajo', 'medio' y 'alto' usando cuantiles (33.3% y 66.7%)
cortes_iniciales = {}

for col in variables:
    minimo = float(df_reducido[col].min())
    maximo = float(df_reducido[col].max())
    q1 = float(df_reducido[col].quantile(0.333))  # corte inferior
    q2 = float(df_reducido[col].median())          # punto medio
    q3 = float(df_reducido[col].quantile(0.667))  # corte superior
    
    cortes_iniciales[col] = {
        'min': round(minimo, 4),
        'm1': round(q1, 4),
        'm2': round(q2, 4),
        'm3': round(q3, 4),
        'max': round(maximo, 4)
    }
    
    # Creamos la versión discreta para PRISM
    limites = [-np.inf, q1, q3, np.inf]
    etiquetas = ['bajo', 'medio', 'alto']
    df_reducido[col + '_disc'] = pd.cut(df_reducido[col], bins=limites, labels=etiquetas)

print("\nCortes calculados para cada variable (m1, m2, m3):")
for col, valores in cortes_iniciales.items():
    print(f"  • {col:17s} -> Bajo: [{valores['min']} a {valores['m1']}] | Medio: [{valores['m1']} a {valores['m3']}] | Alto: [{valores['m3']} a {valores['max']}]")

# 5. Separación 80% Train y 20% Test
df_train = df_reducido.sample(frac=0.80, random_state=42).copy()
df_test = df_reducido.drop(df_train.index).copy()

print(f"\nDatos divididos con éxito:")
print(f"  • Entrenamiento (Train): {len(df_train)} vinos (80%)")
print(f"  • Prueba (Test):         {len(df_test)} vinos (20%)")

# 6. Guardar archivos dentro de la carpeta Fase_1_Preparacion
ruta_preparados = os.path.join(DIRECTORIO_ACTUAL, "datos_preparados.csv")
ruta_train = os.path.join(DIRECTORIO_ACTUAL, "datos_train.csv")
ruta_test = os.path.join(DIRECTORIO_ACTUAL, "datos_test.csv")
ruta_cortes = os.path.join(DIRECTORIO_ACTUAL, "cortes_iniciales.json")

df_reducido.to_csv(ruta_preparados, index=False)
df_train.to_csv(ruta_train, index=False)
df_test.to_csv(ruta_test, index=False)

with open(ruta_cortes, "w") as f:
    json.dump(cortes_iniciales, f, indent=4)

print("\nArchivos generados en Fase_1_Preparacion/:")
print("  -> datos_preparados.csv")
print("  -> datos_train.csv")
print("  -> datos_test.csv")
print("  -> cortes_iniciales.json")
print("--- FASE 1 COMPLETADA CON ÉXITO ---\n")
