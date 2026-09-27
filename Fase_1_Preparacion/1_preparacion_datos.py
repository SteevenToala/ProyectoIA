# Proyecto Inteligencia Artificial
# Fase 1: Preparacion y limpieza de datos para Wine Quality

import os
import json
import pandas as pd
import numpy as np

carpeta_actual = os.path.dirname(os.path.abspath(__file__))
carpeta_proyecto = os.path.abspath(os.path.join(carpeta_actual, ".."))

# 1. Cargamos el archivo original de vino tinto
ruta_archivo_csv = os.path.join(carpeta_proyecto, "Dataset_Wine_Qhality", "winequality-red.csv")
datos_originales = pd.read_csv(ruta_archivo_csv, sep=";")

# 2. Seleccionamos unicamente las 4 caracteristicas quimicas que vamos a usar
# y renombramos 'volatile acidity' a 'acidez_volatil' para que sea mas claro
datos_vino = pd.DataFrame()
datos_vino['alcohol'] = datos_originales['alcohol']
datos_vino['acidez_volatil'] = datos_originales['volatile acidity']
datos_vino['sulfatos'] = datos_originales['sulphates']
datos_vino['ph'] = datos_originales['pH']

# 3. Convertimos la calificacion numerica original (3 a 8) en 3 categorias de calidad
def asignar_categoria_calidad(calificacion_numerica):
    if calificacion_numerica <= 5:
        return 'BAJA'
    elif calificacion_numerica == 6:
        return 'MEDIA'
    else:
        return 'ALTA'

datos_vino['calidad'] = datos_originales['quality'].apply(asignar_categoria_calidad)

print("\nCantidad de vinos por cada categoria de calidad:")
print(datos_vino['calidad'].value_counts())

# 4. Calculamos los puntos de corte iniciales por terciles (33.3% y 66.7%)
# especificamente para cada una de las 4 variables quimicas
cortes_quimicos = {}

# Cortes para Alcohol
alcohol_minimo = float(datos_vino['alcohol'].min())
alcohol_corte_bajo = float(datos_vino['alcohol'].quantile(0.333))
alcohol_punto_medio = float(datos_vino['alcohol'].median())
alcohol_corte_alto = float(datos_vino['alcohol'].quantile(0.667))
alcohol_maximo = float(datos_vino['alcohol'].max())

cortes_quimicos['alcohol'] = {
    'minimo': round(alcohol_minimo, 4),
    'corte_bajo': round(alcohol_corte_bajo, 4),
    'punto_medio': round(alcohol_punto_medio, 4),
    'corte_alto': round(alcohol_corte_alto, 4),
    'maximo': round(alcohol_maximo, 4)
}

# Cortes para Acidez Volatil
acidez_minimo = float(datos_vino['acidez_volatil'].min())
acidez_corte_bajo = float(datos_vino['acidez_volatil'].quantile(0.333))
acidez_punto_medio = float(datos_vino['acidez_volatil'].median())
acidez_corte_alto = float(datos_vino['acidez_volatil'].quantile(0.667))
acidez_maximo = float(datos_vino['acidez_volatil'].max())

cortes_quimicos['acidez_volatil'] = {
    'minimo': round(acidez_minimo, 4),
    'corte_bajo': round(acidez_corte_bajo, 4),
    'punto_medio': round(acidez_punto_medio, 4),
    'corte_alto': round(acidez_corte_alto, 4),
    'maximo': round(acidez_maximo, 4)
}

# Cortes para Sulfatos
sulfatos_minimo = float(datos_vino['sulfatos'].min())
sulfatos_corte_bajo = float(datos_vino['sulfatos'].quantile(0.333))
sulfatos_punto_medio = float(datos_vino['sulfatos'].median())
sulfatos_corte_alto = float(datos_vino['sulfatos'].quantile(0.667))
sulfatos_maximo = float(datos_vino['sulfatos'].max())

cortes_quimicos['sulfatos'] = {
    'minimo': round(sulfatos_minimo, 4),
    'corte_bajo': round(sulfatos_corte_bajo, 4),
    'punto_medio': round(sulfatos_punto_medio, 4),
    'corte_alto': round(sulfatos_corte_alto, 4),
    'maximo': round(sulfatos_maximo, 4)
}

# Cortes para pH
ph_minimo = float(datos_vino['ph'].min())
ph_corte_bajo = float(datos_vino['ph'].quantile(0.333))
ph_punto_medio = float(datos_vino['ph'].median())
ph_corte_alto = float(datos_vino['ph'].quantile(0.667))
ph_maximo = float(datos_vino['ph'].max())

cortes_quimicos['ph'] = {
    'minimo': round(ph_minimo, 4),
    'corte_bajo': round(ph_corte_bajo, 4),
    'punto_medio': round(ph_punto_medio, 4),
    'corte_alto': round(ph_corte_alto, 4),
    'maximo': round(ph_maximo, 4)
}

# 5. Creamos las columnas discretizadas ('bajo', 'medio', 'alto') para el algoritmo PRISM
datos_vino['alcohol_discreto'] = pd.cut(
    datos_vino['alcohol'],
    bins=[-np.inf, alcohol_corte_bajo, alcohol_corte_alto, np.inf],
    labels=['bajo', 'medio', 'alto']
)

datos_vino['acidez_discreto'] = pd.cut(
    datos_vino['acidez_volatil'],
    bins=[-np.inf, acidez_corte_bajo, acidez_corte_alto, np.inf],
    labels=['bajo', 'medio', 'alto']
)

datos_vino['sulfatos_discreto'] = pd.cut(
    datos_vino['sulfatos'],
    bins=[-np.inf, sulfatos_corte_bajo, sulfatos_corte_alto, np.inf],
    labels=['bajo', 'medio', 'alto']
)

datos_vino['ph_discreto'] = pd.cut(
    datos_vino['ph'],
    bins=[-np.inf, ph_corte_bajo, ph_corte_alto, np.inf],
    labels=['bajo', 'medio', 'alto']
)

# 6. Separamos en 80% para entrenamiento y 20% para prueba
datos_entrenamiento = datos_vino.sample(frac=0.80, random_state=42).copy()
datos_prueba = datos_vino.drop(datos_entrenamiento.index).copy()

print(f"Muestras divididas: {len(datos_entrenamiento)} entrenamiento (80%) y {len(datos_prueba)} prueba (20%)")

# 7. Guardamos los archivos para que los lean las siguientes fases
ruta_guardar_preparados = os.path.join(carpeta_actual, "datos_preparados.csv")
ruta_guardar_train = os.path.join(carpeta_actual, "datos_train.csv")
ruta_guardar_test = os.path.join(carpeta_actual, "datos_test.csv")
ruta_guardar_cortes = os.path.join(carpeta_actual, "cortes_iniciales.json")

datos_vino.to_csv(ruta_guardar_preparados, index=False)
datos_entrenamiento.to_csv(ruta_guardar_train, index=False)
datos_prueba.to_csv(ruta_guardar_test, index=False)

with open(ruta_guardar_cortes, "w") as archivo_json:
    json.dump(cortes_quimicos, archivo_json, indent=4)
