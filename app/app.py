import streamlit as st
import pandas as pd
import joblib

# Definir la nueva ruta del modelo
modelo_path = "/workspaces/final_project_rice/pickle/modelo_logistico.pkl"

# Cargar el modelo previamente guardado
modelo = joblib.load(modelo_path)

# Diccionario para asignar nombres manualmente
clase_arroz = {1: "Arroz Cammeo", 2: "Arroz Osmancik"}

st.title("Predicción de Clase de Arroz con Regresión Logística")

# Formulario de entrada con límites corregidos
area = st.number_input("Ingrese el valor de Área", min_value=7551, max_value=18913)
perimeter = st.number_input("Ingrese el valor de Perímetro", min_value=359.1, max_value=548.4459838867188)
major_axis_length = st.number_input("Ingrese el valor de Longitud del Eje Mayor", min_value=145.26446533203125, max_value=239.010498046875)
minor_axis_length = st.number_input("Ingrese el valor de Longitud del Eje Menor", min_value=59.532405853271484, max_value=107.54244995117188)
eccentricity = st.number_input("Ingrese el valor de Excentricidad", min_value=0.7772325873374939, max_value=0.9480069279670715)
convex_area = st.number_input("Ingrese el valor de Área Convexa", min_value=7723, max_value=19099)
extent = st.number_input("Ingrese el valor de Extensión", min_value=0.49741286039352417, max_value=0.8610495328903198)

# Botón de predicción
if st.button("Predecir"):
    entrada = pd.DataFrame([[area, perimeter, major_axis_length, minor_axis_length, eccentricity, convex_area, extent]],
                           columns=["area", "perimeter", "major_axis_length", "minor_axis_length", "eccentricity", "convex_area", "extent"])  

    # Obtener la predicción numérica del modelo
    prediccion_numerica = modelo.predict(entrada)[0]

    # Mostramos la predicción numérica para diagnóstico
    st.write(f"🔢 Predicción numérica antes de conversión: {prediccion_numerica}")

    # Convertimos la predicción en entero (si es necesario)
    prediccion_numerica = int(round(prediccion_numerica))  # Aseguramos que coincida con las claves del diccionario

    # Traducir la predicción al nombre del arroz
    nombre_arroz = clase_arroz.get(prediccion_numerica, "Desconocido")

    st.write(f"🔍 La clase predicha de arroz es: **{nombre_arroz}**")