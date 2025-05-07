import streamlit as st
import pandas as pd
import joblib

# Definir la nueva ruta del modelo
modelo_path = "/workspaces/final_project_rice/pickle/modelo_logistico.pkl"

# MARKUP: Descripción del Proyecto
# =============================================================================
st.markdown("""
## Proyecto Clasificador de los tipos de arroz Cammeo y Osmancik

### Introducción

El arroz es uno de los cultivos más importantes a nivel global, siendo la base alimentaria de más de la mitad de la población mundial. Entre las numerosas variedades existentes (más de 10,000), el **Osmancik** (originario de Turquía) y el **Cammeo** (producido en Italia) destacan por sus características únicas en textura, composición nutricional y usos culinarios. Sin embargo, la creciente demanda y los estándares de calidad en la industria alimentaria han generado la necesidad de desarrollar métodos eficientes para clasificar, autenticar y optimizar estos tipos de arroz, evitando adulteraciones y garantizando su correcta comercialización.

En este contexto, el análisis de datos y el machine learning emergen como herramientas clave para diferenciar automáticamente estas variedades a partir de sus atributos morfológicos (como longitud, área y perímetro del grano) y químicos (contenido de proteínas, humedad y cenizas). Estudios previos demuestran que estas variables presentan patrones distinguibles, lo que abre la puerta a la aplicación de algoritmos de clasificación supervisada.

Este proyecto busca validar si es posible identificar con alta precisión estas variedades mediante modelos predictivos, lo que tendría aplicaciones en:

* Control de calidad en molinos y distribuidores de arroz.
* Detección de fraudes en productos etiquetados como "puros".
* Optimización agrícola, correlacionando características del grano con condiciones de cultivo.
""", unsafe_allow_html=True)

st.markdown("""
## El dataset se compone de lo siguiente:
```plaintext
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3810 entries, 0 to 3809
Data columns (total 8 columns):
 #   Column             Non-Null Count  Dtype   
---  ------             --------------  -----   
 0   area               3810 non-null   int64   
 1   perimeter          3810 non-null   float64  
 2   major_axis_length  3810 non-null   float64  
 3   minor_axis_length  3810 non-null   float64  
 4   eccentricity       3810 non-null   float64  
 5   convex_area        3810 non-null   int64   
 6   extent             3810 non-null   float64  
 7   class              3810 non-null   category
dtypes: category(1), float64(5), int64(2)
memory usage: 212.3 KB """,unsafe_allow_html=True)

st.markdown("""
## El dataset muestra la siguiente relación entre las variables, a través de un gráfico de heatmap:
```plaintext
""",unsafe_allow_html=True)
import streamlit as st
st.image('/workspaces/final_project_rice/pictures/HEATMAP.png')

st.markdown("""
## A continuación se muestra la siguiente relación entre las variables, a través de un gráfico de pairplot:
```plaintext
""",unsafe_allow_html=True)
import streamlit as st
st.image('/workspaces/final_project_rice/pictures/PAIRPLOT.png')

st.markdown("""
## Data Science:
# Se prueban 3 modelos con los siguientes resultados:
"Random Forest" muestra los siguientes resultados: 
            Mejores parámetros: 
            {'rforest__bootstrap': True, 
            'rforest__max_depth': 5, 
            'rforest__min_samples_leaf': 4, 'rforest__min_samples_split': 2, 
            'rforest__n_estimators': 50}
             Random Forest Accuracy: 0.93
             F1 Score: 0.93

"Logistic Regression" muestra los siguientes resultados: 
            Logistic Regression Accuracy: 0.94 
            F1 Score: 0.94

"Gradient Boosting" muestra los siguientes resultados: 
            mejores parametros: {'gradient_boosting__learning_rate': 0.01, 
            'gradient_boosting__max_depth': 5, 'gradient_boosting__n_estimators': 150} 
            Gradient Boosting Accuracy: 0.93 
            F1 Score: 0.93
Por lo que se decide utilizar Logistic Regression para este experimento.
```plaintext
""",unsafe_allow_html=True)

st.markdown("""
## A continuación se muestra el Confusion Matrix y ROC Curve del modelo de Logistic Regression:
```plaintext
""",unsafe_allow_html=True)

import streamlit as st
st.image('/workspaces/final_project_rice/pictures/confusion.png')

import streamlit as st
st.image('/workspaces/final_project_rice/pictures/ROCCURVE.png')

# Cargar el modelo previamente guardado
modelo = joblib.load(modelo_path)

# Diccionario para asignar nombres manualmente
clase_arroz = {0: "Arroz Cammeo", 1: "Arroz Osmancik"}

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
    # =============================================================================


import streamlit as st
import pandas as pd
import joblib




