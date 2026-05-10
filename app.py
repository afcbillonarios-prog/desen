import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Predicción de Crédito", page_icon="💰", layout="centered")

@st.cache_resource
def load_model():
    try:
        model = joblib.load('modelo_regresion.pkl')
        columns = joblib.load('columnas_modelo.pkl')
        return model, columns
    except FileNotFoundError:
        return None, None

def predecir_valor_credito(plazo, periodo_gracia, cuotas, cuotas_capital, tasa_indexacion, model, model_columns):
    # Crear dataframe con entrada
    input_data = pd.DataFrame({
        'plazo': [plazo],
        'periodo_gracia': [periodo_gracia],
        'cuotas': [cuotas],
        'cuotas_capital': [cuotas_capital]
    })
    
    # Añadir dummies de la tasa de indexación
    # Inicializar todas las columnas categóricas esperadas en 0
    for col in model_columns:
        if col not in input_data.columns:
            input_data[col] = 0
            
    # Activar la dummy correspondiente a la tasa seleccionada si existe en las columnas del modelo
    tasa_col = f'tasa_indexacion_{tasa_indexacion}'
    if tasa_col in input_data.columns:
        input_data[tasa_col] = 1
        
    # Reordenar las columnas para coincidir con el modelo
    input_data = input_data[model_columns]
    
    # Realizar predicción
    prediccion = model.predict(input_data)[0]
    return max(0, prediccion) # El valor de crédito no puede ser negativo

# Interfaz
st.title("🚜 Predicción de Crédito Agropecuario")
st.write("Esta aplicación utiliza un modelo de Regresión Lineal entrenado bajo la metodología CRISP-ML para predecir el **Valor de Crédito** estimado de fomento agropecuario.")

model, model_columns = load_model()

if model is None:
    st.warning("⚠️ El modelo no ha sido encontrado. Por favor ejecuta el cuaderno `3_Model.ipynb` primero para generar los archivos `.pkl`.")
else:
    st.header("Ingresa las características del crédito")
    
    col1, col2 = st.columns(2)
    with col1:
        plazo = st.number_input("Plazo (meses)", min_value=1, max_value=360, value=12)
        cuotas = st.number_input("Total de Cuotas", min_value=1, max_value=360, value=12)
    with col2:
        periodo_gracia = st.number_input("Periodo de Gracia (meses)", min_value=0, max_value=120, value=0)
        cuotas_capital = st.number_input("Cuotas de Capital", min_value=1, max_value=360, value=12)
        
    tasa_indexacion = st.selectbox("Tasa de Indexación", ["IBR", "DTF", "IPC", "Fija"])
    
    if st.button("Predecir Valor de Crédito", type="primary"):
        resultado = predecir_valor_credito(plazo, periodo_gracia, cuotas, cuotas_capital, tasa_indexacion, model, model_columns)
        st.success(f"El valor de crédito estimado es: **${resultado:,.2f}**")

st.markdown("---")
st.markdown("*Proyecto desarrollado siguiendo el ciclo de vida de Machine Learning.*")
