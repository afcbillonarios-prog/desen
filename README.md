# 🚜 Predicción de Crédito de Fomento Agropecuario

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B.svg?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-F7931E.svg?logo=scikit-learn&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-Landing_Page-E34F26.svg?logo=html5&logoColor=white)
![CRISP-ML](https://img.shields.io/badge/Metodolog%C3%ADa-CRISP--ML-brightgreen.svg)

Este proyecto implementa el ciclo de vida completo de un modelo de Machine Learning utilizando la metodología **CRISP-ML(Q)**. El objetivo es predecir el **Valor de Crédito** para el fomento agropecuario basándose en variables financieras de la solicitud.

## 🚀 Ciclo de Vida del Modelo (CRISP-ML)

El proyecto está dividido en cuadernos que siguen las fases de esta metodología:

1. **Fase 1: Identificación del Problema** -> Descrito en el HTML y el primer notebook.
2. **Fase 2: Recolección de Datos** -> `1_ETL.ipynb`
3. **Fase 3: Preparación de Datos (ETL)** -> `1_ETL.ipynb`
4. **Fase 4: Análisis y Feature Engineering (EDA)** -> `2_EDA.ipynb`
5. **Fase 5: Ingeniería de Modelos y Evaluación** -> `3_Model.ipynb` (Regresión Lineal)
6. **Fase 6: Despliegue** -> `app.py` (Streamlit) y `index.html` (Landing Page)
7. **Fase 7: Mantenimiento y Actualización** -> Consolidación del repositorio y monitoreo del uso del modelo.

## 📁 Estructura del Proyecto

- `1_ETL.ipynb`: Cuaderno para la extracción, limpieza y transformación de los datos.
- `2_EDA.ipynb`: Cuaderno para el Análisis Exploratorio de Datos.
- `3_Model.ipynb`: Cuaderno donde se entrena y evalúa el modelo de Regresión Lineal. Se generan los archivos `.pkl`.
- `app.py`: Aplicación web funcional en Streamlit para consumir el modelo preentrenado.
- `index.html`: Landing page informativa y estética que explica los propósitos y variables del modelo.
- `requirements.txt`: Lista de dependencias de Python necesarias.
- `sample_data.csv`: Muestra extraída del dataset original (debido al gran tamaño del archivo base) para el desarrollo.

## ⚙️ Instalación y Uso

1. **Clonar/Descargar** los archivos en una carpeta local.
2. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Ejecutar los cuadernos** (en orden) para generar los archivos del modelo:
   - Corre `1_ETL.ipynb`
   - Corre `2_EDA.ipynb`
   - Corre `3_Model.ipynb` (Este paso generará `modelo_regresion.pkl` y `columnas_modelo.pkl`)
4. **Desplegar la App** (Streamlit):
   ```bash
   streamlit run app.py
   ```
5. **Ver la Landing Page**:
   - Abre `index.html` en tu navegador favorito.

---
*Desarrollado para Talento Tech*
