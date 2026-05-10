import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

def create_etl():
    nb = new_notebook()
    nb.cells.extend([
        new_markdown_cell("# Fase 1: Identificación del Problema\n\nEn esta etapa, identificamos el problema de negocio. El objetivo es analizar los desembolsos de crédito de fomento agropecuario y predecir el **valor del crédito** (`valor_credito`) otorgado, basándonos en variables como el plazo, el periodo de gracia y otras características del crédito."),
        new_markdown_cell("# Fase 2: Recolección de Datos\n\nCargamos la muestra de los datos obtenidos del sistema original para comenzar a trabajar (hemos extraído una muestra de 50,000 registros para facilitar el procesamiento)."),
        new_code_cell("import pandas as pd\nimport numpy as np\n\n# Cargar los datos\ndf = pd.read_csv('sample_data.csv')\ndf.head()"),
        new_markdown_cell("# Fase 3: Preparación de Datos (ETL)\n\nRealizamos la limpieza de los datos, tratamos valores nulos y adaptamos los tipos de datos. También seleccionamos las columnas de interés para nuestro análisis y modelo."),
        new_code_cell("# Convertir valores monetarios de formato string con comas a float\nfor col in ['valor_credito', 'valor_inversion']:\n    if df[col].dtype == object:\n        df[col] = df[col].astype(str).str.replace(',', '').astype(float)\n\n# Seleccionar variables numéricas útiles y limpiar nulos\ncolumns_to_keep = ['plazo', 'periodo_gracia', 'cuotas', 'cuotas_capital', 'tasa_indexacion', 'valor_credito']\ndf_clean = df[columns_to_keep].copy()\ndf_clean.dropna(inplace=True)\n\ndf_clean.head()"),
        new_code_cell("# Guardamos los datos limpios para la siguiente etapa\ndf_clean.to_csv('cleaned_data.csv', index=False)\nprint('Datos limpios guardados en cleaned_data.csv')")
    ])
    with open('C:/Users/Windows/Desktop/TalentoTechProject/1_ETL.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def create_eda():
    nb = new_notebook()
    nb.cells.extend([
        new_markdown_cell("# Análisis Exploratorio de Datos (EDA)\n\nEn este cuaderno realizamos un análisis descriptivo de nuestros datos para entender la distribución y la relación entre las variables, en especial con respecto a nuestra variable objetivo: `valor_credito`."),
        new_code_cell("import pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\n# Cargar datos limpios\ndf = pd.read_csv('cleaned_data.csv')\ndf.head()"),
        new_markdown_cell("### Distribución del Valor del Crédito\n\nVeamos cómo se distribuyen los valores de crédito."),
        new_code_cell("plt.figure(figsize=(10,6))\nsns.histplot(df['valor_credito'], bins=50, kde=True)\nplt.title('Distribución de Valor del Crédito')\nplt.xlabel('Valor de Crédito')\nplt.ylabel('Frecuencia')\nplt.show()"),
        new_markdown_cell("### Matriz de Correlación\n\nRevisamos cómo se correlacionan las variables numéricas para identificar cuáles tienen una mayor influencia sobre el valor del crédito."),
        new_code_cell("# Convertir variables categóricas si las hay, en este caso tasa_indexacion\nif 'tasa_indexacion' in df.columns:\n    df['tasa_indexacion_encoded'] = df['tasa_indexacion'].astype('category').cat.codes\n\nnumeric_cols = df.select_dtypes(include=['float64', 'int64', 'int8']).columns\nplt.figure(figsize=(8,6))\nsns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')\nplt.title('Matriz de Correlación')\nplt.show()")
    ])
    with open('C:/Users/Windows/Desktop/TalentoTechProject/2_EDA.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

def create_model():
    nb = new_notebook()
    nb.cells.extend([
        new_markdown_cell("# Fase 4 y 5: Ingeniería de Modelos y Evaluación\n\nEn esta etapa entrenamos un modelo de **Regresión Lineal** para predecir el `valor_credito`. Posteriormente, evaluaremos su rendimiento y lo exportaremos para su uso en producción (Fase 6: Despliegue)."),
        new_code_cell("import pandas as pd\nfrom sklearn.model_selection import train_test_split\nfrom sklearn.linear_model import LinearRegression\nfrom sklearn.metrics import mean_squared_error, r2_score\nimport joblib\n\n# Cargar datos limpios\ndf = pd.read_csv('cleaned_data.csv')\n\n# Preparar variables (One-Hot Encoding para tasa_indexacion)\ndf = pd.get_dummies(df, columns=['tasa_indexacion'], drop_first=True)"),
        new_markdown_cell("### División del Dataset y Entrenamiento"),
        new_code_cell("X = df.drop('valor_credito', axis=1)\ny = df['valor_credito']\n\nX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n\nmodel = LinearRegression()\nmodel.fit(X_train, y_train)"),
        new_markdown_cell("### Evaluación del Modelo"),
        new_code_cell("y_pred = model.predict(X_test)\n\nrmse = mean_squared_error(y_test, y_pred, squared=False)\nr2 = r2_score(y_test, y_pred)\n\nprint(f'RMSE: {rmse}')\nprint(f'R2 Score: {r2}')"),
        new_markdown_cell("### Exportar el Modelo y Nombres de las Columnas\nGuardamos el modelo entrenado y los nombres de las columnas en formato `.pkl` para ser consumidos en nuestra aplicación de Streamlit."),
        new_code_cell("joblib.dump(model, 'modelo_regresion.pkl')\njoblib.dump(list(X.columns), 'columnas_modelo.pkl')\nprint('Modelo exportado exitosamente.')")
    ])
    with open('C:/Users/Windows/Desktop/TalentoTechProject/3_Model.ipynb', 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)

if __name__ == "__main__":
    create_etl()
    create_eda()
    create_model()
    print("Notebooks creados exitosamente.")
