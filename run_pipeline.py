import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import numpy as np

base_path = r'C:\Users\Windows\Desktop\TalentoTechProject'
os.chdir(base_path)

print("Iniciando Fase 3: ETL...")
# Cargar los datos
df = pd.read_csv('sample_data.csv')

# ETL
for col in ['valor_credito', 'valor_inversion']:
    if col in df.columns:
        # Remover cualquier caracter no numérico (excepto punto decimal y signo negativo)
        df[col] = df[col].astype(str).replace(r'[^0-9.-]', '', regex=True).astype(float)

columns_to_keep = ['plazo', 'periodo_gracia', 'cuotas', 'cuotas_capital', 'tasa_indexacion', 'valor_credito']
# Asegurar que existan las columnas
columns_to_keep = [c for c in columns_to_keep if c in df.columns]

df_clean = df[columns_to_keep].copy()

# Forzar conversión de numéricas
for col in ['plazo', 'periodo_gracia', 'cuotas', 'cuotas_capital', 'valor_credito']:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

df_clean.dropna(inplace=True)

df_clean.to_csv('cleaned_data.csv', index=False)
print("Datos limpios guardados en cleaned_data.csv")

print("Iniciando Fase 4 y 5: Modelado...")
# Model
df_clean = pd.get_dummies(df_clean, columns=['tasa_indexacion'], drop_first=True)

X = df_clean.drop('valor_credito', axis=1)
y = df_clean['valor_credito']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Evaluación - RMSE: {rmse}, R2: {r2}")

joblib.dump(model, 'modelo_regresion.pkl')
joblib.dump(list(X.columns), 'columnas_modelo.pkl')
print("¡modelo_regresion.pkl y columnas_modelo.pkl generados exitosamente!")
