import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar datos con cache
@st.cache_data
def load_data():
    try:
        return pd.read_excel('BancoXavantes837.xlsx')
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

data = load_data()
if data is None:
    st.stop()

# Limpiar nombres de columnas
data.columns = data.columns.str.strip()

# Definir pesos para calcular el índice de riesgo cardiovascular
pesos = {
    'CTOTAL': 0.2, 'CLDL': 0.3, 'CHDL': -0.2, 'Triglic': 0.2, 'CVLDL': 0.1,
    'IMC': 0.15, 'BAI': 0.1, 'Cintura': 0.15, 'Cadera': -0.1, 'Grasa': 0.1,
    'Edad': 0.2, 'Leptina': 0.05, 'FTO_Aditivo': 0.05
}

# Verificar existencia de columnas
missing_columns = [col for col in pesos.keys() if col not in data.columns]
if missing_columns:
    st.error(f"Faltan las siguientes columnas: {missing_columns}")
    st.stop()

# Calcular el índice de riesgo cardiovascular
data['Riesgo_Cardiovascular'] = data[list(pesos.keys())].mul(pesos).sum(axis=1)

# Definir umbral basado en el 50% del valor máximo
umbral = data['Riesgo_Cardiovascular'].max() * 0.5
data['Riesgo_Cardiovascular_Binario'] = (data['Riesgo_Cardiovascular'] > umbral).astype(int)

# Mostrar datos
st.title('Análisis de Enfermedades Cardiovasculares en la Población Indígena Xavante de Brasil')
st.write("### Vista previa de los datos")
st.dataframe(data.head())

# Balance de clases
st.write("#### Balance de Clases en la Variable Objetivo")
st.write(data['Riesgo_Cardiovascular_Binario'].value_counts())

# Gráfica del riesgo cardiovascular
fig, ax = plt.subplots()
sns.histplot(data['Riesgo_Cardiovascular'], bins=30, kde=True, ax=ax, color='blue')
ax.set_title('Distribución del Índice de Riesgo Cardiovascular')
ax.set_xlabel('Riesgo Cardiovascular')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)

# Separar datos para entrenamiento
X = data[list(pesos.keys())]
y = data['Riesgo_Cardiovascular_Binario']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Evaluación del Modelo SVM
st.write("### Evaluación del Modelo SVM")
cm_svm = confusion_matrix(y_test, y_pred_svm)
fig, ax = plt.subplots()
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=["Bajo", "Alto"], yticklabels=["Bajo", "Alto"])
ax.set_title("Matriz de Confusión - SVM")
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
st.pyplot(fig)

st.text("**Reporte de Clasificación - SVM**")
st.text(classification_report(y_test, y_pred_svm))

# Modelo de Red Neuronal
nn_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Predicciones
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)

# Evaluación de la Red Neuronal
st.write("### Evaluación del Modelo de Red Neuronal")
cm_nn = confusion_matrix(y_test, y_pred_nn)
fig, ax = plt.subplots()
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Greens', xticklabels=["Bajo", "Alto"], yticklabels=["Bajo", "Alto"])
ax.set_title("Matriz de Confusión - Red Neuronal")
ax.set_xlabel("Predicho")
ax.set_ylabel("Real")
st.pyplot(fig)

st.text("**Reporte de Clasificación - Red Neuronal**")
st.text(classification_report(y_test, y_pred_nn))

# Evolución del entrenamiento
st.write("### Evolución del Entrenamiento de la Red Neuronal")
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label="Entrenamiento")
ax.plot(history.history['val_accuracy'], label="Validación", linestyle="dashed")
ax.set_title("Precisión del Modelo")
ax.set_xlabel("Épocas")
ax.set_ylabel("Precisión")
ax.legend()
st.pyplot(fig)

