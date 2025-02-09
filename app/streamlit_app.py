import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar los datos
@st.cache_data  # Usar cache_data para optimizar carga
def load_data():
    try:
        return  pd.read_excel('/ProyectoFinal/data/BancoXavantes837.xlsx')  # Asegurar la ruta correcta

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

# Verificar que todas las columnas existen antes de calcular el índice de riesgo
missing_columns = [col for col in pesos.keys() if col not in data.columns]
if missing_columns:
    st.error(f"Las siguientes columnas están ausentes en los datos: {missing_columns}")
    st.stop()

# Calcular el índice de riesgo cardiovascular
data['Riesgo_Cardiovascular'] = sum(data[col] * peso for col, peso in pesos.items())

# Crear variable binaria de riesgo cardiovascular
data['Riesgo_Cardiovascular_Binario'] = (data['Riesgo_Cardiovascular'] > data['Riesgo_Cardiovascular'].median()).astype(int)

# Mostrar los datos
st.title('Análisis de Riesgo Cardiovascular')
st.write("### Vista previa de los datos")
st.write(data.head())

# Distribución de la variable objetivo
st.write("#### Distribución de la Variable Objetivo")
fig, ax = plt.subplots()
data['Riesgo_Cardiovascular_Binario'].value_counts().plot(kind='bar', ax=ax)
ax.set_title('Distribución de Riesgo Cardiovascular')
ax.set_xlabel('Riesgo Cardiovascular (0: Bajo, 1: Alto)')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)

# Matriz de correlación
st.write("#### Matriz de Correlación")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Matriz de Correlación')
st.pyplot(fig)

# Preprocesamiento de datos
def preprocess_data(data):
    X = data.drop(columns=['Riesgo_Cardiovascular', 'Riesgo_Cardiovascular_Binario'])
    y = data['Riesgo_Cardiovascular_Binario']
    X = pd.get_dummies(X, drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

X_scaled, y = preprocess_data(data)

# Reducción de dimensionalidad
st.write("### Reducción de Dimensionalidad")
option = st.selectbox('Técnica', ['Datos Originales', 'PCA', 't-SNE'])

if option == 'PCA':
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_scaled)
elif option == 't-SNE':
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X_scaled)
else:
    X_reduced = X_scaled

# Entrenar y evaluar modelos
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

nn_model = create_nn_model(X_train.shape[1])
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

st.write(f"#### Precisión del Modelo SVM: {accuracy_svm:.2f}")
st.write(f"#### Precisión del Modelo de Red Neuronal: {accuracy_nn:.2f}")

