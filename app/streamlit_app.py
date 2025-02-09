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
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Cargar los datos
@st.cache_data  # Usar st.cache_data en lugar de st.cache
def load_data():
    """
    Carga los datos desde un archivo CSV.
    
    Retorna:
        pd.DataFrame: Datos cargados.
    """
    return pd.read_csv('data/BancoXavantes837.csv')  # Asegurar la ruta correcta

data = load_data()

# Título de la aplicación
st.title('Análisis de Riesgo Cardiovascular')

# Mostrar los datos
st.write("### Vista previa de los datos")
st.write(data.head())

# Análisis Exploratorio de Datos (EDA)
st.write("### Análisis Exploratorio de Datos")

# Distribución de la variable objetivo (Riesgo Cardiovascular)
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
    X = data.drop(columns=['IID', 'Riesgo_Cardiovascular', 'Riesgo_Cardiovascular_Binario'])
    y = data['Riesgo_Cardiovascular_Binario']
    X = pd.get_dummies(X, columns=['Sexo'], drop_first=True)
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
    title = 'Visualización de Datos con PCA'
elif option == 't-SNE':
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X_scaled)
    title = 'Visualización de Datos con t-SNE'
else:
    X_reduced = X_scaled
    title = 'Datos Originales'

if option != 'Datos Originales':
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    plt.colorbar(scatter, label='Riesgo Cardiovascular (0: Bajo, 1: Alto)')
    st.pyplot(fig)

# Entrenar y evaluar modelos
st.write("### Entrenamiento y Evaluación de Modelos")
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

# Modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Modelo de Red Neuronal
def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

nn_model = create_nn_model(X_train.shape[1])
history = nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

st.write(f"#### Precisión del Modelo SVM: {accuracy_svm:.2f}")
st.write(f"#### Precisión del Modelo de Red Neuronal: {accuracy_nn:.2f}")

# Gráfica de precisión de Red Neuronal
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='Precisión en entrenamiento')
ax.plot(history.history['val_accuracy'], label='Precisión en validación')
ax.set_title('Precisión del Modelo de Red Neuronal')
ax.set_xlabel('Épocas')
ax.set_ylabel('Precisión')
ax.legend()
st.pyplot(fig)

