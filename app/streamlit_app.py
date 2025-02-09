# app/streamlit_app.py

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
@st.cache_data  # Usar st.cache_data en lugar de st.cache
def load_data():
    """
    Carga los datos desde un archivo Excel.
    
    Retorna:
        pd.DataFrame: Datos cargados.
    """
    # Asegúrate de que la ruta del archivo sea correcta
    return pd.read_csv('../data/BancoXavantes837.csv')  # Ruta relativa al archivo

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
    """
    Preprocesa los datos: codifica variables categóricas y escala las características.
    
    Parámetros:
        data (pd.DataFrame): Datos originales.
    
    Retorna:
        tuple: (X_scaled, y), donde X_scaled son las características escaladas y y es la variable objetivo.
    """
    X = data.drop(columns=['IID', 'Riesgo_Cardiovascular', 'Riesgo_Cardiovascular_Binario'])
    y = data['Riesgo_Cardiovascular_Binario']  # Variable de interés
    X = pd.get_dummies(X, columns=['Sexo'], drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

X_scaled, y = preprocess_data(data)

# Reducción de dimensionalidad
st.write("### Reducción de Dimensionalidad")
st.write("Selecciona una técnica de reducción de dimensionalidad:")
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

# Graficar los datos reducidos
if option != 'Datos Originales':
    fig, ax = plt.subplots()
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='coolwarm', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('Componente 1')
    ax.set_ylabel('Componente 2')
    plt.colorbar(ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='coolwarm', alpha=0.6), label='Riesgo Cardiovascular (0: Bajo, 1: Alto)')
    st.pyplot(fig)

# Entrenar y evaluar modelos
st.write("### Entrenamiento y Evaluación de Modelos")

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

# Entrenar el modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

# Entrenar el modelo de Red Neuronal
def create_nn_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

nn_model = create_nn_model(X_train.shape[1])
history = nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

# Mostrar resultados
st.write(f"#### Precisión del Modelo SVM: {accuracy_svm:.2f}")
st.write(f"#### Precisión del Modelo de Red Neuronal: {accuracy_nn:.2f}")

# Gráfica de precisión del modelo de Red Neuronal
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='Precisión en entrenamiento')
ax.plot(history.history['val_accuracy'], label='Precisión en validación')
ax.set_title('Precisión del Modelo de Red Neuronal')
ax.set_xlabel('Épocas')
ax.set_ylabel('Precisión')
ax.legend()
st.pyplot(fig)

# Comparación de modelos en diferentes representaciones
st.write("### Comparación de Modelos en Diferentes Representaciones")

# Crear un DataFrame para la comparación
comparison_data = {
    'Representación': ['Datos Originales', 'PCA', 't-SNE'],
    'SVM': [0.0, 0.0, 0.0],  # Valores de precisión para SVM
    'Red Neuronal': [0.0, 0.0, 0.0]  # Valores de precisión para Red Neuronal
}

# Calcular la precisión para cada representación
representations = {
    'Datos Originales': X_scaled,
    'PCA': PCA(n_components=2).fit_transform(X_scaled),
    't-SNE': TSNE(n_components=2, random_state=42).fit_transform(X_scaled)
}

for i, (name, X_rep) in enumerate(representations.items()):
    X_train, X_test, y_train, y_test = train_test_split(X_rep, y, test_size=0.3, random_state=42)
    
    # Entrenar y evaluar SVM
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    comparison_data['SVM'][i] = accuracy_score(y_test, y_pred_svm)
    
    # Entrenar y evaluar Red Neuronal
    nn_model = create_nn_model(X_train.shape[1])
    nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
    comparison_data['Red Neuronal'][i] = accuracy_score(y_test, y_pred_nn)

# Mostrar la comparación
st.write("#### Comparación de Precisión en Diferentes Representaciones")
comparison_df = pd.DataFrame(comparison_data)
st.write(comparison_df)

# Gráfica de comparación
fig, ax = plt.subplots()
comparison_df.set_index('Representación').plot(kind='bar', ax=ax)
ax.set_title('Comparación de Precisión entre Modelos')
ax.set_xlabel('Representación de Datos')
ax.set_ylabel('Precisión')
st.pyplot(fig)
