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

# Verificar la existencia de columnas necesarias
missing_columns = [col for col in pesos.keys() if col not in data.columns]
if missing_columns:
    st.error(f"Faltan las siguientes columnas: {missing_columns}")
    st.stop()

# Calcular el índice de riesgo cardiovascular
data['Riesgo_Cardiovascular'] = data[list(pesos.keys())].mul(pesos).sum(axis=1)

data['Riesgo_Cardiovascular_Binario'] = (data['Riesgo_Cardiovascular'] > data['Riesgo_Cardiovascular'].median()).astype(int)
# Calcular el índice de riesgo cardiovascular
data['Riesgo_Cardiovascular'] = sum(data[col] * peso for col, peso in pesos.items())

# Calcular el índice de riesgo cardiovascular
data['Riesgo_Cardiovascular'] = sum(data[col] * peso for col, peso in pesos.items())

# Calcular el índice de riesgo cardiovascular
data['Riesgo_Cardiovascular'] = sum(data[col] * peso for col, peso in pesos.items())

# Definir un umbral fijo basado en una fracción del máximo
umbral = data['Riesgo_Cardiovascular'].max() * 0.5  # Ajustar según necesidad

# Crear variable binaria de riesgo cardiovascular
data['Riesgo_Cardiovascular_Binario'] = (data['Riesgo_Cardiovascular'] > umbral).astype(int)



# Mostrar datos
st.title('Análisis de Enfermedades Cardiovasculares en la Población Indígena Xavante de Brasil')
st.write('La siguiente base de datos pertenece a una población de indígenas Xavantes de Brasil, la cual cuenta con variables importantes para determinar enfermedades cardiovasculares en la población. La base de datos incluye las siguientes variables:
•	Sexo: Género de los individuos (hombre o mujer).
•	Edad: Edad en años.
•	Leptina: Nivel de leptina, una hormona relacionada con la regulación del apetito y el metabolismo.
•	Grasa: Porcentaje de grasa corporal.
•	IMC: Índice de Masa Corporal, una medida de la relación entre peso y altura.
•	BAI: Índice de Adiposidad Corporal, una medida alternativa al IMC.
•	Cintura: Circunferencia de la cintura en centímetros.
•	Cadera: Circunferencia de la cadera en centímetros.
•	CVLDL: Colesterol de lipoproteínas de muy baja densidad.
•	Triglic: Nivel de triglicéridos en sangre.
•	CTOTAL: Colesterol total.
•	CLDL: Colesterol de lipoproteínas de baja densidad (colesterol "malo").
•	CHDL: Colesterol de lipoproteínas de alta densidad (colesterol "bueno").
•	FTO_Aditivo: Variante genética asociada con la obesidad y el riesgo cardiovascular.
')
st.write("### Vista previa de los datos")
st.dataframe(data.head())
st.write("### Información de los datos")
st.dataframe(data.describe())
# Mostrar información sobre el umbral seleccionado
st.write(f"Se ha utilizado un umbral de **{umbral:.2f}** basado en el 50% del valor máximo del índice.")

# Mostrar el balance de clases
st.write("#### Balance de Clases en la Variable Objetivo")
st.write(data['Riesgo_Cardiovascular_Binario'].value_counts())

st.write("### Histograma de la Distribución del Riesgo Cardiovascular")

# Crear el histograma
fig, ax = plt.subplots()
sns.histplot(data['Riesgo_Cardiovascular'], bins=30, kde=True, ax=ax, color='blue')
ax.set_title('Distribución del Índice de Riesgo Cardiovascular')
ax.set_xlabel('Riesgo Cardiovascular')
ax.set_ylabel('Frecuencia')

# Mostrar la gráfica en Streamlit
st.pyplot(fig)


st.write("### Distribución de Variables Numéricas")

# Obtener solo las columnas numéricas
numerical_columns = data.select_dtypes(include=['number']).columns.tolist()

# Validar que hay columnas numéricas antes de continuar
if not numerical_columns:
    st.error("No se encontraron variables numéricas en el dataset.")
    st.stop()

# Seleccionar la variable a visualizar
selected_variable = st.selectbox("Selecciona una variable numérica:", numerical_columns)

# Botón para mostrar la gráfica
if st.button(f"Ver distribución de {selected_variable}"):
    fig, ax = plt.subplots()
    sns.histplot(data[selected_variable], bins=30, kde=True, ax=ax)
    ax.set_title(f'Distribución de {selected_variable}')
    ax.set_xlabel(selected_variable)
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
    if X_scaled.shape[1] > 50:
        st.warning("t-SNE puede ser lento con muchas dimensiones, considere PCA primero.")
    tsne = TSNE(n_components=2, random_state=42)
    X_reduced = tsne.fit_transform(X_scaled)
else:
    X_reduced = X_scaled

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)

# Entrenar modelo SVM
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

def create_nn_model(input_dim):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Entrenar modelo de red neuronal
nn_model = create_nn_model(X_train.shape[1])
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
accuracy_nn = accuracy_score(y_test, y_pred_nn)

# Mostrar resultados
st.write(f"#### Precisión del Modelo SVM: {accuracy_svm:.2f}")
st.write(f"#### Precisión del Modelo de Red Neuronal: {accuracy_nn:.2f}")
