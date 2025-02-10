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
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import roc_curve, auc


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
st.write("""
La siguiente base de datos pertenece a una población de indígenas Xavantes de Brasil, 
la cual cuenta con variables importantes para determinar enfermedades cardiovasculares en la población.  
La base de datos incluye las siguientes variables:

- **Sexo**: Género de los individuos (hombre o mujer).
- **Edad**: Edad en años.
- **Leptina**: Nivel de leptina, una hormona relacionada con la regulación del apetito y el metabolismo.
- **Grasa**: Porcentaje de grasa corporal.
- **IMC**: Índice de Masa Corporal, una medida de la relación entre peso y altura.
- **BAI**: Índice de Adiposidad Corporal, una medida alternativa al IMC.
- **Cintura**: Circunferencia de la cintura en centímetros.
- **Cadera**: Circunferencia de la cadera en centímetros.
- **CVLDL**: Colesterol de lipoproteínas de muy baja densidad.
- **Triglic**: Nivel de triglicéridos en sangre.
- **CTOTAL**: Colesterol total.
- **CLDL**: Colesterol de lipoproteínas de baja densidad (colesterol "malo").
- **CHDL**: Colesterol de lipoproteínas de alta densidad (colesterol "bueno").
- **FTO_Aditivo**: Variante genética asociada con la obesidad y el riesgo cardiovascular.
""")

st.write("### Vista previa de los datos")
st.dataframe(data.head())
st.write("### Información de los datos")
st.dataframe(data.describe())
# Capturar la información del DataFrame
buffer = io.StringIO()
data.info(buf=buffer)  # Captura la salida de data.info()
info_str = buffer.getvalue()  # Convierte el buffer en string

st.write("### Información del DataFrame")
st.text(info_str)  # Muestra el contenido de info() en formato de texto

st.write("""
### Variable Objetivo: Riesgo Cardiovascular

La variable objetivo de este estudio es el **Riesgo Cardiovascular**, que se determina en función de los siguientes criterios clínicos y umbrales establecidos:

- **Colesterol Total (CTOTAL)**: Alto riesgo si **CTOTAL > 200 mg/dL**.
- **Triglicéridos (Triglic)**: Alto riesgo si **Triglic > 150 mg/dL**.
- **Colesterol LDL (CLDL)**: Alto riesgo si **CLDL > 130 mg/dL**.
- **Colesterol HDL (CHDL)**: **Bajo riesgo** si **CHDL < 40 mg/dL (hombres)** o **< 50 mg/dL (mujeres)**.
- **Índice de Masa Corporal (IMC)**: Alto riesgo si **IMC > 30** (obesidad).
- **Circunferencia de Cintura**: Alto riesgo si **Cintura > 102 cm (hombres)** o **> 88 cm (mujeres)**.
- **Relación Cintura-Cadera**: Alto riesgo si **Relación > 0.9 (hombres)** o **> 0.85 (mujeres)**.

#### Definición de la Variable de Interés:
- **0 (Bajo riesgo):** El individuo **no cumple con ninguno** de los criterios de alto riesgo.
- **1 (Alto riesgo):** El individuo **cumple con al menos uno** de los criterios de alto riesgo mencionados anteriormente.

Esta variable se calcula automáticamente en el análisis utilizando los umbrales clínicos establecidos.
""")

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

############################################################################################


def preprocess_data(data):
    X = data.drop(columns=['IID', 'Riesgo_Cardiovascular', 'Riesgo_Cardiovascular_Binario'], errors='ignore')
    y = data['Riesgo_Cardiovascular_Binario']  # Variable de interés
    X = pd.get_dummies(X, columns=['Sexo'], drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def plot_roc_curve(y_true, y_pred_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    st.pyplot(plt)

def plot_scatter(X, y, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm', alpha=0.7)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.title(title)
    st.pyplot(plt)

data = pd.read_csv('data.csv')  # Cargar datos
X_scaled, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

st.write("### SVM con Datos Originales")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]
st.write(f"Precisión (SVM): {accuracy_score(y_test, y_pred_svm):.2f}")
st.write("Matriz de Confusión (SVM):")
st.write(confusion_matrix(y_test, y_pred_svm))
plot_roc_curve(y_test, y_pred_proba_svm, 'Curva ROC - SVM (Datos Originales)')

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.3, random_state=42)
st.write("### SVM con PCA")
plot_scatter(X_pca, y, "Distribución PCA")
svm_model_pca = SVC(kernel='linear', probability=True, random_state=42)
svm_model_pca.fit(X_train_pca, y_train_pca)
y_pred_svm_pca = svm_model_pca.predict(X_test_pca)
y_pred_proba_svm_pca = svm_model_pca.predict_proba(X_test_pca)[:, 1]
st.write(f"Precisión (SVM con PCA): {accuracy_score(y_test_pca, y_pred_svm_pca):.2f}")
st.write("Matriz de Confusión (SVM con PCA):")
st.write(confusion_matrix(y_test_pca, y_pred_svm_pca))
plot_roc_curve(y_test_pca, y_pred_proba_svm_pca, 'Curva ROC - SVM (PCA)')

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(X_tsne, y, test_size=0.3, random_state=42)
st.write("### SVM con t-SNE")
plot_scatter(X_tsne, y, "Distribución t-SNE")
svm_model_tsne = SVC(kernel='linear', probability=True, random_state=42)
svm_model_tsne.fit(X_train_tsne, y_train_tsne)
y_pred_svm_tsne = svm_model_tsne.predict(X_test_tsne)
y_pred_proba_svm_tsne = svm_model_tsne.predict_proba(X_test_tsne)[:, 1]
st.write(f"Precisión (SVM con t-SNE): {accuracy_score(y_test_tsne, y_pred_svm_tsne):.2f}")
st.write("Matriz de Confusión (SVM con t-SNE):")
st.write(confusion_matrix(y_test_tsne, y_pred_svm_tsne))
plot_roc_curve(y_test_tsne, y_pred_proba_svm_tsne, 'Curva ROC - SVM (t-SNE)')
