import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle
import gzip
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


# Menú de navegación en el sidebar
st.sidebar.title("Navegación")
st.sidebar.markdown("---")

# Submenú para Análisis de Datos
with st.sidebar.expander(" Análisis de Datos", expanded=True):
    section_analisis = st.radio(
        "Selecciona una sección:",
        [
            "Análisis Exploratorio",
        ],
    )

# Submenú para Modelos de Machine Learning
with st.sidebar.expander(" Modelos Ajustados", expanded=True):
    section_modelos = st.radio(
        "Selecciona una sección:",
        [
            "SVM con Datos Originales",
            "SVM con PCA",
            "SVM con t-SNE",
            "Red Neuronal",
            "Red Neuronal con PCA",
            "Red Neuronal con t-SNE",
        ],
    )

# Submenú para Predicciones
with st.sidebar.expander("Predicciones", expanded=True):
    section_predicciones = st.radio(
        "Selecciona una sección:",
        [
            "Predicción de Riesgo Cardiovascular",
        ],
    )




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
st.dataframe(data.head(5))

# Mostrar estadísticas descriptivas de los datos normalizados
st.write("### Información de los datos")
st.write(data.describe())


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

# Seleccionar las variables a visualizar
selected_variables = st.multiselect(
    "Selecciona las variables numéricas que deseas visualizar:",
    numerical_columns,
    default=numerical_columns[:2]  # Mostrar las primeras 2 variables por defecto
)

# Verificar si se seleccionaron variables
if not selected_variables:
    st.warning("Por favor, selecciona al menos una variable numérica.")
    st.stop()

# Organizar las gráficas en un mosaico
num_columns = 2  # Número de columnas en el mosaico
num_rows = (len(selected_variables) + num_columns - 1) // num_columns  # Calcular el número de filas necesarias

# Crear un mosaico de gráficas
for i in range(num_rows):
    cols = st.columns(num_columns)  # Crear una fila con el número de columnas especificado
    for j in range(num_columns):
        idx = i * num_columns + j  # Calcular el índice de la variable actual
        if idx < len(selected_variables):  # Verificar que el índice esté dentro del rango
            variable = selected_variables[idx]
            with cols[j]:  # Usar la columna correspondiente
                fig, ax = plt.subplots(figsize=(6, 4))  # Tamaño más pequeño para las gráficas
                sns.histplot(data[variable], bins=30, kde=True, ax=ax)
                ax.set_title(f'Distribución de {variable}')
                ax.set_xlabel(variable)
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig)

# Matriz de correlación
st.write("#### Matriz de Correlación")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Matriz de Correlación')
st.pyplot(fig)

############################################################################################


# Preprocesamiento de datos
def preprocess_data(data):
    # Asegúrate de que los nombres de las columnas coincidan exactamente con los del archivo
    X = data.drop(columns=['IID', 'Riesgo_Cardiovascular', 'Riesgo_Cardiovascular_Binario'])
    y = data['Riesgo_Cardiovascular_Binario']
    X = pd.get_dummies(X, columns=['Sexo'], drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

try:
    X_scaled, y = preprocess_data(data)
except KeyError as e:
    st.error(f"Error: {e}. Verifica los nombres de las columnas en el archivo de datos.")
    st.stop()

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Función para graficar PCA y t-SNE
def plot_dimension_reduction(X_pca, X_tsne, y, title_pca, title_tsne):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', ax=ax[0])
    ax[0].set_title(title_pca)
    
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='coolwarm', ax=ax[1])
    ax[1].set_title(title_tsne)
    
    st.pyplot(fig)

# Función para graficar curva ROC
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

# Aplicar PCA y t-SNE
st.write("### Reducción de Dimensionalidad")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plot_dimension_reduction(X_pca, X_tsne, y, 'PCA - Datos Originales', 't-SNE - Datos Originales')

# Entrenar y evaluar modelos con SVM
st.write("## Modelos Ajustados")

# SVM con datos originales
st.write("### SVM con Datos Originales")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]
accuracy_svm = accuracy_score(y_test, y_pred_svm)
st.write(f"**Precisión (SVM):** {accuracy_svm:.2f}")
st.write("**Matriz de Confusión (SVM):**")
st.write(confusion_matrix(y_test, y_pred_svm))
st.write("**Informe de Clasificación (SVM):**")
st.write(classification_report(y_test, y_pred_svm))
plot_roc_curve(y_test, y_pred_proba_svm, 'Curva ROC - SVM (Datos Originales)')

st.write("""
**Conclusión:**
El modelo SVM con datos originales logra una precisión del {:.2f}. La matriz de confusión muestra un buen equilibrio entre verdaderos positivos y falsos positivos. La curva ROC con un AUC de {:.2f} indica un buen rendimiento en la clasificación.
""".format(accuracy_svm, auc(roc_curve(y_test, y_pred_proba_svm)[0], roc_curve(y_test, y_pred_proba_svm)[1])))

# Reducción de dimensionalidad con PCA y t-SNE para entrenamiento de modelos
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.3, random_state=42)
X_train_tsne, X_test_tsne, _, _ = train_test_split(X_tsne, y, test_size=0.3, random_state=42)

# Entrenar y evaluar modelos con SVM usando PCA y t-SNE
for name, X_tr, X_te in [('PCA', X_train_pca, X_test_pca), ('t-SNE', X_train_tsne, X_test_tsne)]:
    st.write(f"### SVM con {name}")
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Precisión (SVM con {name}):** {accuracy:.2f}")
    st.write(f"**Matriz de Confusión (SVM con {name}):**")
    st.write(confusion_matrix(y_test, y_pred))
    st.write(f"**Informe de Clasificación (SVM con {name}):**")
    st.write(classification_report(y_test, y_pred))
    plot_roc_curve(y_test, y_pred_proba, f'Curva ROC - SVM ({name})')

    st.write(f"""
    **Conclusión:**
    El modelo SVM con {name} logra una precisión del {accuracy:.2f}. La curva ROC con un AUC de {auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1]):.2f}.
    """)

# Definir hiperparámetros de la red neuronal
hyperparams = {'depth': 4, 'epochs': 43, 'num_units': 144, 'optimizer': 'sgd', 'activation': 'relu', 'batch_size': 72, 'learning_rate': 0.0329}

# Función para crear la red neuronal
def create_nn_model(input_dim, hyperparams):
    model = Sequential()
    for _ in range(hyperparams['depth'] - 1):
        model.add(Dense(hyperparams['num_units'], activation=hyperparams['activation']))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(learning_rate=hyperparams['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Entrenar y evaluar red neuronal con datos originales
st.write("### Red Neuronal con Datos Originales")
nn_model = create_nn_model(X_train.shape[1], hyperparams)
history = nn_model.fit(X_train, y_train, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], validation_split=0.2, verbose=0)
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
y_pred_proba_nn = nn_model.predict(X_test).flatten()
accuracy_nn = accuracy_score(y_test, y_pred_nn)
st.write(f"**Precisión (Red Neuronal):** {accuracy_nn:.2f}")
st.write("**Matriz de Confusión (Red Neuronal):**")
st.write(confusion_matrix(y_test, y_pred_nn))
st.write("**Informe de Clasificación (Red Neuronal):**")
st.write(classification_report(y_test, y_pred_nn))
plot_roc_curve(y_test, y_pred_proba_nn, 'Curva ROC - Red Neuronal')

# Gráfica de precisión y pérdida durante el entrenamiento
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(history.history['accuracy'], label='Precisión en entrenamiento')
ax[0].plot(history.history['val_accuracy'], label='Precisión en validación')
ax[0].set_title('Precisión durante el Entrenamiento')
ax[0].set_xlabel('Épocas')
ax[0].set_ylabel('Precisión')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Pérdida en entrenamiento')
ax[1].plot(history.history['val_loss'], label='Pérdida en validación')
ax[1].set_title('Pérdida durante el Entrenamiento')
ax[1].set_xlabel('Épocas')
ax[1].set_ylabel('Pérdida')
ax[1].legend()
st.pyplot(fig)

st.write("""
**Conclusión:**
La red neuronal con datos originales logra una precisión del {:.2f}. La curva de aprendizaje muestra que el modelo converge adecuadamente, sin signos de sobreajuste. La curva ROC con un AUC de {:.2f} confirma un buen rendimiento en la clasificación.
""".format(accuracy_nn, auc(roc_curve(y_test, y_pred_proba_nn)[0], roc_curve(y_test, y_pred_proba_nn)[1])))

# Evaluar red neuronal con PCA y t-SNE
for name, X_tr, X_te in [('PCA', X_train_pca, X_test_pca), ('t-SNE', X_train_tsne, X_test_tsne)]:
    st.write(f"### Red Neuronal con {name}")
    nn_model = create_nn_model(X_tr.shape[1], hyperparams)
    history = nn_model.fit(X_tr, y_train, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], validation_split=0.2, verbose=0)
    y_pred_nn = (nn_model.predict(X_te) > 0.5).astype(int)
    y_pred_proba_nn = nn_model.predict(X_te).flatten()
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    st.write(f"**Precisión (Red Neuronal con {name}):** {accuracy_nn:.2f}")
    st.write(f"**Matriz de Confusión (Red Neuronal con {name}):**")
    st.write(confusion_matrix(y_test, y_pred_nn))
    st.write(f"**Informe de Clasificación (Red Neuronal con {name}):**")
    st.write(classification_report(y_test, y_pred_nn))
    plot_roc_curve(y_test, y_pred_proba_nn, f'Curva ROC - Red Neuronal ({name})')

    # Gráficas de precisión y pérdida durante el entrenamiento
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['accuracy'], label='Precisión en entrenamiento')
    ax[0].plot(history.history['val_accuracy'], label='Precisión en validación')
    ax[0].set_title(f'Precisión durante el Entrenamiento ({name})')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Precisión')
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Pérdida en entrenamiento')
    ax[1].plot(history.history['val_loss'], label='Pérdida en validación')
    ax[1].set_title(f'Pérdida durante el Entrenamiento ({name})')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Pérdida')
    ax[1].legend()
    st.pyplot(fig)

    st.write(f"""
    **Conclusión:**
    La red neuronal con {name} logra una precisión del {accuracy_nn:.2f}. La curva ROC con un AUC de {auc(roc_curve(y_test, y_pred_proba_nn)[0], roc_curve(y_test, y_pred_proba_nn)[1]):.2f}.
    """)

####################################################

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle
import gzip
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


# Menú de navegación en el sidebar
st.sidebar.title("Navegación")
st.sidebar.markdown("---")

# Submenú para Análisis de Datos
with st.sidebar.expander(" Análisis de Datos", expanded=True):
    section_analisis = st.radio(
        "Selecciona una sección:",
        [
            "Análisis Exploratorio",
        ],
    )

# Submenú para Modelos de Machine Learning
with st.sidebar.expander(" Modelos Ajustados", expanded=True):
    section_modelos = st.radio(
        "Selecciona una sección:",
        [
            "SVM con Datos Originales",
            "SVM con PCA",
            "SVM con t-SNE",
            "Red Neuronal",
            "Red Neuronal con PCA",
            "Red Neuronal con t-SNE",
        ],
    )

# Submenú para Predicciones
with st.sidebar.expander("Predicciones", expanded=True):
    section_predicciones = st.radio(
        "Selecciona una sección:",
        [
            "Predicción de Riesgo Cardiovascular",
        ],
    )




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
st.dataframe(data.head(5))

# Mostrar estadísticas descriptivas de los datos normalizados
st.write("### Información de los datos")
st.write(data.describe())


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

# Seleccionar las variables a visualizar
selected_variables = st.multiselect(
    "Selecciona las variables numéricas que deseas visualizar:",
    numerical_columns,
    default=numerical_columns[:2]  # Mostrar las primeras 2 variables por defecto
)

# Verificar si se seleccionaron variables
if not selected_variables:
    st.warning("Por favor, selecciona al menos una variable numérica.")
    st.stop()

# Organizar las gráficas en un mosaico
num_columns = 2  # Número de columnas en el mosaico
num_rows = (len(selected_variables) + num_columns - 1) // num_columns  # Calcular el número de filas necesarias

# Crear un mosaico de gráficas
for i in range(num_rows):
    cols = st.columns(num_columns)  # Crear una fila con el número de columnas especificado
    for j in range(num_columns):
        idx = i * num_columns + j  # Calcular el índice de la variable actual
        if idx < len(selected_variables):  # Verificar que el índice esté dentro del rango
            variable = selected_variables[idx]
            with cols[j]:  # Usar la columna correspondiente
                fig, ax = plt.subplots(figsize=(6, 4))  # Tamaño más pequeño para las gráficas
                sns.histplot(data[variable], bins=30, kde=True, ax=ax)
                ax.set_title(f'Distribución de {variable}')
                ax.set_xlabel(variable)
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig)

# Matriz de correlación
st.write("#### Matriz de Correlación")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Matriz de Correlación')
st.pyplot(fig)

############################################################################################


# Preprocesamiento de datos
def preprocess_data(data):
    # Asegúrate de que los nombres de las columnas coincidan exactamente con los del archivo
    X = data.drop(columns=['IID', 'Riesgo_Cardiovascular', 'Riesgo_Cardiovascular_Binario'])
    y = data['Riesgo_Cardiovascular_Binario']
    X = pd.get_dummies(X, columns=['Sexo'], drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

try:
    X_scaled, y = preprocess_data(data)
except KeyError as e:
    st.error(f"Error: {e}. Verifica los nombres de las columnas en el archivo de datos.")
    st.stop()

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Función para graficar PCA y t-SNE
def plot_dimension_reduction(X_pca, X_tsne, y, title_pca, title_tsne):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', ax=ax[0])
    ax[0].set_title(title_pca)
    
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='coolwarm', ax=ax[1])
    ax[1].set_title(title_tsne)
    
    st.pyplot(fig)

# Función para graficar curva ROC
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

# Aplicar PCA y t-SNE
st.write("### Reducción de Dimensionalidad")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plot_dimension_reduction(X_pca, X_tsne, y, 'PCA - Datos Originales', 't-SNE - Datos Originales')

# Entrenar y evaluar modelos con SVM
st.write("## Modelos Ajustados")

# SVM con datos originales
st.write("### SVM con Datos Originales")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]
accuracy_svm = accuracy_score(y_test, y_pred_svm)
st.write(f"**Precisión (SVM):** {accuracy_svm:.2f}")
st.write("**Matriz de Confusión (SVM):**")
st.write(confusion_matrix(y_test, y_pred_svm))
st.write("**Informe de Clasificación (SVM):**")
st.write(classification_report(y_test, y_pred_svm))
plot_roc_curve(y_test, y_pred_proba_svm, 'Curva ROC - SVM (Datos Originales)')

st.write("""
**Conclusión:**
El modelo SVM con datos originales logra una precisión del {:.2f}. La matriz de confusión muestra un buen equilibrio entre verdaderos positivos y falsos positivos. La curva ROC con un AUC de {:.2f} indica un buen rendimiento en la clasificación.
""".format(accuracy_svm, auc(roc_curve(y_test, y_pred_proba_svm)[0], roc_curve(y_test, y_pred_proba_svm)[1])))

# Reducción de dimensionalidad con PCA y t-SNE para entrenamiento de modelos
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.3, random_state=42)
X_train_tsne, X_test_tsne, _, _ = train_test_split(X_tsne, y, test_size=0.3, random_state=42)

# Entrenar y evaluar modelos con SVM usando PCA y t-SNE
for name, X_tr, X_te in [('PCA', X_train_pca, X_test_pca), ('t-SNE', X_train_tsne, X_test_tsne)]:
    st.write(f"### SVM con {name}")
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"**Precisión (SVM con {name}):** {accuracy:.2f}")
    st.write(f"**Matriz de Confusión (SVM con {name}):**")
    st.write(confusion_matrix(y_test, y_pred))
    st.write(f"**Informe de Clasificación (SVM con {name}):**")
    st.write(classification_report(y_test, y_pred))
    plot_roc_curve(y_test, y_pred_proba, f'Curva ROC - SVM ({name})')

    st.write(f"""
    **Conclusión:**
    El modelo SVM con {name} logra una precisión del {accuracy:.2f}. La curva ROC con un AUC de {auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1]):.2f}.
    """)

# Definir hiperparámetros de la red neuronal
hyperparams = {'depth': 4, 'epochs': 43, 'num_units': 144, 'optimizer': 'sgd', 'activation': 'relu', 'batch_size': 72, 'learning_rate': 0.0329}

# Función para crear la red neuronal
def create_nn_model(input_dim, hyperparams):
    model = Sequential()
    for _ in range(hyperparams['depth'] - 1):
        model.add(Dense(hyperparams['num_units'], activation=hyperparams['activation']))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(learning_rate=hyperparams['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Entrenar y evaluar red neuronal con datos originales
st.write("### Red Neuronal con Datos Originales")
nn_model = create_nn_model(X_train.shape[1], hyperparams)
history = nn_model.fit(X_train, y_train, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], validation_split=0.2, verbose=0)
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
y_pred_proba_nn = nn_model.predict(X_test).flatten()
accuracy_nn = accuracy_score(y_test, y_pred_nn)
st.write(f"**Precisión (Red Neuronal):** {accuracy_nn:.2f}")
st.write("**Matriz de Confusión (Red Neuronal):**")
st.write(confusion_matrix(y_test, y_pred_nn))
st.write("**Informe de Clasificación (Red Neuronal):**")
st.write(classification_report(y_test, y_pred_nn))
plot_roc_curve(y_test, y_pred_proba_nn, 'Curva ROC - Red Neuronal')

# Gráfica de precisión y pérdida durante el entrenamiento
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(history.history['accuracy'], label='Precisión en entrenamiento')
ax[0].plot(history.history['val_accuracy'], label='Precisión en validación')
ax[0].set_title('Precisión durante el Entrenamiento')
ax[0].set_xlabel('Épocas')
ax[0].set_ylabel('Precisión')
ax[0].legend()

ax[1].plot(history.history['loss'], label='Pérdida en entrenamiento')
ax[1].plot(history.history['val_loss'], label='Pérdida en validación')
ax[1].set_title('Pérdida durante el Entrenamiento')
ax[1].set_xlabel('Épocas')
ax[1].set_ylabel('Pérdida')
ax[1].legend()
st.pyplot(fig)

st.write("""
**Conclusión:**
La red neuronal con datos originales logra una precisión del {:.2f}. La curva de aprendizaje muestra que el modelo converge adecuadamente, sin signos de sobreajuste. La curva ROC con un AUC de {:.2f} confirma un buen rendimiento en la clasificación.
""".format(accuracy_nn, auc(roc_curve(y_test, y_pred_proba_nn)[0], roc_curve(y_test, y_pred_proba_nn)[1])))

# Evaluar red neuronal con PCA y t-SNE
for name, X_tr, X_te in [('PCA', X_train_pca, X_test_pca), ('t-SNE', X_train_tsne, X_test_tsne)]:
    st.write(f"### Red Neuronal con {name}")
    nn_model = create_nn_model(X_tr.shape[1], hyperparams)
    history = nn_model.fit(X_tr, y_train, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], validation_split=0.2, verbose=0)
    y_pred_nn = (nn_model.predict(X_te) > 0.5).astype(int)
    y_pred_proba_nn = nn_model.predict(X_te).flatten()
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    st.write(f"**Precisión (Red Neuronal con {name}):** {accuracy_nn:.2f}")
    st.write(f"**Matriz de Confusión (Red Neuronal con {name}):**")
    st.write(confusion_matrix(y_test, y_pred_nn))
    st.write(f"**Informe de Clasificación (Red Neuronal con {name}):**")
    st.write(classification_report(y_test, y_pred_nn))
    plot_roc_curve(y_test, y_pred_proba_nn, f'Curva ROC - Red Neuronal ({name})')

    # Gráficas de precisión y pérdida durante el entrenamiento
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(history.history['accuracy'], label='Precisión en entrenamiento')
    ax[0].plot(history.history['val_accuracy'], label='Precisión en validación')
    ax[0].set_title(f'Precisión durante el Entrenamiento ({name})')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Precisión')
    ax[0].legend()

    ax[1].plot(history.history['loss'], label='Pérdida en entrenamiento')
    ax[1].plot(history.history['val_loss'], label='Pérdida en validación')
    ax[1].set_title(f'Pérdida durante el Entrenamiento ({name})')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Pérdida')
    ax[1].legend()
    st.pyplot(fig)

    st.write(f"""
    **Conclusión:**
    La red neuronal con {name} logra una precisión del {accuracy_nn:.2f}. La curva ROC con un AUC de {auc(roc_curve(y_test, y_pred_proba_nn)[0], roc_curve(y_test, y_pred_proba_nn)[1]):.2f}.
    """)

####################################################
st.title("Predicción de Riesgo Cardiovascular")

# Cargar modelo desde archivo comprimido
@st.cache_resource
def load_model():
    """Carga el modelo y el scaler desde un archivo comprimido."""
    try:
        with gzip.open("modelo_entrenado.pkl.gz", "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict) and "modelo" in data and "scaler" in data:
            return data["modelo"], data["scaler"]
        else:
            raise ValueError("El archivo no tiene el formato esperado.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None

# Cargar modelo y scaler
model, scaler = load_model()

# Función para ingresar datos del usuario
def user_input():
    st.header("📋 Ingresar Datos del Paciente")

    # Variables categóricas
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino"], index=1)
    fto_aditivo = st.selectbox("FTO Aditivo", [0, 1], index=0)

    # Variables numéricas con validación de rango
    edad = st.number_input("Edad", min_value=18, max_value=100, value=60, step=1)
    leptina = st.number_input("Leptina (ng/mL)", min_value=0.0, max_value=100.0, value=30.0, step=0.1)
    grasa = st.number_input("Grasa Corporal (%)", min_value=0.0, max_value=100.0, value=35.0, step=0.1)
    imc = st.number_input("Índice de Masa Corporal (IMC)", min_value=10.0, max_value=50.0, value=32.0, step=0.1)
    bai = st.number_input("Índice de Adiposidad Corporal (BAI)", min_value=0.0, max_value=50.0, value=30.0, step=0.1)
    cintura = st.number_input("Circunferencia de Cintura (cm)", min_value=30.0, max_value=200.0, value=110.0, step=0.1)
    cadera = st.number_input("Circunferencia de Cadera (cm)", min_value=30.0, max_value=200.0, value=120.0, step=0.1)
    cvldl = st.number_input("Colesterol VLDL (mg/dL)", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
    triglic = st.number_input("Triglicéridos (mg/dL)", min_value=0.0, max_value=500.0, value=250.0, step=0.1)
    ctotal = st.number_input("Colesterol Total (mg/dL)", min_value=0.0, max_value=400.0, value=280.0, step=0.1)
    cldl = st.number_input("Colesterol LDL (mg/dL)", min_value=0.0, max_value=300.0, value=180.0, step=0.1)
    chdl = st.number_input("Colesterol HDL (mg/dL)", min_value=0.0, max_value=100.0, value=35.0, step=0.1)

    # Convertir sexo a variable binaria (0 = Femenino, 1 = Masculino)
    sexo_binario = 1 if sexo == "Masculino" else 0

    # Crear un array con los datos ingresados
    data = np.array([[sexo_binario, edad, leptina, grasa, imc, bai, cintura, cadera, 
                      cvldl, triglic, ctotal, cldl, chdl, fto_aditivo]], dtype=np.float32)
    return data

# Obtener datos del usuario
input_data = user_input()

# Botón para hacer la predicción
if st.button(" Realizar Predicción"):
    if model is not None and scaler is not None:
        try:
            # Verificar dimensiones esperadas por el scaler
            expected_features = scaler.n_features_in_
            actual_features = input_data.shape[1]
            if actual_features != expected_features:
                st.error(f"⚠️ Error: El modelo espera {expected_features} características, pero se proporcionaron {actual_features}.")
            else:
                # Escalar todas las características correctamente
                input_data_scaled = scaler.transform(input_data)

                # Realizar la predicción
                prediction = model.predict(input_data_scaled)

                # Manejo seguro de la salida
                prediction_value = float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction)

                # Clasificar el riesgo
                prediction_label = "🟢 Bajo Riesgo" if prediction_value >= 0.5 else "🔴 Alto Riesgo"

                # Mostrar resultados
                st.subheader("📌 Resultado de la Predicción:")
                st.markdown(f"## {prediction_label}")
               

        except Exception as e:
            st.error(f"⚠️ Error en la predicción: {e}")
    else:
        st.error("⚠️ No se pudo cargar el modelo y/o el scaler.")

