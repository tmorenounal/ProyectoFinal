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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


st.sidebar.title("Navegación")
st.sidebar.markdown("---")

# Submenú para Análisis de Datos
with st.sidebar.expander("Análisis de Datos", expanded=True):
    section_analisis = st.radio(
        "Selecciona una sección:",
        ["Análisis Exploratorio"],
        key="analisis_radio"  # clave única asignada
    )

# Submenú para Modelos de Machine Learning
with st.sidebar.expander("Modelos Ajustados", expanded=True):
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
        key="modelos_radio"  # clave única asignada
    )

# Submenú para Predicciones
with st.sidebar.expander("Predicciones", expanded=True):
    section_predicciones = st.radio(
        "Selecciona una sección:",
        ["Predicción de Riesgo Cardiovascular"],
        key="predicciones_radio"  # clave única asignada
    )



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

# Calcular el índice de riesgo cardiovascular (se realiza una única vez)
data['Riesgo_Cardiovascular'] = data[list(pesos.keys())].mul(pesos).sum(axis=1)

# Definir un umbral fijo basado en una fracción del máximo
umbral = data['Riesgo_Cardiovascular'].max() * 0.5  # Ajustar según necesidad

# Crear variable binaria de riesgo cardiovascular
data['Riesgo_Cardiovascular_Binario'] = (data['Riesgo_Cardiovascular'] > umbral).astype(int)



st.title('Análisis de Enfermedades Cardiovasculares en la Población Indígena Xavante de Brasil')
st.write("""
La siguiente base de datos pertenece a una población de indígenas Xavantes de Brasil, 
la cual cuenta con variables importantes para determinar enfermedades cardiovasculares en la población.  
La base de datos incluye las siguientes variables:

- Sexo: Género de los individuos (hombre o mujer).
- Edad: Edad en años.
- Leptina: Nivel de leptina, una hormona relacionada con la regulación del apetito y el metabolismo.
- Grasa: Porcentaje de grasa corporal.
- IMC: Índice de Masa Corporal, una medida de la relación entre peso y altura.
- BAI: Índice de Adiposidad Corporal, una medida alternativa al IMC.
- Cintura: Circunferencia de la cintura en centímetros.
- Cadera: Circunferencia de la cadera en centímetros.
- CVLDL: Colesterol de lipoproteínas de muy baja densidad.
- Triglic: Nivel de triglicéridos en sangre.
- CTOTAL: Colesterol total.
- CLDL: Colesterol de lipoproteínas de baja densidad (colesterol "malo").
- CHDL: Colesterol de lipoproteínas de alta densidad (colesterol "bueno").
- FTO_Aditivo: Variante genética asociada con la obesidad y el riesgo cardiovascular.
""")

st.write("### Vista previa de los datos")
st.dataframe(data.head(5))

st.write("### Información de los datos")
st.write(data.describe())

st.write("""
### Variable Objetivo: Riesgo Cardiovascular

La variable objetivo de este estudio es el Riesgo Cardiovascular, que se determina en función de umbrales establecidos a partir de los valores calculados:

- Se utiliza un umbral de {:.2f} (50% del valor máximo) para clasificar el riesgo.
- 0 (Bajo riesgo): El individuo no supera el umbral.
- 1 (Alto riesgo): El individuo supera el umbral.
""".format(umbral))

st.write("#### Balance de Clases en la Variable Objetivo")
st.write(data['Riesgo_Cardiovascular_Binario'].value_counts())

st.write("### Histograma de la Distribución del Riesgo Cardiovascular")
fig, ax = plt.subplots()
sns.histplot(data['Riesgo_Cardiovascular'], bins=30, kde=True, ax=ax, color='blue')
ax.set_title('Distribución del Índice de Riesgo Cardiovascular')
ax.set_xlabel('Riesgo Cardiovascular')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)

st.write("### Distribución de Variables Numéricas")
numerical_columns = data.select_dtypes(include=['number']).columns.tolist()

if not numerical_columns:
    st.error("No se encontraron variables numéricas en el dataset.")
    st.stop()

selected_variables = st.multiselect(
    "Selecciona las variables numéricas que deseas visualizar:",
    numerical_columns,
    default=numerical_columns[:2]
)

if not selected_variables:
    st.warning("Por favor, selecciona al menos una variable numérica.")
    st.stop()

num_columns = 2
num_rows = (len(selected_variables) + num_columns - 1) // num_columns

for i in range(num_rows):
    cols = st.columns(num_columns)
    for j in range(num_columns):
        idx = i * num_columns + j
        if idx < len(selected_variables):
            variable = selected_variables[idx]
            with cols[j]:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(data[variable], bins=30, kde=True, ax=ax)
                ax.set_title(f'Distribución de {variable}')
                ax.set_xlabel(variable)
                ax.set_ylabel('Frecuencia')
                st.pyplot(fig)

st.write("#### Matriz de Correlación")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Matriz de Correlación')
st.pyplot(fig)



def preprocess_data(data):
    # Asegurarse de que los nombres de las columnas coincidan con los del archivo
    X = data.drop(columns=['IID', 'Riesgo_Cardiovascular', 'Riesgo_Cardiovascular_Binario'], errors='ignore')
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

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)



def plot_dimension_reduction(X_pca, X_tsne, y, title_pca, title_tsne):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', ax=ax[0])
    ax[0].set_title(title_pca)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='coolwarm', ax=ax[1])
    ax[1].set_title(title_tsne)
    st.pyplot(fig)

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



st.write("### Reducción de Dimensionalidad")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)
plot_dimension_reduction(X_pca, X_tsne, y, 'PCA - Datos Originales', 't-SNE - Datos Originales')



st.write("## Modelos Ajustados")

# --- SVM con datos originales ---
st.write("### SVM con Datos Originales")
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
y_pred_proba_svm = svm_model.predict_proba(X_test)[:, 1]
accuracy_svm = accuracy_score(y_test, y_pred_svm)
st.write(f"Precisión (SVM): {accuracy_svm:.2f}")
st.write("Matriz de Confusión (SVM):")
st.write(confusion_matrix(y_test, y_pred_svm))
st.write("Informe de Clasificación (SVM):")
st.write(classification_report(y_test, y_pred_svm))
plot_roc_curve(y_test, y_pred_proba_svm, 'Curva ROC - SVM (Datos Originales)')

st.write(f"""
Conclusión:
El modelo SVM con datos originales logra una precisión del {accuracy_svm:.2f}. La curva ROC presenta un AUC de {auc(roc_curve(y_test, y_pred_proba_svm)[0], roc_curve(y_test, y_pred_proba_svm)[1]):.2f}.
""")

# --- SVM con PCA y t-SNE ---
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.3, random_state=42)
X_train_tsne, X_test_tsne, _, _ = train_test_split(X_tsne, y, test_size=0.3, random_state=42)

for name, X_tr, X_te in [('PCA', X_train_pca, X_test_pca), ('t-SNE', X_train_tsne, X_test_tsne)]:
    st.write(f"### SVM con {name}")
    model = SVC(kernel='linear', probability=True, random_state=42)
    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Precisión (SVM con {name}): {accuracy:.2f}")
    st.write("Matriz de Confusión:")
    st.write(confusion_matrix(y_test, y_pred))
    st.write("Informe de Clasificación:")
    st.write(classification_report(y_test, y_pred))
    plot_roc_curve(y_test, y_pred_proba, f'Curva ROC - SVM ({name})')
    st.write(f"""
    Conclusión:
    El modelo SVM con {name} logra una precisión del {accuracy:.2f}. La curva ROC presenta un AUC de {auc(roc_curve(y_test, y_pred_proba)[0], roc_curve(y_test, y_pred_proba)[1]):.2f}.
    """)

# --- Red Neuronal ---
hyperparams = {
    'depth': 4,
    'epochs': 43,
    'num_units': 144,
    'optimizer': 'sgd',
    'activation': 'relu',
    'batch_size': 72,
    'learning_rate': 0.0329
}

def create_nn_model(input_dim, hyperparams):
    model = Sequential()
    for _ in range(hyperparams['depth'] - 1):
        model.add(Dense(hyperparams['num_units'], activation=hyperparams['activation']))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(learning_rate=hyperparams['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Red Neuronal con Datos Originales
st.write("### Red Neuronal con Datos Originales")
nn_model = create_nn_model(X_train.shape[1], hyperparams)
history = nn_model.fit(X_train, y_train, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], validation_split=0.2, verbose=0)
y_pred_nn = (nn_model.predict(X_test) > 0.5).astype(int)
y_pred_proba_nn = nn_model.predict(X_test).flatten()
accuracy_nn = accuracy_score(y_test, y_pred_nn)
st.write(f"Precisión (Red Neuronal): {accuracy_nn:.2f}")
st.write("Matriz de Confusión (Red Neuronal):")
st.write(confusion_matrix(y_test, y_pred_nn))
st.write("Informe de Clasificación (Red Neuronal):")
st.write(classification_report(y_test, y_pred_nn))
plot_roc_curve(y_test, y_pred_proba_nn, 'Curva ROC - Red Neuronal')

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

st.write(f"""
Conclusión:
La red neuronal con datos originales logra una precisión del {accuracy_nn:.2f}. La curva ROC presenta un AUC de {auc(roc_curve(y_test, y_pred_proba_nn)[0], roc_curve(y_test, y_pred_proba_nn)[1]):.2f}.
""")

# Red Neuronal con PCA y t-SNE
for name, X_tr, X_te in [('PCA', X_train_pca, X_test_pca), ('t-SNE', X_train_tsne, X_test_tsne)]:
    st.write(f"### Red Neuronal con {name}")
    nn_model = create_nn_model(X_tr.shape[1], hyperparams)
    history = nn_model.fit(X_tr, y_train, epochs=hyperparams['epochs'], batch_size=hyperparams['batch_size'], validation_split=0.2, verbose=0)
    y_pred_nn = (nn_model.predict(X_te) > 0.5).astype(int)
    y_pred_proba_nn = nn_model.predict(X_te).flatten()
    accuracy_nn = accuracy_score(y_test, y_pred_nn)
    st.write(f"Precisión (Red Neuronal con {name}): {accuracy_nn:.2f}")
    st.write("Matriz de Confusión:")
    st.write(confusion_matrix(y_test, y_pred_nn))
    st.write("Informe de Clasificación:")
    st.write(classification_report(y_test, y_pred_nn))
    plot_roc_curve(y_test, y_pred_proba_nn, f'Curva ROC - Red Neuronal ({name})')
    
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
    Conclusión:
    La red neuronal con {name} logra una precisión del {accuracy_nn:.2f}. La curva ROC presenta un AUC de {auc(roc_curve(y_test, y_pred_proba_nn)[0], roc_curve(y_test, y_pred_proba_nn)[1]):.2f}.
    """)



st.title("Predicción de Riesgo Cardiovascular")

@st.cache_resource
def load_model():
    """Carga el modelo y el scaler desde un archivo comprimido."""
    try:
        with gzip.open("modelo_entrenado.pkl.gz", "rb") as f:
            data_model = pickle.load(f)
        if isinstance(data_model, dict) and "modelo" in data_model and "scaler" in data_model:
            return data_model["modelo"], data_model["scaler"]
        else:
            raise ValueError("El archivo no tiene el formato esperado.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None

model, scaler = load_model()

def user_input():
    st.header(" Ingresar Datos")

    # Variables categóricas
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino"], index=1)
    fto_aditivo = st.selectbox("FTO Aditivo", [0, 1], index=0)

    # Variables numéricas
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

    # Convertir Sexo a variable binaria (0 = Femenino, 1 = Masculino)
    sexo_binario = 1 if sexo == "Masculino" else 0

    # Crear array con los datos ingresados (asegurarse del orden que espera el modelo)
    input_data = np.array([[sexo_binario, edad, leptina, grasa, imc, bai, cintura, cadera, 
                            cvldl, triglic, ctotal, cldl, chdl, fto_aditivo]], dtype=np.float32)
    return input_data

input_data = user_input()

if st.button("Realizar Predicción"):
    if model is not None and scaler is not None:
        try:
            expected_features = scaler.n_features_in_
            actual_features = input_data.shape[1]
            if actual_features != expected_features:
                st.error(f"⚠ Error: El modelo espera {expected_features} características, pero se proporcionaron {actual_features}.")
            else:
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)
                prediction_value = float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction)
                prediction_label = "🟢 Bajo Riesgo" if prediction_value >= 0.5 else "🔴 Alto Riesgo"
                st.subheader("📌 Resultado de la Predicción:")
                st.markdown(f"## {prediction_label}")
        except Exception as e:
            st.error(f"⚠ Error en la predicción: {e}")
    else:
        st.error("⚠ No se pudo cargar el modelo y/o el scaler.")
