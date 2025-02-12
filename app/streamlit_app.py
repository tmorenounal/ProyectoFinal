import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle, gzip
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# ================================
# Configuración General
# ================================

CONFIG = {
    'data_path': 'BancoXavantes837.xlsx',
    'model_path': 'modelo_entrenado.pkl.gz',
    'umbral_factor': 0.5,
    'nn_hyperparams': {
        'depth': 4,
        'epochs': 43,
        'num_units': 144,
        'optimizer': 'sgd',
        'activation': 'relu',
        'batch_size': 72,
        'learning_rate': 0.0329,
    },
    'pesos': {
        'CTOTAL': 0.2, 'CLDL': 0.3, 'CHDL': -0.2, 'Triglic': 0.2, 'CVLDL': 0.1,
        'IMC': 0.15, 'BAI': 0.1, 'Cintura': 0.15, 'Cadera': -0.1, 'Grasa': 0.1,
        'Edad': 0.2, 'Leptina': 0.05, 'FTO_Aditivo': 0.05
    }
}

# ================================
# Funciones de Carga y Preprocesamiento
# ================================

def load_data(path):
    """Carga la base de datos desde un archivo Excel y limpia los nombres de las columnas."""
    try:
        data = pd.read_excel(path)
        data.columns = data.columns.str.strip()
        return data
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

@st.cache_data
def get_data():
    data = load_data(CONFIG['data_path'])
    if data is None:
        st.stop()
    return data

def calculate_riesgo(data, pesos, umbral_factor=0.5):
    """Calcula el índice de riesgo cardiovascular y define la variable binaria."""
    missing_columns = [col for col in pesos.keys() if col not in data.columns]
    if missing_columns:
        st.error(f"Faltan las siguientes columnas: {missing_columns}")
        st.stop()
    # Cálculo único del índice
    data['Riesgo_Cardiovascular'] = data[list(pesos.keys())].mul(pesos).sum(axis=1)
    umbral = data['Riesgo_Cardiovascular'].max() * umbral_factor
    data['Riesgo_Cardiovascular_Binario'] = (data['Riesgo_Cardiovascular'] > umbral).astype(int)
    return data, umbral

def preprocess_data(data):
    """
    Preprocesa los datos:
    - Elimina columnas irrelevantes.
    - Convierte la variable 'Sexo' en dummy.
    - Escala los datos numéricos.
    """
    X = data.drop(columns=['IID', 'Riesgo_Cardiovascular', 'Riesgo_Cardiovascular_Binario'], errors='ignore')
    y = data['Riesgo_Cardiovascular_Binario']
    X = pd.get_dummies(X, columns=['Sexo'], drop_first=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# ================================
# Funciones de Visualización
# ================================

def plot_histogram(data, column, title, xlabel):
    fig, ax = plt.subplots()
    sns.histplot(data[column], bins=30, kde=True, ax=ax, color='blue')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frecuencia')
    st.pyplot(fig)

def plot_dimension_reduction(X_pca, X_tsne, y):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='coolwarm', ax=axs[0])
    axs[0].set_title("PCA")
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette='coolwarm', ax=axs[1])
    axs[1].set_title("t-SNE")
    st.pyplot(fig)

def plot_roc_curve(y_true, y_pred_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.2f})', color='blue', lw=2)
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    st.pyplot(fig)

# ================================
# Funciones para Entrenar Modelos
# ================================

def train_svm(X_train, y_train, X_test, y_test):
    """Entrena un modelo SVM y retorna predicciones y métricas."""
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_pred_proba = svm.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    return svm, y_pred, y_pred_proba, acc

def create_nn_model(input_dim, hyperparams):
    """Crea y compila una red neuronal según los hiperparámetros definidos."""
    model = Sequential()
    for _ in range(hyperparams['depth'] - 1):
        model.add(Dense(hyperparams['num_units'], activation=hyperparams['activation']))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = SGD(learning_rate=hyperparams['learning_rate'])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_nn(X_train, y_train, X_test, y_test, hyperparams):
    """Entrena la red neuronal y retorna el modelo, el historial y métricas."""
    model = create_nn_model(X_train.shape[1], hyperparams)
    history = model.fit(X_train, y_train, epochs=hyperparams['epochs'], 
                        batch_size=hyperparams['batch_size'], validation_split=0.2, verbose=0)
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    y_pred_proba = model.predict(X_test).flatten()
    acc = accuracy_score(y_test, y_pred)
    return model, history, y_pred, y_pred_proba, acc

# ================================
# Función para Cargar Modelo Pre-entrenado
# ================================

@st.cache_resource
def load_pretrained_model(path):
    try:
        with gzip.open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict) and "modelo" in data and "scaler" in data:
            return data["modelo"], data["scaler"]
        else:
            raise ValueError("Formato del modelo incorrecto.")
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None, None

# ================================
# Función para Ingresar Datos del Usuario (Predicción)
# ================================

def user_input():
    st.header("📋 Ingresar Datos del Paciente")
    sexo = st.selectbox("Sexo", ["Femenino", "Masculino"], index=1)
    fto_aditivo = st.selectbox("FTO Aditivo", [0, 1], index=0)
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
    sexo_binario = 1 if sexo == "Masculino" else 0
    input_data = np.array([[sexo_binario, edad, leptina, grasa, imc, bai, cintura, cadera, 
                             cvldl, triglic, ctotal, cldl, chdl, fto_aditivo]], dtype=np.float32)
    return input_data

# ================================
# Aplicación Principal con Pestañas (Tabs)
# ================================

st.sidebar.title("Navegación")
tabs = st.tabs(["Análisis Exploratorio", "Modelos Ajustados", "Predicción"])

# ----- TAB 1: Análisis Exploratorio -----
with tabs[0]:
    st.header("Análisis Exploratorio")
    data = get_data()
    data, umbral = calculate_riesgo(data, CONFIG['pesos'], CONFIG['umbral_factor'])
    
    st.subheader("Vista Previa de Datos")
    st.dataframe(data.head(5))
    
    st.subheader("Estadísticas Descriptivas")
    st.write(data.describe())
    
    st.markdown(f"### Umbral utilizado: *{umbral:.2f}*")
    st.subheader("Balance de Clases")
    st.write(data['Riesgo_Cardiovascular_Binario'].value_counts())
    
    st.subheader("Histograma de Riesgo Cardiovascular")
    plot_histogram(data, 'Riesgo_Cardiovascular', 'Distribución del Índice de Riesgo', 'Riesgo Cardiovascular')
    
    st.subheader("Matriz de Correlación")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Matriz de Correlación")
    st.pyplot(fig)
    
    # Distribución de Variables Numéricas
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    if numerical_cols:
        st.subheader("Distribución de Variables Numéricas")
        selected_vars = st.multiselect("Selecciona variables", numerical_cols, default=numerical_cols[:2])
        for var in selected_vars:
            plot_histogram(data, var, f'Distribución de {var}', var)
    
    # Reducción de Dimensionalidad
    X_scaled, y, _ = preprocess_data(data)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    st.subheader("Reducción de Dimensionalidad")
    plot_dimension_reduction(X_pca, X_tsne, y)

# ----- TAB 2: Modelos Ajustados -----
with tabs[1]:
    st.header("Modelos Ajustados")
    data = get_data()
    data, umbral = calculate_riesgo(data, CONFIG['pesos'], CONFIG['umbral_factor'])
    X_scaled, y, _ = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # SVM con Datos Originales
    st.subheader("SVM con Datos Originales")
    _, y_pred_svm, y_pred_proba_svm, acc_svm = train_svm(X_train, y_train, X_test, y_test)
    st.write(f"Precisión: *{acc_svm:.2f}*")
    st.write("Matriz de Confusión")
    st.write(confusion_matrix(y_test, y_pred_svm))
    st.write("Reporte de Clasificación")
    st.write(classification_report(y_test, y_pred_svm))
    plot_roc_curve(y_test, y_pred_proba_svm, "Curva ROC - SVM (Datos Originales)")
    
    # SVM con PCA y t-SNE
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    X_train_tsne, X_test_tsne, _, _ = train_test_split(X_tsne, y, test_size=0.3, random_state=42)
    
    for name, X_tr, X_te in [('PCA', X_train_pca, X_test_pca), ('t-SNE', X_train_tsne, X_test_tsne)]:
        st.subheader(f"SVM con {name}")
        _, y_pred, y_pred_proba, acc = train_svm(X_tr, y_train, X_te, y_test)
        st.write(f"Precisión: *{acc:.2f}*")
        st.write("Matriz de Confusión")
        st.write(confusion_matrix(y_test, y_pred))
        st.write("Reporte de Clasificación")
        st.write(classification_report(y_test, y_pred))
        plot_roc_curve(y_test, y_pred_proba, f"Curva ROC - SVM ({name})")
    
    # Red Neuronal con Datos Originales
    st.subheader("Red Neuronal con Datos Originales")
    nn_model, history_nn, y_pred_nn, y_pred_proba_nn, acc_nn = train_nn(X_train, y_train, X_test, y_test, CONFIG['nn_hyperparams'])
    st.write(f"Precisión: *{acc_nn:.2f}*")
    st.write("Matriz de Confusión")
    st.write(confusion_matrix(y_test, y_pred_nn))
    st.write("Reporte de Clasificación")
    st.write(classification_report(y_test, y_pred_nn))
    plot_roc_curve(y_test, y_pred_proba_nn, "Curva ROC - Red Neuronal (Datos Originales)")
    
    st.subheader("Curvas de Entrenamiento - Red Neuronal")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(history_nn.history['accuracy'], label='Entrenamiento')
    axs[0].plot(history_nn.history['val_accuracy'], label='Validación')
    axs[0].set_title("Precisión durante el Entrenamiento")
    axs[0].set_xlabel("Épocas")
    axs[0].set_ylabel("Precisión")
    axs[0].legend()
    axs[1].plot(history_nn.history['loss'], label='Entrenamiento')
    axs[1].plot(history_nn.history['val_loss'], label='Validación')
    axs[1].set_title("Pérdida durante el Entrenamiento")
    axs[1].set_xlabel("Épocas")
    axs[1].set_ylabel("Pérdida")
    axs[1].legend()
    st.pyplot(fig)
    
    # Red Neuronal con PCA y t-SNE
    for name, X_tr, X_te in [('PCA', X_train_pca, X_test_pca), ('t-SNE', X_train_tsne, X_test_tsne)]:
        st.subheader(f"Red Neuronal con {name}")
        nn_model, history_nn, y_pred_nn, y_pred_proba_nn, acc_nn = train_nn(X_tr, y_train, X_te, y_test, CONFIG['nn_hyperparams'])
        st.write(f"Precisión: *{acc_nn:.2f}*")
        st.write("Matriz de Confusión")
        st.write(confusion_matrix(y_test, y_pred_nn))
        st.write("Reporte de Clasificación")
        st.write(classification_report(y_test, y_pred_nn))
        plot_roc_curve(y_test, y_pred_proba_nn, f"Curva ROC - Red Neuronal ({name})")
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].plot(history_nn.history['accuracy'], label='Entrenamiento')
        axs[0].plot(history_nn.history['val_accuracy'], label='Validación')
        axs[0].set_title(f"Precisión durante el Entrenamiento ({name})")
        axs[0].set_xlabel("Épocas")
        axs[0].set_ylabel("Precisión")
        axs[0].legend()
        axs[1].plot(history_nn.history['loss'], label='Entrenamiento')
        axs[1].plot(history_nn.history['val_loss'], label='Validación')
        axs[1].set_title(f"Pérdida durante el Entrenamiento ({name})")
        axs[1].set_xlabel("Épocas")
        axs[1].set_ylabel("Pérdida")
        axs[1].legend()
        st.pyplot(fig)

# ----- TAB 3: Predicción -----
with tabs[2]:
    st.header("Predicción de Riesgo Cardiovascular")
    model_pretrained, scaler_pretrained = load_pretrained_model(CONFIG['model_path'])
    input_data = user_input()
    if st.button("Realizar Predicción"):
        if model_pretrained is not None and scaler_pretrained is not None:
            try:
                if input_data.shape[1] != scaler_pretrained.n_features_in_:
                    st.error(f"El modelo espera {scaler_pretrained.n_features_in_} características, pero se proporcionaron {input_data.shape[1]}.")
                else:
                    input_data_scaled = scaler_pretrained.transform(input_data)
                    prediction = model_pretrained.predict(input_data_scaled)
                    prediction_value = float(prediction[0]) if isinstance(prediction, np.ndarray) else float(prediction)
                    result = "🟢 Bajo Riesgo" if prediction_value >= 0.5 else "🔴 Alto Riesgo"
                    st.subheader("Resultado de la Predicción")
                    st.markdown(f"## {result}")
            except Exception as e:
                st.error(f"Error en la predicción: {e}")
        else:
            st.error("No se pudo cargar el modelo y/o el escalador.")
