import pandas as pd
import re
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔽 Descargar datos de NLTK si es la primera vez
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Función para preprocesar el título
def preprocesar_titulo(titulo):
    titulo = str(titulo).lower()
    titulo = re.sub(r"http\S+|www\S+|@\S+", "", titulo)  # Eliminar URLs y menciones
    titulo = re.sub(r"[^\w\s]", "", titulo)  # Eliminar caracteres especiales
    tokens = word_tokenize(titulo)
    tokens = [t for t in tokens if t not in stopwords.words('english')]  # Eliminar stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  # Lemmatizar
    return " ".join(tokens)

# 🔄 Cargar el modelo y el vectorizador
def cargar_modelo_y_vectorizador():
    modelo_random_forest = joblib.load('modelo_random_forest.pkl')  # Cargar el modelo guardado
    vectorizer = joblib.load('vectorizer.pkl')  # Cargar el vectorizador
    return modelo_random_forest, vectorizer

# 🔄 Función para hacer predicciones
def hacer_predicciones(df_nuevos_datos, modelo_random_forest, vectorizer):
    # Preprocesar los títulos de los nuevos datos
    df_nuevos_datos['titulo_procesado'] = df_nuevos_datos['titulo'].apply(preprocesar_titulo)
    
    # Transformar los títulos procesados con el vectorizador
    X_nuevos_vect = vectorizer.transform(df_nuevos_datos['titulo_procesado'])
    
    # Realizar las predicciones con el modelo cargado
    predicciones = modelo_random_forest.predict(X_nuevos_vect)
    
    # Añadir las predicciones al dataframe
    df_nuevos_datos['prediccion_sentimiento'] = predicciones
    return df_nuevos_datos

# 🖥 Función principal que ejecuta todo
def main():
    # 🔄 Cargar los nuevos datos
    df_nuevos_datos = pd.read_csv("nuevos_datos.csv")  # Asegúrate de tener este archivo
    
    # 🔄 Cargar el modelo y el vectorizador
    modelo_random_forest, vectorizer = cargar_modelo_y_vectorizador()
    
    # 🔄 Hacer las predicciones
    df_nuevos_datos_con_predicciones = hacer_predicciones(df_nuevos_datos, modelo_random_forest, vectorizer)
    
    # 🔄 Guardar las predicciones en un archivo CSV
    df_nuevos_datos_con_predicciones.to_csv('predicciones.csv', index=False)
    print("Predicciones guardadas en 'predicciones.csv'")

if __name__ == '__main__':
    main()
