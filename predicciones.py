import pandas as pd
import re
import nltk
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Función para preprocesar el título
def preprocesar_titulo(titulo):
    titulo = str(titulo).lower()
    titulo = re.sub(r"http\S+|www\S+|@\S+", "", titulo)  
    titulo = re.sub(r"[^\w\s]", "", titulo)  
    tokens = word_tokenize(titulo)
    tokens = [t for t in tokens if t not in stopwords.words('english')]  
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]  
    return " ".join(tokens)

# Cargar el modelo y el vectorizador
def cargar_modelo_y_vectorizador():
    modelo_random_forest = joblib.load('models/modelo_random_forest.pkl')  
    vectorizer = joblib.load('models/vectorizer.pkl')  
    return modelo_random_forest, vectorizer

# Función para hacer predicciones
def hacer_predicciones(df_nuevos_datos, modelo_random_forest, vectorizer):
    df_nuevos_datos['titulo_procesado'] = df_nuevos_datos['titulo'].apply(preprocesar_titulo)
    
    X_nuevos_vect = vectorizer.transform(df_nuevos_datos['titulo_procesado'])
    
    predicciones = modelo_random_forest.predict(X_nuevos_vect)
    
    df_nuevos_datos['prediccion_sentimiento'] = predicciones
    return df_nuevos_datos

def main():
    df_nuevos_datos = pd.read_csv("data/nuevos_datos_limpios.csv")  
    
    modelo_random_forest, vectorizer = cargar_modelo_y_vectorizador()
    
    df_nuevos_datos_con_predicciones = hacer_predicciones(df_nuevos_datos, modelo_random_forest, vectorizer)
    
    df_nuevos_datos_con_predicciones.to_csv('data/predicciones.csv', index=False)
    print("Predicciones guardadas en 'predicciones.csv'")

if __name__ == '__main__':
    main()