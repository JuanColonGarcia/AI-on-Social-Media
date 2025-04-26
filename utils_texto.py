import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Descarga de recursos (solo se ejecuta una vez)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
analyzer = SentimentIntensityAnalyzer()

def preprocesar_titulo(titulo):
    titulo = str(titulo).lower()
    titulo = re.sub(r"http\S+|www\S+|@\S+", "", titulo)
    titulo = re.sub(r'[^\w\s]', '', titulo)
    tokens = word_tokenize(titulo)
    tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def analizar_sentimiento_completo(titulo_procesado):
    scores = analyzer.polarity_scores(titulo_procesado)
    return scores['compound']
