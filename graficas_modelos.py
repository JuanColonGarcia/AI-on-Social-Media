import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Cargar el DataFrame procesado
df = pd.read_csv('reddit_posts.csv')

# Agrupar por fecha y calcular el sentimiento promedio del título
sentimiento_por_fecha = df.groupby(df['fecha'].dt.date)['sentimiento_titulo'].mean().reset_index()

# Visualizar la evolución del sentimiento
plt.figure(figsize=(12, 6))
sns.lineplot(x='fecha', y='sentimiento_titulo', data=sentimiento_por_fecha)
plt.title('Evolución del Sentimiento hacia IA en Reddit (basado en títulos)')
plt.xlabel('Fecha')
plt.ylabel('Sentimiento Promedio del Título')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Función para generar WordCloud
def generar_wordcloud(palabras, titulo):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(palabras))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(titulo)
    plt.tight_layout()
    plt.show()

# Separar tokens por sentimiento
tokens_neg = []
tokens_neu = []
tokens_pos = []

for _, row in df.iterrows():
    tokens = word_tokenize(row['titulo_procesado'])
    if row['clasificacion_titulo'] == 'Negativo':
        tokens_neg.extend(tokens)
    elif row['clasificacion_titulo'] == 'Neutral':
        tokens_neu.extend(tokens)
    elif row['clasificacion_titulo'] == 'Positivo':
        tokens_pos.extend(tokens)

# Remover stopwords para nubes de palabras
stop_words = set(stopwords.words('english'))
tokens_neg = [t for t in tokens_neg if t not in stop_words]
tokens_neu = [t for t in tokens_neu if t not in stop_words]
tokens_pos = [t for t in tokens_pos if t not in stop_words]

# Mostrar WordClouds
generar_wordcloud(tokens_pos, 'Palabras Clave - Sentimiento Positivo')
generar_wordcloud(tokens_neu, 'Palabras Clave - Sentimiento Neutral')
generar_wordcloud(tokens_neg, 'Palabras Clave - Sentimiento Negativo')
