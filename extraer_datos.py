import praw
import pandas as pd
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Conexión con la API de Reddit
reddit = praw.Reddit(
    client_id='BeOrSEQopbVXbBDqAi5-cQ',
    client_secret='uTiRYEUGlUdRsHqWTZGkUcZpCvrH8g',
    user_agent='IA_sentiment_app by /u/Jumpy_Tie_1645',
    username='Jumpy_Tie_1645',
    password='12345BGL.'
)

# Función asíncrona para extraer los posts de un subreddit
async def extraer_posts_async(subreddit_name, keyword, limite=100):
    reddit = asyncpraw.Reddit(
        client_id='BeOrSEQopbVXbBDqAi5-cQ',
        client_secret='uTiRYEUGlUdRsHqWTZGkUcZpCvrH8g',
        user_agent='IA_sentiment_app by /u/Jumpy_Tie_1645',
        username='Jumpy_Tie_1645',
        password='12345BGL.'
    )
    posts = []
    subreddit = await reddit.subreddit(subreddit_name)
    async for post in subreddit.search(keyword, limit=limite):
        posts.append({
            'titulo': post.title,
            'texto': post.selftext,
            'fecha': post.created_utc,
            'score': post.score
        })
    await reddit.close() 
    return posts


# Función síncrona para extraer posts de un subreddit
def extraer_posts(subreddit_name, keyword, limite=500):
    posts = []
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.search(keyword, limit=limite):
        posts.append({
            'titulo': post.title,
            'texto': post.selftext,
            'fecha': post.created_utc,
            'score': post.score
        })
    return posts

# Función síncrona para extraer posts con palabras clave de opinión
def extraer_posts_con_opinion(subreddit_name, keywords, limite=1000):
    posts = []
    subreddit = reddit.subreddit(subreddit_name)
    for keyword in keywords:
        for post in subreddit.search(keyword, limit=limite):
            posts.append({
                'titulo': post.title,
                'texto': post.selftext,
                'fecha': post.created_utc,
                'score': post.score
            })
    return posts


# Palabras clave para buscar opiniones más polarizadas
keywords_opinion = [
    "ChatGPT helps students", "ChatGPT can be dangerous", "ChatGPT improves communication", "ChatGPT makes mistakes",
    "ChatGPT can't replace humans", "ChatGPT saves time", "ChatGPT is overrated", "ChatGPT is addictive",
    "ChatGPT enhances learning", "ChatGPT should be banned", "ChatGPT needs regulation", "Gemini is better than ChatGPT",
    "Gemini saves time", "Gemini is creative", "Gemini helps students", "Gemini improves productivity",
    "Gemini enhances creativity", "Gemini should be banned", "Gemini is disappointing", "Copilot improves productivity",
    "Copilot makes coding easier", "Copilot is innovative", "Copilot is overrated", "Copilot has potential",
    "Copilot is not accurate", "Copilot saves time", "AI threatens creative jobs", "AI helps with homework",
    "AI is the end of creativity", "AI will destroy education", "AI is misunderstood", "AI can boost learning",
    "AI creates new possibilities", "AI helps in healthcare", "AI threatens artists", "AI helps in art",
    "AI is for lazy people", "AI enhances writing", "AI is replacing writers", "AI needs regulation",
    "AI threatens freedom", "AI enhances creativity", "AI creates more problems", "AI improves customer experience",
    "AI affects mental health", "AI should not replace people", "AI is not fair", "AI affects social interaction",
    "AI is misused", "AI threatens journalism", "AI replaces basic tasks"
]


subreddits = ['technology', 'Gemini', 'MachineLearning', 'ChatGPT']
posts = [] 
for subreddit in subreddits:
    posts.extend(extraer_posts_con_opinion(subreddit, keywords_opinion, 5000))

df = pd.DataFrame(posts)
df['fecha'] = pd.to_datetime(df['fecha'], unit='s')
fecha_limite = datetime.utcnow() - timedelta(days=1825)
df = df[df['fecha'] >= fecha_limite]

print(f"Posts después del filtro por fecha ({len(df)} en total):")
print(df['fecha'].dt.date.value_counts().sort_index())

# Preprocesamiento del texto
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
print(df.head())

df.to_csv("data/reddit_posts.csv", index=False)