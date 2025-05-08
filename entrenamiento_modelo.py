import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_curve, auc
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

df = pd.read_csv("data/reddit_posts.csv")

# Preprocesar títulos
def preprocesar_titulo(titulo):
    titulo = str(titulo).lower()
    titulo = re.sub(r"http\S+|www\S+|@\S+", "", titulo)
    titulo = re.sub(r"[^\w\s]", "", titulo)
    tokens = word_tokenize(titulo)
    tokens = [t for t in tokens if t not in stopwords.words('english')]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

# Aplicar preprocesamiento si no existe
if 'titulo_procesado' not in df.columns:
    df['titulo_procesado'] = df['titulo'].apply(preprocesar_titulo)

# Calcular sentimiento si no existe
if 'sentimiento_titulo' not in df.columns:
    analyzer = SentimentIntensityAnalyzer()
    df['sentimiento_titulo'] = df['titulo_procesado'].apply(lambda t: analyzer.polarity_scores(t)['compound'])

# Clasificar sentimiento
df['clasificacion_titulo'] = pd.cut(df['sentimiento_titulo'],
                                     bins=[-1, -0.1, 0.1, 1],
                                     labels=['Negativo', 'Neutral', 'Positivo'])

# Eliminar posibles nulos
df = df.dropna(subset=['titulo_procesado', 'clasificacion_titulo'])

# Vectorización TF-IDF
X = df['titulo_procesado']
y = df['clasificacion_titulo']
vectorizer = TfidfVectorizer(max_features=3000)
X_vect = vectorizer.fit_transform(X)

# División
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, stratify=y, random_state=42)

# Modelos
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True)
}

# Entrenamiento y evaluación
for nombre, modelo in modelos.items():
    print(f"\n Modelo: {nombre}")
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=['Negativo', 'Neutral', 'Positivo'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negativo', 'Neutral', 'Positivo'],
                yticklabels=['Negativo', 'Neutral', 'Positivo'])
    plt.title(f'Matriz de Confusión - {nombre}')
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.show()

    # Calcular Curva ROC para cada clase
if nombre == "SVM":
    # Convertir las etiquetas de clase en formato binario (one-vs-rest) para la curva ROC
    y_test_bin = label_binarize(y_test, classes=['Negativo', 'Neutral', 'Positivo'])
    y_prob_svm = modelo.predict_proba(X_test) 

    # Calcular el AUC para cada clase
    fpr, tpr, thresholds = {}, {}, {}
    roc_auc = {}
    
    for i, label in enumerate(['Negativo', 'Neutral', 'Positivo']):
        fpr[label], tpr[label], thresholds[label] = roc_curve(y_test_bin[:, i], y_prob_svm[:, i])
        roc_auc[label] = auc(fpr[label], tpr[label])

    plt.figure()
    for label in ['Negativo', 'Neutral', 'Positivo']:
        plt.plot(fpr[label], tpr[label], lw=2, label=f'{label} (AUC = {roc_auc[label]:0.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - {nombre}')
    plt.legend(loc='lower right')
    plt.show()

# Guardar el vectorizador y los modelos entrenados en la carpeta models/
joblib.dump(vectorizer, 'models/vectorizer.pkl')
joblib.dump(modelos["Logistic Regression"], 'models/modelo_logistico.pkl')
joblib.dump(modelos["Random Forest"], 'models/modelo_random_forest.pkl')
joblib.dump(modelos["SVM"], 'models/modelo_svm.pkl')

print("\n Modelos y vectorizador guardados correctamente.")