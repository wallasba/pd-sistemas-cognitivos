import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List

# Download dos recursos do NLTK (executado uma vez, com silêncio)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

stop_en = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Limpeza básica: minúsculas, remoção de HTML, URLs, números e pontuação."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>|http\S+|\d+|[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_and_lemmatize(text: str) -> List[str]:
    """Tokeniza e aplica lematização, removendo stopwords e tokens com menos de 3 caracteres."""
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_en and len(t) > 2]
    return tokens

def preprocess_abstracts(df: pd.DataFrame, column: str = 'abstract') -> pd.DataFrame:
    """Adiciona colunas 'abstract_clean' e 'tokens' ao DataFrame."""
    df['abstract_clean'] = df[column].fillna('').apply(clean_text)
    df['tokens'] = df['abstract_clean'].apply(tokenize_and_lemmatize)
    return df