import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List
import pandas as pd

# ============================================================
# Download automático dos recursos do NLTK (executado uma vez)
# ============================================================
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

# ============================================================
# Configurações
# ============================================================
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
    """Tokeniza, remove stopwords e aplica lematização."""
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_en and len(t) > 2]
    return tokens

def preprocess_abstracts(df: pd.DataFrame, column: str = 'abstract') -> pd.DataFrame:
    """Aplica limpeza e tokenização a todos os abstracts do DataFrame."""
    df['abstract_clean'] = df[column].fillna('').apply(clean_text)
    df['tokens'] = df['abstract_clean'].apply(tokenize_and_lemmatize)
    return df

def ensure_string_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Converte colunas para string e substitui NaN por string vazia."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    return df

def preprocess_abstracts(df: pd.DataFrame, column: str = 'abstract') -> pd.DataFrame:
    # Garantir que colunas essenciais sejam strings
    text_cols = ['abstract', 'authors', 'affiliations', 'source', 'title', 'year']
    df = ensure_string_columns(df, text_cols)
    
    # Limpeza do abstract
    df['abstract_clean'] = df[column].fillna('').apply(clean_text)
    df['tokens'] = df['abstract_clean'].apply(tokenize_and_lemmatize)
    
    # Converte 'year' para inteiro (se possível)
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    
    return df