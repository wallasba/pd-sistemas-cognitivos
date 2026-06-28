import pandas as pd
import re
from typing import Optional

def clean_text(text: str) -> str:
    """Limpeza de texto (minúsculas, remoção de HTML, URLs, números, pontuação)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'<.*?>|http\S+|\d+|[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_corpus(csv_path: str, text_column: str = 'abstract') -> pd.DataFrame:
    """Carrega o corpus e aplica limpeza básica."""
    df = pd.read_csv(csv_path)
    
    # Garante que a coluna de texto exista
    if text_column not in df.columns:
        raise ValueError(f"Coluna '{text_column}' não encontrada no CSV.")
    
    # Cria a coluna limpa
    clean_col = f"{text_column}_clean"
    df[clean_col] = df[text_column].fillna('').apply(clean_text)
    
    # Remove linhas com texto vazio ou muito curto
    df = df[df[clean_col].str.len() > 20]
    print(f"✅ Corpus carregado: {len(df)} documentos válidos.")
    return df