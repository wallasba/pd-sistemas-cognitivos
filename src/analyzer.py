import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List

def get_stats(df: pd.DataFrame) -> dict:
    return {
        'total': len(df),
        'sources': df['source'].value_counts().to_dict(),
        'years': df['year'].value_counts().sort_index().to_dict(),
        'avg_abstract_len': df['abstract'].str.len().mean()
    }

def generate_wordcloud(df: pd.DataFrame, column: str = 'abstract_clean') -> WordCloud:
    text = ' '.join(df[column].fillna('').tolist())
    wc = WordCloud(width=800, height=400, background_color='white', max_words=150)
    return wc.generate(text)

def plot_distributions(df: pd.DataFrame):
    # Retorna figuras matplotlib para exibição no Streamlit
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    df['source'].value_counts().plot(kind='bar', ax=ax1, color='salmon')
    ax1.set_title('Distribuição por Fonte')
    df['year'].value_counts().sort_index().plot(kind='bar', ax=ax2, color='skyblue')
    ax2.set_title('Distribuição por Ano')
    plt.tight_layout()
    return fig