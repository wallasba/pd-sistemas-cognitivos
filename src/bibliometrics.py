# src/bibliometrics.py
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocessor import clean_text
import streamlit as st
import re

def get_top_authors(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Retorna os autores mais frequentes (proxy de produtividade)."""
    all_authors = []
    for _, row in df.iterrows():
        authors = row.get('authors', '')
        if authors:
            all_authors.extend([a.strip() for a in authors.split(',') if a.strip()])
    if not all_authors:
        return {'error': 'Nenhum autor encontrado.'}
    counter = Counter(all_authors)
    top = counter.most_common(top_n)
    return {
        'top_authors': top,
        'total_unique_authors': len(counter),
        'total_author_occurrences': sum(counter.values())
    }

def get_top_terms(df: pd.DataFrame, top_n: int = 15, column: str = 'abstract_clean') -> dict:
    """Retorna os termos mais frequentes (proxy de temas principais)."""
    if column not in df.columns:
        return {'error': f'Coluna {column} não encontrada.'}
    text = ' '.join(df[column].fillna(''))
    if not text:
        return {'error': 'Texto vazio.'}
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform([text])
    terms = vectorizer.get_feature_names_out()
    counts = X.toarray()[0]
    term_freq = list(zip(terms, counts))
    term_freq.sort(key=lambda x: x[1], reverse=True)
    return {
        'top_terms': term_freq,
        'total_terms': len(set(text.split()))
    }

def get_top_institutions(df: pd.DataFrame, top_n: int = 10) -> dict:
    """Retorna as instituições mais frequentes (proxy de produção institucional)."""
    all_affs = []
    for _, row in df.iterrows():
        affs = row.get('affiliations', [])
        if affs:
            if isinstance(affs, list):
                all_affs.extend([a.strip() for a in affs if a.strip()])
            elif isinstance(affs, str):
                all_affs.extend([a.strip() for a in affs.split(';') if a.strip()])
    if not all_affs:
        return {'error': 'Nenhuma instituição encontrada. Use OpenAlex para dados de afiliação.'}
    counter = Counter(all_affs)
    top = counter.most_common(top_n)
    return {
        'top_institutions': top,
        'total_unique_inst': len(counter),
        'total_inst_occurrences': sum(counter.values())
    }

def get_most_similar_articles(df: pd.DataFrame, query: str, top_n: int = 5) -> dict:
    """Retorna os artigos mais similares à query usando TF-IDF + cosseno."""
    if 'abstract_clean' not in df.columns:
        return {'error': 'Coluna abstract_clean não encontrada.'}
    texts = df['abstract_clean'].fillna('').tolist()
    if not texts or all(t == '' for t in texts):
        return {'error': 'Textos vazios.'}
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(texts + [query])
    # O último é a query
    query_vec = X[-1]
    doc_vecs = X[:-1]
    sims = cosine_similarity(query_vec, doc_vecs).flatten()
    top_indices = sims.argsort()[-top_n:][::-1]
    results = []
    for idx in top_indices:
        if sims[idx] > 0:
            results.append({
                'index': idx,
                'title': df.iloc[idx].get('title', 'Sem título'),
                'score': float(sims[idx]),
                'abstract_preview': df.iloc[idx].get('abstract', '')[:200] + '...'
            })
    return {
        'query': query,
        'top_articles': results,
        'average_similarity': np.mean(sims) if len(sims) > 0 else 0
    }

def get_temporal_trends(df: pd.DataFrame, top_terms: int = 5) -> dict:
    """Analisa evolução temporal de termos por ano (proxy de tendências)."""
    if 'year' not in df.columns or 'abstract_clean' not in df.columns:
        return {'error': 'Colunas year ou abstract_clean não encontradas.'}
    # Agrupa por ano
    years = sorted(df['year'].dropna().unique())
    if len(years) < 2:
        return {'error': 'Poucos anos para análise temporal.'}
    year_term_counts = {}
    for year in years:
        year_df = df[df['year'] == year]
        text = ' '.join(year_df['abstract_clean'].fillna(''))
        vectorizer = CountVectorizer(stop_words='english', max_features=50)
        X = vectorizer.fit_transform([text])
        terms = vectorizer.get_feature_names_out()
        counts = X.toarray()[0]
        term_freq = dict(zip(terms, counts))
        year_term_counts[year] = term_freq
    # Identifica termos com maior crescimento (último ano vs primeiro)
    if len(years) >= 2:
        first_year = years[0]
        last_year = years[-1]
        first_terms = year_term_counts.get(first_year, {})
        last_terms = year_term_counts.get(last_year, {})
        all_terms = set(first_terms.keys()) | set(last_terms.keys())
        growth = {}
        for term in all_terms:
            f_count = first_terms.get(term, 0)
            l_count = last_terms.get(term, 0)
            if f_count > 0:
                growth[term] = (l_count - f_count) / (f_count + 1)
            else:
                growth[term] = l_count
        top_growth = sorted(growth.items(), key=lambda x: x[1], reverse=True)[:top_terms]
        return {
            'years': years,
            'year_term_counts': year_term_counts,
            'top_growing_terms': top_growth
        }
    return {'error': 'Dados insuficientes.'}

def get_collaboration_metrics(df: pd.DataFrame) -> dict:
    """Calcula métricas de colaboração (proxy de redes)."""
    # Número médio de autores por artigo
    avg_authors = []
    for _, row in df.iterrows():
        authors = row.get('authors', '')
        if authors:
            avg_authors.append(len([a for a in authors.split(',') if a.strip()]))
    avg_authors = np.mean(avg_authors) if avg_authors else 0
    # Número de artigos com múltiplas afiliações
    multi_aff = 0
    for _, row in df.iterrows():
        affs = row.get('affiliations', [])
        if isinstance(affs, list) and len(affs) > 1:
            multi_aff += 1
        elif isinstance(affs, str) and len(affs.split(';')) > 1:
            multi_aff += 1
    return {
        'avg_authors_per_article': round(avg_authors, 2),
        'multi_institution_articles': multi_aff,
        'total_articles': len(df),
        'collaboration_rate': round(multi_aff / len(df) * 100, 1) if len(df) > 0 else 0
    }

def get_bibliometric_insights(df: pd.DataFrame, query: str = "") -> dict:
    """Função principal que agrega todas as análises."""
    insights = {}
    insights['authors'] = get_top_authors(df)
    insights['terms'] = get_top_terms(df)
    insights['institutions'] = get_top_institutions(df)
    if query:
        insights['similar_articles'] = get_most_similar_articles(df, query)
    else:
        insights['similar_articles'] = {'error': 'Nenhuma query fornecida.'}
    insights['temporal'] = get_temporal_trends(df)
    insights['collaboration'] = get_collaboration_metrics(df)
    return insights

if st.button("📌 Mostrar apenas os top 10 autores no grafo"):
    top_authors = [a for a, _ in st.session_state.biblio_insights['authors']['top_authors']]
    # Recria o grafo com apenas esses autores (ou destaca no grafo atual)