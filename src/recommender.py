import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def extract_keywords(df: pd.DataFrame, top_n: int = 20) -> list:
    """Retorna as palavras mais frequentes no corpus (após limpeza)."""
    if 'abstract_clean' not in df.columns:
        return []
    text = ' '.join(df['abstract_clean'].fillna(''))
    vectorizer = CountVectorizer(stop_words='english', max_features=top_n)
    X = vectorizer.fit_transform([text])
    terms = vectorizer.get_feature_names_out()
    return terms.tolist()

def suggest_objectives(context: str, df: pd.DataFrame = None) -> list:
    """Sugere objetivos genéricos com base no contexto e, se houver corpus, nos tópicos."""
    base_suggestions = [
        "Mapear o estado da arte sobre {topic}",
        "Identificar lacunas de pesquisa em {topic}",
        "Analisar tendências e evolução de {topic} nos últimos anos",
        "Comparar abordagens e metodologias empregadas em {topic}",
        "Propor um novo framework ou modelo para {topic}"
    ]
    # Se houver corpus, extrai tópicos para personalizar
    if df is not None and not df.empty:
        keywords = extract_keywords(df, top_n=5)
        topic_str = ', '.join(keywords[:3]) if keywords else "o tema"
    else:
        topic_str = "o tema"
    
    suggestions = [s.format(topic=topic_str) for s in base_suggestions]
    return suggestions

def suggest_hypotheses(objectives: list, df: pd.DataFrame = None) -> list:
    """Gera hipóteses genéricas a partir dos objetivos."""
    if not objectives:
        return []
    hypotheses = []
    for obj in objectives[:2]:
        hypotheses.append(f"O aumento de publicações sobre {obj.split()[2:][0]} está correlacionado com avanços tecnológicos.")
        hypotheses.append(f"Existem diferenças significativas entre as abordagens adotadas por instituições de diferentes regiões.")
    return hypotheses[:4]  # limita a 4

def suggest_search_terms(context: str, df: pd.DataFrame = None) -> list:
    """Sugere termos de busca com base no contexto e, se houver corpus, nos termos mais frequentes."""
    base_terms = [context]
    if df is not None and not df.empty:
        keywords = extract_keywords(df, top_n=10)
        base_terms.extend(keywords[:5])
    # Adiciona operadores booleanos genéricos
    terms_with_operators = []
    for term in base_terms[:3]:
        terms_with_operators.append(f'"{term}"')
        terms_with_operators.append(f'"{term}" AND (methodology OR approach)')
        terms_with_operators.append(f'"{term}" AND (review OR survey)')
    return terms_with_operators