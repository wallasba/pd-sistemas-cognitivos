from typing import List, Dict, Optional
import networkx as nx
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import re

def build_coauthorship_graph(df: pd.DataFrame, min_coauthors: int = 2) -> nx.Graph:
    """Nós = autores; arestas = colaboração (peso = número de artigos em coautoria)."""
    G = nx.Graph()
    for _, row in df.iterrows():
        authors = row.get('authors', '')
        if not authors:
            continue
        author_list = [a.strip() for a in authors.split(',') if a.strip()]
        if len(author_list) < min_coauthors:
            continue
        for i in range(len(author_list)):
            for j in range(i+1, len(author_list)):
                u, v = author_list[i], author_list[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
    # Remove nós isolados
    G.remove_nodes_from([n for n, d in G.degree() if d == 0])
    return G

def build_institution_collaboration_graph(df: pd.DataFrame) -> nx.Graph:
    """Nós = instituições; arestas = colaboração (autores de instituições diferentes no mesmo artigo)."""
    G = nx.Graph()
    for _, row in df.iterrows():
        affs = row.get('affiliations', [])
        if not affs or len(affs) < 2:
            continue
        # Remove duplicatas e vazios
        inst_list = list(set([a for a in affs if a]))
        for i in range(len(inst_list)):
            for j in range(i+1, len(inst_list)):
                u, v = inst_list[i], inst_list[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
    G.remove_nodes_from([n for n, d in G.degree() if d == 0])
    return G

def build_term_institution_graph(df: pd.DataFrame, top_terms: int = 30) -> nx.Graph:
    """
    Nós = termos (frequentes) + instituições.
    Aresta = termo aparece em artigo de determinada instituição.
    """
    # Extrai termos mais frequentes usando CountVectorizer
    abstracts = df['abstract_clean'].fillna('').tolist()
    vectorizer = CountVectorizer(stop_words='english', max_features=top_terms)
    X = vectorizer.fit_transform(abstracts)
    terms = vectorizer.get_feature_names_out()
    
    G = nx.Graph()
    # Para cada artigo, conecta instituições aos termos presentes
    for idx, row in df.iterrows():
        insts = row.get('affiliations', [])
        if not insts:
            continue
        # Palavras do abstract que estão entre os termos selecionados
        doc_terms = set(terms) & set(row['abstract_clean'].split())
        for inst in insts:
            G.add_node(inst, type='institution')
            for term in doc_terms:
                G.add_node(term, type='term')
                if G.has_edge(inst, term):
                    G[inst][term]['weight'] += 1
                else:
                    G.add_edge(inst, term, weight=1)
    G.remove_nodes_from([n for n, d in G.degree() if d == 0])
    return G