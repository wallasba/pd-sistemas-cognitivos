import networkx as nx
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_coauthorship_graph(df: pd.DataFrame, min_coauthors: int = 2) -> nx.Graph:
    """Já existente – mantido."""
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
    G.remove_nodes_from([n for n, d in G.degree() if d == 0])
    return G

def build_institution_collaboration_graph(df: pd.DataFrame) -> nx.Graph:
    """Já existente – mantido."""
    G = nx.Graph()
    for _, row in df.iterrows():
        affs = row.get('affiliations', [])
        if not affs or len(affs) < 2:
            continue
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
    """Já existente – mantido."""
    abstracts = df['abstract_clean'].fillna('').tolist()
    vectorizer = CountVectorizer(stop_words='english', max_features=top_terms)
    X = vectorizer.fit_transform(abstracts)
    terms = vectorizer.get_feature_names_out()
    G = nx.Graph()
    for idx, row in df.iterrows():
        insts = row.get('affiliations', [])
        if not insts:
            continue
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

# ============================================================
# NOVOS GRAFOS
# ============================================================

def build_term_cooccurrence_graph(df: pd.DataFrame, top_terms: int = 50, min_cooccurrence: int = 2) -> nx.Graph:
    """
    Grafo de co-ocorrência de termos: nós = termos; arestas = aparecem juntos no mesmo resumo.
    Peso = número de documentos em que co-ocorrem.
    """
    abstracts = df['abstract_clean'].fillna('').tolist()
    vectorizer = CountVectorizer(stop_words='english', max_features=top_terms)
    X = vectorizer.fit_transform(abstracts)
    terms = vectorizer.get_feature_names_out()
    
    # Matriz de co-ocorrência (produto interno)
    cooccur = X.T @ X
    cooccur.setdiag(0)  # remove auto-coocorrência
    
    G = nx.Graph()
    for i, term_i in enumerate(terms):
        for j, term_j in enumerate(terms):
            if i < j and cooccur[i, j] >= min_cooccurrence:
                G.add_edge(term_i, term_j, weight=int(cooccur[i, j]))
    G.remove_nodes_from([n for n, d in G.degree() if d == 0])
    return G

def build_author_citation_proxy_graph(df: pd.DataFrame, top_authors: int = 30) -> nx.Graph:
    """
    Proxy de cocitação: autores que são mencionados juntos nos resumos (co-ocorrência de nomes).
    Isso simula autores que são citados em conjunto.
    """
    # Extrai nomes de autores do campo 'authors'
    author_list = []
    for _, row in df.iterrows():
        authors = row.get('authors', '')
        if authors:
            author_list.extend([a.strip() for a in authors.split(',') if a.strip()])
    
    # Conta frequência de autores
    author_freq = Counter(author_list)
    top_authors_set = set([a for a, _ in author_freq.most_common(top_authors)])
    
    # Cria grafo de co-ocorrência de autores nos resumos
    G = nx.Graph()
    for _, row in df.iterrows():
        authors = row.get('authors', '')
        if not authors:
            continue
        authors_present = [a.strip() for a in authors.split(',') if a.strip() and a.strip() in top_authors_set]
        if len(authors_present) < 2:
            continue
        for i in range(len(authors_present)):
            for j in range(i+1, len(authors_present)):
                u, v = authors_present[i], authors_present[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
    G.remove_nodes_from([n for n, d in G.degree() if d == 0])
    return G

def build_article_similarity_graph(df: pd.DataFrame, top_n: int = 20, threshold: float = 0.3) -> nx.Graph:
    """
    Proxy de acoplamento bibliográfico: artigos que compartilham termos semelhantes (cosine similarity).
    Isso simula artigos que compartilham referências.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    abstracts = df['abstract_clean'].fillna('').tolist()
    vectorizer = CountVectorizer(stop_words='english', max_features=100)
    X = vectorizer.fit_transform(abstracts)
    sim_matrix = cosine_similarity(X)
    
    G = nx.Graph()
    n = len(df)
    # Limita a top_n artigos (para performance)
    indices = list(range(n))
    for i in indices:
        for j in indices[i+1:]:
            if sim_matrix[i, j] >= threshold:
                G.add_edge(i, j, weight=float(sim_matrix[i, j]))
    
    # Adiciona atributos de título para os nós
    for idx, row in df.iterrows():
        if idx in G.nodes:
            G.nodes[idx]['title'] = row.get('title', f'Artigo {idx}')
    
    G.remove_nodes_from([n for n, d in G.degree() if d == 0])
    return G

def prepare_node_attributes(G, metric='degree', community_detection=True):
    """Adiciona atributos de tamanho e cor aos nós."""
    if metric == 'degree':
        deg = dict(G.degree())
        max_deg = max(deg.values()) if deg else 1
        for node in G.nodes:
            G.nodes[node]['size'] = 10 + (deg[node] / max_deg) * 50
    elif metric == 'betweenness':
        try:
            betweenness = nx.betweenness_centrality(G)
            max_val = max(betweenness.values()) if betweenness else 1
            for node in G.nodes:
                G.nodes[node]['size'] = 10 + (betweenness[node] / max_val) * 50
        except:
            # fallback
            for node in G.nodes:
                G.nodes[node]['size'] = 30
    else:
        for node in G.nodes:
            G.nodes[node]['size'] = 30
    
    # Cor por comunidade (usando Louvain)
    if community_detection and G.number_of_nodes() > 2:
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, seed=42)
            color_map = {}
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#DDA0DD', '#F0E68C', '#FFD700']
            for i, comm in enumerate(communities):
                for node in comm:
                    color_map[node] = colors[i % len(colors)]
            for node in G.nodes:
                G.nodes[node]['color'] = color_map.get(node, '#D3D3D3')
        except:
            for node in G.nodes:
                G.nodes[node]['color'] = '#4A90D9'  # cor padrão
    else:
        for node in G.nodes:
            G.nodes[node]['color'] = '#4A90D9'
    
    return G