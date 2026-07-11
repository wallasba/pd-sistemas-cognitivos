import networkx as nx
import pandas as pd
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# src/graph_builder.py

def build_coauthorship_graph(df: pd.DataFrame, min_coauthors: int = 2) -> nx.Graph:
    """
    Constrói grafo de coautoria.
    Nós = autores. Arestas = coautoria (peso = número de artigos em colaboração).
    Adiciona atributo 'frequency' a cada nó = número total de artigos do autor.
    """
    G = nx.Graph()
    all_authors = []  # Lista para acumular todos os autores (para frequência)
    
    for _, row in df.iterrows():
        authors = row.get('authors', '')
        if not authors:
            continue
        author_list = [a.strip() for a in authors.split(',') if a.strip()]
        if len(author_list) < min_coauthors:
            continue
        # Acumula autores para frequência
        all_authors.extend(author_list)
        # Adiciona arestas de coautoria
        for i in range(len(author_list)):
            for j in range(i+1, len(author_list)):
                u, v = author_list[i], author_list[j]
                if G.has_edge(u, v):
                    G[u][v]['weight'] += 1
                else:
                    G.add_edge(u, v, weight=1)
    
    # Calcula frequência de cada autor
    author_freq = Counter(all_authors)
    
    # Adiciona atributo 'frequency' a cada nó
    for node in G.nodes:
        G.nodes[node]['frequency'] = author_freq.get(node, 0)
    
    # Remove nós isolados (sem arestas)
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

def filter_top_nodes_by_criterion(G, top_n=30, criterion='degree'):
    """
    Filtra os N nós mais importantes de acordo com o critério.
    - degree: maior grau
    - frequency: maior frequência (atributo 'frequency' do nó)
    - betweenness: maior centralidade
    """
    if G is None or G.number_of_nodes() == 0:
        return G
    if G.number_of_nodes() <= top_n:
        return G
    
    if criterion == 'degree':
        scores = dict(G.degree())
    elif criterion == 'betweenness':
        scores = nx.betweenness_centrality(G)
    elif criterion == 'frequency':
        scores = {}
        for node in G.nodes:
            freq = G.nodes[node].get('frequency', 0)
            scores[node] = freq if freq > 0 else G.degree(node)  # fallback
    else:
        scores = dict(G.degree())
    
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in sorted_nodes[:top_n]]
    return G.subgraph(top_nodes).copy()

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

def compute_network_statistics(G: nx.Graph) -> dict:
    """
    Calcula estatísticas completas da rede para análise científica.
    Retorna um dicionário com todas as métricas.
    """
    stats = {}
    
    # Métricas básicas
    stats['nodes'] = G.number_of_nodes()
    stats['edges'] = G.number_of_edges()
    stats['density'] = nx.density(G)
    stats['is_connected'] = nx.is_connected(G)
    
    # Componentes
    if not nx.is_connected(G):
        stats['num_components'] = nx.number_connected_components(G)
        components = list(nx.connected_components(G))
        stats['largest_component_size'] = max(len(c) for c in components)
        stats['component_sizes'] = [len(c) for c in components]
    else:
        stats['num_components'] = 1
        stats['largest_component_size'] = G.number_of_nodes()
        stats['component_sizes'] = [G.number_of_nodes()]
    
    # Transitividade (clustering)
    stats['transitivity'] = nx.transitivity(G)
    stats['average_clustering'] = nx.average_clustering(G)
    
    # Diâmetro e raio (apenas se conectado)
    if nx.is_connected(G):
        try:
            stats['diameter'] = nx.diameter(G)
            stats['radius'] = nx.radius(G)
            stats['average_shortest_path'] = nx.average_shortest_path_length(G)
        except:
            stats['diameter'] = None
            stats['radius'] = None
            stats['average_shortest_path'] = None
    else:
        stats['diameter'] = None
        stats['radius'] = None
        stats['average_shortest_path'] = None
    
    # Centralidades (nós mais importantes)
    try:
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)
    except:
        degree_cent = {}
        betweenness_cent = {}
        closeness_cent = {}
        eigenvector_cent = {}
    
    # Top 5 nós por cada centralidade
    stats['top_degree'] = sorted(degree_cent.items(), key=lambda x: x[1], reverse=True)[:5]
    stats['top_betweenness'] = sorted(betweenness_cent.items(), key=lambda x: x[1], reverse=True)[:5]
    stats['top_closeness'] = sorted(closeness_cent.items(), key=lambda x: x[1], reverse=True)[:5]
    stats['top_eigenvector'] = sorted(eigenvector_cent.items(), key=lambda x: x[1], reverse=True)[:5]
    
    # Resumo dos valores de centralidade
    if degree_cent:
        stats['degree_mean'] = np.mean(list(degree_cent.values()))
        stats['degree_max'] = max(degree_cent.values())
        stats['degree_min'] = min(degree_cent.values())
    
    # Assortatividade (se houver arestas)
    if G.number_of_edges() > 0:
        try:
            stats['assortativity'] = nx.degree_assortativity_coefficient(G)
        except:
            stats['assortativity'] = None
    else:
        stats['assortativity'] = None
    
    return stats

def stats_to_dataframe(stats: dict) -> pd.DataFrame:
    """Converte estatísticas em DataFrame para exportação."""
    rows = []
    
    # Métricas gerais
    rows.append({'Metric': 'Número de nós', 'Value': stats.get('nodes', 0)})
    rows.append({'Metric': 'Número de arestas', 'Value': stats.get('edges', 0)})
    rows.append({'Metric': 'Densidade', 'Value': f"{stats.get('density', 0):.4f}"})
    rows.append({'Metric': 'Conectado', 'Value': stats.get('is_connected', False)})
    rows.append({'Metric': 'Componentes', 'Value': stats.get('num_components', 0)})
    rows.append({'Metric': 'Maior componente', 'Value': stats.get('largest_component_size', 0)})
    rows.append({'Metric': 'Transitividade', 'Value': f"{stats.get('transitivity', 0):.4f}"})
    rows.append({'Metric': 'Clustering médio', 'Value': f"{stats.get('average_clustering', 0):.4f}"})
    
    if stats.get('diameter'):
        rows.append({'Metric': 'Diâmetro', 'Value': stats['diameter']})
        rows.append({'Metric': 'Raio', 'Value': stats['radius']})
        rows.append({'Metric': 'Distância média', 'Value': f"{stats['average_shortest_path']:.3f}"})
    
    if 'degree_mean' in stats:
        rows.append({'Metric': 'Grau médio', 'Value': f"{stats['degree_mean']:.3f}"})
        rows.append({'Metric': 'Grau máximo', 'Value': f"{stats['degree_max']:.3f}"})
        rows.append({'Metric': 'Grau mínimo', 'Value': f"{stats['degree_min']:.3f}"})
    
    if stats.get('assortativity'):
        rows.append({'Metric': 'Assortatividade', 'Value': f"{stats['assortativity']:.3f}"})
    
    # Top nós por centralidade
    for label, top_list in [
        ('Top grau', stats.get('top_degree', [])),
        ('Top betweenness', stats.get('top_betweenness', [])),
        ('Top closeness', stats.get('top_closeness', [])),
        ('Top eigenvector', stats.get('top_eigenvector', []))
    ]:
        if top_list:
            rows.append({'Metric': f"{label} (1º)", 'Value': f"{top_list[0][0]} ({top_list[0][1]:.4f})"})
            if len(top_list) > 1:
                rows.append({'Metric': f"{label} (2º)", 'Value': f"{top_list[1][0]} ({top_list[1][1]:.4f})"})
            if len(top_list) > 2:
                rows.append({'Metric': f"{label} (3º)", 'Value': f"{top_list[2][0]} ({top_list[2][1]:.4f})"})
    
    return pd.DataFrame(rows)