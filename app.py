import streamlit as st
import pandas as pd
import tempfile
import os
from src.data_collector import collect_articles
from src.preprocessor import preprocess_abstracts
from src.analyzer import get_stats, generate_wordcloud, plot_distributions
from src.graph_builder import (
    build_coauthorship_graph,
    build_institution_collaboration_graph,
    build_term_institution_graph,
    build_term_cooccurrence_graph,
    build_author_citation_proxy_graph,
    build_article_similarity_graph,
    prepare_node_attributes
)

from src.recommender import (
    suggest_objectives,
    suggest_hypotheses,
    suggest_search_terms,
    extract_keywords
)
from src.rag_pipeline import RAGPipeline
import matplotlib.pyplot as plt
from pyvis.network import Network
import networkx as nx
import streamlit.components.v1 as components
import nltk

# ============================================================
# Garante que os recursos do NLTK estejam disponíveis
# ============================================================
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

for f in ["graph_coauth.html", "graph_inst.html", "graph_terms.html"]:
    if os.path.exists(f):
        os.remove(f)

# ============================================================
# Configuração da página
# ============================================================
st.set_page_config(page_title="Assistente de Pesquisa Científica", layout="wide")
st.title("🧠 Assistente de Pesquisa Científica (Universal)")
st.caption("Guia você desde a definição do problema até a análise de dados, em qualquer área do conhecimento.")

# ============================================================
# Inicialização do estado
# ============================================================
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'df' not in st.session_state:
    st.session_state.df = None
if 'research_context' not in st.session_state:
    st.session_state.research_context = ""
if 'objectives' not in st.session_state:
    st.session_state.objectives = []
if 'hypotheses' not in st.session_state:
    st.session_state.hypotheses = []
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None

# ============================================================
# Funções auxiliares para navegação
# ============================================================
def go_to_step(step):
    st.session_state.step = step
    st.rerun()

def next_step():
    st.session_state.step += 1
    st.rerun()

def prev_step():
    st.session_state.step -= 1
    st.rerun()

# ============================================================
# BARRA LATERAL - PROGRESSO E UPLOAD DE CORPUS
# ============================================================
with st.sidebar:
    st.header("📌 Progresso da Pesquisa")
    steps = ["Definição", "Objetivos", "Estratégia", "Coleta", "Análise"]
    current = st.session_state.step - 1
    for i, step_name in enumerate(steps):
        if i < current:
            st.success(f"✅ {step_name}")
        elif i == current:
            st.info(f"🔵 {step_name}")
        else:
            st.caption(f"⬜ {step_name}")
    
    st.divider()
    
    # ============================================================
    # UPLOAD DE CORPUS EXISTENTE (CSV, XLSX, XLS)
    # ============================================================
    st.header("📂 Carregar Corpus Existente")
    
    with st.expander("📋 Estrutura necessária do arquivo"):
        st.markdown("""
        O arquivo (CSV, XLSX ou XLS) deve conter as seguintes colunas:
        - **`title`** (texto): Título do artigo.
        - **`abstract`** (texto): Resumo do artigo.
        - **`source`** (texto): Fonte (ex: arxiv, pubmed).
        - **`year`** (inteiro): Ano de publicação.
        - **`authors`** (texto): Autores separados por vírgula.
        - **`affiliations`** (texto): Instituições separadas por ponto e vírgula `;`.
        """)
        st.caption("💡 Colunas adicionais são permitidas, mas não obrigatórias.")
    
    uploaded_file = st.file_uploader(
        "Faça upload do seu corpus (CSV, XLSX ou XLS)",
        type=['csv', 'xlsx', 'xls'],
        help="Arquivo com a estrutura descrita acima."
    )
    
    if uploaded_file is not None:
        # Verifica se o arquivo não está vazio
        if uploaded_file.size == 0:
            st.error("❌ O arquivo está vazio. Por favor, envie um arquivo com dados.")
        else:
            try:
                # ============================================================
                # DETECTA O FORMATO E LÊ O ARQUIVO
                # ============================================================
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    df_raw = pd.read_csv(
                        uploaded_file,
                        encoding='utf-8',
                        engine='python',
                        low_memory=False
                    )
                elif file_extension == 'xlsx':
                    df_raw = pd.read_excel(
                        uploaded_file,
                        engine='openpyxl'
                    )
                elif file_extension == 'xls':
                    df_raw = pd.read_excel(
                        uploaded_file,
                        engine='xlrd'
                    )
                else:
                    st.error(f"❌ Formato '{file_extension}' não suportado. Use CSV, XLSX ou XLS.")
                    st.stop()
                
                # ============================================================
                # VALIDAÇÃO E PRÉ-PROCESSAMENTO
                # ============================================================
                if df_raw.empty:
                    st.error("❌ O arquivo não contém dados. Verifique o conteúdo.")
                else:
                    # Converte colunas de texto para string (evita erro de .split())
                    text_cols = ['title', 'abstract', 'source', 'authors', 'affiliations']
                    for col in text_cols:
                        if col in df_raw.columns:
                            df_raw[col] = df_raw[col].fillna('').astype(str)
                        else:
                            # Se a coluna não existir, cria com valores vazios
                            df_raw[col] = ''
                    
                    # Verifica colunas obrigatórias (mínimo: title, abstract, source, year)
                    required_cols = ['title', 'abstract', 'source', 'year']
                    missing_cols = [col for col in required_cols if col not in df_raw.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Colunas obrigatórias ausentes: {', '.join(missing_cols)}")
                    else:
                        st.success(f"✅ Arquivo válido! {len(df_raw)} registros encontrados.")
                        
                        # Botão para carregar
                        if st.button("📥 Carregar este Corpus", use_container_width=True):
                            with st.spinner("Pré-processando o corpus..."):
                                # Aplica o pré-processamento (limpeza, tokenização)
                                df = preprocess_abstracts(df_raw)
                                
                                # Garante que a coluna 'affiliations' exista como lista
                                if 'affiliations' not in df.columns:
                                    df['affiliations'] = [[] for _ in range(len(df))]
                                else:
                                    # Se já existe, converte strings com ';' para lista
                                    df['affiliations'] = df['affiliations'].apply(
                                        lambda x: [a.strip() for a in str(x).split(';') if a.strip()]
                                    )
                                
                                # Garante que 'authors' seja string
                                if 'authors' not in df.columns:
                                    df['authors'] = ""
                                else:
                                    df['authors'] = df['authors'].fillna('').astype(str)
                                
                                # Armazena no session_state
                                st.session_state.df = df
                                
                                # Avança diretamente para a etapa de análise (Step 4)
                                st.session_state.step = 4
                                st.success(f"✅ Corpus carregado com sucesso! {len(df)} artigos prontos para análise.")
                                st.rerun()
            
            except pd.errors.EmptyDataError:
                st.error("❌ O arquivo está vazio ou corrompido. Verifique o conteúdo.")
            except UnicodeDecodeError:
                # Fallback para encoding alternativo (apenas CSV)
                try:
                    uploaded_file.seek(0)
                    if file_extension == 'csv':
                        df_raw = pd.read_csv(
                            uploaded_file,
                            encoding='latin1',
                            engine='python',
                            low_memory=False
                        )
                        st.info("🔁 Leitura bem-sucedida com encoding 'latin1'. Considere salvar seu CSV em UTF-8.")
                        # Repete a validação (simplificada aqui, mas você pode chamar uma função)
                        # Para evitar duplicação, sugiro extrair a validação para uma função separada.
                    else:
                        st.error("❌ Erro de encoding em arquivo Excel. Verifique o formato.")
                except Exception as e2:
                    st.error(f"❌ Erro de encoding: {e2}")
            except Exception as e:
                st.error(f"❌ Erro ao ler o arquivo: {e}")
    
    # Métricas e informações na barra lateral
    if st.session_state.df is not None:
        st.metric("Artigos coletados", len(st.session_state.df))
    st.caption("Sessão: " + (st.session_state.research_context[:50] + "..." if st.session_state.research_context else ""))

# ============================================================
# ETAPA 1: Definição do Problema
# ============================================================
if st.session_state.step == 1:
    st.header("1️⃣ Defina seu problema de pesquisa")
    st.markdown("Descreva livremente o tema, a área e as principais questões que você deseja investigar.")
    
    context = st.text_area(
        "Qual é o seu tema/problema de pesquisa?",
        value=st.session_state.research_context,
        placeholder="Ex: Impacto da inteligência artificial na engenharia de transportes, com foco em otimização de tráfego.",
        height=150
    )
    st.session_state.research_context = context
    
    col1, col2 = st.columns(2)
    with col1:
        area = st.text_input("Área do conhecimento (opcional)", placeholder="Ex: Engenharia, Ciências da Saúde, Física...")
    with col2:
        keywords = st.text_input("Palavras-chave iniciais (separadas por vírgula)", placeholder="Ex: deep learning, otimização, tráfego")
    
    if st.button("Próximo →", type="primary"):
        if context.strip():
            st.session_state.research_context = context
            # Armazena palavras-chave para uso posterior
            st.session_state.initial_keywords = [k.strip() for k in keywords.split(',') if k.strip()]
            next_step()
        else:
            st.warning("Por favor, descreva o problema de pesquisa.")

# ============================================================
# ETAPA 2: Objetivos e Hipóteses
# ============================================================
elif st.session_state.step == 2:
    st.header("2️⃣ Objetivos e Hipóteses")
    st.markdown("Com base no contexto, defina os objetivos e hipóteses da sua pesquisa. Use as sugestões ou crie os seus.")
    
    # Gera recomendações (se houver corpus prévio, usa; senão, usa contexto)
    if st.session_state.df is not None:
        df_for_recommend = st.session_state.df
    else:
        df_for_recommend = None
    
    suggested_objs = suggest_objectives(st.session_state.research_context, df_for_recommend)
    st.markdown("**💡 Sugestões de objetivos (clique para adicionar):**")
    cols = st.columns(3)
    for i, obj in enumerate(suggested_objs):
        with cols[i % 3]:
            if st.button(f"📌 {obj[:50]}...", key=f"obj_{i}"):
                if obj not in st.session_state.objectives:
                    st.session_state.objectives.append(obj)
    
    # Área para editar/inserir objetivos
    objectives_text = st.text_area(
        "Seus objetivos (um por linha):",
        value="\n".join(st.session_state.objectives),
        height=100,
        placeholder="Ex: \nMapear o estado da arte...\nIdentificar lacunas..."
    )
    if st.button("Atualizar Objetivos"):
        st.session_state.objectives = [line.strip() for line in objectives_text.split('\n') if line.strip()]
        st.success("Objetivos atualizados!")
    
    # Hipóteses
    suggested_hyp = suggest_hypotheses(st.session_state.objectives, df_for_recommend)
    st.markdown("**💡 Sugestões de hipóteses (clique para adicionar):**")
    cols_h = st.columns(3)
    for i, hyp in enumerate(suggested_hyp):
        with cols_h[i % 3]:
            if st.button(f"🧪 {hyp[:50]}...", key=f"hyp_{i}"):
                if hyp not in st.session_state.hypotheses:
                    st.session_state.hypotheses.append(hyp)
    
    hypotheses_text = st.text_area(
        "Suas hipóteses (um por linha):",
        value="\n".join(st.session_state.hypotheses),
        height=80,
        placeholder="Ex: O uso de transformers melhora a acurácia..."
    )
    if st.button("Atualizar Hipóteses"):
        st.session_state.hypotheses = [line.strip() for line in hypotheses_text.split('\n') if line.strip()]
        st.success("Hipóteses atualizadas!")
    
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("← Voltar"):
            prev_step()
    with col_next:
        if st.button("Próximo →", type="primary"):
            if st.session_state.objectives:
                next_step()
            else:
                st.warning("Defina pelo menos um objetivo.")

# ============================================================
# ETAPA 3: Estratégia de Busca
# ============================================================
elif st.session_state.step == 3:
    # Se já houver um corpus carregado, sugere pular para a análise
    if st.session_state.df is not None and not st.session_state.df.empty:
        st.info("ℹ️ Você já possui um corpus carregado. Pode prosseguir diretamente para a análise.")
        if st.button("📊 Ir para Análise", type="primary"):
            st.session_state.step = 4
            st.rerun()
        st.markdown("---")
elif st.session_state.step == 3:
    st.header("3️⃣ Estratégia de Busca Bibliográfica")
    st.markdown("Construa sua query de busca com operadores booleanos e defina os parâmetros de coleta.")
    
    # Sugestões de termos de busca
    if st.session_state.df is not None:
        df_for_recommend = st.session_state.df
    else:
        df_for_recommend = None
    
    suggested_terms = suggest_search_terms(st.session_state.research_context, df_for_recommend)
    st.markdown("**💡 Termos e operadores sugeridos (clique para usar):**")
    cols_t = st.columns(4)
    for i, term in enumerate(suggested_terms[:8]):
        with cols_t[i % 4]:
            if st.button(f"🔍 {term}", key=f"term_{i}"):
                if st.session_state.query:
                    st.session_state.query += f" OR {term}"
                else:
                    st.session_state.query = term
                st.rerun()
    
    # Campo de query
    query = st.text_input(
        "Query de busca (use AND, OR, NOT e aspas):",
        value=st.session_state.query,
        placeholder='Ex: ("machine learning" OR "deep learning") AND "transportation" NOT "robotics"'
    )
    st.session_state.query = query
    
    # Parâmetros
    col1, col2 = st.columns(2)
    with col1:
        year_start = st.slider("Ano inicial", 2000, 2026, 2015)
        year_end = st.slider("Ano final", 2000, 2026, 2023)
    with col2:
        max_results = st.slider("Máximo de artigos por fonte", 100, 1000, 300, step=50)
        sources = st.multiselect(
            "Fontes (selecione uma ou mais):",
            ['arxiv', 'openalex', 'pubmed', 'crossref', 'europe_pmc', 'zenodo', 'doaj'],
            default=['arxiv', 'openalex', 'pubmed']
        )
    
    # Botão para coletar (já na etapa 3, podemos executar e ir para a 4)
    col_prev, col_collect, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Voltar"):
            prev_step()
    with col_collect:
        if st.button("🚀 Coletar e Avançar", type="primary"):
            if not query.strip():
                st.warning("Digite uma query de busca.")
            elif not sources:
                st.warning("Selecione pelo menos uma fonte.")
            else:
                with st.spinner("Coletando dados – isso pode levar alguns minutos..."):
                    df_raw = collect_articles(
                        query=query,
                        sources=sources,
                        max_results=max_results,
                        year_start=year_start,
                        year_end=year_end
                    )
                    if df_raw.empty:
                        st.error("Nenhum artigo encontrado. Tente ajustar a query ou período.")
                    else:
                        df = preprocess_abstracts(df_raw)
                        st.session_state.df = df
                        st.success(f"✅ {len(df)} artigos coletados e pré-processados!")
                        next_step()
    with col_next:
        pass  # não há próximo manual, o botão de coleta já avança

# ============================================================
# ETAPA 4: Análise e Visualização (Grafos, Estatísticas)
# ============================================================
elif st.session_state.step == 4:
    st.header("4️⃣ Análise Exploratória e Refinamento")
    st.markdown("Explore os resultados da sua coleta. Use os grafos e estatísticas para refinar sua pesquisa.")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Estatísticas
        stats = get_stats(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de artigos", stats['total'])
        col2.metric("Fontes", len(stats['sources']))
        col3.metric("Período", f"{min(stats['years'])} - {max(stats['years'])}")
        
        # Distribuições
        fig = plot_distributions(df)
        st.pyplot(fig)
        
        # Nuvem de palavras
        st.subheader("☁️ Nuvem de Palavras (termos mais frequentes)")
        wc = generate_wordcloud(df)
        fig_wc, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc)
        
        # ============================================================
        # GRAFOS COM MÚLTIPLAS PERSPECTIVAS E ATRIBUTOS VISUAIS
        # ============================================================
        st.subheader("🕸️ Análise de Redes Científicas")
        st.markdown("""
        **Escolha o tipo de grafo e as métricas para visualização:**
        - **Tamanho do nó** pode representar grau, betweenness ou número de publicações.
        - **Cor** pode representar comunidades (grupos de pesquisa) ou ser fixa.
        """)

        # Seleção do tipo de grafo
        graph_type = st.selectbox(
            "Tipo de grafo:",
            [
                "Coautoria",
                "Colaboração entre Instituições",
                "Termos × Instituições",
                "Co-ocorrência de Termos",
                "Cocitação (proxy - autores)",
                "Acoplamento (proxy - similaridade de artigos)"
            ]
        )

        # Parâmetros específicos para cada grafo
        with st.expander("⚙️ Parâmetros do grafo"):
            top_n = st.slider("Número máximo de nós a exibir", 10, 100, 30, step=5)
            min_weight = st.slider("Peso mínimo da aresta", 1, 10, 2, step=1)
            if graph_type == "Coautoria":
                min_coauthors = st.slider("Mínimo de coautores por artigo", 2, 5, 2)
            elif graph_type == "Co-ocorrência de Termos":
                top_terms = st.slider("Número de termos", 20, 100, 50, step=5)

        # Métricas para atributos visuais
        col_vis1, col_vis2 = st.columns(2)
        with col_vis1:
            size_metric = st.selectbox(
                "Tamanho do nó baseado em:",
                ["grau (conexões)", "betweenness (centralidade)", "fixo"]
            )
        with col_vis2:
            color_mode = st.selectbox(
                "Cor do nó:",
                ["comunidades (Louvain)", "fixa (azul)"]
            )

        # Botão para gerar
        if st.button("📊 Gerar Grafo", type="primary"):
            with st.spinner(f"Construindo grafo de {graph_type}..."):
                G = None
                if graph_type == "Coautoria":
                    G = build_coauthorship_graph(df, min_coauthors=min_coauthors)
                elif graph_type == "Colaboração entre Instituições":
                    G = build_institution_collaboration_graph(df)
                elif graph_type == "Termos × Instituições":
                    G = build_term_institution_graph(df, top_terms=top_n)
                elif graph_type == "Co-ocorrência de Termos":
                    G = build_term_cooccurrence_graph(df, top_terms=top_terms, min_cooccurrence=min_weight)
                elif graph_type == "Cocitação (proxy - autores)":
                    G = build_author_citation_proxy_graph(df, top_authors=top_n)
                elif graph_type == "Acoplamento (proxy - similaridade de artigos)":
                    G = build_article_similarity_graph(df, top_n=top_n, threshold=min_weight/10)
                
                if G is None or G.number_of_nodes() == 0:
                    st.warning("Grafo vazio. Tente ajustar os parâmetros.")
                else:
                    # Aplica atributos visuais
                    metric_map = {
                        "grau (conexões)": "degree",
                        "betweenness (centralidade)": "betweenness",
                        "fixo": "fixed"
                    }
                    community_detection = (color_mode == "comunidades (Louvain)")
                    G = prepare_node_attributes(G, metric=metric_map[size_metric], community_detection=community_detection)
                    
                    # Exibe o grafo com a função display_graph (que já temos)
                    display_graph(G, graph_type, "custom", view_mode, top_n=top_n)
        
        # Função para filtrar os N nós mais importantes (por grau)
        def filter_top_nodes(G, top_n=30):
            if G.number_of_nodes() <= top_n:
                return G
            deg = dict(G.degree())
            sorted_nodes = sorted(deg.items(), key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in sorted_nodes[:top_n]]
            return G.subgraph(top_nodes).copy()
        
        # Função para exibir grafo com cache e controle de física
        def filter_top_nodes(G, top_n=30):
            """Filtra os N nós mais importantes (por grau)."""
            if G is None or G.number_of_nodes() == 0:
                return G
            if G.number_of_nodes() <= top_n:
                return G
            deg = dict(G.degree())
            sorted_nodes = sorted(deg.items(), key=lambda x: x[1], reverse=True)
            top_nodes = [node for node, _ in sorted_nodes[:top_n]]
            return G.subgraph(top_nodes).copy()

        def prepare_node_attributes(G, metric='degree', community_detection=True):
            """
            Adiciona atributos 'size' e 'color' aos nós do grafo.
            - size: baseado em grau, betweenness ou fixo (30)
            - color: baseado em comunidades (Louvain) ou fixo (#4A90D9)
            """
            if G is None or G.number_of_nodes() == 0:
                return G
            
            G_copy = G.copy()
            
            # 1. Define tamanho dos nós
            if metric == 'degree':
                deg = dict(G_copy.degree())
                max_deg = max(deg.values()) if deg else 1
                for node in G_copy.nodes:
                    # Tamanho entre 10 e 60, proporcional ao grau
                    size = 10 + (deg.get(node, 0) / max_deg) * 50
                    G_copy.nodes[node]['size'] = size
            elif metric == 'betweenness':
                try:
                    betweenness = nx.betweenness_centrality(G_copy)
                    max_val = max(betweenness.values()) if betweenness else 1
                    for node in G_copy.nodes:
                        size = 10 + (betweenness.get(node, 0) / max_val) * 50
                        G_copy.nodes[node]['size'] = size
                except:
                    # Fallback para grau se betweenness falhar
                    deg = dict(G_copy.degree())
                    max_deg = max(deg.values()) if deg else 1
                    for node in G_copy.nodes:
                        size = 10 + (deg.get(node, 0) / max_deg) * 50
                        G_copy.nodes[node]['size'] = size
            else:  # 'fixed'
                for node in G_copy.nodes:
                    G_copy.nodes[node]['size'] = 30
            
            # 2. Define cor dos nós
            if community_detection and G_copy.number_of_nodes() > 2:
                try:
                    from networkx.algorithms.community import louvain_communities
                    communities = louvain_communities(G_copy, seed=42)
                    color_map = {}
                    colors = [
                        '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', 
                        '#98D8C8', '#DDA0DD', '#F0E68C', '#FFD700',
                        '#87CEEB', '#FF69B4', '#32CD32', '#FF4500'
                    ]
                    for i, comm in enumerate(communities):
                        color = colors[i % len(colors)]
                        for node in comm:
                            color_map[node] = color
                    for node in G_copy.nodes:
                        G_copy.nodes[node]['color'] = color_map.get(node, '#D3D3D3')
                except Exception as e:
                    # Fallback: cor fixa se detecção de comunidades falhar
                    for node in G_copy.nodes:
                        G_copy.nodes[node]['color'] = '#4A90D9'
            else:
                for node in G_copy.nodes:
                    G_copy.nodes[node]['color'] = '#4A90D9'
            
            return G_copy

        def display_graph(G, title, key_suffix, view_mode, top_n=30):
            """
            Exibe grafo no Streamlit com suporte a:
            - Modo interativo (PyVis) com física desativada
            - Modo estático (Matplotlib) como fallback
            - Cache do HTML no session_state
            - Tamanho e cor dos nós (preparados por prepare_node_attributes)
            """
            if G is None or G.number_of_nodes() == 0:
                st.info(f"ℹ️ Sem dados para gerar o grafo '{title}'.")
                return
            
            # Filtra para evitar sobrecarga
            G_filtered = filter_top_nodes(G, top_n)
            if G_filtered is None or G_filtered.number_of_nodes() < 2:
                st.info(f"ℹ️ Grafo muito pequeno para visualização (menos de 2 nós).")
                return
            
            # Aplica atributos visuais (size e color) se ainda não tiverem sido aplicados
            # (se já tiverem sido aplicados, mantém)
            if G_filtered.nodes and 'size' not in G_filtered.nodes[next(iter(G_filtered.nodes))]:
                # Se não tiver atributos, aplica padrão (grau + comunidades)
                G_filtered = prepare_node_attributes(G_filtered, metric='degree', community_detection=True)
            
            total_nodes = G.number_of_nodes()
            displayed_nodes = G_filtered.number_of_nodes()
            st.caption(f"📊 Exibindo {displayed_nodes} nós (dos {total_nodes} totais).")
            
            cache_key = f"graph_{key_suffix}"
            
            # ============================================================
            # MODO INTERATIVO (PyVis)
            # ============================================================
            if view_mode == "Interativo (com zoom e arraste)":
                try:
                    # Verifica se já está em cache e não precisa regenerar
                    if cache_key not in st.session_state or st.session_state.get(f"{cache_key}_force", False):
                        net = Network(
                            height="600px", 
                            width="100%", 
                            notebook=False, 
                            bgcolor="#ffffff",
                            font_color="black"
                        )
                        net.from_nx(G_filtered)
                        
                        # Aplica tamanho e cor dos nós manualmente (pyvis nem sempre transfere atributos)
                        for node in G_filtered.nodes:
                            if node in net.nodes:
                                net.nodes[node]['size'] = G_filtered.nodes[node].get('size', 30)
                                net.nodes[node]['color'] = G_filtered.nodes[node].get('color', '#4A90D9')
                                # Adiciona título para tooltip
                                if 'title' in G_filtered.nodes[node]:
                                    net.nodes[node]['title'] = G_filtered.nodes[node]['title']
                        
                        # Desativa física para evitar movimentação automática
                        net.set_options("""
                        var options = {
                        "physics": {
                            "enabled": false
                        },
                        "interaction": {
                            "hover": true,
                            "tooltipDelay": 200
                        }
                        }
                        """)
                        
                        # Gera HTML
                        filename = f"graph_{key_suffix}.html"
                        net.write_html(filename)
                        with open(filename, "r", encoding="utf-8") as f:
                            html_content = f.read()
                        
                        # Limpa arquivo temporário
                        try:
                            os.remove(filename)
                        except:
                            pass
                        
                        if len(html_content) > 1000:
                            st.session_state[cache_key] = html_content
                            st.session_state[f"{cache_key}_force"] = False
                        else:
                            raise Exception("HTML gerado é muito pequeno, provavelmente incompleto.")
                    
                    # Exibe o HTML do cache
                    if cache_key in st.session_state:
                        st.components.v1.html(st.session_state[cache_key], height=650)
                        st.caption("🖱️ Grafo interativo: arraste para mover, role para zoom. Física desativada.")
                        
                        # Botão para forçar recarga
                        col_reload, _ = st.columns([1, 5])
                        with col_reload:
                            if st.button("🔄 Recarregar grafo", key=f"reload_{key_suffix}", use_container_width=True):
                                st.session_state[f"{cache_key}_force"] = True
                                st.rerun()
                    else:
                        st.warning("Grafo não disponível no cache. Tente novamente.")
                        
                except Exception as e:
                    st.warning(f"⚠️ Falha ao gerar grafo interativo: {e}")
                    st.info("🔄 Alternando para o modo estático (imagem fixa)...")
                    # Fallback: chama a função novamente no modo estático
                    display_graph(G, title, key_suffix, "Estático (imagem fixa)", top_n)
            
            # ============================================================
            # MODO ESTÁTICO (Matplotlib)
            # ============================================================
            else:
                try:
                    # Extrai tamanhos e cores dos nós
                    sizes = []
                    colors = []
                    labels = {}
                    for node in G_filtered.nodes:
                        sizes.append(G_filtered.nodes[node].get('size', 30))
                        colors.append(G_filtered.nodes[node].get('color', '#4A90D9'))
                        # Trunca labels longos
                        label = str(node)
                        if len(label) > 25:
                            label = label[:22] + "..."
                        labels[node] = label
                    
                    fig, ax = plt.subplots(figsize=(14, 10))
                    
                    # Layout com menos iterações para rapidez
                    pos = nx.spring_layout(G_filtered, seed=42, k=0.3, iterations=80)
                    
                    # Desenha arestas (com transparência e espessura baseada no peso)
                    edges = G_filtered.edges(data=True)
                    edge_weights = [data.get('weight', 1) for _, _, data in edges]
                    if edge_weights:
                        max_weight = max(edge_weights) if edge_weights else 1
                        edge_widths = [1 + (w / max_weight) * 2 for w in edge_weights]
                    else:
                        edge_widths = [1] * len(edges)
                    
                    nx.draw_networkx_edges(
                        G_filtered, pos, ax=ax, 
                        alpha=0.3, width=edge_widths, edge_color='gray'
                    )
                    
                    # Desenha nós (com tamanho e cor variáveis)
                    nx.draw_networkx_nodes(
                        G_filtered, pos, ax=ax,
                        node_size=sizes, node_color=colors, alpha=0.8, edgecolors='black', linewidths=0.5
                    )
                    
                    # Desenha labels (com fonte pequena para não poluir)
                    nx.draw_networkx_labels(
                        G_filtered, pos, ax=ax, labels=labels, font_size=7, font_weight='bold'
                    )
                    
                    ax.set_title(f"{title} (top {displayed_nodes} nós)", fontsize=14, fontweight='bold')
                    ax.axis('off')
                    
                    # Ajusta layout para não cortar labels
                    plt.tight_layout()
                    
                    # Exibe no Streamlit
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    st.caption("📌 Versão estática (imagem fixa). Clique em 'Interativo' para explorar.")
                    
                except Exception as e:
                    st.error(f"❌ Erro ao gerar visualização estática: {e}")
        
        # Botões com spinner e feedback
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            if st.button("📌 Coautoria (top 30)", key="btn_coauth"):
                with st.spinner("Construindo grafo de coautoria..."):
                    G = build_coauthorship_graph(df, min_coauthors=2)
                    display_graph(G, "Rede de Coautoria", "coauth", view_mode, top_n=30)
        with col_g2:
            if st.button("🏛️ Instituições (top 30)", key="btn_inst"):
                with st.spinner("Construindo grafo de instituições..."):
                    G = build_institution_collaboration_graph(df)
                    display_graph(G, "Colaboração entre Instituições", "inst", view_mode, top_n=30)
        with col_g3:
            if st.button("🔗 Termos × Instituições (top 20)", key="btn_terms"):
                with st.spinner("Construindo grafo termos-instituições..."):
                    G = build_term_institution_graph(df, top_terms=20)
                    display_graph(G, "Termos Frequentes vs Instituições", "terms", view_mode, top_n=25)
        
        # ============================================================
        # Recomendações e refinamento
        # ============================================================
        st.subheader("💡 Recomendações para Refinamento")
        top_keywords = extract_keywords(df, top_n=10)
        if top_keywords:
            st.markdown("**Termos mais frequentes no corpus coletado:** " + ", ".join(top_keywords))
            st.markdown("**Sugestão:** Considere adicionar ou remover termos na sua query para refinar o escopo.")
        
        # Botões de navegação
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("← Voltar para Estratégia de Busca"):
                st.session_state.step = 3
                st.rerun()
        with col_next:
            if st.button("✅ Concluir e Ir para o Resumo"):
                st.session_state.step = 5
                st.rerun()
    else:
        st.warning("Nenhum corpus coletado. Volte para a etapa anterior e realize a coleta.")

# ============================================================
# ETAPA 5: Resumo e Próximos Passos
# ============================================================
elif st.session_state.step == 5:
    st.header("5️⃣ Resumo da Pesquisa e Próximos Passos")
    st.markdown("Parabéns! Você completou o ciclo de pesquisa assistida. Aqui está um resumo do seu projeto.")
    
    if st.session_state.research_context:
        st.subheader("Problema de Pesquisa")
        st.write(st.session_state.research_context)
    
    if st.session_state.objectives:
        st.subheader("Objetivos")
        for obj in st.session_state.objectives:
            st.write(f"- {obj}")
    
    if st.session_state.hypotheses:
        st.subheader("Hipóteses")
        for hyp in st.session_state.hypotheses:
            st.write(f"- {hyp}")
    
    if st.session_state.query:
        st.subheader("Query de Busca Utilizada")
        st.code(st.session_state.query)
    
    if st.session_state.df is not None:
        st.subheader(f"Corpus Coletado ({len(st.session_state.df)} artigos)")
        st.dataframe(st.session_state.df[['title', 'source', 'year']].head(10))
    
    st.markdown("---")
    st.markdown("**📌 Próximos passos recomendados:**")
    st.markdown("""
    - Refine sua query e repita a coleta para explorar novas facetas.
    - Use o chat RAG (se configurado) para fazer perguntas específicas sobre o corpus.
    - Exporte os dados e grafos para uso em publicações.
    - Considere expandir o período ou incluir outras fontes.
    """)
    
    if st.button("🔄 Recomeçar do início"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()