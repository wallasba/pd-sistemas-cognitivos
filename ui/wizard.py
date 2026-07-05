# ui/wizard.py
import streamlit as st
from src.data_collector import collect_articles
from src.preprocessor import preprocess_abstracts
from src.recommender import suggest_objectives, suggest_hypotheses, suggest_search_terms, extract_keywords
from src.graph_builder import (
    build_coauthorship_graph,
    build_institution_collaboration_graph,
    build_term_institution_graph,
    build_term_cooccurrence_graph,
    build_author_citation_proxy_graph,
    build_article_similarity_graph
)
from .analysis import render_analysis
from .graph_viewer import display_graph, prepare_node_attributes
from .rag_chat import render_rag_chat
import pandas as pd
import networkx as nx
from src.bibliometrics import get_bibliometric_insights

def render_wizard():
    """Renderiza o wizard de 5 etapas."""
    
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
                st.session_state.initial_keywords = [k.strip() for k in keywords.split(',') if k.strip()]
                st.session_state.step = 2
                st.rerun()
            else:
                st.warning("Por favor, descreva o problema de pesquisa.")
    
    # ============================================================
    # ETAPA 2: Objetivos e Hipóteses
    # ============================================================
    elif st.session_state.step == 2:
        st.header("2️⃣ Objetivos e Hipóteses")
        st.markdown("Com base no contexto, defina os objetivos e hipóteses da sua pesquisa.")
        
        df_for_recommend = st.session_state.df
        
        suggested_objs = suggest_objectives(st.session_state.research_context, df_for_recommend)
        st.markdown("**💡 Sugestões de objetivos (clique para adicionar):**")
        cols = st.columns(3)
        for i, obj in enumerate(suggested_objs):
            with cols[i % 3]:
                if st.button(f"📌 {obj[:50]}...", key=f"obj_{i}"):
                    if obj not in st.session_state.objectives:
                        st.session_state.objectives.append(obj)
        
        objectives_text = st.text_area(
            "Seus objetivos (um por linha):",
            value="\n".join(st.session_state.objectives),
            height=100,
            placeholder="Ex: \nMapear o estado da arte...\nIdentificar lacunas..."
        )
        if st.button("Atualizar Objetivos"):
            st.session_state.objectives = [line.strip() for line in objectives_text.split('\n') if line.strip()]
            st.success("Objetivos atualizados!")
        
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
                st.session_state.step = 1
                st.rerun()
        with col_next:
            if st.button("Próximo →", type="primary"):
                if st.session_state.objectives:
                    st.session_state.step = 3
                    st.rerun()
                else:
                    st.warning("Defina pelo menos um objetivo.")
    
    # ============================================================
    # ETAPA 3: Estratégia de Busca
    # ============================================================
    elif st.session_state.step == 3:
        st.header("3️⃣ Estratégia de Busca Bibliográfica")
        st.markdown("Construa sua query de busca com operadores booleanos e defina os parâmetros de coleta.")
        
        # Se já houver corpus, sugere pular
        if st.session_state.df is not None and not st.session_state.df.empty:
            st.info("ℹ️ Você já possui um corpus carregado. Pode prosseguir diretamente para a análise.")
            if st.button("📊 Ir para Análise", type="primary"):
                st.session_state.step = 4
                st.rerun()
            st.markdown("---")
        
        df_for_recommend = st.session_state.df
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
        
        query = st.text_input(
            "Query de busca (use AND, OR, NOT e aspas):",
            value=st.session_state.query,
            placeholder='Ex: ("machine learning" OR "deep learning") AND "transportation" NOT "robotics"'
        )
        st.session_state.query = query
        
        col1, col2 = st.columns(2)
        with col1:
            year_start = st.slider("Ano inicial", 2000, 2026, 2015)
            year_end = st.slider("Ano final", 2000, 2026, 2023)
        with col2:
            max_results = st.slider("Máximo de artigos por fonte", 100, 1000, 300, step=50)
            sources = st.multiselect(
                "Fontes:",
                ['arxiv', 'openalex', 'pubmed', 'crossref', 'europe_pmc', 'zenodo', 'doaj'],
                default=['arxiv', 'openalex', 'pubmed']
            )
        
        col_prev, col_collect = st.columns([1, 2])
        with col_prev:
            if st.button("← Voltar"):
                st.session_state.step = 2
                st.rerun()
        with col_collect:
            if st.button("🚀 Coletar e Avançar", type="primary"):
                if not query.strip():
                    st.warning("Digite uma query de busca.")
                elif not sources:
                    st.warning("Selecione pelo menos uma fonte.")
                else:
                    with st.spinner("Coletando dados – pode levar alguns minutos..."):
                        df_raw = collect_articles(
                            query=query,
                            sources=sources,
                            max_results=max_results,
                            year_start=year_start,
                            year_end=year_end
                        )
                        if df_raw.empty:
                            st.error("Nenhum artigo encontrado.")
                        else:
                            df = preprocess_abstracts(df_raw)
                            st.session_state.df = df
                            st.success(f"✅ {len(df)} artigos coletados!")
                            st.session_state.step = 4
                            st.rerun()
    
    # ============================================================
    # ETAPA 4: Análise
    # ============================================================

    elif st.session_state.step == 4:
        st.header("4️⃣ Análise Exploratória e Refinamento")
        st.markdown("Explore os resultados da sua coleta. Use os grafos e estatísticas para refinar sua pesquisa.")
        
        if st.session_state.df is not None:
            df = st.session_state.df
            
            # Análises descritivas (mantido)
            render_analysis(df)
            
            # ============================================================
            # GRAFOS – CONSTRUÇÃO E VISUALIZAÇÃO SEPARADAS
            # ============================================================
            st.subheader("🕸️ Análise de Redes Científicas")
            
            # --- 1. Configuração do grafo (parâmetros de construção) ---
            with st.expander("⚙️ Parâmetros do grafo", expanded=False):
                col_config1, col_config2 = st.columns(2)
                with col_config1:
                    graph_type = st.selectbox(
                        "Tipo de grafo:",
                        [
                            "Coautoria",
                            "Colaboração entre Instituições",
                            "Termos × Instituições",
                            "Co-ocorrência de Termos",
                            "Cocitação (proxy - autores)",
                            "Acoplamento (proxy - similaridade de artigos)"
                        ],
                        key="graph_type_config"
                    )

                    filter_criterion = st.selectbox(
                        "Critério para selecionar os nós:",
                        ["degree", "frequency", "betweenness"],
                        format_func=lambda x: {
                            "degree": "Grau (mais conectados)",
                            "frequency": "Frequência (mais produtivos)",
                            "betweenness": "Centralidade (pontes)"
                        }[x],
                        key="filter_criterion"
                    )
                
                with col_config2:
                    top_n = st.slider("Nº máximo de nós", 10, 100, 40, step=5, key="top_n_config")
                    min_weight = st.slider("Peso mínimo da aresta", 1, 10, 2, step=1, key="min_weight_config")
                
                # Parâmetros específicos por tipo
                if graph_type == "Coautoria":
                    min_coauthors = st.slider("Mínimo de coautores", 2, 5, 2, key="min_coauthors_config")
                elif graph_type == "Co-ocorrência de Termos":
                    top_terms = st.slider("Nº de termos", 20, 100, 50, step=5, key="top_terms_config")
                elif graph_type == "Acoplamento (proxy - similaridade de artigos)":
                    similarity_threshold = st.slider("Limiar de similaridade", 0.1, 0.9, 0.3, 0.05, key="sim_threshold_config")
            
            # Botão para construir o grafo
            if st.button("📊 Gerar Grafo", type="primary", key="build_graph_button"):
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
                        threshold = similarity_threshold if 'similarity_threshold' in locals() else 0.3
                        G = build_article_similarity_graph(df, top_n=top_n, threshold=threshold)
                    
                    if G is None or G.number_of_nodes() == 0:
                        st.warning("Grafo vazio. Tente ajustar os parâmetros.")
                        st.session_state.current_graph = None
                    else:
                        st.session_state.current_graph = G
                        st.session_state.graph_title = graph_type
                        st.success(f"Grafo construído! {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")
                        # Força re-exibição imediata
                        st.rerun()
            
            # --- 2. Ajustes visuais (interativos, sem reconstrução) ---
            if st.session_state.get('current_graph') is not None:
                st.markdown("---")
                st.markdown("**🎨 Ajustes de Visualização (interativos – não regeneram o grafo)**")
                
                # Modo de visualização
                view_mode = st.radio(
                    "Modo de visualização:",
                    ["Interativo (com zoom e arraste)", "Estático (imagem fixa)"],
                    index=1,
                    horizontal=True,
                    key="view_mode_interactive"
                )
                
                col_a, col_b = st.columns(2)
                with col_a:
                    node_metric = st.selectbox(
                        "Tamanho do nó baseado em:",
                        ["degree", "betweenness", "closeness", "uniform"],
                        format_func=lambda x: {
                            "degree": "Grau (conexões)",
                            "betweenness": "Betweenness (centralidade)",
                            "closeness": "Closeness (proximidade)",
                            "uniform": "Uniforme (fixo)"
                        }[x],
                        key="node_metric_vis"
                    )
                    node_scale = st.slider("Fator de escala dos nós", 0.5, 3.0, 1.0, 0.1, key="node_scale_vis")
                    node_min_size = st.slider("Tamanho mínimo do nó", 5, 30, 10, key="node_min_vis")
                    node_max_size = st.slider("Tamanho máximo do nó", 30, 120, 80, key="node_max_vis")
                
                with col_b:
                    edge_scale = st.slider("Fator de escala das arestas", 0.5, 3.0, 1.5, 0.1, key="edge_scale_vis")
                    edge_min_width = st.slider("Largura mínima da aresta", 0.2, 2.0, 0.5, 0.1, key="edge_min_vis")
                    edge_max_width = st.slider("Largura máxima da aresta", 2.0, 8.0, 6.0, 0.5, key="edge_max_vis")
                
                col_c, col_d = st.columns(2)
                with col_c:
                    layout_type = st.selectbox(
                        "Layout do grafo:",
                        ["spring", "circular", "kamada_kawai", "random"],
                        format_func=lambda x: {
                            "spring": "Spring (elástico)",
                            "circular": "Circular",
                            "kamada_kawai": "Kamada-Kawai",
                            "random": "Aleatório"
                        }[x],
                        key="layout_vis"
                    )
                    if layout_type == "spring":
                        layout_k = st.slider("K (distância entre nós)", 0.1, 1.0, 0.3, 0.05, key="layout_k_vis")
                        layout_iterations = st.slider("Iterações", 50, 500, 100, 50, key="layout_iter_vis")
                    else:
                        layout_k = 0.3
                        layout_iterations = 100
                
                with col_d:
                    font_size = st.slider("Tamanho da fonte dos rótulos", 6, 20, 10, key="font_vis")
                    fig_width = st.slider("Largura da figura (pol)", 8, 20, 14, key="fig_width_vis")
                    fig_height = st.slider("Altura da figura (pol)", 6, 16, 10, key="fig_height_vis")
                    show_labels = st.checkbox("Exibir rótulos", value=True, key="show_labels_vis")
                    label_trim = st.slider("Truncar rótulos com >", 10, 40, 25, key="label_trim_vis")
                
                # Botão para reaplicar visualização (opcional, pois sliders já re-rodam)
                if st.button("🔄 Atualizar Visualização", key="update_vis_button"):
                    st.rerun()
                
                # --- 3. Exibição do grafo (usando os parâmetros visuais atuais) ---
                G = st.session_state.current_graph
                title = st.session_state.graph_title
                
                # Aplica atributos visuais (prepare_node_attributes) com os parâmetros atuais
                G_prepared = prepare_node_attributes(
                    G,
                    metric=node_metric,
                    scale_factor=node_scale,
                    min_size=node_min_size,
                    max_size=node_max_size
                )
                
                # Chama display_graph com todos os parâmetros (incluindo edge_scale)
                display_graph(
                    G_prepared,
                    title,
                    "current_graph",
                    view_mode,
                    top_n=top_n,
                    filter_criterion=filter_criterion,
                    node_metric=node_metric,
                    node_scale=node_scale,
                    node_min_size=node_min_size,
                    node_max_size=node_max_size,
                    edge_scale=edge_scale,
                    edge_min_width=edge_min_width,
                    edge_max_width=edge_max_width,
                    font_size=font_size,
                    fig_width=fig_width,
                    fig_height=fig_height,
                    layout_type=layout_type,
                    layout_k=layout_k if layout_type == 'spring' else 0.3,
                    layout_iterations=layout_iterations if layout_type == 'spring' else 100,
                    show_labels=show_labels,
                    label_trim=label_trim
                )
                
                # Adicional: mostrar estatísticas do grafo
                with st.expander("📊 Estatísticas do grafo"):
                    st.write(f"**Nós:** {G.number_of_nodes()}")
                    st.write(f"**Arestas:** {G.number_of_edges()}")
                    if G.number_of_nodes() > 0:
                        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
                        st.write(f"**Grau médio:** {avg_degree:.2f}")
                        st.write(f"**Densidade:** {nx.density(G):.4f}")
                        if nx.is_connected(G):
                            st.write("**Conectado:** Sim")
                            st.write(f"**Diâmetro:** {nx.diameter(G)}")
                        else:
                            st.write("**Conectado:** Não")
                            st.write(f"**Componentes:** {nx.number_connected_components(G)}")
            
        # ============================================================
        # ANÁLISE BIBLIOMÉTRICA AUTOMÁTICA
        # ============================================================
        st.subheader("📊 Análise Bibliométrica Automática")
        st.markdown("""
        O sistema realiza automaticamente análises essenciais para pesquisadores seniores, utilizando **proxies** (medidas indiretas) 
        baseadas nos dados disponíveis. Abaixo estão os principais insights.
        """)

        # Explicação sobre proxy
        with st.expander("ℹ️ O que é um proxy em bibliometria?", expanded=False):
            st.markdown("""
            Em bibliometria, um **proxy** é uma medida indireta utilizada para representar ou estimar uma característica de interesse 
            quando não é possível medi-la diretamente. Exemplos comuns:
            - **Número de citações** → proxy de **impacto científico**.
            - **Frequência de coocorrência de termos** → proxy de **relevância temática**.
            - **Centralidade em redes** → proxy de **influência** de autores ou instituições.
            
            Nesta aplicação, como não dispomos de dados de citações, utilizamos **proxies** baseados nos metadados disponíveis:
            - **Frequência de autores** → proxy de **produtividade**.
            - **Frequência de termos** → proxy de **temas principais**.
            - **Coautoria e afiliações** → proxy de **redes de colaboração**.
            """)

        # Botão para executar análises
        if st.button("🔍 Executar Análises Bibliométricas", key="run_biblio"):
            with st.spinner("Processando dados..."):
                query = st.session_state.query if st.session_state.query else ""
                insights = get_bibliometric_insights(df, query)
                st.session_state.biblio_insights = insights
                st.success("Análises concluídas!")

        # Exibir insights se disponíveis
        if st.session_state.get('biblio_insights') is not None:
            insights = st.session_state.biblio_insights
            biblio_col1, biblio_col2 = st.columns(2)
        
            # Coluna 1: Autores, Instituições, Colaboração
            with biblio_col1:
                st.markdown("**👥 Autores mais frequentes (proxy de produtividade)**")
                if 'error' in insights.get('authors', {}):
                    st.info(insights['authors']['error'])
                else:
                    authors_data = insights['authors']
                    for i, (author, count) in enumerate(authors_data['top_authors'], 1):
                        st.write(f"{i}. **{author}** – {count} menções")
                    st.caption(f"Total de autores únicos: {authors_data['total_unique_authors']}")
                
                st.markdown("---")
                st.markdown("**🏛️ Instituições mais frequentes (proxy de produção institucional)**")
                if 'error' in insights.get('institutions', {}):
                    st.info(insights['institutions']['error'])
                else:
                    inst_data = insights['institutions']
                    for i, (inst, count) in enumerate(inst_data['top_institutions'], 1):
                        st.write(f"{i}. **{inst}** – {count} menções")
                    st.caption(f"Total de instituições únicas: {inst_data['total_unique_inst']}")
            
            # Coluna 2: Termos, Similaridade, Tendências
            with biblio_col2:
                st.markdown("**📝 Termos mais frequentes (proxy de temas principais)**")
                if 'error' in insights.get('terms', {}):
                    st.info(insights['terms']['error'])
                else:
                    terms_data = insights['terms']
                    for term, count in terms_data['top_terms']:
                        st.write(f"- **{term}** ({count})")
                    st.caption(f"Total de termos únicos: {terms_data['total_terms']}")
                
                st.markdown("---")
                st.markdown("**📈 Evolução temporal dos termos (tendências emergentes)**")
                temporal = insights.get('temporal', {})
                if 'error' in temporal:
                    st.info(temporal['error'])
                else:
                    years = temporal.get('years', [])
                    if years:
                        st.write(f"Período analisado: {min(years)} – {max(years)}")
                        growing = temporal.get('top_growing_terms', [])
                        if growing:
                            st.write("**Termos com maior crescimento:**")
                            for term, growth in growing[:5]:
                                st.write(f"- **{term}** (crescimento: {growth:.2f})")
                        else:
                            st.info("Nenhum termo com crescimento significativo.")
        
        if st.session_state.get('biblio_insights') and 'authors' in st.session_state.biblio_insights:
            if 'top_authors' in st.session_state.biblio_insights['authors']:
                top_authors = [a for a, _ in st.session_state.biblio_insights['authors']['top_authors'][:10]]
                if st.button("📌 Mostrar apenas os top 10 autores no grafo", key="filter_top_authors"):
                    if st.session_state.get('current_graph') is not None:
                        # Filtra o grafo atual para manter apenas os top autores
                        G = st.session_state.current_graph
                        nodes_to_keep = [n for n in G.nodes if n in top_authors]
                        # Também inclui vizinhos para não isolar (opcional)
                        # Para manter apenas os top autores e suas conexões diretas:
                        subgraph = G.subgraph(nodes_to_keep).copy()
                        st.session_state.current_graph = subgraph
                        st.session_state.graph_title = f"Coautoria (top 10 autores)"
                        st.rerun()
                    else:
                        st.warning("Gere um grafo de coautoria primeiro.")
            
            # Linha extra para similaridade
            st.markdown("---")
            st.markdown("**🔍 Artigos mais similares ao tema pesquisado**")
            sim = insights.get('similar_articles', {})
            if 'error' in sim:
                st.info(sim['error'])
            elif sim.get('top_articles'):
                for i, article in enumerate(sim['top_articles'], 1):
                    st.write(f"{i}. **{article['title']}** (similaridade: {article['score']:.3f})")
                    st.caption(f"Resumo: {article['abstract_preview']}")
            else:
                st.info("Nenhum artigo similar encontrado.")
            
            # Colaboração e estatísticas
            st.markdown("---")
            st.markdown("**🤝 Métricas de Colaboração**")
            collab = insights.get('collaboration', {})
            if collab:
                col1, col2, col3 = st.columns(3)
                col1.metric("Média de autores/artigo", collab.get('avg_authors_per_article', 0))
                col2.metric("Artigos com múltiplas afiliações", collab.get('multi_institution_articles', 0))
                col3.metric("Taxa de colaboração", f"{collab.get('collaboration_rate', 0)}%")

            
            # Recomendações
            st.subheader("💡 Recomendações para Refinamento")
            top_keywords = extract_keywords(df, top_n=10)
            if top_keywords:
                st.markdown("**Termos mais frequentes no corpus:** " + ", ".join(top_keywords))
                st.markdown("**Sugestão:** Considere adicionar ou remover termos na query para refinar o escopo.")
            
            # Navegação
            col_prev, col_next = st.columns(2)
            with col_prev:
                if st.button("← Voltar para Estratégia"):
                    st.session_state.step = 3
                    st.rerun()
            with col_next:
                if st.button("✅ Concluir e Ir para o Resumo"):
                    st.session_state.step = 5
                    st.rerun()
        else:
            st.warning("Nenhum corpus coletado. Volte e realize a coleta.")
    
    # ============================================================
    # ETAPA 5: Resumo
    # ============================================================
    elif st.session_state.step == 5:
        st.header("5️⃣ Resumo da Pesquisa e Próximos Passos")
        st.markdown("Parabéns! Você completou o ciclo de pesquisa assistida.")
        
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
        - Refine sua query e repita a coleta.
        - Use o chat RAG (se configurado) para perguntas específicas.
        - Exporte os dados e grafos para publicações.
        - Considere expandir o período ou incluir outras fontes.
        """)
        
        # Chat RAG (opcional)
        st.markdown("---")
        render_rag_chat(st.session_state.df)
        
        if st.button("🔄 Recomeçar do início"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()