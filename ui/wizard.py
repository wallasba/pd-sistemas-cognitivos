# ui/wizard.py
# ============================================================
# VERSÃO COMPLETA E CORRIGIDA
# ============================================================

import streamlit as st
import pandas as pd
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
from src.graph_builder import compute_network_statistics, stats_to_dataframe

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
            
            # ============================================================
            # PRÉ-VISUALIZAÇÃO DO CORPUS
            # ============================================================
            st.subheader("📄 Pré-visualização do Corpus")
            with st.expander("Ver primeiras 10 linhas do corpus carregado", expanded=False):
                st.dataframe(df[['title', 'source', 'year', 'abstract']].head(10))
                st.caption(f"Total de artigos: {len(df)}")
            st.divider()
            
            # Análises descritivas
            render_analysis(df)
            
            # ============================================================
            # GRAFOS – COM ATUALIZAÇÃO AUTOMÁTICA E ESTATÍSTICAS
            # ============================================================
            st.subheader("🕸️ Análise de Redes Científicas")

            # --- Parâmetros do grafo (armazenados em session_state para auto-atualização) ---

            col_grafo_tipo, col_grafo_params = st.columns([1, 2])

            with col_grafo_tipo:
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
                    key="graph_type_selector"
                )
                
                filter_criterion = st.selectbox(
                    "Critério para selecionar os nós:",
                    ["degree", "frequency", "betweenness"],
                    format_func=lambda x: {
                        "degree": "Grau (mais conectados)",
                        "frequency": "Frequência (mais produtivos)",
                        "betweenness": "Centralidade (pontes)"
                    }[x],
                    key="filter_criterion_selector"
                )
                
                with st.expander("⚙️ Parâmetros do grafo"):
                    top_n = st.slider("Nº máximo de nós", 10, 100, 40, step=5, key="top_n_graph")
                    min_weight = st.slider("Peso mínimo da aresta", 1, 10, 2, step=1, key="min_weight_graph")
                    top_edges_to_highlight = st.slider(
                        "Nº de arestas mais fortes para destacar",
                        min_value=0, max_value=50, value=10, step=1,
                        key="top_edges_highlight",
                        help="Selecione quantas arestas com maior peso serão destacadas com cores vibrantes."
                    )
                    if graph_type == "Coautoria":
                        min_coauthors = st.slider("Mínimo de coautores", 2, 5, 2, key="min_coauthors_graph")
                    elif graph_type == "Co-ocorrência de Termos":
                        top_terms = st.slider("Nº de termos", 20, 100, 50, step=5, key="top_terms_graph")

            with col_grafo_params:
                st.markdown("**🎨 Ajustes de Visualização**")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    node_metric = st.selectbox(
                        "Tamanho do nó baseado em:",
                        ["degree", "betweenness", "closeness", "frequency", "uniform"],
                        format_func=lambda x: {
                            "degree": "Grau (conexões)",
                            "betweenness": "Betweenness (centralidade)",
                            "closeness": "Closeness (proximidade)",
                            "frequency": "Frequência (produtividade)",
                            "uniform": "Uniforme (fixo)"
                        }[x],
                        key="node_metric_vis"
                    )
                    node_scale = st.slider(
                        "Multiplicador de tamanho dos nós", 
                        0.5, 3.0, 1.0, 0.1, 
                        key="node_scale_vis"
                    )
                    node_min_size = st.slider("Tamanho mínimo do nó", 20, 60, 30, key="node_min_vis")
                    node_max_size = st.slider("Tamanho máximo do nó", 60, 200, 120, key="node_max_vis")
                
                with col_b:
                    edge_scale = st.slider(
                        "Multiplicador de espessura das arestas", 
                        0.5, 3.0, 1.0, 0.1,
                        key="edge_scale_vis"
                    )
                    edge_min_width = st.slider("Largura mínima da aresta", 0.2, 2.0, 0.5, 0.1, key="edge_min_vis")
                    edge_max_width = st.slider("Largura máxima da aresta", 2.0, 8.0, 5.0, 0.5, key="edge_max_vis")
                
                col_c, col_d = st.columns(2)
                with col_c:
                    layout_type = st.selectbox(
                        "Layout do grafo:",
                        ["spring", "circular", "kamada_kawai", "shell"],
                        format_func=lambda x: {
                            "spring": "Spring (elástico)",
                            "circular": "Circular",
                            "kamada_kawai": "Kamada-Kawai",
                            "shell": "Shell (camadas)"
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
                    font_size = st.slider("Tamanho da fonte dos rótulos", 8, 24, 12, key="font_vis")
                    fig_width = st.slider("Largura da figura (pol)", 10, 24, 16, key="fig_width_vis")
                    fig_height = st.slider("Altura da figura (pol)", 8, 18, 12, key="fig_height_vis")
                    show_labels = st.checkbox("Exibir rótulos", value=True, key="show_labels_vis")
                    label_trim = st.slider("Truncar rótulos com >", 10, 40, 25, key="label_trim_vis")

            # ============================================================
            # CONSTRUÇÃO DO GRAFO (AUTOMÁTICA)
            # ============================================================

            # Função para construir o grafo com base nos parâmetros atuais
            def build_current_graph():
                """Constrói o grafo com os parâmetros atuais."""
                if df is None:
                    return None, None
                
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
                
                return G, graph_type

            # ============================================================
            # EXIBIÇÃO AUTOMÁTICA DO GRAFO
            # ============================================================

            # Usamos uma chave de cache baseada em todos os parâmetros relevantes
            cache_key = f"{graph_type}_{filter_criterion}_{top_n}_{min_weight}_{top_edges_to_highlight}_{node_metric}_{node_scale}_{layout_type}"

            # Se o grafo não estiver em cache ou os parâmetros mudaram, reconstrói
            if 'cached_graph' not in st.session_state or st.session_state.get('graph_key') != cache_key:
                G, title = build_current_graph()
                if G and G.number_of_nodes() > 0:
                    st.session_state.cached_graph = G
                    st.session_state.graph_title = title
                    st.session_state.graph_key = cache_key
                else:
                    st.session_state.cached_graph = None

            # Exibe o grafo se existir
            if st.session_state.get('cached_graph') is not None:
                G = st.session_state.cached_graph
                title = st.session_state.graph_title
                
                # Aplica atributos visuais
                st.session_state['color_community'] = True
                display_graph(
                    G, title, "custom",
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
                    label_trim=label_trim,
                    top_edges_to_highlight=top_edges_to_highlight,
                    use_edge_color_gradient=True
                )
                
                # ============================================================
                # ESTATÍSTICAS DA REDE (com exportação)
                # ============================================================
                st.subheader("📊 Estatísticas da Rede")
                
                # Botão para gerar estatísticas
                if st.button("📈 Gerar Estatísticas da Rede", key="generate_stats"):
                    with st.spinner("Calculando estatísticas..."):
                        stats = compute_network_statistics(G)
                        st.session_state.network_stats = stats
                        st.success("Estatísticas calculadas!")
                
                # Exibe estatísticas se disponíveis
                if 'network_stats' in st.session_state and st.session_state.network_stats:
                    stats = st.session_state.network_stats
                    df_stats = stats_to_dataframe(stats)
                    
                    # Exibe tabela
                    st.dataframe(df_stats, use_container_width=True)
                    
                    # Botão de exportação
                    col_exp1, col_exp2 = st.columns(2)
                    with col_exp1:
                        csv = df_stats.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Baixar CSV",
                            data=csv,
                            file_name="network_statistics.csv",
                            mime="text/csv"
                        )
                    with col_exp2:
                        from io import BytesIO
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df_stats.to_excel(writer, index=False, sheet_name='Network Stats')
                        excel_data = output.getvalue()
                        st.download_button(
                            label="📥 Baixar Excel",
                            data=excel_data,
                            file_name="network_statistics.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            else:
                st.info("ℹ️ Ajuste os parâmetros acima para gerar um grafo. O grafo será atualizado automaticamente.")
    
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
        
        st.markdown("---")
        render_rag_chat(st.session_state.df)
        
        if st.button("🔄 Recomeçar do início"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()