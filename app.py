import streamlit as st
import pandas as pd
import os
from src.data_collector import collect_articles
from src.preprocessor import preprocess_abstracts
from src.analyzer import get_stats, generate_wordcloud, plot_distributions
from src.graph_builder import (
    build_coauthorship_graph,
    build_institution_collaboration_graph,
    build_term_institution_graph
)
from src.rag_pipeline import RAGPipeline
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import tempfile
import time

st.set_page_config(page_title="Assistente de Pesquisa IA - Integrado", layout="wide")
st.title("🔬 Assistente de Pesquisa em Inteligência Artificial")

# ============================================================
# Inicialização do session_state
# ============================================================
if 'df' not in st.session_state:
    st.session_state.df = None
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None

# ============================================================
# Máscaras de busca (placeholders para pesquisador sênior)
# ============================================================
MASKS = {
    "Deep Learning em Saúde": '"deep learning" AND ("medical" OR "health")',
    "PLN e Transformers": '"natural language processing" AND ("transformer" OR "BERT")',
    "IA na Educação": '"artificial intelligence" AND "education"',
    "Robótica e Aprendizado por Reforço": '"reinforcement learning" AND "robotics"',
    "Ética em IA": '"AI ethics" OR "bias"',
    "Tendências em IA (2023)": '"artificial intelligence" AND "2023"'
}

# ============================================================
# SIDEBAR – Informações e estados
# ============================================================
with st.sidebar:
    st.header("📊 Status")
    if st.session_state.df is not None:
        st.success(f"Corpus carregado: {len(st.session_state.df)} artigos")
    else:
        st.warning("Nenhum corpus carregado.")
    st.divider()
    st.caption("Desenvolvido para pesquisadores seniores.")

# ============================================================
# ABA 1: COLETA
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["📥 Coleta", "📈 Análise", "🕸️ Grafos", "💬 Chat RAG"])

with tab1:
    st.header("1️⃣ Definição da Consulta")
    # Placeholder com máscaras
    mask_option = st.selectbox("Escolha uma máscara de exemplo (ou digite sua própria query):",
                               ["(Personalizada)"] + list(MASKS.keys()))
    if mask_option != "(Personalizada)":
        query_placeholder = MASKS[mask_option]
    else:
        query_placeholder = '"artificial intelligence" AND "civil engineering"'
    
    query = st.text_input("Query de pesquisa (use AND, OR, NOT e aspas):",
                          value=query_placeholder,
                          help="Exemplo: 'machine learning' AND 'healthcare' NOT 'robotics'")
    
    col1, col2 = st.columns(2)
    with col1:
        year_start = st.slider("Ano inicial", 2000, 2026, 2015)
        year_end = st.slider("Ano final", 2000, 2026, 2023)
    with col2:
        max_results = st.slider("Máximo de artigos por fonte", 100, 1000, 300, step=50)
        sources = st.multiselect(
            "Fontes (selecione uma ou mais)",
            ['arxiv', 'openalex', 'pubmed'],
            default=['arxiv', 'openalex']
        )
    
    st.caption("💡 Para fontes adicionais (Crossref, Zenodo, etc.), consulte a documentação.")
    
    # Botão para coletar
    if st.button("🚀 Coletar Artigos", type="primary"):
        if not query or not sources:
            st.warning("Preencha a query e selecione pelo menos uma fonte.")
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
                    st.error("Nenhum artigo encontrado. Tente ajustar a query ou período.")
                else:
                    # Pré-processa
                    df = preprocess_abstracts(df_raw)
                    st.session_state.df = df
                    st.success(f"✅ {len(df)} artigos coletados e pré-processados!")
                    
                    # Salva em cache (opcional)
                    df.to_csv("data/corpus_cache.csv", index=False)
                    st.info("Corpus salvo em cache (data/corpus_cache.csv).")
    
    # Opção para carregar corpus existente
    st.divider()
    st.subheader("📂 Carregar corpus existente")
    uploaded_file = st.file_uploader("Carregue um arquivo CSV (ex: corpus_ia.csv)", type=['csv'])
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
        df = preprocess_abstracts(df_raw)
        st.session_state.df = df
        st.success(f"✅ Corpus carregado: {len(df)} artigos.")

# ============================================================
# ABA 2: ANÁLISE
# ============================================================
with tab2:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("📊 Estatísticas Descritivas")
        stats = get_stats(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de artigos", stats['total'])
        col2.metric("Fontes distintas", len(stats['sources']))
        col3.metric("Período", f"{min(stats['years'])} - {max(stats['years'])}")
        
        # Gráficos
        fig = plot_distributions(df)
        st.pyplot(fig)
        
        # Nuvem de palavras
        st.subheader("☁️ Nuvem de Palavras")
        wc = generate_wordcloud(df)
        fig_wc, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc)
    else:
        st.info("Nenhum corpus carregado. Vá para a aba 'Coleta'.")

# ============================================================
# ABA 3: GRAFOS
# ============================================================
with tab3:
    if st.session_state.df is not None:
        df = st.session_state.df
        st.subheader("🕸️ Grafos de Conhecimento")
        st.markdown("""
        **Clique nos botões abaixo para gerar diferentes tipos de grafos interativos:**
        - **Coautoria**: autores que publicaram juntos.
        - **Instituições**: colaboração entre instituições.
        - **Termos × Instituições**: termos frequentes associados a instituições.
        """)
        
        col_grafos = st.columns(3)
        with col_grafos[0]:
            if st.button("📌 Gerar Grafo de Coautoria"):
                with st.spinner("Construindo grafo de coautoria..."):
                    G = build_coauthorship_graph(df, min_coauthors=2)
                    if G.number_of_nodes() == 0:
                        st.warning("Nenhuma coautoria encontrada.")
                    else:
                        # Converte para PyVis
                        net = Network(height="600px", width="100%", notebook=False)
                        net.from_nx(G)
                        net.show("graph_coauthorship.html")
                        with open("graph_coauthorship.html", "r", encoding="utf-8") as f:
                            html = f.read()
                        components.html(html, height=700)
        
        with col_grafos[1]:
            if st.button("🏛️ Gerar Grafo de Instituições"):
                with st.spinner("Construindo grafo de instituições..."):
                    G = build_institution_collaboration_graph(df)
                    if G.number_of_nodes() == 0:
                        st.warning("Nenhuma colaboração entre instituições encontrada.")
                    else:
                        net = Network(height="600px", width="100%", notebook=False)
                        net.from_nx(G)
                        net.show("graph_institutions.html")
                        with open("graph_institutions.html", "r", encoding="utf-8") as f:
                            html = f.read()
                        components.html(html, height=700)
        
        with col_grafos[2]:
            if st.button("🔗 Gerar Grafo Termos × Instituições"):
                with st.spinner("Construindo grafo termos-instituições..."):
                    G = build_term_institution_graph(df, top_terms=30)
                    if G.number_of_nodes() == 0:
                        st.warning("Nenhuma relação termo-instituição encontrada.")
                    else:
                        net = Network(height="600px", width="100%", notebook=False)
                        net.from_nx(G)
                        net.show("graph_terms_institutions.html")
                        with open("graph_terms_institutions.html", "r", encoding="utf-8") as f:
                            html = f.read()
                        components.html(html, height=700)
    else:
        st.info("Nenhum corpus carregado para gerar grafos.")

# ============================================================
# ABA 4: CHAT RAG
# ============================================================
with tab4:
    st.subheader("💬 Assistente de Pesquisa (RAG)")
    st.markdown("""
    Faça perguntas em linguagem natural sobre o corpus. O assistente usará os artigos coletados para fundamentar suas respostas.
    """)
    
    if st.session_state.df is not None and not st.session_state.df.empty:
        # Inicializa o pipeline RAG se ainda não existir
        if st.session_state.rag_pipeline is None:
            with st.spinner("Carregando o pipeline RAG (embeddings + LLM) – primeira vez pode demorar..."):
                # Salva o corpus limpo em um CSV temporário para o RAG
                temp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                st.session_state.df[['title', 'abstract_clean', 'source', 'year']].to_csv(temp_csv.name, index=False)
                temp_csv.close()
                try:
                    st.session_state.rag_pipeline = RAGPipeline(
                        corpus_path=temp_csv.name,
                        llm_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
                        embedding_model_name="all-MiniLM-L6-v2"
                    )
                    st.success("Pipeline RAG carregado!")
                except Exception as e:
                    st.error(f"Erro ao carregar RAG: {e}")
        
        # Interface de chat
        if st.session_state.rag_pipeline is not None:
            # Máscaras de perguntas para pesquisador sênior
            question_masks = [
                "Quais são as principais instituições que publicam sobre deep learning?",
                "Quem são os autores mais prolíficos na área de PLN?",
                "Quais termos estão mais associados à instituição MIT?",
                "Quais as tendências recentes (2023) em IA?",
                "Como a IA está sendo aplicada na saúde?",
                "Quais são os desafios éticos mencionados nos artigos?"
            ]
            selected_question = st.selectbox("Escolha uma pergunta de exemplo (ou digite a sua):",
                                             ["(Digitar própria)"] + question_masks)
            user_question = st.text_input("Sua pergunta:", 
                                          value="" if selected_question == "(Digitar própria)" else selected_question)
            
            if st.button("Enviar pergunta"):
                if user_question and st.session_state.rag_pipeline:
                    with st.spinner("Processando..."):
                        result = st.session_state.rag_pipeline.answer(user_question)
                        st.markdown("**Resposta:**")
                        st.write(result['answer'])
                        with st.expander("📚 Ver trechos recuperados"):
                            for i, ctx in enumerate(result['retrieved_context']):
                                st.caption(f"Trecho {i+1} (score: {ctx['score']:.4f})")
                                st.caption(f"Fonte: {ctx['metadata']['title']}")
                                st.text(ctx['chunk'][:300] + "...")
                                st.divider()
                else:
                    st.warning("Digite uma pergunta.")
    else:
        st.info("Carregue um corpus na aba 'Coleta' para usar o chat RAG.")