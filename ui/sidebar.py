# ui/sidebar.py
import streamlit as st
import pandas as pd
from src.preprocessor import preprocess_abstracts

def render_sidebar():
    """Renderiza a barra lateral com progresso, upload e estatísticas."""
    st.sidebar.header("📌 Progresso da Pesquisa")
    steps = ["Definição", "Objetivos", "Estratégia", "Coleta", "Análise"]
    current = st.session_state.step - 1
    for i, step_name in enumerate(steps):
        if i < current:
            st.sidebar.success(f"✅ {step_name}")
        elif i == current:
            st.sidebar.info(f"🔵 {step_name}")
        else:
            st.sidebar.caption(f"⬜ {step_name}")
    
    st.sidebar.divider()
    
    # Estatísticas do corpus (se carregado)
    if st.session_state.df is not None:
        st.sidebar.metric("Artigos coletados", len(st.session_state.df))
        if st.session_state.research_context:
            st.sidebar.caption(f"Tema: {st.session_state.research_context[:50]}...")
    
    st.sidebar.divider()
    
    # ============================================================
    # UPLOAD DE CORPUS (CSV, XLS, XLSX)
    # ============================================================
    st.sidebar.header("📂 Carregar Corpus Existente")

    with st.sidebar.expander("📋 Estrutura necessária do arquivo"):
        st.markdown("""
        O arquivo (CSV, XLS ou XLSX) deve conter as seguintes colunas:
        - **`title`** (texto): Título do artigo.
        - **`abstract`** (texto): Resumo do artigo.
        - **`source`** (texto): Fonte (ex: arxiv, pubmed).
        - **`year`** (inteiro): Ano de publicação.
        - **`authors`** (texto): Autores separados por vírgula.
        - **`affiliations`** (texto): Instituições separadas por ponto e vírgula `;`.
        """)
        st.caption("💡 Colunas adicionais são permitidas, mas não obrigatórias.")

    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do seu corpus (CSV, XLS ou XLSX)",
        type=['csv', 'xls', 'xlsx'],
        help="Arquivo com a estrutura descrita acima."
    )

    if uploaded_file is not None:
        try:
            # Detecta extensão e lê adequadamente
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df_raw = pd.read_csv(uploaded_file)
            elif file_extension in ['xls', 'xlsx']:
                df_raw = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else 'xlrd')
            else:
                st.sidebar.error("Formato não suportado. Use CSV, XLS ou XLSX.")
                st.stop()
            
            # Verifica colunas obrigatórias
            required_cols = ['title', 'abstract', 'source', 'year']
            missing_cols = [col for col in required_cols if col not in df_raw.columns]
            
            if missing_cols:
                st.sidebar.error(f"❌ Colunas obrigatórias ausentes: {', '.join(missing_cols)}")
            else:
                st.sidebar.success(f"✅ Arquivo válido! {len(df_raw)} registros.")
                
                # Pré-visualização
                with st.sidebar.expander("📄 Pré-visualização (primeiras 5 linhas)"):
                    st.dataframe(df_raw.head(5))
                
                if st.sidebar.button("📥 Carregar este Corpus", use_container_width=True):
                    with st.spinner("Pré-processando..."):
                        # Aplica pré-processamento (limpeza, tokenização)
                        df = preprocess_abstracts(df_raw)
                        
                        # Garante colunas adicionais
                        if 'affiliations' not in df.columns:
                            df['affiliations'] = [[] for _ in range(len(df))]
                        if 'authors' not in df.columns:
                            df['authors'] = ""
                        
                        st.session_state.df = df
                        st.session_state.step = 4  # vai direto para análise
                        st.success(f"✅ Corpus carregado! {len(df)} artigos prontos.")
                        st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao ler o arquivo: {e}")
    
    st.sidebar.divider()
    st.sidebar.caption("🧠 Assistente de Pesquisa Científica - v2.0")