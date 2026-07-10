# ui/sidebar.py
import streamlit as st
import pandas as pd
import os
from src.preprocessor import preprocess_abstracts

def render_sidebar():
    """Renderiza a barra lateral com progresso, upload e configurações."""
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
    
    # ============================================================
    # CONFIGURAÇÃO DA API KEY (GROQ)
    # ============================================================
    st.sidebar.header("🔑 Configuração da API")
    
    # Verifica se a chave já está no ambiente ou session_state
    default_key = os.getenv("GROQ_API_KEY") or st.session_state.get("groq_api_key", "")
    
    groq_key = st.sidebar.text_input(
        "Chave da API Groq (opcional se estiver no .env):",
        value=default_key,
        type="password",
        help="Insira sua chave da Groq. Se já estiver no arquivo .env, deixe em branco."
    )
    
    # Armazena a chave no session_state para uso global
    if groq_key:
        st.session_state.groq_api_key = groq_key
    elif default_key:
        st.session_state.groq_api_key = default_key
    
    st.sidebar.divider()
    
    # ============================================================
    # ESTATÍSTICAS DO CORPUS (se carregado)
    # ============================================================
    if st.session_state.df is not None and not st.session_state.df.empty:
        st.sidebar.metric("📄 Artigos carregados", len(st.session_state.df))
        if st.session_state.research_context:
            st.sidebar.caption(f"Tema: {st.session_state.research_context[:50]}...")
    
    st.sidebar.divider()
    
    # ============================================================
    # UPLOAD DE CORPUS
    # ============================================================
    st.sidebar.header("📂 Carregar Corpus Existente")
    
    with st.sidebar.expander("📋 Estrutura necessária do arquivo"):
        st.markdown("""
        O arquivo (CSV, XLS ou XLSX) deve conter:
        - **`title`** (texto)
        - **`abstract`** (texto)
        - **`source`** (texto)
        - **`year`** (inteiro)
        - **`authors`** (texto)
        - **`affiliations`** (texto, separado por `;`)
        """)
        st.caption("Colunas adicionais são permitidas.")
    
    uploaded_file = st.sidebar.file_uploader(
        "Faça upload do seu corpus (CSV, XLS, XLSX)",
        type=['csv', 'xls', 'xlsx'],
        help="Arquivo com a estrutura descrita acima."
    )
    
    if uploaded_file is not None:
        try:
            # Detecta extensão
            file_extension = uploaded_file.name.split('.')[-1].lower()
            if file_extension == 'csv':
                df_raw = pd.read_csv(uploaded_file)
            elif file_extension in ['xls', 'xlsx']:
                df_raw = pd.read_excel(uploaded_file, engine='openpyxl' if file_extension == 'xlsx' else 'xlrd')
            else:
                st.sidebar.error("Formato não suportado.")
                st.stop()
            
            required_cols = ['title', 'abstract', 'source', 'year']
            missing_cols = [col for col in required_cols if col not in df_raw.columns]
            
            if missing_cols:
                st.sidebar.error(f"❌ Colunas obrigatórias ausentes: {', '.join(missing_cols)}")
            else:
                st.sidebar.success(f"✅ Arquivo válido! {len(df_raw)} registros.")
                
                if st.sidebar.button("📥 Carregar este Corpus", use_container_width=True):
                    with st.spinner("Pré-processando..."):
                        df = preprocess_abstracts(df_raw)
                        if 'affiliations' not in df.columns:
                            df['affiliations'] = [[] for _ in range(len(df))]
                        if 'authors' not in df.columns:
                            df['authors'] = ""
                        st.session_state.df = df
                        st.session_state.step = 4  # vai direto para análise
                        st.success(f"✅ Corpus carregado! {len(df)} artigos.")
                        st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Erro: {e}")
    
    st.sidebar.divider()
    st.sidebar.caption("🧠 Assistente de Pesquisa Científica - v2.0")