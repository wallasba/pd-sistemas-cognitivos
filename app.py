import streamlit as st
from ui.sidebar import render_sidebar
from ui.wizard import render_wizard
import os
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Assistente de Pesquisa Científica",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🧠 Assistente de Pesquisa Científica (Universal)")
st.caption("Guia você desde a definição do problema até a análise de dados, em qualquer área do conhecimento.")

# ============================================================
# INICIALIZAÇÃO DO ESTADO
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
if 'initial_keywords' not in st.session_state:
    st.session_state.initial_keywords = []

# ============================================================
# BARRA LATERAL
# ============================================================
render_sidebar()

# ============================================================
# CONTEÚDO PRINCIPAL (WIZARD)
# ============================================================
render_wizard()