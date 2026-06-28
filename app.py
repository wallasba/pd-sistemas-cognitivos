import streamlit as st
from src.rag_pipeline import RAGPipeline
from src.config import MODEL_CACHE_DIR

st.set_page_config(page_title="Assistente de Pesquisa em IA", layout="wide")
st.title("🤖 Assistente de Pesquisa em Inteligência Artificial")

# Inicializa o pipeline com cache local
@st.cache_resource
def load_pipeline():
    return RAGPipeline(
        corpus_path="./data/corpus_ia.csv",
        llm_model_name="microsoft/Phi-3-mini-4k-instruct",
        cache_dir=MODEL_CACHE_DIR
    )

if 'pipeline' not in st.session_state:
    with st.spinner("Carregando modelos (use cache local se disponível)..."):
        st.session_state.pipeline = load_pipeline()

# ... (restante do código do app permanece igual)

# Sidebar com configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    prompt_type = st.selectbox("Técnica de Prompt", ["zero_shot", "few_shot", "cot"])
    k = st.slider("Chunks recuperados (k)", 1, 10, 5)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.7)

# Chat
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Pergunte sobre o corpus de IA!"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Digite sua pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Gerando resposta..."):
            result = st.session_state.pipeline.answer(
                prompt, 
                prompt_type=prompt_type,
                k=k,
                temperature=temperature
            )
            st.markdown(result['answer'])
            with st.expander("📚 Ver contexto recuperado"):
                for i, r in enumerate(result['retrieved_context']):
                    st.caption(f"Trecho {i+1} (score: {r['score']:.3f}) - {r['metadata']['title']}")
                    st.text(r['chunk'][:300])
            st.session_state.messages.append({"role": "assistant", "content": result['answer']})