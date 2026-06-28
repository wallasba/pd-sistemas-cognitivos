import streamlit as st
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="Assistente de Pesquisa em IA", layout="wide")
st.title("🤖 Assistente de Pesquisa em Inteligência Artificial")

@st.cache_resource
def load_pipeline():
    return RAGPipeline(
        corpus_path="./data/corpus_ia.csv",
        llm_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
        embedding_model_name="all-MiniLM-L6-v2",
        chunk_size=300,
        chunk_overlap=50,
        k_retrieval=5
    )

if 'pipeline' not in st.session_state:
    with st.spinner("Carregando modelos (pode levar alguns minutos na primeira execução)..."):
        st.session_state.pipeline = load_pipeline()

# Sidebar com configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    prompt_type = st.selectbox(
        "Técnica de Prompt",
        ["zero_shot", "few_shot", "cot"],
        format_func=lambda x: {
            "zero_shot": "Zero-Shot",
            "few_shot": "Few-Shot",
            "cot": "Chain-of-Thought"
        }[x]
    )
    k = st.slider("Chunks recuperados (k)", 1, 10, 5)
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.7, 0.1)

# Inicializa histórico de mensagens
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Pergunte sobre o corpus de IA!"}]

# Exibe histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada do usuário
if prompt := st.chat_input("Digite sua pergunta sobre o corpus de IA..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("Pesquisando e gerando resposta..."):
            try:
                result = st.session_state.pipeline.answer(
                    query=prompt,
                    prompt_type=prompt_type,
                    k=k,
                    temperature=temperature
                )
                st.markdown(result['answer'])
                
                # Expansor com contexto recuperado
                with st.expander("📚 Ver trechos recuperados (contexto)"):
                    for i, r in enumerate(result['retrieved_context']):
                        st.caption(f"**Trecho {i+1}** (Score: {r['score']:.4f})")
                        st.caption(f"*Fonte:* {r['metadata'].get('title', 'Sem título')}")
                        st.text(r['chunk'][:400] + "...")
                        st.divider()
                
                # Adiciona resposta ao histórico
                st.session_state.messages.append({"role": "assistant", "content": result['answer']})
            
            except Exception as e:
                st.error(f"Erro ao gerar resposta: {e}")

# Rodapé
st.sidebar.divider()
st.sidebar.caption("""
**Sobre:** Assistente de pesquisa com RAG local.
- Modelos: Hugging Face Transformers + Sentence-Transformers
- Busca: FAISS (similaridade por cosseno)
- 100% local, sem APIs externas
""")