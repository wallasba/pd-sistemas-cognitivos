import streamlit as st
from src.rag_pipeline import RAGPipeline

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================
st.set_page_config(
    page_title="Assistente de Pesquisa",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# TÍTULO E DESCRIÇÃO PRINCIPAL
# =====================================================
st.title("🤖 Assistente de Pesquisa")
st.caption("Baseado em artigos científicos de 7 repositórios (arXiv, OpenAlex, PubMed, Crossref, Europe PMC, Zenodo, DOAJ).")

# =====================================================
# SEÇÃO DE INSTRUÇÕES (MAPA DE INSTRUÇÕES)
# =====================================================
with st.expander("📖 Como usar este assistente - Mapa de Instruções", expanded=False):
    st.markdown("""
    ### 🎯 Objetivo
    Este assistente permite que você faça perguntas em **linguagem natural** sobre o corpus de artigos científicos.  
    Ele usa um pipeline RAG (Recuperação + Geração) para buscar os trechos mais relevantes e gerar uma resposta fundamentada.

    ---

    ### 🛠️ Configurações (Barra Lateral)
    - **Técnica de Prompt**: 
      - `Zero-Shot`: Pergunta direta, sem exemplos.
      - `Few-Shot`: Fornece exemplos de perguntas e respostas ideais (melhor para perguntas específicas).
      - `Chain-of-Thought`: Pede que o modelo raciocine passo a passo (útil para perguntas complexas).
    - **Chunks recuperados (k)**: Número de trechos de artigos usados como contexto (quanto maior, mais informação, mas pode diluir a resposta).
    - **Temperatura**: Controla a criatividade da resposta (0.0 = mais precisa/factual; 1.0 = mais criativa/divergente).

    ---

    ### 💬 Como fazer uma pergunta
    1. Digite sua pergunta no campo de chat (ex: *"Quais são as aplicações de deep learning na infraestrutura de transportes?"*).
    2. Pressione Enter ou clique no ícone de enviar.
    3. Aguarde enquanto o sistema recupera os trechos relevantes e gera a resposta.

    ---

    ### 📚 Entendendo a resposta
    - **Resposta principal**: Texto gerado pelo modelo com base nos trechos recuperados.
    - **Expansor "📚 Ver trechos recuperados"**: Mostra os trechos exatos dos artigos usados como fonte, com o **score de similaridade** (quanto maior, mais relevante).
    - Se a resposta disser que não encontrou informação, tente reformular a pergunta com termos mais específicos.

    ---

    ### 💡 Exemplos de perguntas (use as máscaras abaixo ou copie e cole)
    Abaixo estão perguntas pré-definidas para testar rapidamente o sistema. Clique em **"Usar esta pergunta"** para preenchê-la automaticamente no chat.
    """)

    # =====================================================
    # MÁSCARAS DE CONSULTA (TEMPLATES)
    # =====================================================
    st.markdown("#### 🔍 Máscaras de Consulta (Exemplos)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🧠 Principais aplicações de IA", key="ex1", use_container_width=True):
            st.session_state.query_mask = "Quais são as principais aplicações de inteligência artificial mencionadas nos artigos?"
            st.rerun()
        
        if st.button("📊 Tendências", key="ex2", use_container_width=True):
            st.session_state.query_mask = "Quais foram os tópicos de IA mais pesquisados segundo o corpus?"
            st.rerun()
    
    with col2:
        if st.button("🧬 PLN com Transformers", key="ex3", use_container_width=True):
            st.session_state.query_mask = "O que os artigos dizem sobre o uso de transformers em processamento de linguagem natural?"
            st.rerun()
        
        if st.button("🤖 Aprendizado por Reforço", key="ex4", use_container_width=True):
            st.session_state.query_mask = "Quais são as limitações e avanços do aprendizado por reforço para a engenharia civil?"
            st.rerun()
    
    with col3:
        if st.button("🏥 Ética em IA", key="ex5", use_container_width=True):
            st.session_state.query_mask = "Questões éticas e de viés em IA são discutidas nos artigos do corpus?"
            st.rerun()
        
        if st.button("📈 Modelos Preditivos", key="ex6", use_container_width=True):
            st.session_state.query_mask = "Quais técnicas de modelagem preditiva e estatística são mais citadas?"
            st.rerun()
    
    st.markdown("---")
    st.info("💡 **Dica:** Você pode copiar qualquer uma das perguntas acima e colar no chat, ou clicar no botão para preenchê-la automaticamente.")

# =====================================================
# CARREGAMENTO DO PIPELINE (CACHEADO)
# =====================================================
@st.cache_resource
def load_pipeline():
    return RAGPipeline(
        corpus_path="./data/corpus_ia.csv",
        llm_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",  # Modelo leve
        embedding_model_name="all-MiniLM-L6-v2",
        chunk_size=300,
        chunk_overlap=50,
        k_retrieval=5
    )

if 'pipeline' not in st.session_state:
    with st.spinner("⏳ Carregando modelos (pode levar alguns minutos na primeira execução)..."):
        st.session_state.pipeline = load_pipeline()
        st.success("✅ Pipeline carregado com sucesso!")

# =====================================================
# BARRA LATERAL - GUIA RÁPIDO E CONFIGURAÇÕES
# =====================================================
with st.sidebar:
    st.header("⚙️ Configurações")
    
    st.markdown("""
    **📌 Guia Rápido**
    - Ajuste os parâmetros abaixo para controlar a qualidade e o comportamento das respostas.
    - Consulte a seção **"📖 Como usar"** na página principal para mais detalhes.
    """)
    
    prompt_type = st.selectbox(
        "Técnica de Prompt",
        ["zero_shot", "few_shot", "cot"],
        format_func=lambda x: {
            "zero_shot": "Zero-Shot (direto)",
            "few_shot": "Few-Shot (com exemplos)",
            "cot": "Chain-of-Thought (raciocínio)"
        }[x],
        help="Zero-Shot: pergunta direta. Few-Shot: inclui exemplos. CoT: raciocínio passo a passo."
    )
    
    k = st.slider(
        "Chunks recuperados (k)", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Número de trechos de artigos usados como contexto. Valores mais altos trazem mais informação, mas podem diluir a precisão."
    )
    
    temperature = st.slider(
        "Temperatura", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.1,
        help="0.0 = respostas mais precisas/factuais. 1.0 = mais criativas/divergentes."
    )
    
    st.divider()
    
    st.caption("""
    **Sobre o sistema:**
    - 🤖 Modelo: SmolLM2-360M-Instruct (local)
    - 🔍 Embeddings: all-MiniLM-L6-v2
    - 📂 Busca: FAISS (similaridade por cosseno)
    - 🔒 100% local, sem APIs externas
    """)

# =====================================================
# INICIALIZAÇÃO DO HISTÓRICO DE MENSAGENS
# =====================================================
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Olá! Sou seu assistente de pesquisa. Pergunte-me sobre tendências, métodos ou artigos específicos do corpus de IA.\n\n💡 *Experimente usar uma das máscaras de consulta acima ou digite sua própria pergunta.*"}
    ]

# =====================================================
# EXIBIÇÃO DO HISTÓRICO
# =====================================================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =====================================================
# GERENCIAMENTO DA MÁSCARA (QUERY MASK)
# =====================================================
# Se o usuário clicou em um botão de máscara, preenche o prompt
if 'query_mask' in st.session_state and st.session_state.query_mask:
    prompt = st.session_state.query_mask
    st.session_state.query_mask = None  # Reseta após usar
else:
    prompt = st.chat_input("Digite sua pergunta sobre o corpus de IA...")

# =====================================================
# PROCESSAMENTO DA CONSULTA
# =====================================================
if prompt:
    # Adiciona pergunta do usuário ao histórico
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gera a resposta
    with st.chat_message("assistant"):
        with st.spinner("🔍 Pesquisando no corpus e gerando resposta..."):
            try:
                result = st.session_state.pipeline.answer(
                    query=prompt,
                    prompt_type=prompt_type,
                    k=k,
                    temperature=temperature
                )
                
                # Exibe a resposta
                st.markdown(result['answer'])
                
                # Expansor com o contexto recuperado (transparência)
                with st.expander("📚 Ver trechos recuperados (contexto)"):
                    if result['retrieved_context']:
                        for i, r in enumerate(result['retrieved_context']):
                            st.caption(f"**Trecho {i+1}** (Score: {r['score']:.4f})")
                            st.caption(f"*Fonte:* {r['metadata'].get('title', 'Sem título')}")
                            st.text(r['chunk'][:500] + "..." if len(r['chunk']) > 500 else r['chunk'])
                            st.divider()
                    else:
                        st.info("Nenhum trecho recuperado para esta consulta.")
                
                # Adiciona resposta ao histórico
                st.session_state.messages.append({"role": "assistant", "content": result['answer']})
            
            except Exception as e:
                st.error(f"⚠️ Erro ao processar a consulta: {e}")
                st.stop()

# =====================================================
# RODAPÉ
# =====================================================
st.sidebar.divider()
st.sidebar.caption("📚 Projeto de Sistemas Cognitivos com LLMs - 2026")