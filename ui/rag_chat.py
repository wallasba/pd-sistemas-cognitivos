import streamlit as st
import tempfile
from src.rag_pipeline import RAGPipeline
from src.preprocessor import preprocess_abstracts

# ui/rag_chat.py (trecho modificado)

def render_rag_chat(df):
    if df is None or df.empty:
        st.info("Carregue um corpus para usar o chat RAG.")
        return

    if st.session_state.rag_pipeline is None:
        with st.spinner("Carregando pipeline RAG (embeddings + LLM) – primeira vez pode demorar..."):
            temp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            # Salva as colunas necessárias – incluindo 'abstract_clean'
            df[['title', 'abstract_clean', 'source', 'year']].to_csv(temp_csv.name, index=False)
            temp_csv.close()
            try:
                st.session_state.rag_pipeline = RAGPipeline(
                    corpus_path=temp_csv.name,
                    text_column='abstract_clean',  # <-- INFORMAR A COLUNA CORRETA
                    llm_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
                    embedding_model_name="all-MiniLM-L6-v2"
                )
                st.success("Pipeline RAG carregado!")
            except Exception as e:
                st.error(f"Erro ao carregar RAG: {e}")
                return
    
    # Máscaras de perguntas
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
            st.warning("Digite uma pergunta.")# ui/rag_chat.py (versão melhorada)

import streamlit as st
import tempfile
from src.rag_pipeline import RAGPipeline

def render_rag_chat(df):
    """Renderiza a interface do chat RAG."""
    st.subheader("💬 Assistente de Pesquisa (RAG)")
    st.markdown("Faça perguntas em linguagem natural sobre o corpus. O assistente usará os artigos para fundamentar suas respostas.")
    
    if df is None or df.empty:
        st.info("Carregue um corpus para usar o chat RAG.")
        return

    if st.session_state.rag_pipeline is None:
        with st.spinner("Carregando pipeline RAG (embeddings + LLM) – primeira vez pode demorar..."):
            temp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            # Salva as colunas necessárias – incluindo 'abstract_clean'
            df[['title', 'abstract_clean', 'source', 'year']].to_csv(temp_csv.name, index=False)
            temp_csv.close()
            try:
                st.session_state.rag_pipeline = RAGPipeline(
                    corpus_path=temp_csv.name,
                    text_column='abstract_clean',
                    llm_model_name="HuggingFaceTB/SmolLM2-360M-Instruct",
                    embedding_model_name="all-MiniLM-L6-v2",
                    k_retrieval=7  # Aumenta o número de chunks
                )
                st.success("Pipeline RAG carregado!")
            except Exception as e:
                st.error(f"Erro ao carregar RAG: {e}")
                return

    # Máscaras de perguntas atualizadas
    question_masks = [
        "Quais são os desafios éticos mencionados nos artigos?",
        "Quais são as principais instituições que publicam sobre deep learning?",
        "Quem são os autores mais prolíficos na área de PLN?",
        "Quais termos estão mais associados à instituição MIT?",
        "Quais as tendências recentes (2026) em IA?",
        "Como a IA está sendo aplicada na engenharia?",
    ]
    selected_question = st.selectbox("Escolha uma pergunta de exemplo (ou digite a sua):",
                                     ["(Digitar própria)"] + question_masks)
    user_question = st.text_input("Sua pergunta:", 
                                  value="" if selected_question == "(Digitar própria)" else selected_question)
    
    if st.button("Enviar pergunta", type="primary"):
        if user_question and st.session_state.rag_pipeline:
            with st.spinner("Processando..."):
                result = st.session_state.rag_pipeline.answer(
                    user_question,
                    prompt_type='zero_shot',
                    temperature=0.3,
                    max_new_tokens=350
                )
                answer = result['answer']
                st.markdown("**Resposta:**")
                st.write(answer)
                
                # Se a resposta for a mensagem padrão, sugere reformular
                if "Não encontrei informações" in answer:
                    st.info("💡 Tente reformular a pergunta com termos mais específicos ou amplie o corpus.")
                
                with st.expander("📚 Ver trechos recuperados (contexto)"):
                    if result['retrieved_context']:
                        for i, ctx in enumerate(result['retrieved_context']):
                            st.caption(f"Trecho {i+1} (score: {ctx['score']:.4f})")
                            st.caption(f"Fonte: {ctx['metadata']['title']}")
                            st.text(ctx['chunk'][:400] + "...")
                            st.divider()
                    else:
                        st.info("Nenhum trecho recuperado.")
        else:
            st.warning("Digite uma pergunta.")