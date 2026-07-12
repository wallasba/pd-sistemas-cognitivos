# ui/rag_chat.py
import streamlit as st
import tempfile
import os
from src.rag_pipeline import RAGPipeline

def render_rag_chat(df):
    st.subheader("💬 Assistente de Pesquisa (RAG)")
    st.markdown("Faça perguntas em linguagem natural sobre o corpus. O assistente usará os artigos para fundamentar suas respostas.")
    
    if df is None or df.empty:
        st.info("Carregue um corpus para usar o chat RAG.")
        return

    if st.session_state.rag_pipeline is None:
        with st.spinner("Carregando pipeline RAG (embeddings + Groq)..."):
            temp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            df[['title', 'abstract_clean', 'source', 'year']].to_csv(temp_csv.name, index=False)
            temp_csv.close()
            try:
                groq_api_key = st.session_state.get("groq_api_key") or os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    st.warning("⚠️ Chave da API Groq não encontrada. Configure-a na barra lateral.")
                    return

                st.session_state.rag_pipeline = RAGPipeline(
                    corpus_path=temp_csv.name,
                    text_column='abstract_clean',
                    embedding_model_name="all-MiniLM-L6-v2",
                    k_retrieval=10,
                    groq_api_key=groq_api_key,
                    api_model="llama-3.1-8b-instant",
                    response_language="português"
                )
                st.success("Pipeline RAG carregado (Groq)!")
                
                # ===== MENSAGEM REMOVIDA =====
                # A exibição dos anos foi removida para não poluir a interface.
                # ================================
                
            except Exception as e:
                st.error(f"Erro: {e}")
                return

    # Perguntas de exemplo
    question_masks = [
        # ===== ANÁLISE TEMPORAL =====
        "Quais são os anos disponíveis no corpus?",
        "Qual a distribuição temporal dos artigos?",
        "Quantos artigos foram publicados em 2023?",
        "Qual o artigo mais recente do corpus?",
        
        # ===== MÉTRICAS GERAIS =====
        "Quantos artigos o corpus contém?",
        "Quais são as principais fontes (repositórios) dos artigos?",
        "Quantos autores únicos existem no corpus?",
        "Quem são os autores mais frequentes?",
        
        # ===== ANÁLISE TEMÁTICA =====
        "Quais são os principais temas abordados nos artigos?",
        "Quais as tendências recentes (2026) em IA?",
        "Quais são as principais aplicações da IA mencionadas?",
        
        # ===== ANÁLISE ÉTICA E DESAFIOS =====
        "Quais são os desafios éticos mencionados nos artigos?",
        "Quais são as principais limitações da IA citadas?",
        
        # ===== ANÁLISE POR ÁREA (exemplos personalizáveis) =====
        "Qual o impacto da IA na engenharia de transportes?",
        "Como a IA está sendo aplicada na área da saúde?",
        "Quais são as principais colaborações institucionais mencionadas?"
    ]
    selected_question = st.selectbox("Escolha uma pergunta de exemplo (ou digite a sua):",
                                     ["(Digitar própria)"] + question_masks)
    user_question = st.text_input("Sua pergunta:", 
                                  value="" if selected_question == "(Digitar própria)" else selected_question)
    
    if st.button("Enviar pergunta", type="primary"):
        if user_question and st.session_state.rag_pipeline:
            with st.spinner("Processando no Groq..."):
                result = st.session_state.rag_pipeline.answer(user_question)
                st.markdown("**Resposta:**")
                st.write(result['answer'])
                
                with st.expander("📚 Ver trechos recuperados"):
                    if result.get('retrieved_context'):
                        for i, ctx in enumerate(result['retrieved_context']):
                            st.caption(f"Trecho {i+1} (score: {ctx['score']:.4f})")
                            st.caption(f"Fonte: {ctx['metadata']['title']} ({ctx['metadata']['year']})")
                            st.text(ctx['chunk'][:300] + "...")
                            st.divider()
                    else:
                        st.info("Nenhum trecho recuperado.")
        else:
            st.warning("Digite uma pergunta.")