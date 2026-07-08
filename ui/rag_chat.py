# ui/rag_chat.py
import streamlit as st
import tempfile
from src.rag_pipeline import RAGPipeline

def render_rag_chat(df):
    st.subheader("💬 Assistente de Pesquisa (RAG)")
    st.markdown("Faça perguntas em linguagem natural sobre o corpus. O assistente usará os artigos para fundamentar suas respostas.")
    
    if df is None or df.empty:
        st.info("Carregue um corpus para usar o chat RAG.")
        return

    if st.session_state.rag_pipeline is None:
        with st.spinner("Carregando pipeline RAG (embeddings + LLM) – primeira vez pode demorar..."):
            temp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
            df[['title', 'abstract_clean', 'source', 'year']].to_csv(temp_csv.name, index=False)
            temp_csv.close()
            try:
                st.session_state.rag_pipeline = RAGPipeline(
                    corpus_path=temp_csv.name,
                    text_column='abstract_clean',
                    llm_model_name="HuggingFaceTB/SmolLM2-1.7B-Instruct",
                    embedding_model_name="all-MiniLM-L6-v2",
                    k_retrieval=10
                )
                st.success("Pipeline RAG carregado!")
                if hasattr(st.session_state.rag_pipeline, 'df'):
                    years = st.session_state.rag_pipeline.df['year'].dropna().unique()
                    years_sorted = sorted(years)
                    st.info(f"📊 Anos disponíveis no corpus: {years_sorted}")
            except Exception as e:
                st.error(f"Erro ao carregar RAG: {e}")
                return

    question_masks = [
        "Quais são os desafios éticos mencionados nos artigos?",
        "Quais as tendências recentes (2026) em IA?",
        "Quais são as principais aplicações da IA mencionadas?",
        "Quais são as principais instituições que publicam sobre deep learning?",
        "Quem são os autores mais prolíficos na área de PLN?",
        "qual o artigo mais recente do corpus? qual o ano?"
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
                    temperature=0.2,
                    max_new_tokens=600
                )
                answer = result['answer']
                st.markdown("**Resposta:**")
                st.write(answer)
                
                if "Não encontrei" in answer or "não foi possível" in answer:
                    st.info("💡 Tente reformular a pergunta com termos mais específicos ou amplie o corpus.")
                
                with st.expander("📚 Ver trechos recuperados (contexto)"):
                    if result.get('retrieved_context'):
                        for i, ctx in enumerate(result['retrieved_context']):
                            st.caption(f"Trecho {i+1} (score: {ctx['score']:.4f})")
                            st.caption(f"Fonte: {ctx['metadata']['title']} ({ctx['metadata']['year']})")
                            st.text(ctx['chunk'][:500] + "...")
                            st.divider()
                    else:
                        st.info("Nenhum trecho recuperado.")
        else:
            st.warning("Digite uma pergunta.")