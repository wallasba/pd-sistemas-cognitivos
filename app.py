import streamlit as st
import pandas as pd
import tempfile
import os
from src.data_collector import collect_articles
from src.preprocessor import preprocess_abstracts
from src.analyzer import get_stats, generate_wordcloud, plot_distributions
from src.graph_builder import (
    build_coauthorship_graph,
    build_institution_collaboration_graph,
    build_term_institution_graph
)
from src.recommender import (
    suggest_objectives,
    suggest_hypotheses,
    suggest_search_terms,
    extract_keywords
)
from src.rag_pipeline import RAGPipeline
import matplotlib.pyplot as plt
from pyvis.network import Network
import streamlit.components.v1 as components
import nltk

# ============================================================
# Garante que os recursos do NLTK estejam disponíveis
# ============================================================
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# (restante do código)

# ============================================================
# Configuração da página
# ============================================================
st.set_page_config(page_title="Assistente de Pesquisa Científica", layout="wide")
st.title("🧠 Assistente de Pesquisa Científica (Universal)")
st.caption("Guia você desde a definição do problema até a análise de dados, em qualquer área do conhecimento.")

# ============================================================
# Inicialização do estado
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

# ============================================================
# Funções auxiliares para navegação
# ============================================================
def go_to_step(step):
    st.session_state.step = step
    st.rerun()

def next_step():
    st.session_state.step += 1
    st.rerun()

def prev_step():
    st.session_state.step -= 1
    st.rerun()

# ============================================================
# BARRA LATERAL - PROGRESSO E UPLOAD DE CORPUS
# ============================================================
with st.sidebar:
    st.header("📌 Progresso da Pesquisa")
    steps = ["Definição", "Objetivos", "Estratégia", "Coleta", "Análise"]
    current = st.session_state.step - 1
    for i, step_name in enumerate(steps):
        if i < current:
            st.success(f"✅ {step_name}")
        elif i == current:
            st.info(f"🔵 {step_name}")
        else:
            st.caption(f"⬜ {step_name}")
    
    st.divider()
    
    # ============================================================
    # UPLOAD DE CORPUS EXISTENTE (CSV, XLSX, XLS)
    # ============================================================
    st.header("📂 Carregar Corpus Existente")
    
    with st.expander("📋 Estrutura necessária do arquivo"):
        st.markdown("""
        O arquivo (CSV, XLSX ou XLS) deve conter as seguintes colunas:
        - **`title`** (texto): Título do artigo.
        - **`abstract`** (texto): Resumo do artigo.
        - **`source`** (texto): Fonte (ex: arxiv, pubmed).
        - **`year`** (inteiro): Ano de publicação.
        - **`authors`** (texto): Autores separados por vírgula.
        - **`affiliations`** (texto): Instituições separadas por ponto e vírgula `;`.
        """)
        st.caption("💡 Colunas adicionais são permitidas, mas não obrigatórias.")
    
    uploaded_file = st.file_uploader(
        "Faça upload do seu corpus (CSV, XLSX ou XLS)",
        type=['csv', 'xlsx', 'xls'],
        help="Arquivo com a estrutura descrita acima."
    )
    
    if uploaded_file is not None:
        # Verifica se o arquivo não está vazio
        if uploaded_file.size == 0:
            st.error("❌ O arquivo está vazio. Por favor, envie um arquivo com dados.")
        else:
            try:
                # ============================================================
                # DETECTA O FORMATO E LÊ O ARQUIVO
                # ============================================================
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    df_raw = pd.read_csv(
                        uploaded_file,
                        encoding='utf-8',
                        engine='python',
                        low_memory=False
                    )
                elif file_extension == 'xlsx':
                    df_raw = pd.read_excel(
                        uploaded_file,
                        engine='openpyxl'
                    )
                elif file_extension == 'xls':
                    df_raw = pd.read_excel(
                        uploaded_file,
                        engine='xlrd'
                    )
                else:
                    st.error(f"❌ Formato '{file_extension}' não suportado. Use CSV, XLSX ou XLS.")
                    st.stop()
                
                # ============================================================
                # VALIDAÇÃO E PRÉ-PROCESSAMENTO
                # ============================================================
                if df_raw.empty:
                    st.error("❌ O arquivo não contém dados. Verifique o conteúdo.")
                else:
                    # Converte colunas de texto para string (evita erro de .split())
                    text_cols = ['title', 'abstract', 'source', 'authors', 'affiliations']
                    for col in text_cols:
                        if col in df_raw.columns:
                            df_raw[col] = df_raw[col].fillna('').astype(str)
                        else:
                            # Se a coluna não existir, cria com valores vazios
                            df_raw[col] = ''
                    
                    # Verifica colunas obrigatórias (mínimo: title, abstract, source, year)
                    required_cols = ['title', 'abstract', 'source', 'year']
                    missing_cols = [col for col in required_cols if col not in df_raw.columns]
                    
                    if missing_cols:
                        st.error(f"❌ Colunas obrigatórias ausentes: {', '.join(missing_cols)}")
                    else:
                        st.success(f"✅ Arquivo válido! {len(df_raw)} registros encontrados.")
                        
                        # Botão para carregar
                        if st.button("📥 Carregar este Corpus", use_container_width=True):
                            with st.spinner("Pré-processando o corpus..."):
                                # Aplica o pré-processamento (limpeza, tokenização)
                                df = preprocess_abstracts(df_raw)
                                
                                # Garante que a coluna 'affiliations' exista como lista
                                if 'affiliations' not in df.columns:
                                    df['affiliations'] = [[] for _ in range(len(df))]
                                else:
                                    # Se já existe, converte strings com ';' para lista
                                    df['affiliations'] = df['affiliations'].apply(
                                        lambda x: [a.strip() for a in str(x).split(';') if a.strip()]
                                    )
                                
                                # Garante que 'authors' seja string
                                if 'authors' not in df.columns:
                                    df['authors'] = ""
                                else:
                                    df['authors'] = df['authors'].fillna('').astype(str)
                                
                                # Armazena no session_state
                                st.session_state.df = df
                                
                                # Avança diretamente para a etapa de análise (Step 4)
                                st.session_state.step = 4
                                st.success(f"✅ Corpus carregado com sucesso! {len(df)} artigos prontos para análise.")
                                st.rerun()
            
            except pd.errors.EmptyDataError:
                st.error("❌ O arquivo está vazio ou corrompido. Verifique o conteúdo.")
            except UnicodeDecodeError:
                # Fallback para encoding alternativo (apenas CSV)
                try:
                    uploaded_file.seek(0)
                    if file_extension == 'csv':
                        df_raw = pd.read_csv(
                            uploaded_file,
                            encoding='latin1',
                            engine='python',
                            low_memory=False
                        )
                        st.info("🔁 Leitura bem-sucedida com encoding 'latin1'. Considere salvar seu CSV em UTF-8.")
                        # Repete a validação (simplificada aqui, mas você pode chamar uma função)
                        # Para evitar duplicação, sugiro extrair a validação para uma função separada.
                    else:
                        st.error("❌ Erro de encoding em arquivo Excel. Verifique o formato.")
                except Exception as e2:
                    st.error(f"❌ Erro de encoding: {e2}")
            except Exception as e:
                st.error(f"❌ Erro ao ler o arquivo: {e}")
    
    # Métricas e informações na barra lateral
    if st.session_state.df is not None:
        st.metric("Artigos coletados", len(st.session_state.df))
    st.caption("Sessão: " + (st.session_state.research_context[:50] + "..." if st.session_state.research_context else ""))

# ============================================================
# ETAPA 1: Definição do Problema
# ============================================================
if st.session_state.step == 1:
    st.header("1️⃣ Defina seu problema de pesquisa")
    st.markdown("Descreva livremente o tema, a área e as principais questões que você deseja investigar.")
    
    context = st.text_area(
        "Qual é o seu tema/problema de pesquisa?",
        value=st.session_state.research_context,
        placeholder="Ex: Impacto da inteligência artificial na engenharia de transportes, com foco em otimização de tráfego.",
        height=150
    )
    st.session_state.research_context = context
    
    col1, col2 = st.columns(2)
    with col1:
        area = st.text_input("Área do conhecimento (opcional)", placeholder="Ex: Engenharia, Ciências da Saúde, Física...")
    with col2:
        keywords = st.text_input("Palavras-chave iniciais (separadas por vírgula)", placeholder="Ex: deep learning, otimização, tráfego")
    
    if st.button("Próximo →", type="primary"):
        if context.strip():
            st.session_state.research_context = context
            # Armazena palavras-chave para uso posterior
            st.session_state.initial_keywords = [k.strip() for k in keywords.split(',') if k.strip()]
            next_step()
        else:
            st.warning("Por favor, descreva o problema de pesquisa.")

# ============================================================
# ETAPA 2: Objetivos e Hipóteses
# ============================================================
elif st.session_state.step == 2:
    st.header("2️⃣ Objetivos e Hipóteses")
    st.markdown("Com base no contexto, defina os objetivos e hipóteses da sua pesquisa. Use as sugestões ou crie os seus.")
    
    # Gera recomendações (se houver corpus prévio, usa; senão, usa contexto)
    if st.session_state.df is not None:
        df_for_recommend = st.session_state.df
    else:
        df_for_recommend = None
    
    suggested_objs = suggest_objectives(st.session_state.research_context, df_for_recommend)
    st.markdown("**💡 Sugestões de objetivos (clique para adicionar):**")
    cols = st.columns(3)
    for i, obj in enumerate(suggested_objs):
        with cols[i % 3]:
            if st.button(f"📌 {obj[:50]}...", key=f"obj_{i}"):
                if obj not in st.session_state.objectives:
                    st.session_state.objectives.append(obj)
    
    # Área para editar/inserir objetivos
    objectives_text = st.text_area(
        "Seus objetivos (um por linha):",
        value="\n".join(st.session_state.objectives),
        height=100,
        placeholder="Ex: \nMapear o estado da arte...\nIdentificar lacunas..."
    )
    if st.button("Atualizar Objetivos"):
        st.session_state.objectives = [line.strip() for line in objectives_text.split('\n') if line.strip()]
        st.success("Objetivos atualizados!")
    
    # Hipóteses
    suggested_hyp = suggest_hypotheses(st.session_state.objectives, df_for_recommend)
    st.markdown("**💡 Sugestões de hipóteses (clique para adicionar):**")
    cols_h = st.columns(3)
    for i, hyp in enumerate(suggested_hyp):
        with cols_h[i % 3]:
            if st.button(f"🧪 {hyp[:50]}...", key=f"hyp_{i}"):
                if hyp not in st.session_state.hypotheses:
                    st.session_state.hypotheses.append(hyp)
    
    hypotheses_text = st.text_area(
        "Suas hipóteses (um por linha):",
        value="\n".join(st.session_state.hypotheses),
        height=80,
        placeholder="Ex: O uso de transformers melhora a acurácia..."
    )
    if st.button("Atualizar Hipóteses"):
        st.session_state.hypotheses = [line.strip() for line in hypotheses_text.split('\n') if line.strip()]
        st.success("Hipóteses atualizadas!")
    
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("← Voltar"):
            prev_step()
    with col_next:
        if st.button("Próximo →", type="primary"):
            if st.session_state.objectives:
                next_step()
            else:
                st.warning("Defina pelo menos um objetivo.")

# ============================================================
# ETAPA 3: Estratégia de Busca
# ============================================================
elif st.session_state.step == 3:
    # Se já houver um corpus carregado, sugere pular para a análise
    if st.session_state.df is not None and not st.session_state.df.empty:
        st.info("ℹ️ Você já possui um corpus carregado. Pode prosseguir diretamente para a análise.")
        if st.button("📊 Ir para Análise", type="primary"):
            st.session_state.step = 4
            st.rerun()
        st.markdown("---")
elif st.session_state.step == 3:
    st.header("3️⃣ Estratégia de Busca Bibliográfica")
    st.markdown("Construa sua query de busca com operadores booleanos e defina os parâmetros de coleta.")
    
    # Sugestões de termos de busca
    if st.session_state.df is not None:
        df_for_recommend = st.session_state.df
    else:
        df_for_recommend = None
    
    suggested_terms = suggest_search_terms(st.session_state.research_context, df_for_recommend)
    st.markdown("**💡 Termos e operadores sugeridos (clique para usar):**")
    cols_t = st.columns(4)
    for i, term in enumerate(suggested_terms[:8]):
        with cols_t[i % 4]:
            if st.button(f"🔍 {term}", key=f"term_{i}"):
                if st.session_state.query:
                    st.session_state.query += f" OR {term}"
                else:
                    st.session_state.query = term
                st.rerun()
    
    # Campo de query
    query = st.text_input(
        "Query de busca (use AND, OR, NOT e aspas):",
        value=st.session_state.query,
        placeholder='Ex: ("machine learning" OR "deep learning") AND "transportation" NOT "robotics"'
    )
    st.session_state.query = query
    
    # Parâmetros
    col1, col2 = st.columns(2)
    with col1:
        year_start = st.slider("Ano inicial", 2000, 2026, 2015)
        year_end = st.slider("Ano final", 2000, 2026, 2023)
    with col2:
        max_results = st.slider("Máximo de artigos por fonte", 100, 1000, 300, step=50)
        sources = st.multiselect(
            "Fontes (selecione uma ou mais):",
            ['arxiv', 'openalex', 'pubmed', 'crossref', 'europe_pmc', 'zenodo', 'doaj'],
            default=['arxiv', 'openalex', 'pubmed']
        )
    
    # Botão para coletar (já na etapa 3, podemos executar e ir para a 4)
    col_prev, col_collect, col_next = st.columns([1, 2, 1])
    with col_prev:
        if st.button("← Voltar"):
            prev_step()
    with col_collect:
        if st.button("🚀 Coletar e Avançar", type="primary"):
            if not query.strip():
                st.warning("Digite uma query de busca.")
            elif not sources:
                st.warning("Selecione pelo menos uma fonte.")
            else:
                with st.spinner("Coletando dados – isso pode levar alguns minutos..."):
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
                        df = preprocess_abstracts(df_raw)
                        st.session_state.df = df
                        st.success(f"✅ {len(df)} artigos coletados e pré-processados!")
                        next_step()
    with col_next:
        pass  # não há próximo manual, o botão de coleta já avança

# ============================================================
# ETAPA 4: Análise e Visualização (Grafos, Estatísticas)
# ============================================================
elif st.session_state.step == 4:
    st.header("4️⃣ Análise Exploratória e Refinamento")
    st.markdown("Explore os resultados da sua coleta. Use os grafos e estatísticas para refinar sua pesquisa.")
    
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Estatísticas
        stats = get_stats(df)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total de artigos", stats['total'])
        col2.metric("Fontes", len(stats['sources']))
        col3.metric("Período", f"{min(stats['years'])} - {max(stats['years'])}")
        
        # Distribuições
        fig = plot_distributions(df)
        st.pyplot(fig)
        
        # Nuvem de palavras
        st.subheader("☁️ Nuvem de Palavras (termos mais frequentes)")
        wc = generate_wordcloud(df)
        fig_wc, ax = plt.subplots(figsize=(10,5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig_wc)
        
        # Grafos interativos (em colunas)
        st.subheader("🕸️ Grafos de Conhecimento")
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            if st.button("📌 Coautoria"):
                G = build_coauthorship_graph(df)
                if G.number_of_nodes() > 0:
                    net = Network(height="500px", width="100%")
                    net.from_nx(G)
                    try:
                        net.write_html("graph_coauth.html", notebook=True)
                        with open("graph_coauth.html", "r", encoding="utf-8") as f:
                            html = f.read()
                        components.html(html, height=600)
                    except Exception as e:
                        st.error(f"Erro ao gerar o grafo: {e}")
                else:
                    st.info("Grafo pequeno ou sem coautoria.")
        with col_g2:
            if st.button("🏛️ Instituições"):
                G = build_institution_collaboration_graph(df)
                if G.number_of_nodes() > 0:
                    net = Network(height="500px", width="100%")
                    net.from_nx(G)
                    try:
                        net.write_html("graph_inst.html", notebook=True)
                        with open("graph_inst.html", "r", encoding="utf-8") as f:
                            html = f.read()
                        components.html(html, height=600)
                    except Exception as e:
                        st.error(f"Erro ao gerar o grafo: {e}")
                else:
                    st.info("Poucas instituições ou sem colaboração.")
        with col_g3:
            if st.button("🔗 Termos × Instituições"):
                G = build_term_institution_graph(df, top_terms=20)
                if G.number_of_nodes() > 0:
                    net = Network(height="500px", width="100%")
                    net.from_nx(G)
                    try:
                        net.write_html("graph_terms.html", notebook=True)
                        with open("graph_terms.html", "r", encoding="utf-8") as f:
                            html = f.read()
                        components.html(html, height=600)
                    except Exception as e:
                        st.error(f"Erro ao gerar o grafo: {e}")
                else:
                    st.info("Relações insuficientes.")
        
        # Refinamento: sugestões de novos termos com base no corpus
        st.subheader("💡 Recomendações para Refinamento")
        top_keywords = extract_keywords(df, top_n=10)
        if top_keywords:
            st.markdown("**Termos mais frequentes no corpus coletado:** " + ", ".join(top_keywords))
            st.markdown("**Sugestão:** Considere adicionar ou remover termos na sua query para refinar o escopo.")
        
        # Botão para voltar e refinar a busca
        col_prev, col_next = st.columns(2)
        with col_prev:
            if st.button("← Voltar para Estratégia de Busca"):
                st.session_state.step = 3
                st.rerun()
        with col_next:
            if st.button("✅ Concluir e Ir para o Resumo"):
                st.session_state.step = 5
                st.rerun()
    else:
        st.warning("Nenhum corpus coletado. Volte para a etapa anterior e realize a coleta.")

# ============================================================
# ETAPA 5: Resumo e Próximos Passos
# ============================================================
elif st.session_state.step == 5:
    st.header("5️⃣ Resumo da Pesquisa e Próximos Passos")
    st.markdown("Parabéns! Você completou o ciclo de pesquisa assistida. Aqui está um resumo do seu projeto.")
    
    if st.session_state.research_context:
        st.subheader("Problema de Pesquisa")
        st.write(st.session_state.research_context)
    
    if st.session_state.objectives:
        st.subheader("Objetivos")
        for obj in st.session_state.objectives:
            st.write(f"- {obj}")
    
    if st.session_state.hypotheses:
        st.subheader("Hipóteses")
        for hyp in st.session_state.hypotheses:
            st.write(f"- {hyp}")
    
    if st.session_state.query:
        st.subheader("Query de Busca Utilizada")
        st.code(st.session_state.query)
    
    if st.session_state.df is not None:
        st.subheader(f"Corpus Coletado ({len(st.session_state.df)} artigos)")
        st.dataframe(st.session_state.df[['title', 'source', 'year']].head(10))
    
    st.markdown("---")
    st.markdown("**📌 Próximos passos recomendados:**")
    st.markdown("""
    - Refine sua query e repita a coleta para explorar novas facetas.
    - Use o chat RAG (se configurado) para fazer perguntas específicas sobre o corpus.
    - Exporte os dados e grafos para uso em publicações.
    - Considere expandir o período ou incluir outras fontes.
    """)
    
    if st.button("🔄 Recomeçar do início"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()