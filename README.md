# 1. Assistente de Pesquisa Científica com RAG

[![Streamlit](https://img.shields.io/badge/Streamlit-1.37.0-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python)](https://python.org)
[![Groq](https://img.shields.io/badge/Groq-API-FF6B00?style=flat-square)](https://groq.com)

Assistente de pesquisa para análise e exploração de literatura científica utilizando **RAG (Retrieval-Augmented Generation)** com embeddings semânticos e modelos de linguagem de alta performance via **API Groq**.

---

## 📌 Visão Geral

Este projeto é uma aplicação **end‑to‑end** desenvolvida para auxiliar pesquisadores (doutorandos, pós‑docs e pesquisadores seniores) na análise de grandes volumes de artigos científicos. O sistema combina:

- **Coleta de artigos** de repositólogos abertos (arXiv, OpenAlex, PubMed, etc.)
- **Pré‑processamento** e indexação vetorial com **FAISS**
- **Pipeline RAG** com recuperação de contexto e geração de respostas fundamentadas
- **Análise bibliométrica** (grafos de coautoria, instituições, termos, estatísticas de rede)
- **Interface interativa** com Streamlit e fluxo guiado (wizard em 5 etapas)

A aplicação permite que o pesquisador faça perguntas em linguagem natural e obtenha respostas **baseadas exclusivamente no conteúdo do corpus**, com transparência sobre os trechos recuperados.

---

## 🎯 Funcionalidades

- ✅ **Wizard de 5 etapas**: definição do problema, objetivos, estratégia de busca, análise exploratória e chat RAG.
- ✅ **Coleta de artigos** de 7 repositórios com filtros de ano, fonte e limite.
- ✅ **Upload de corpus próprio** (CSV, XLS, XLSX) com validação de estrutura.
- ✅ **Embeddings semânticos** com Sentence‑Transformers (`all‑MiniLM‑L6‑v2`).
- ✅ **Busca vetorial** com FAISS (similaridade por cosseno).
- ✅ **Pipeline RAG** com LLM (Llama 3.1 8B via Groq) e prompts otimizados para respostas sintetizadas.
- ✅ **Respostas factuais** diretas (anos, fontes, autores) sem uso do LLM.
- ✅ **Grafos estáticos** interativos com NetworkX + Matplotlib:
  - Coautoria, colaboração institucional, termos × instituições
  - Destaque das top‑N arestas mais fortes (com gradiente de cores)
  - Nós com tamanho proporcional à métrica (grau, betweenness, frequência)
  - Atualização automática dos parâmetros
- ✅ **Estatísticas de rede** completas (densidade, transitividade, centralidades, componentes, etc.) com exportação para CSV/XLSX.
- ✅ **Chat RAG** com perguntas e respostas em português, exibindo os trechos recuperados.
- ✅ **Máscaras de perguntas** para pesquisador sênior (tendências, desafios, aplicações, etc.).
- ✅ **Segurança**: chave API gerenciada via `.env` ou interface, nunca exposta no código.

---

## 🛠️ Tecnologias Utilizadas

| Componente | Tecnologia | Versão |
| :--- | :--- | :--- |
| **LLM** | Groq (Llama 3.1 8B) | – |
| **Embeddings** | Sentence‑Transformers | 3.0.0+ |
| **Vector Store** | FAISS (CPU) | 1.8.0+ |
| **Interface** | Streamlit | 1.37.0+ |
| **Grafos** | NetworkX + Matplotlib | 3.3+ / 3.9+ |
| **Processamento** | Pandas, NumPy, NLTK | 2.2+ / 1.26+ / 3.9+ |
| **Coleta** | arxiv, pyalex, pymed, requests | – |

---

## 📦 Requisitos

- **Python** 3.12 ou superior
- **Pip** e ambiente virtual (recomendado)
- **Conexão com internet** para a API Groq e download de modelos
- **Conta no Groq** (gratuita) para obter a chave API

---

## 🚀 Instalação e Configuração

### 1. Clone o repositório

```bash
git clone https://github.com/wallasb85/apd-sistemas-cognitivos.git
cd apd-sistemas-cognitivos
```
### 2. Crie e ative um ambiente virtual
```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate         # Windows
```

### 3. Instale as dependências
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure a chave da API Groq

```bash
GROQ_API_KEY=sua_chave_aqui
```

### Executando a Aplicação

```bash
Gstreamlit run app.py
```

# Como Usar
## Etapas do Wizard
1. Definição do Problema – descreva seu tema e área de pesquisa.
2. Objetivos e Hipóteses – defina objetivos e hipóteses com sugestões automáticas.
3. Estratégia de Busca – construa a query booleana, selecione fontes, período e limite.
4. Análise Exploratória – visualize estatísticas, nuvem de palavras, grafos e estatísticas de rede.
5. Resumo e Chat RAG – faça perguntas em linguagem natural sobre o corpus.

## Atalhos Rápidos
1. Na barra lateral: após carregar o corpus, botões para Ir para Análise e Ir para o Chat RAG.
2. Na Etapa 4: botão "💬 Ir para o Chat RAG" para pular diretamente para as perguntas.

## Upload de Corpus Próprio
Na barra lateral, você pode carregar um arquivo CSV, XLS ou XLSX com as colunas:

1. title (texto)
2. abstract (texto)
3. source (texto)
4. year (inteiro)
5. authors (texto, separados por vírgula)
6. affiliations (texto, separados por ;)

# Estrutura do Projeto

```bash
assistente-pesquisa-rag/
├── app.py                          # Ponto de entrada da aplicação
├── src/                            # Lógica de negócio
│   ├── data_collector.py           # Coleta de artigos via APIs
│   ├── preprocessor.py             # Limpeza e lematização
│   ├── analyzer.py                 # Estatísticas e nuvem de palavras
│   ├── graph_builder.py            # Construção de grafos e estatísticas
│   ├── recommender.py              # Sugestões de objetivos e hipóteses
│   └── rag_pipeline.py             # Pipeline RAG completo
├── ui/                             # Interface Streamlit
│   ├── sidebar.py                  # Barra lateral (upload, API key, atalhos)
│   ├── wizard.py                   # Wizard de 5 etapas
│   ├── analysis.py                 # Visualizações descritivas
│   ├── graph_viewer.py             # Exibição de grafos estáticos
│   └── rag_chat.py                 # Chat RAG interativo
├── notebooks/                      # Notebooks demonstrativos (c01 a c05)
├── data/                           # (Opcional) para armazenar corpus
├── requirements.txt                # Dependências
├── .env.example                    # Exemplo de variáveis de ambiente
└── README.md                       # Este arquivo
```

# Notebooks Demonstrativos
A pasta notebooks/ contém 5 notebooks que demonstram cada competência do projeto:

## Notebook	Conteúdo

```bash
c01_modelos_llm.ipynb	Uso da API Groq, comparação de temperatura e modelos disponíveis.
c02_prompting.ipynb	Comparação de zero‑shot, few‑shot e chain‑of‑thought.
c03_embeddings_busca.ipynb	Geração de embeddings, indexação FAISS e busca semântica.
c04_inferencia_local_remota.ipynb	Comparação entre inferência local e remota (Groq).
c05_rag_pipeline.ipynb	Pipeline RAG completo, com e sem contexto.
Execute-os em ordem para validar o funcionamento de cada componente.
```