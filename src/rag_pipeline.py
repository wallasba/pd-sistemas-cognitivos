# src/rag_pipeline.py
# ============================================================
# VERSÃO SIMPLIFICADA COM LOGS PARA DEPURAÇÃO
# ============================================================

import os
import pandas as pd
import re
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
from huggingface_hub import InferenceClient

class RAGPipeline:
    def __init__(
        self,
        corpus_path: str,
        text_column: str = 'abstract_clean',
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        k_retrieval: int = 10,
        hf_token: Optional[str] = None,
        api_model: str = "microsoft/Phi-3-mini-4k-instruct",
        response_language: str = "português"
    ):
        self.text_column = text_column
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        self.api_model = api_model
        self.response_language = response_language

        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN não encontrado.")
        self.client = InferenceClient(token=self.hf_token)

        print("[1/4] Carregando corpus...")
        self.df = pd.read_csv(corpus_path)
        if 'year' in self.df.columns:
            self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
            self.df = self.df.dropna(subset=['year'])
            self.df['year'] = self.df['year'].astype(int)
            print(f"   Anos disponíveis: {sorted(self.df['year'].unique())}")

        if text_column not in self.df.columns:
            if 'abstract' in self.df.columns:
                self.text_column = 'abstract'
            else:
                raise ValueError(f"Coluna de texto não encontrada.")
        else:
            self.text_column = text_column

        self.df[self.text_column] = self.df[self.text_column].fillna('')

        print("[2/4] Preparando chunks...")
        self.chunks = []
        self.metadata = []
        self._prepare_chunks()

        print("[3/4] Carregando modelo de embeddings...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index, self.chunk_vectors = self._build_faiss_index()

        print("[4/4] Pipeline RAG pronto (usando API)!")
        print("✅ Pipeline RAG pronto!")

    def _prepare_chunks(self):
        for idx, row in self.df.iterrows():
            text = row[self.text_column]
            if not text or len(text) < 20:
                continue
            words = text.split()
            if len(words) <= self.chunk_size:
                self.chunks.append(text)
                self.metadata.append({
                    'doc_id': idx,
                    'title': row.get('title', 'Sem título'),
                    'source': row.get('source', 'desconhecido'),
                    'year': row.get('year', '')
                })
            else:
                for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
                    chunk_words = words[i:i + self.chunk_size]
                    chunk_text = ' '.join(chunk_words)
                    self.chunks.append(chunk_text)
                    self.metadata.append({
                        'doc_id': idx,
                        'title': row.get('title', 'Sem título'),
                        'source': row.get('source', 'desconhecido'),
                        'year': row.get('year', ''),
                        'chunk_offset': i
                    })

    def _build_faiss_index(self):
        vectors = self.embedding_model.encode(
            self.chunks,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        faiss.normalize_L2(vectors)
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(vectors)
        return index, vectors

    def retrieve(self, query: str, k: int = None) -> List[Dict]:
        if k is None:
            k = self.k_retrieval
        query_vec = self.embedding_model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        distances, indices = self.index.search(query_vec, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'score': float(dist),
                    'chunk': self.chunks[idx],
                    'metadata': self.metadata[idx]
                })
        return results

    # ============================================================
    # FACTUAIS
    # ============================================================
    def _get_latest_article(self) -> str:
        if 'year' not in self.df.columns or 'title' not in self.df.columns:
            return None
        df_year = self.df.dropna(subset=['year'])
        if df_year.empty:
            return None
        max_year = int(df_year['year'].max())
        latest = df_year[df_year['year'] == max_year]
        if latest.empty:
            return None
        titles = latest['title'].dropna().tolist()
        if not titles:
            return f"Existem {len(latest)} artigos do ano {max_year}, mas nenhum título foi encontrado."
        if len(titles) == 1:
            return f"O artigo mais recente do corpus é **'{titles[0]}'** publicado em **{max_year}**."
        else:
            title_list = "\n".join(f"- {t}" for t in titles[:5])
            return f"Artigos mais recentes ({max_year}):\n{title_list}"

    def _get_latest_article_year(self) -> str:
        if 'year' not in self.df.columns:
            return None
        df_year = self.df.dropna(subset=['year'])
        if df_year.empty:
            return None
        max_year = int(df_year['year'].max())
        return f"O ano do artigo mais recente é **{max_year}**."

    def _get_total_articles(self) -> str:
        return f"O corpus contém **{len(self.df)}** artigos."

    # ============================================================
    # GERAÇÃO VIA API
    # ============================================================
    def _generate_with_api(self, prompt: str, max_tokens: int = 600, temperature: float = 0.2) -> Optional[str]:
        """Gera resposta usando a Inference API."""
        try:
            system_msg = (
                f"Você é um assistente de pesquisa. Responda em {self.response_language}. "
                "Sua resposta deve ser baseada APENAS nos trechos fornecidos. Use parágrafos."
            )
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ]
            print(f"\n[LOG] Enviando prompt para API (modelo: {self.api_model})...")
            print(f"[LOG] Prompt (primeiros 300 caracteres): {prompt[:300]}...")
            completion = self.client.chat.completions.create(
                model=self.api_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
            )
            answer = completion.choices[0].message.content.strip()
            print(f"[LOG] Resposta recebida (primeiros 200 caracteres): {answer[:200]}...")
            return answer
        except Exception as e:
            print(f"[ERRO] Falha na API: {e}")
            return None

    # ============================================================
    # RAG PARA PERGUNTAS ABERTAS
    # ============================================================
    def _answer_with_rag(self, query: str) -> Dict:
        # Recupera chunks
        retrieved = self.retrieve(query, self.k_retrieval)
        print(f"[LOG] {len(retrieved)} chunks recuperados.")
        if not retrieved:
            return {
                "answer": "Não encontrei informações relevantes no corpus para responder a essa pergunta.",
                "retrieved_context": []
            }
        
        # Mostra os primeiros chunks no terminal
        for i, r in enumerate(retrieved[:3]):
            print(f"[LOG] Chunk {i+1} (score: {r['score']:.4f}): {r['chunk'][:100]}...")
        
        # Constrói contexto resumido (para não estourar o contexto)
        context_parts = []
        for r in retrieved[:8]:
            meta = r['metadata']
            year = meta.get('year', 'ano desconhecido')
            title = meta.get('title', 'Sem título')
            context_parts.append(f"[{title} ({year})]\n{r['chunk'][:500]}")  # Limita a 500 caracteres por chunk
        context_text = "\n\n---\n\n".join(context_parts)
        
        # Prompt direto
        prompt = f"""
        Trechos de artigos:
        {context_text}
        
        Pergunta: {query}
        
        Responda em {self.response_language} com base APENAS nos trechos.
        """
        
        answer = self._generate_with_api(prompt)
        if not answer:
            answer = "Não foi possível obter uma resposta da API. Verifique sua conexão e token."
        elif len(answer) < 20:
            answer = "Não encontrei informações suficientes no corpus para responder a essa pergunta."
        
        return {
            "answer": answer,
            "retrieved_context": retrieved
        }

    # ============================================================
    # ANSWER PRINCIPAL
    # ============================================================
    def answer(self, query: str, prompt_type: str = "zero_shot",
               include_context: bool = True, k: int = None,
               max_new_tokens: int = 600, temperature: float = 0.2) -> Dict:
        q_lower = query.lower().strip()
        print(f"\n[LOG] Nova pergunta: {query}")

        # FACTUAL
        if "artigo mais recente" in q_lower or "último artigo" in q_lower:
            resposta = self._get_latest_article()
            if resposta:
                return {"answer": resposta, "retrieved_context": [], "metadata": {"type": "factual"}}
        if "ano do artigo mais recente" in q_lower:
            resposta = self._get_latest_article_year()
            if resposta:
                return {"answer": resposta, "retrieved_context": [], "metadata": {"type": "factual"}}
        if "quantos artigos" in q_lower or "total de artigos" in q_lower:
            resposta = self._get_total_articles()
            return {"answer": resposta, "retrieved_context": [], "metadata": {"type": "factual"}}

        # ABERTAS: RAG
        result = self._answer_with_rag(query)
        return {
            "answer": result["answer"],
            "retrieved_context": result["retrieved_context"],
            "metadata": {"type": "rag"}
        }