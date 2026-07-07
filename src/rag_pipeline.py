# src/rag_pipeline.py
# ============================================================
# VERSÃO HÍBRIDA: RAG + ESTATÍSTICAS DO CORPUS
# ============================================================

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class RAGPipeline:
    def __init__(
        self,
        corpus_path: str,
        text_column: str = 'abstract_clean',
        llm_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        k_retrieval: int = 7
    ):
        self.text_column = text_column
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name

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

        # Pré-calcula termos frequentes (para respostas estatísticas)
        self._compute_term_frequencies()

        print("[2/4] Preparando chunks...")
        self.chunks = []
        self.metadata = []
        self._prepare_chunks()

        print("[3/4] Carregando modelo de embeddings...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index, self.chunk_vectors = self._build_faiss_index()

        print("[4/4] Carregando LLM local...")
        self.llm_pipe, self.tokenizer = self._load_llm()

        print("✅ Pipeline RAG pronto!")

    def _compute_term_frequencies(self):
        """Pré-calcula termos mais frequentes para respostas estatísticas."""
        all_text = ' '.join(self.df[self.text_column].fillna(''))
        vectorizer = CountVectorizer(stop_words='english', max_features=100)
        X = vectorizer.fit_transform([all_text])
        terms = vectorizer.get_feature_names_out()
        counts = X.toarray()[0]
        self.term_freq = list(zip(terms, counts))
        self.term_freq.sort(key=lambda x: x[1], reverse=True)

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

    def _load_llm(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        return generator, tokenizer

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
    # RESPOSTAS ESTATÍSTICAS (SEM LLM)
    # ============================================================
    def _answer_trends(self) -> str:
        """Responde 'principais tendências' com base nos termos mais frequentes."""
        top_terms = [term for term, _ in self.term_freq[:15]]
        return "**Principais tópicos (termos mais frequentes no corpus):**\n" + "\n".join(f"- {t}" for t in top_terms)

    def _answer_applications(self) -> str:
        """Responde 'aplicações' buscando termos que co-ocorrem com 'application'."""
        # Busca trechos com 'application' ou 'applied' e extrai termos próximos
        relevant_chunks = []
        for chunk in self.chunks:
            if re.search(r'\b(application|applied|use|using)\b', chunk, re.IGNORECASE):
                relevant_chunks.append(chunk)
        if not relevant_chunks:
            return "Não encontrei menções a aplicações no corpus."
        # Extrai termos frequentes nesses chunks
        text = ' '.join(relevant_chunks)
        vectorizer = CountVectorizer(stop_words='english', max_features=20)
        X = vectorizer.fit_transform([text])
        terms = vectorizer.get_feature_names_out()
        return "**Aplicações mencionadas:**\n" + "\n".join(f"- {t}" for t in terms[:10])

    def _answer_challenges(self) -> str:
        """Responde 'desafios' buscando termos com 'challenge' ou 'limitation'."""
        relevant_chunks = []
        for chunk in self.chunks:
            if re.search(r'\b(challenge|limitation|issue|problem)\b', chunk, re.IGNORECASE):
                relevant_chunks.append(chunk)
        if not relevant_chunks:
            return "Não encontrei menções a desafios no corpus."
        text = ' '.join(relevant_chunks)
        vectorizer = CountVectorizer(stop_words='english', max_features=20)
        X = vectorizer.fit_transform([text])
        terms = vectorizer.get_feature_names_out()
        return "**Desafios mencionados:**\n" + "\n".join(f"- {t}" for t in terms[:10])

    # ============================================================
    # MÉTODO ANSWER PRINCIPAL
    # ============================================================
    def answer(self, query: str, prompt_type: str = "zero_shot",
               include_context: bool = True, k: int = None,
               max_new_tokens: int = 250, temperature: float = 0.3) -> Dict:
        """Pipeline com fallback estatístico para perguntas abertas."""
        q_lower = query.lower().strip()

        # ========== FALLBACK FACTUAL ==========
        if "artigo mais recente" in q_lower or "último artigo" in q_lower:
            resposta = self._get_latest_article()
            if resposta:
                return {"answer": resposta, "retrieved_context": [], "metadata": {"type": "factual"}, "full_response": {"raw_answer": resposta}}
        if "ano do artigo mais recente" in q_lower:
            resposta = self._get_latest_article_year()
            if resposta:
                return {"answer": resposta, "retrieved_context": [], "metadata": {"type": "factual"}, "full_response": {"raw_answer": resposta}}
        if "quantos artigos" in q_lower or "total de artigos" in q_lower:
            resposta = f"O corpus contém **{len(self.df)}** artigos."
            return {"answer": resposta, "retrieved_context": [], "metadata": {"type": "factual"}, "full_response": {"raw_answer": resposta}}

        # ========== FALLBACK ESTATÍSTICO ==========
        if "tendência" in q_lower or "tendências" in q_lower:
            return {"answer": self._answer_trends(), "retrieved_context": [], "metadata": {"type": "statistical"}, "full_response": {"raw_answer": self._answer_trends()}}
        if "aplicação" in q_lower or "aplicações" in q_lower:
            return {"answer": self._answer_applications(), "retrieved_context": [], "metadata": {"type": "statistical"}, "full_response": {"raw_answer": self._answer_applications()}}
        if "desafio" in q_lower or "desafios" in q_lower or "limitação" in q_lower:
            return {"answer": self._answer_challenges(), "retrieved_context": [], "metadata": {"type": "statistical"}, "full_response": {"raw_answer": self._answer_challenges()}}

        # ========== RAG (LLM) PARA OUTRAS PERGUNTAS ==========
        if k is None:
            k = self.k_retrieval
        retrieved = self.retrieve(query, k) if include_context else []
        context_text = "\n\n---\n\n".join([r['chunk'] for r in retrieved]) if retrieved else "Nenhum contexto encontrado."

        # Prompt mais estruturado
        prompt = self._build_prompt(query, context_text, prompt_type)
        outputs = self.llm_pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.4,
            eos_token_id=self.tokenizer.eos_token_id
        )
        raw_text = outputs[0]['generated_text']
        if prompt in raw_text:
            answer = raw_text.replace(prompt, "").strip()
        else:
            answer = raw_text.strip()

        # Pós-processamento: detecta repetição
        if len(answer) < 15 or answer.count("o ano que") > 2 or answer.count("Pesquisa") > 10:
            answer = "Não encontrei informações suficientes no corpus para responder a essa pergunta."

        return {
            "answer": answer,
            "retrieved_context": retrieved,
            "metadata": {"type": "rag", "prompt_type": prompt_type, "k_retrieved": k},
            "full_response": {"raw_answer": answer}
        }

    def _build_prompt(self, query: str, context: str, prompt_type: str = "zero_shot") -> str:
        system_prompt = (
            "Você é um assistente de pesquisa. Responda APENAS com base no contexto fornecido. "
            "Se o contexto não tiver a informação, diga: 'Não encontrei essa informação no corpus.' "
            "Seja objetivo, conciso e use bullet points quando apropriado. "
            "NUNCA repita frases ou palavras."
        )
        user_content = f"Contexto:\n{context}\n\nPergunta: {query}\n\nResposta (objetiva, sem repetições):"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ============================================================
    # FUNÇÕES FACTUAIS
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