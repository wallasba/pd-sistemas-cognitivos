# src/rag_pipeline.py
# ============================================================
# VERSÃO DEFINITIVA – SEM ESTATÍSTICAS, APENAS RAG COM LLM
# ============================================================

import pandas as pd
import re
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class RAGPipeline:
    def __init__(
        self,
        corpus_path: str,
        text_column: str = 'abstract_clean',
        llm_model_name: str = "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        k_retrieval: int = 10
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
    # RESPOSTAS FACTUAIS (DIRETO DO DATAFRAME)
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
    # NÚCLEO DO RAG – RESPOSTAS EM LINGUAGEM NATURAL
    # ============================================================
    def _answer_with_rag(self, query: str) -> Dict:
        """Usa RAG para responder perguntas abertas com síntese em linguagem natural."""
        # 1. Recupera chunks
        retrieved = self.retrieve(query, self.k_retrieval)
        if not retrieved:
            return {
                "answer": "Não encontrei informações relevantes no corpus para responder a essa pergunta.",
                "retrieved_context": []
            }
        
        # 2. Constrói contexto com metadados
        context_parts = []
        for r in retrieved:
            meta = r['metadata']
            year = meta.get('year', 'ano desconhecido')
            title = meta.get('title', 'Sem título')
            context_parts.append(f"[{title} ({year})]\n{r['chunk']}")
        context_text = "\n\n---\n\n".join(context_parts)
        
        # 3. Prompt principal – extremamente rigoroso
        system_prompt = (
            "Você é um assistente de pesquisa especializado em análise de artigos científicos. "
            "Sua tarefa é responder à pergunta do usuário com base EXCLUSIVAMENTE nos trechos de artigos fornecidos. "
            "Você DEVE produzir uma resposta em linguagem natural, coerente, bem estruturada e organizada em parágrafos. "
            "NUNCA liste palavras-chave, termos isolados, tokens ou qualquer coisa que não seja uma frase completa. "
            "NUNCA use frases como 'Desafios mencionados:' e depois liste palavras. "
            "Se os trechos mencionarem múltiplos pontos, escreva um texto corrido consolidando-os. "
            "Se houver divergências entre os artigos, destaque essas diferenças. "
            "Se a informação solicitada não estiver presente nos trechos, diga explicitamente: "
            "'Não encontrei informações suficientes no corpus para responder a essa pergunta.' "
            "NUNCA invente informações. NUNCA use conhecimento externo. "
            "Sua resposta deve ser útil para um pesquisador."
        )
        user_content = f"""
        Trechos de artigos (com título e ano):
        {context_text}
        
        Pergunta do usuário: {query}
        
        Resposta (em linguagem natural, baseada EXCLUSIVAMENTE nos trechos):
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 4. Geração com parâmetros conservadores
        outputs = self.llm_pipe(
            prompt,
            max_new_tokens=600,
            temperature=0.2,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            eos_token_id=self.tokenizer.eos_token_id
        )
        raw_text = outputs[0]['generated_text']
        if prompt in raw_text:
            answer = raw_text.replace(prompt, "").strip()
        else:
            answer = raw_text.strip()
        
        # 5. Verificação e fallback se a resposta ainda for uma lista
        if self._is_list_response(answer):
            # Fallback com prompt alternativo
            fallback_prompt = self._build_fallback_prompt(query, retrieved)
            outputs2 = self.llm_pipe(
                fallback_prompt,
                max_new_tokens=600,
                temperature=0.2,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.3,
                eos_token_id=self.tokenizer.eos_token_id
            )
            raw_text2 = outputs2[0]['generated_text']
            if fallback_prompt in raw_text2:
                answer = raw_text2.replace(fallback_prompt, "").strip()
            else:
                answer = raw_text2.strip()
            
            # Se ainda for lista, resposta padrão
            if self._is_list_response(answer):
                answer = "Não foi possível gerar uma resposta coerente com os trechos recuperados. Tente reformular a pergunta ou amplie o corpus."
        
        # 6. Garantir tamanho mínimo
        if len(answer) < 30:
            answer = "Não encontrei informações suficientes no corpus para responder a essa pergunta."
        
        return {
            "answer": answer,
            "retrieved_context": retrieved
        }

    def _is_list_response(self, text: str) -> bool:
        """Verifica se a resposta parece uma lista de palavras-chave."""
        if not text:
            return True
        # Divide em linhas
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        # Se todas as linhas têm 1-2 palavras e nenhuma contém verbos
        if len(lines) > 2:
            word_counts = [len(l.split()) for l in lines]
            if all(c <= 3 for c in word_counts):
                return True
        # Se o texto tem menos de 15 palavras e contém palavras comuns de lista
        words = text.split()
        if len(words) < 15:
            list_indicators = ['analysis', 'artificial', 'based', 'data', 'development', 'infrastructure', 
                              'intelligence', 'management', 'model', 'models', 'research', 'systems']
            if any(word in text.lower() for word in list_indicators):
                return True
        return False

    def _build_fallback_prompt(self, query: str, retrieved: List[Dict]) -> str:
        """Prompt alternativo para forçar resposta em parágrafos."""
        context_parts = []
        for r in retrieved[:5]:
            meta = r['metadata']
            context_parts.append(f"- {meta.get('title', 'Sem título')} ({meta.get('year', '')}): {r['chunk'][:400]}...")
        context_text = "\n".join(context_parts)
        
        system_prompt = (
            "Com base nos trechos abaixo, responda à pergunta em UM PARÁGRAFO conciso. "
            "NÃO use listas, bullets ou palavras-chave. Apenas um texto corrido e coerente."
        )
        user_content = f"""
        Trechos:
        {context_text}
        
        Pergunta: {query}
        
        Resposta (um parágrafo apenas):
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ============================================================
    # MÉTODO ANSWER PRINCIPAL – TODAS AS PERGUNTAS VÃO PARA RAG
    # ============================================================
    def answer(self, query: str, prompt_type: str = "zero_shot",
               include_context: bool = True, k: int = None,
               max_new_tokens: int = 600, temperature: float = 0.2) -> Dict:
        """
        Método principal: para perguntas factuais, responde diretamente.
        Para TODAS as outras, usa RAG com LLM.
        """
        q_lower = query.lower().strip()

        # ===== FACTUAIS: respondidas diretamente =====
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

        # ===== TODAS AS OUTRAS: RAG =====
        # Se a pergunta contém palavras-chave abertas, usa RAG
        # Caso contrário, também usa RAG (porque não temos outra forma)
        result = self._answer_with_rag(query)
        return {
            "answer": result["answer"],
            "retrieved_context": result["retrieved_context"],
            "metadata": {"type": "rag"}
        }