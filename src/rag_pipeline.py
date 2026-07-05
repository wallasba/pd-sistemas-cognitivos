import pandas as pd
import numpy as np
import re
import json
from typing import List, Dict, Any, Optional, Union
import nltk
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class RAGPipeline:
    def __init__(
        self,
        corpus_path: str,
        text_column: str = 'abstract_clean',  # <-- NOVO
        llm_model_name: str = "HuggingFaceTB/SmolLM2-360M-Instruct",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        k_retrieval: int = 5
    ):
        self.text_column = text_column
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.k_retrieval = k_retrieval
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name

        # 1. Carrega o corpus
        print("[1/4] Carregando corpus...")
        self.df = pd.read_csv(corpus_path)
        # Verifica se a coluna de texto existe
        if text_column not in self.df.columns:
            # Tenta usar 'abstract' como fallback
            if 'abstract' in self.df.columns:
                self.text_column = 'abstract'
                print(f"Coluna '{text_column}' não encontrada. Usando 'abstract' como fallback.")
            else:
                raise ValueError(f"Coluna de texto não encontrada. Esperado: '{text_column}' ou 'abstract'.")
        else:
            self.text_column = text_column

        # Garante que a coluna de texto esteja preenchida
        self.df[self.text_column] = self.df[self.text_column].fillna('')

        # 2. Prepara chunks
        print("[2/4] Preparando chunks...")
        self.chunks = []
        self.metadata = []
        self._prepare_chunks()

        # 3. Embeddings
        print("[3/4] Carregando modelo de embeddings...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index, self.chunk_vectors = self._build_faiss_index()

        # 4. LLM
        print("[4/4] Carregando LLM local...")
        self.llm_pipe, self.tokenizer = self._load_llm()

        print("✅ Pipeline RAG pronto!")

    def _prepare_chunks(self):
        """Divide cada documento em chunks."""
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
        """Gera embeddings e constrói índice FAISS."""
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
        """Carrega o modelo de linguagem local."""
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
        """Recupera os k chunks mais similares."""
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

    def build_prompt(self, query: str, context: str, prompt_type: str = "zero_shot") -> str:
        """Constrói prompt com instruções rigorosas para evitar alucinações."""
        system_prompt = (
            "Você é um assistente de pesquisa especializado em análise de artigos científicos. "
            "Responda APENAS com base no contexto fornecido. "
            "Se o contexto não contiver informações suficientes para responder, diga exatamente: "
            "'Não encontrei informações suficientes no corpus para responder a essa pergunta.' "
            "NUNCA invente ou repita informações sem base no contexto."
        )
        if prompt_type == "zero_shot":
            user_content = f"""
            Contexto (trechos de artigos):
            {context}
            
            Pergunta: {query}
            
            Resposta (use apenas o contexto acima, seja objetivo e não repita informações):
            """
        elif prompt_type == "few_shot":
            examples = (
                "Exemplo 1:\nPergunta: Quais são as principais aplicações?\nResposta: As principais aplicações são A, B e C, conforme mencionado no contexto.\n"
                "Exemplo 2:\nPergunta: Quais são os desafios?\nResposta: Os desafios incluem X, Y e Z, de acordo com os artigos analisados."
            )
            user_content = f"""
            {examples}
            
            Agora responda a pergunta usando APENAS o contexto fornecido.
            
            Contexto:
            {context}
            
            Pergunta: {query}
            
            Resposta:
            """
        elif prompt_type == "cot":
            user_content = f"""
            Contexto:
            {context}
            
            Pergunta: {query}
            
            Vamos analisar o contexto passo a passo:
            1. Identifique os trechos que mencionam desafios éticos.
            2. Extraia as informações relevantes.
            3. Formule uma resposta concisa.
            
            Resposta (baseada no contexto):
            """
        else:
            user_content = f"Contexto:\n{context}\n\nPergunta: {query}\nResposta:"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_response(self, query: str, context: str, prompt_type: str = "zero_shot",
                        max_new_tokens: int = 350, temperature: float = 0.3) -> Dict:
        """Gera resposta com parâmetros mais conservadores."""
        prompt = self.build_prompt(query, context, prompt_type)
        outputs = self.llm_pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,  # Evita repetições
            eos_token_id=self.tokenizer.eos_token_id
        )
        raw_text = outputs[0]['generated_text']
        if prompt in raw_text:
            answer = raw_text.replace(prompt, "").strip()
        else:
            answer = raw_text.strip()
        
        # Se a resposta for muito curta ou parecer alucinação, indica falta de informação
        if len(answer) < 20 or "cercos" in answer.lower():  # palavra detectada como alucinação
            answer = "Não encontrei informações suficientes no corpus para responder a essa pergunta."
        
        return {
            "raw_answer": answer,
            "prompt_used": prompt,
            "model": self.llm_model_name
        }

    def answer(self, query: str, prompt_type: str = "zero_shot",
               include_context: bool = True, k: int = None,
               max_new_tokens: int = 512, temperature: float = 0.7) -> Dict:
        """Pipeline RAG completo."""
        if k is None:
            k = self.k_retrieval
        retrieved = self.retrieve(query, k) if include_context else []
        context_text = "\n\n---\n\n".join([r['chunk'] for r in retrieved]) if retrieved else "Nenhum contexto."
        response = self.generate_response(query, context_text, prompt_type, max_new_tokens, temperature)
        return {
            "answer": response["raw_answer"],
            "retrieved_context": retrieved,
            "metadata": {
                "prompt_type": prompt_type,
                "k_retrieved": k,
                "model": self.llm_model_name,
                "include_context": include_context
            },
            "full_response": response
        }