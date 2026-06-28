from src.data_loader import load_corpus
from src.chunking import prepare_chunks
from src.embeddings import EmbeddingIndex
from src.llm_inference import load_llm, generate_text
from src.prompt_builder import build_prompt
from src.config import MODEL_CACHE_DIR
from typing import Dict, Any, List, Optional

class RAGPipeline:
    def __init__(
        self,
        corpus_path: str,
        llm_model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        k_retrieval: int = 5,
        cache_dir: Optional[str] = None
    ):
        self.cache_dir = cache_dir or MODEL_CACHE_DIR
        self.llm_model_name = llm_model_name
        self.k_retrieval = k_retrieval
        
        # 1. Carrega corpus
        print("[1/4] Carregando corpus...")
        self.df = load_corpus(corpus_path)
        
        # 2. Prepara chunks
        print("[2/4] Preparando chunks...")
        self.chunks = prepare_chunks(self.df, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # 3. Constrói índice de embeddings (usando cache)
        print("[3/4] Construindo índice de embeddings...")
        self.embed_index = EmbeddingIndex(embedding_model_name, cache_dir=self.cache_dir)
        self.embed_index.build_index(self.chunks)
        
        # 4. Carrega LLM (usando cache)
        print("[4/4] Carregando LLM...")
        self.llm_pipe, self.tokenizer = load_llm(
            llm_model_name,
            cache_dir=self.cache_dir
        )
        
        print("✅ Pipeline RAG pronto!")
    
    def answer(
        self,
        query: str,
        prompt_type: str = "zero_shot",
        include_context: bool = True,
        k: int = None,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Executa o pipeline RAG para uma consulta."""
        if k is None:
            k = self.k_retrieval
        
        # Recuperação
        retrieved = self.embed_index.search(query, k) if include_context else []
        
        context_text = ""
        if retrieved:
            context_text = "\n\n---\n\n".join([r['chunk'] for r in retrieved])
        else:
            context_text = "Nenhum contexto recuperado."
        
        # Constrói o prompt
        messages = build_prompt(query, context_text, prompt_type)
        prompt_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Geração
        answer = generate_text(
            self.llm_pipe,
            self.tokenizer,
            prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        return {
            "answer": answer,
            "retrieved_context": retrieved,
            "metadata": {
                "prompt_type": prompt_type,
                "k_retrieved": k,
                "model": self.llm_model_name,
                "include_context": include_context
            },
            "prompt_used": prompt_text
        }
    
    def compare_without_context(self, query: str, **kwargs) -> Dict[str, Any]:
        """Versão sem contexto para demonstrar a eficácia do RAG."""
        return self.answer(query, include_context=False, **kwargs)