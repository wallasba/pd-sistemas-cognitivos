import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any, Tuple

class EmbeddingIndex:
    """Gerencia embeddings e índice FAISS."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []
        self.vectors = None
    
    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Gera embeddings e constrói índice FAISS."""
        self.chunks = chunks
        texts = [c['text'] for c in chunks]
        
        print(f"🔄 Gerando embeddings para {len(texts)} chunks...")
        self.vectors = self.model.encode(
            texts, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        # Normaliza para similaridade por cosseno (Inner Product)
        faiss.normalize_L2(self.vectors)
        
        dimension = self.vectors.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.vectors)
        print(f"✅ Índice FAISS construído (dimensão: {dimension}).")
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Recupera os k chunks mais similares à consulta."""
        if self.index is None:
            raise ValueError("Índice não construído. Execute build_index() primeiro.")
        
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    'score': float(dist),
                    'chunk': self.chunks[idx]['text'],
                    'metadata': self.chunks[idx]['metadata']
                })
        return results
    
    def save_index(self, path: str) -> None:
        """Salva o índice FAISS e os metadados."""
        if self.index is None:
            raise ValueError("Índice vazio.")
        faiss.write_index(self.index, f"{path}.faiss")
        import pickle
        with open(f"{path}_meta.pkl", 'wb') as f:
            pickle.dump({'chunks': self.chunks, 'model_name': self.model_name}, f)
        print(f"✅ Índice salvo em {path}")
    
    def load_index(self, path: str) -> None:
        """Carrega índice e metadados."""
        import pickle
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}_meta.pkl", 'rb') as f:
            data = pickle.load(f)
        self.chunks = data['chunks']
        self.model_name = data['model_name']
        self.model = SentenceTransformer(self.model_name)
        print(f"✅ Índice carregado de {path}")