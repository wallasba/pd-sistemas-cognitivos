import pandas as pd
from typing import List, Dict, Any

def prepare_chunks(
    df: pd.DataFrame, 
    text_column: str = 'abstract_clean',
    chunk_size: int = 300, 
    chunk_overlap: int = 50
) -> List[Dict[str, Any]]:
    """
    Divide cada resumo em chunks com overlap.
    Retorna lista de dicionários: {'text': str, 'metadata': dict}
    """
    chunks = []
    
    for idx, row in df.iterrows():
        text = row[text_column]
        if not text or len(text) < 20:
            continue
        
        words = text.split()
        if len(words) <= chunk_size:
            chunks.append({
                'text': text,
                'metadata': {
                    'doc_id': idx,
                    'title': row.get('title', 'Sem título'),
                    'source': row.get('source', 'desconhecido'),
                    'year': row.get('year', '')
                }
            })
        else:
            for i in range(0, len(words), chunk_size - chunk_overlap):
                chunk_words = words[i:i + chunk_size]
                chunk_text = ' '.join(chunk_words)
                chunks.append({
                    'text': chunk_text,
                    'metadata': {
                        'doc_id': idx,
                        'title': row.get('title', 'Sem título'),
                        'source': row.get('source', 'desconhecido'),
                        'year': row.get('year', ''),
                        'chunk_offset': i
                    }
                })
    
    print(f"✅ {len(chunks)} chunks criados a partir de {len(df)} documentos.")
    return chunks