from typing import List, Optional, Dict, Any
import pandas as pd
import requests
import time
import re
import arxiv
from pyalex import Works
from pymed import PubMed
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ============================================
# 1. Funções auxiliares
# ============================================
def reconstruct_abstract(index: Dict) -> str:
    if not index:
        return ""
    pos_map = {}
    for word, positions in index.items():
        for pos in positions:
            pos_map[pos] = word
    max_pos = max(pos_map.keys()) if pos_map else -1
    return ' '.join(pos_map.get(i, '') for i in range(max_pos + 1))

def safe_get(data: Any, key: str, default: str = "") -> str:
    return data.get(key, default) if isinstance(data, dict) else default

# ============================================
# 2. Coleta por fonte
# ============================================
def fetch_openalex(query: str, max_results: int = 300,
                   year_start: Optional[int] = None,
                   year_end: Optional[int] = None) -> List[Dict]:
    papers = []
    try:
        # Construção do filtro de data
        filters = {}
        if year_start:
            filters['publication_year'] = {'gte': year_start}
        if year_end:
            filters['publication_year'] = {**filters.get('publication_year', {}), 'lte': year_end}
        
        for work in Works().search(query).filter(**filters).paginate(per_page=200, n_max=max_results):
            # Extrai afiliações
            affiliations = []
            for authorship in work.get('authorships', []):
                for inst in authorship.get('institutions', []):
                    name = inst.get('display_name', '')
                    if name:
                        affiliations.append(name)
            affiliations = list(set(affiliations))
            
            papers.append({
                'title': work.get('title', ''),
                'authors': ', '.join([a.get('author', {}).get('display_name', '') 
                                      for a in work.get('authorships', [])]),
                'abstract': reconstruct_abstract(work.get('abstract_inverted_index')),
                'year': work.get('publication_year'),
                'source': 'openalex',
                'affiliations': affiliations,
                'url': work.get('id', '')
            })
        print(f"OpenAlex: {len(papers)} artigos coletados.")
    except Exception as e:
        print(f"Erro no OpenAlex: {e}")
    return papers

def fetch_arxiv(query: str, max_results: int = 300,
                year_start: Optional[int] = None,
                year_end: Optional[int] = None) -> List[Dict]:
    papers = []
    try:
        # ArXiv não suporta filtro por ano diretamente; faremos pós-filtro
        client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate)
        for result in client.results(search):
            year = result.updated.year if result.updated else None
            if year_start and year and year < year_start:
                continue
            if year_end and year and year > year_end:
                continue
            papers.append({
                'title': result.title,
                'authors': ', '.join([a.name for a in result.authors]),
                'abstract': result.summary,
                'year': year,
                'source': 'arxiv',
                'affiliations': [],  # ArXiv não fornece afiliação estruturada
                'url': result.entry_id
            })
        print(f"ArXiv: {len(papers)} artigos coletados.")
    except Exception as e:
        print(f"Erro no arXiv: {e}")
    return papers

def fetch_pubmed(query: str, max_results: int = 300,
                 year_start: Optional[int] = None,
                 year_end: Optional[int] = None,
                 email: str = "seuemail@dominio.com") -> List[Dict]:
    papers = []
    try:
        pubmed = PubMed(tool="research_assistant", email=email)
        # PubMed não tem filtro de ano na query, faremos pós-filtro
        results = pubmed.query(query, max_results=max_results)
        for article in results:
            year = article.publication_date.year if article.publication_date else None
            if year_start and year and year < year_start:
                continue
            if year_end and year and year > year_end:
                continue
            # Extrai afiliações (campo 'affiliation' pode estar em authors)
            affiliations = []
            for author in article.authors:
                if isinstance(author, dict) and 'affiliation' in author:
                    aff = author['affiliation']
                    if aff:
                        affiliations.append(aff)
            affiliations = list(set(affiliations))
            papers.append({
                'title': article.title,
                'authors': ', '.join([a.get('name', '') if isinstance(a, dict) else str(a) for a in article.authors]),
                'abstract': article.abstract,
                'year': year,
                'source': 'pubmed',
                'affiliations': affiliations,
                'url': f"https://pubmed.ncbi.nlm.nih.gov/{article.pubmed_id}"
            })
        print(f"PubMed: {len(papers)} artigos coletados.")
    except Exception as e:
        print(f"Erro no PubMed: {e}")
    return papers

# (demais fontes – Crossref, Europe PMC, Zenodo, DOAJ – podem ser adaptadas de forma similar,
#  mas para brevidade, focamos nas três principais)

# ============================================
# 3. Orquestrador principal
# ============================================
def collect_articles(
    query: str,
    sources: List[str] = None,
    max_results: int = 300,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    email: str = "seuemail@dominio.com"
) -> pd.DataFrame:
    if sources is None:
        sources = ['arxiv', 'openalex', 'pubmed']
    
    all_papers = []
    if 'openalex' in sources:
        all_papers.extend(fetch_openalex(query, max_results, year_start, year_end))
    if 'arxiv' in sources:
        all_papers.extend(fetch_arxiv(query, max_results, year_start, year_end))
    if 'pubmed' in sources:
        all_papers.extend(fetch_pubmed(query, max_results, year_start, year_end, email))
    # Adicione outras fontes se necessário
    
    # Remove duplicatas (baseado em título + ano)
    df = pd.DataFrame(all_papers)
    if not df.empty:
        df.drop_duplicates(subset=['title', 'year'], inplace=True)
    return df