from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import Tuple, Any, Optional
from src.config import MODEL_CACHE_DIR, USE_LOCAL_CACHE_ONLY

def load_llm(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    device_map: str = "auto",
    torch_dtype = None,
    cache_dir: Optional[str] = None
) -> Tuple[Any, Any]:
    """
    Carrega modelo de linguagem local via Hugging Face Transformers.
    Utiliza cache local para evitar redownloads.
    
    Args:
        model_name: nome do modelo no Hub
        device_map: estratégia de alocação de dispositivo
        torch_dtype: tipo de dados (ex: torch.bfloat16)
        cache_dir: diretório de cache (padrão: MODEL_CACHE_DIR)
    
    Returns:
        (generator_pipeline, tokenizer)
    """
    if cache_dir is None:
        cache_dir = MODEL_CACHE_DIR

    print(f"🔄 Carregando modelo: {model_name} (cache em {cache_dir})...")
    
    # Configuração para modo offline (se desejado)
    if USE_LOCAL_CACHE_ONLY:
        print("⚠️  Modo offline ativado – apenas cache local será usado.")
        # O Hugging Face respeita a variável de ambiente HF_OFFLINE=1
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True,
        local_files_only=USE_LOCAL_CACHE_ONLY
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=USE_LOCAL_CACHE_ONLY
    )
    
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    
    print(f"✅ LLM carregado. Dispositivo: {model.device}")
    return generator, tokenizer