from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
from typing import Tuple, Any

def load_llm(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    device_map: str = "auto",
    torch_dtype = None
) -> Tuple[Any, Any]:
    """
    Carrega modelo de linguagem local via Hugging Face Transformers.
    Retorna: (generator_pipeline, tokenizer)
    """
    print(f"🔄 Carregando modelo: {model_name}...")
    
    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True
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

def generate_text(
    pipe: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95
) -> str:
    """Gera texto com parâmetros controlados."""
    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    raw = outputs[0]['generated_text']
    # Remove o prompt da resposta
    if prompt in raw:
        return raw.replace(prompt, "").strip()
    return raw.strip()