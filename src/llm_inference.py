"""
Módulo para carregamento e inferência de LLMs locais via Hugging Face Transformers.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Tuple, Any, Optional


def load_llm(
    model_name: str = "microsoft/Phi-3-mini-4k-instruct",
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """
    Carrega modelo de linguagem local via Hugging Face Transformers.

    Args:
        model_name: Nome do modelo no Hugging Face Hub.
        device_map: Estratégia de alocação de dispositivo ('auto', 'cpu', 'cuda').
        torch_dtype: Tipo de dado para o tensor (ex: torch.float16).
        trust_remote_code: Permite carregar código remoto (necessário para alguns modelos).

    Returns:
        pipeline (transformers.pipeline): Pipeline de geração de texto.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer associado.
    """
    print(f"🔄 Carregando modelo: {model_name} ...")

    if torch_dtype is None:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    # Garante que o token de padding exista
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
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
    top_p: float = 0.95,
    do_sample: bool = True,
) -> str:
    """
    Gera texto com parâmetros controlados.

    Args:
        pipe: Pipeline de geração.
        tokenizer: Tokenizer.
        prompt: Texto de entrada (já formatado com chat template, se necessário).
        max_new_tokens: Número máximo de tokens a gerar.
        temperature: Temperatura da amostragem.
        top_p: Probabilidade cumulativa para nucleus sampling.
        do_sample: Se False, usa greedy decoding.

    Returns:
        str: Texto gerado (sem o prompt).
    """
    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,  # Evita warnings
    )
    raw = outputs[0]['generated_text']

    # Remove o prompt da resposta (para ficar apenas o texto gerado)
    if prompt in raw:
        return raw.replace(prompt, "").strip()
    return raw.strip()