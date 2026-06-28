from typing import Dict, Any

def build_prompt(
    query: str,
    context: str,
    prompt_type: str = "zero_shot",
    system_prompt: str = None
) -> str:
    """
    Constrói prompt com diferentes técnicas.
    Tipos: 'zero_shot', 'few_shot', 'cot' (Chain-of-Thought)
    """
    if system_prompt is None:
        system_prompt = (
            "Você é um assistente de pesquisa especializado em Inteligência Artificial. "
            "Responda APENAS com base no contexto fornecido. Se não souber, diga que não sabe."
        )
    
    if prompt_type == "zero_shot":
        user_content = f"""
        Contexto:
        {context}
        
        Pergunta: {query}
        
        Responda de forma concisa e direta.
        """
    
    elif prompt_type == "few_shot":
        examples = """
        Exemplo 1:
        Pergunta: Quais são aplicações de deep learning em saúde?
        Resposta: Deep learning é usado em imagens médicas para diagnóstico de câncer, detecção de retinopatia e análise de patologia.
        
        Exemplo 2:
        Pergunta: Diferença entre LDA e NMF?
        Resposta: LDA é probabilístico, NMF é fatorização de matrizes. Ambos para tópicos, mas NMF tende a tópicos mais coesos.
        """
        user_content = f"""
        {examples}
        
        Agora responda a pergunta usando APENAS o contexto:
        
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
        
        Vamos pensar passo a passo:
        1. Identifique os principais conceitos no contexto.
        2. Relacione com a pergunta.
        3. Formule uma resposta clara.
        
        Resposta (com raciocínio):
        """
    
    else:
        raise ValueError(f"Tipo de prompt inválido: {prompt_type}")
    
    # Retorna a estrutura em formato de mensagens (para chat template)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    return messages  # Retorna a lista de mensagens para ser aplicada pelo tokenizer