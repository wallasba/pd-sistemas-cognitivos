import os

# Diretório onde os modelos serão armazenados (cache local)
MODEL_CACHE_DIR = os.environ.get("HF_MODEL_CACHE_DIR", "./models_cache")

# Cria o diretório se não existir
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Opções adicionais
# Para Execução offline, após o primeiro download, com HF_OFFLINE=1, o sistema roda sem internet.
USE_LOCAL_CACHE_ONLY = os.environ.get("HF_OFFLINE", "false").lower() == "true"