import re
# Ler o arquivo original
with open('enriched_nature.py', 'r', encoding='utf-8') as f:
    content = f.read()
# Corrigir o problema da string multilinha não terminada
# Corrigir o problema do bloco try sem except
fixed_content = fixed_content.replace('    try:\\n        # Verificar se o arquivo existe', '    try:\\n        # Verificar se o arquivo existe')
fixed_content = fixed_content.replace('return True\\n\\nif __name__', 'return True\\n\\nexcept Exception as e:\\n    logger.error(f\
Erro
ao
integrar
função
de
enriquecimento:
str(e)
\)\\n    return False\\n\\nif __name__')
