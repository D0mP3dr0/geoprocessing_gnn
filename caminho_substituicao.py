# Instruções para substituir caminhos existentes no notebook PIPELINECOMPLETO.ipynb

"""
No notebook PIPELINECOMPLETO.ipynb, você precisa substituir todos os caminhos absolutos existentes 
pelos caminhos relativos usando as variáveis BASE_DIR, DATA_DIR, OUTPUT_DIR e REPORT_DIR.

1. PRIMEIRO PASSO: Adicione a célula de configuração no início do notebook
   Copie o código do arquivo 'caminho_configuracao.py' e adicione como uma nova célula 
   logo após a célula que monta o Google Drive.

2. SUBSTITUIÇÕES A FAZER:
   
   a) Qualquer caminho para arquivos GPKG brutos:
      ANTIGO: qualquer_caminho/arquivo.gpkg
      NOVO: os.path.join(DATA_DIR, "arquivo.gpkg")
   
   b) Qualquer caminho para salvar arquivos processados:
      ANTIGO: qualquer_caminho/resultado.gpkg
      NOVO: os.path.join(OUTPUT_DIR, "resultado.gpkg")
   
   c) Qualquer caminho para relatórios:
      ANTIGO: qualquer_caminho/relatorio.json
      NOVO: os.path.join(REPORT_DIR, "relatorio.json")

3. EXEMPLOS COMUNS DE SUBSTITUIÇÃO:

   - Se tiver código como: pd.read_file("/content/drive/MyDrive/algum_caminho/dados.gpkg")
     Substitua por: pd.read_file(os.path.join(DATA_DIR, "dados.gpkg"))
   
   - Se tiver código como: gdf.to_file("/content/drive/MyDrive/algum_caminho/resultado.gpkg")
     Substitua por: gdf.to_file(os.path.join(OUTPUT_DIR, "resultado.gpkg"))
   
   - Se tiver código como: with open("/content/drive/MyDrive/algum_caminho/relatorio.json", "w") as f:
     Substitua por: with open(os.path.join(REPORT_DIR, "relatorio.json"), "w") as f:

4. LEMBRE-SE DE:
   - Verificar se 'import os' está presente no topo do notebook
   - Ajustar caminhos em todas as células que fazem operações de I/O com arquivos
   - Testar o notebook após as alterações para garantir que tudo funciona corretamente
""" 