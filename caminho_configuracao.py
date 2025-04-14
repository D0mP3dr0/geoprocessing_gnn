import os

# Caminhos de diretórios no Google Drive
BASE_DIR = "/content/drive/MyDrive/geoprocessamento_gnn"
DATA_DIR = os.path.join(BASE_DIR, "DATA")  # Arquivos GPKG brutos
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT")  # Arquivos processados/editados
REPORT_DIR = os.path.join(BASE_DIR, "QUALITY_REPORT")  # Relatórios de qualidade

# Criar os diretórios se não existirem
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print(f"Diretório de dados brutos: {DATA_DIR}")
print(f"Diretório de saída: {OUTPUT_DIR}")
print(f"Diretório de relatórios: {REPORT_DIR}")

# Importante: Todas as referências a caminhos no restante do notebook devem usar estas variáveis
# Por exemplo:
# - Para carregar dados brutos: os.path.join(DATA_DIR, "nome_do_arquivo.gpkg")
# - Para salvar resultados: os.path.join(OUTPUT_DIR, "resultado.gpkg")
# - Para salvar relatórios: os.path.join(REPORT_DIR, "relatorio.json") 