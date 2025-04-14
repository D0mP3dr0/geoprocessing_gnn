"""
Script para executar o pipeline de processamento de rede viária no Google Colab.
Este script usa os caminhos exatos fornecidos pelo usuário e executa o pipeline completo.
"""

import os
import sys
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('colab_run')

# Verificar se estamos no Colab
try:
    import google.colab
    from google.colab import drive
    IN_COLAB = True
    logger.info("Ambiente Google Colab detectado")
except ImportError:
    IN_COLAB = False
    logger.warning("Este script deve ser executado no Google Colab.")
    sys.exit(1)

# Montar o Google Drive
try:
    drive.mount('/content/drive')
    logger.info("Google Drive montado com sucesso")
except:
    logger.error("Erro ao montar o Google Drive. Execute este script em uma célula do Colab.")
    sys.exit(1)

# Verificar diretórios necessários
DIRS = [
    '/content/drive/MyDrive/geoprocessamento_gnn',
    '/content/drive/MyDrive/geoprocessamento_gnn/data',
    '/content/drive/MyDrive/geoprocessamento_gnn/OUTPUT',
    '/content/drive/MyDrive/geoprocessamento_gnn/QUALITY_REPORT',
    '/content/drive/MyDrive/geoprocessamento_gnn/VISUALIZACOES'
]

for directory in DIRS:
    if not os.path.exists(directory):
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Diretório criado: {directory}")
        except Exception as e:
            logger.error(f"Erro ao criar diretório {directory}: {e}")
    else:
        logger.info(f"Diretório verificado: {directory}")

# Verificar arquivo de entrada
INPUT_FILE = '/content/drive/MyDrive/geoprocessamento_gnn/data/roads_enriched_20250412_230707.gpkg'
if not os.path.exists(INPUT_FILE):
    logger.error(f"Arquivo de entrada não encontrado: {INPUT_FILE}")
    logger.info("Verifique se o arquivo está no caminho correto.")
    sys.exit(1)
else:
    logger.info(f"Arquivo de entrada encontrado: {INPUT_FILE}")

# Verificar arquivos contextuais
CONTEXT_FILES = {
    'setores': '/content/drive/MyDrive/geoprocessamento_gnn/data/setores_censitarios_enriched_20250413_175729.gpkg',
    'landuse': '/content/drive/MyDrive/geoprocessamento_gnn/data/landuse_enriched_20250413_105344.gpkg',
    'buildings': '/content/drive/MyDrive/geoprocessamento_gnn/data/buildings_enriched_20250413_131208.gpkg'
}

for key, file_path in CONTEXT_FILES.items():
    if os.path.exists(file_path):
        logger.info(f"Arquivo contextual encontrado: {key} ({file_path})")
    else:
        logger.warning(f"Arquivo contextual não encontrado: {key} ({file_path})")

# Adicionar diretório do projeto ao path
sys.path.append('/content/drive/MyDrive/geoprocessamento_gnn')

# Importar e executar o pipeline
try:
    logger.info("Importando módulos de processamento...")
    from src.graph.road.pipeline.03_Inicializacao_Carregamento_Pre_processamento import run_preprocessing_pipeline
    
    logger.info("Executando pipeline completo...")
    result = run_preprocessing_pipeline()
    
    if result:
        logger.info("Pipeline executado com sucesso!")
        logger.info(f"Arquivo processado: {result['output_gpkg_path']}")
        logger.info(f"Nós para GNN: {result['output_nodes_csv_path']}")
        logger.info(f"Arestas para GNN: {result['output_edges_csv_path']}")
        logger.info(f"Tempo total de processamento: {result['processing_time']:.2f} segundos")
    else:
        logger.error("Falha na execução do pipeline.")
except Exception as e:
    logger.exception(f"Erro durante a execução: {str(e)}")
    sys.exit(1)

logger.info("Script concluído com sucesso!") 