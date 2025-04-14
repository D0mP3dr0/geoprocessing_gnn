# -*- coding: utf-8 -*-
"""
Pipeline de Análise de Redes Viárias

Este pacote fornece componentes modulares para análise de redes viárias 
utilizando métodos baseados em grafos e GNN, mantendo os mesmos caminhos
e estrutura de arquivos do código original no Google Drive:
/content/drive/MyDrive/geoprocessamento_gnn

Os módulos preservam as funcionalidades originais, organizados de forma mais modular.
"""

__version__ = '0.1.0'

# Importar módulos principais para facilitar o acesso
from .config import (
    DRIVE_PATH, DATA_DIR, OUTPUT_DIR, REPORT_DIR,
    ROADS_PROCESSED_PATH, ROADS_ENRICHED_PATH
)

# Para carregar os dados originais
from .data_loading import load_road_data, load_from_osm, mount_google_drive

# Importar funções do módulo de pré-processamento
# Usamos importação com nome de arquivo como string para evitar problemas com o número no nome
import importlib
preprocessing_module = importlib.import_module("src.graph.road.pipeline.03_Inicializacao_Carregamento_Pre_processamento")

# Importar funções específicas do módulo
load_contextual_data = preprocessing_module.load_contextual_data
explode_multilines = preprocessing_module.explode_multilines
calculate_sinuosity = preprocessing_module.calculate_sinuosity
clean_road_data = preprocessing_module.clean_road_data
check_connectivity = preprocessing_module.check_connectivity
prepare_node_features = preprocessing_module.prepare_node_features
prepare_edge_features = preprocessing_module.prepare_edge_features
normalize_features = preprocessing_module.normalize_features
preprocess_road_data = preprocessing_module.preprocess_road_data
run_preprocessing_pipeline = preprocessing_module.run_preprocessing_pipeline

# Exportar funções específicas para facilitar a importação
__all__ = [
    'load_road_data',
    'load_contextual_data',
    'explode_multilines',
    'calculate_sinuosity',
    'clean_road_data',
    'check_connectivity',
    'prepare_node_features',
    'prepare_edge_features',
    'normalize_features',
    'preprocess_road_data',
    'run_preprocessing_pipeline'
]

# Informar sobre caminhos e uso
print(f"Pipeline de análise de redes viárias inicializado.")
print(f"Usando caminhos do Google Drive: {DRIVE_PATH}")
print(f"Para começar, use mount_google_drive() para montar o drive e garantir acesso aos arquivos.") 