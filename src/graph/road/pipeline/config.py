# -*- coding: utf-8 -*-
"""
Configuration Settings

Este módulo contém as configurações e caminhos originais usados no script grapho_intra_roads.py.
"""

import os
import torch
from datetime import datetime

# Caminhos originais do Google Drive
DRIVE_PATH = "/content/drive/MyDrive/geoprocessamento_gnn"
DATA_DIR = os.path.join(DRIVE_PATH, "DATA")
OUTPUT_DIR = os.path.join(DRIVE_PATH, "OUTPUT")
REPORT_DIR = os.path.join(DRIVE_PATH, "QUALITY_REPORT")

# Caminhos para arquivos específicos - mantendo os nomes originais
ROADS_PROCESSED_PATH = os.path.join(DATA_DIR, "processed/roads_processed.gpkg")
ROADS_ENRICHED_PATH = os.path.join(DATA_DIR, "enriched/roads_enriched.gpkg")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
ROAD_GRAPHS_DIR = os.path.join(OUTPUT_DIR, "road_graphs")

# Garantir que os diretórios existam
for directory in [DATA_DIR, OUTPUT_DIR, REPORT_DIR, RESULTS_DIR, 
                VISUALIZATIONS_DIR, MODELS_DIR, ROAD_GRAPHS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configurações de processamento de dados
DEFAULT_CRS = "EPSG:4674"  # CRS usado no Brasil
METRIC_CRS = "EPSG:31983"  # Projeção UTM para região de São Paulo (adaptar conforme necessário)
BATCH_SIZE = 64
NUM_WORKERS = 4

# Configurações do modelo
MODEL_TYPES = ["GNN", "ImprovedGNN", "AttentionGNN"]
DEFAULT_MODEL = "GNN"
DEFAULT_HIDDEN_DIM = 64
DEFAULT_DROPOUT = 0.5
DEFAULT_LEARNING_RATE = 0.01
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_EPOCHS = 200
DEFAULT_PATIENCE = 20

# Mapeamento de classes de nós
DEFAULT_NODE_CLASSES = {
    "motorway": 0,
    "trunk": 1,
    "primary": 2,
    "secondary": 3,
    "tertiary": 4,
    "residential": 5,
    "unclassified": 6,
    "service": 7,
    "other": 8
}

# Mapeamento de features de arestas
DEFAULT_EDGE_FEATURES = [
    "length",
    "speed_kph", 
    "lanes",
    "width",
    "oneway"
]

# Mapeamento de features de nós
DEFAULT_NODE_FEATURES = [
    "x",
    "y",
    "degree",
    "betweenness",
    "closeness"
]

# Configurações de visualização
VIZ_FIGSIZE = (15, 15)
DEFAULT_DPI = 300
NODE_CMAP = "viridis"
EDGE_CMAP = "plasma"
NODE_SIZE = 50
EDGE_WIDTH = 1.5

# Configurações de dispositivo
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Formato de nome de arquivo com timestamp
def get_timestamp():
    """Obter timestamp no formato padrão."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def get_output_filename(prefix, extension):
    """Gerar nome de arquivo com timestamp."""
    return f"{prefix}_{get_timestamp()}.{extension}"

# Caminhos de arquivo padrão
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, get_output_filename("model", "pt"))
DEFAULT_REPORT_PATH = os.path.join(REPORT_DIR, get_output_filename("report", "json"))
DEFAULT_VIZ_PATH = os.path.join(VISUALIZATIONS_DIR, get_output_filename("viz", "png"))

# Categorias de tipos de rodovias (conforme original)
HIGHWAY_CATEGORIES = {
    'motorway': 'highway',
    'trunk': 'highway',
    'primary': 'highway',
    'secondary': 'highway',
    'tertiary': 'highway',
    'unclassified': 'minor',
    'residential': 'minor',
    'service': 'service',
    'footway': 'pedestrian',
    'cycleway': 'bicycle',
    'path': 'pedestrian',
    'track': 'minor',
    'steps': 'pedestrian',
    'living_street': 'minor',
    'pedestrian': 'pedestrian',
    'bus_guideway': 'service',
    'road': 'unclassified'
}

# Configuração de logging
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO" 