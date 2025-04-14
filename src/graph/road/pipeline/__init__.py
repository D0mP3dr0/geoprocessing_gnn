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

# Informar sobre caminhos e uso
print(f"Pipeline de análise de redes viárias inicializado.")
print(f"Usando caminhos do Google Drive: {DRIVE_PATH}")
print(f"Para começar, use mount_google_drive() para montar o drive e garantir acesso aos arquivos.") 