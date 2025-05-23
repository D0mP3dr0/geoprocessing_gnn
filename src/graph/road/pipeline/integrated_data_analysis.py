# -*- coding: utf-8 -*-
"""
Integrated Data Analysis Module

Este módulo integra as análises de dados dos diferentes conjuntos de dados:
- Buildings
- Hidrografia
- Land Use
- Nature/Elevation
- Railways
- Roads
- Setores Censitários

Permite análises cruzadas e preparação de dados para GNN.
"""

import os
import json
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import networkx as nx
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv

# Configuração do ambiente
if os.path.exists('F:/TESE_MESTRADO'):
    # Configuração para ambiente local
    BASE_DIR = 'F:/TESE_MESTRADO'
    GEOPROCESSING_DIR = os.path.join(BASE_DIR, 'geoprocessing')
    DATA_DIR = os.path.join(GEOPROCESSING_DIR, 'data')
    ENRICHED_DATA_DIR = os.path.join(DATA_DIR, 'enriched_data')
    QUALITY_REPORTS_DIR = os.path.join(ENRICHED_DATA_DIR, 'quality_reports_completo')
    OUTPUT_DIR = os.path.join(GEOPROCESSING_DIR, 'outputs')
else:
    # Configuração para Google Colab
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_DIR = '/content/drive/MyDrive/geoprocessamento_gnn'
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    ENRICHED_DATA_DIR = os.path.join(DATA_DIR, 'enriched_data')
    QUALITY_REPORTS_DIR = os.path.join(ENRICHED_DATA_DIR, 'quality_reports_completo')
    OUTPUT_DIR = os.path.join(BASE_DIR, 'OUTPUT')

# Configuração de logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs(OUTPUT_DIR, exist_ok=True)
log_file_path = os.path.join(OUTPUT_DIR, f"integrated_analysis_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Paths para os relatórios de qualidade
quality_reports = {
    'buildings': os.path.join(QUALITY_REPORTS_DIR, 'buildings_qualidade_20250413_131208.json'),
    'hidrografia': os.path.join(QUALITY_REPORTS_DIR, 'hidrografia_enrichment_report_20250412_233009.json'),
    'landuse': os.path.join(QUALITY_REPORTS_DIR, 'landuse_quality_report_20250413_105354.json'),
    'nature': os.path.join(QUALITY_REPORTS_DIR, 'nature_elevation_quality_report_20250413_144444.json'),
    'railways': os.path.join(QUALITY_REPORTS_DIR, 'railways_quality_report_20250413_134853.json'), 
    'roads': os.path.join(QUALITY_REPORTS_DIR, 'road_enrichment_report_20250413_144948.json'),
    'setores': os.path.join(QUALITY_REPORTS_DIR, 'setores_censitarios_enrichment_report_20250412_235225.json')
}

# Paths para os dados processados
processed_data = {
    'buildings': os.path.join(ENRICHED_DATA_DIR, 'buildings_enriched_20250413_131208.gpkg'),
    'hidrografia': os.path.join(ENRICHED_DATA_DIR, 'hidrografia_enriched_20250412_233008.gpkg'),
    'landuse': os.path.join(ENRICHED_DATA_DIR, 'landuse_enriched_20250413_105344.gpkg'),
    'nature': os.path.join(ENRICHED_DATA_DIR, 'nature_elevation_enriched_20250413_144443.gpkg'),
    'railways': os.path.join(ENRICHED_DATA_DIR, 'railways_enriched_20250413_134853.gpkg'),
    'roads': os.path.join(ENRICHED_DATA_DIR, 'roads_enriched_20250412_230707.gpkg'),
    'setores': os.path.join(ENRICHED_DATA_DIR, 'setores_censitarios_enriched_20250413_175729.gpkg')
}

def load_quality_reports():
    """
    Carrega todos os relatórios de qualidade disponíveis
    
    Returns:
        dict: Dicionário com os relatórios de qualidade carregados
    """
    loaded_reports = {}
    
    for name, path in quality_reports.items():
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                loaded_reports[name] = data
                logger.info(f"Relatório de qualidade de {name} carregado com sucesso.")
            except Exception as e:
                logger.error(f"Erro ao carregar relatório de {name}: {e}")
        else:
            logger.warning(f"Relatório de qualidade de {name} não encontrado: {path}")
    
    return loaded_reports

def load_spatial_data():
    """
    Carrega os dados espaciais enriquecidos
    
    Returns:
        dict: Dicionário com os GeoDataFrames carregados
    """
    loaded_data = {}
    
    for name, path in processed_data.items():
        if os.path.exists(path):
            try:
                gdf = gpd.read_file(path)
                loaded_data[name] = gdf
                logger.info(f"Dados de {name} carregados: {len(gdf)} geometrias")
            except Exception as e:
                logger.error(f"Erro ao carregar dados de {name}: {e}")
        else:
            logger.warning(f"Arquivo de dados de {name} não encontrado: {path}")
    
    return loaded_data

def create_integrated_graph(spatial_data):
    """
    Cria um grafo integrado a partir dos diferentes conjuntos de dados
    
    Args:
        spatial_data: Dicionário com os GeoDataFrames carregados
        
    Returns:
        networkx.Graph: Grafo integrado
    """
    G = nx.Graph()
    
    # Adicionar nós de roads (base principal)
    if 'roads' in spatial_data:
        roads_gdf = spatial_data['roads']
        
        # Adicionar nós para cada segmento de estrada
        for idx, row in roads_gdf.iterrows():
            node_id = f"road_{idx}"
            attrs = {
                'type': 'road',
                'geometry': row.geometry,
                'length': row.geometry.length if hasattr(row.geometry, 'length') else 0
            }
            
            # Adicionar outros atributos disponíveis
            for col in roads_gdf.columns:
                if col != 'geometry' and not pd.isna(row[col]):
                    attrs[col] = row[col]
            
            G.add_node(node_id, **attrs)
    
    # Conectar pontos próximos (exemplo: roads com buildings)
    if 'roads' in spatial_data and 'buildings' in spatial_data:
        roads_gdf = spatial_data['roads']
        buildings_gdf = spatial_data['buildings']
        
        # Para cada edifício, encontrar a estrada mais próxima
        for b_idx, b_row in buildings_gdf.iterrows():
            building_id = f"building_{b_idx}"
            
            # Adicionar o nó do edifício
            attrs = {
                'type': 'building',
                'geometry': b_row.geometry,
                'area': b_row.geometry.area if hasattr(b_row.geometry, 'area') else 0
            }
            
            # Adicionar outros atributos disponíveis
            for col in buildings_gdf.columns:
                if col != 'geometry' and not pd.isna(b_row[col]):
                    attrs[col] = b_row[col]
            
            G.add_node(building_id, **attrs)
            
            # Encontrar a estrada mais próxima
            min_dist = float('inf')
            closest_road = None
            
            for r_idx, r_row in roads_gdf.iterrows():
                dist = b_row.geometry.distance(r_row.geometry)
                if dist < min_dist:
                    min_dist = dist
                    closest_road = f"road_{r_idx}"
            
            # Conectar o edifício à estrada mais próxima
            if closest_road and min_dist < 100:  # Limiar de 100 metros
                G.add_edge(building_id, closest_road, weight=min_dist, type='proximity')
    
    # Similar para outras camadas de dados
    # ...
    
    return G

def create_geometric_data(G):
    """
    Converte um grafo NetworkX para o formato PyTorch Geometric
    
    Args:
        G: Grafo NetworkX
        
    Returns:
        torch_geometric.data.Data: Dados para PyTorch Geometric
    """
    # Mapear IDs de nós para índices contíguos
    node_mapping = {node: i for i, node in enumerate(G.nodes())}
    
    # Criar lista de arestas
    edge_index = []
    edge_attr = []
    
    for u, v, data in G.edges(data=True):
        edge_index.append([node_mapping[u], node_mapping[v]])
        edge_index.append([node_mapping[v], node_mapping[u]])  # Grafo não direcionado
        
        # Atributos da aresta (peso)
        weight = data.get('weight', 1.0)
        edge_attr.append([weight])
        edge_attr.append([weight])  # Duplicado para aresta não direcionada
    
    # Converter para tensores PyTorch
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    
    # Atributos dos nós
    node_feats = []
    for node in G.nodes():
        # Extrair características do nó (exemplo simples)
        node_type = G.nodes[node].get('type', 'unknown')
        
        # One-hot encoding do tipo
        if node_type == 'road':
            feat = [1, 0, 0, 0]
        elif node_type == 'building':
            feat = [0, 1, 0, 0]
        elif node_type == 'landuse':
            feat = [0, 0, 1, 0]
        else:
            feat = [0, 0, 0, 1]
        
        node_feats.append(feat)
    
    # Converter para tensor
    x = torch.tensor(node_feats, dtype=torch.float)
    
    # Criar objeto Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data

def analyze_model_performance(reports):
    """
    Analisa o desempenho dos modelos com base nos relatórios
    
    Args:
        reports: Dicionário com os relatórios carregados
        
    Returns:
        pd.DataFrame: DataFrame com métricas comparativas
    """
    metrics = []
    
    for name, report in reports.items():
        # Extrair métricas gerais (diferente para cada tipo de relatório)
        if name == 'roads':
            network_length = report.get('road_network', {}).get('total_length_km', 0)
            feature_count = report.get('road_network', {}).get('feature_count', 0)
            metrics.append({
                'dataset': name,
                'features': feature_count,
                'length_km': network_length,
                'elevation_range': report.get('topography', {}).get('elevation', {}).get('range_m', 0)
            })
        
        elif name == 'buildings':
            buildings_count = report.get('dados_enriquecidos', {}).get('numero_edificacoes', 0)
            area_total = report.get('dados_enriquecidos', {}).get('metricas', {}).get('area', {}).get('area_total', 0)
            metrics.append({
                'dataset': name,
                'features': buildings_count,
                'area_total': area_total,
                'altura_media': report.get('dados_enriquecidos', {}).get('metricas', {}).get('altura', {}).get('media', 0)
            })
            
        # Adicionar outros tipos conforme necessário
        # ...
    
    return pd.DataFrame(metrics)

def main():
    """Função principal para executar a análise integrada"""
    logger.info("Iniciando análise integrada dos dados")
    
    # 1. Carregar relatórios de qualidade
    reports = load_quality_reports()
    logger.info(f"Carregados {len(reports)} relatórios de qualidade")
    
    # 2. Carregar dados espaciais
    spatial_data = load_spatial_data()
    logger.info(f"Carregados {len(spatial_data)} conjuntos de dados espaciais")
    
    # 3. Criar grafo integrado
    G = create_integrated_graph(spatial_data)
    logger.info(f"Grafo integrado criado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas")
    
    # 4. Analisar o desempenho do modelo
    performance_df = analyze_model_performance(reports)
    logger.info("Análise de desempenho concluída")
    
    # 5. Converter para formato PyTorch Geometric
    geometric_data = create_geometric_data(G)
    logger.info(f"Dados convertidos para PyTorch Geometric com {geometric_data.num_nodes} nós")
    
    # 6. Salvar resultados
    results_path = os.path.join(OUTPUT_DIR, f"integrated_analysis_results_{timestamp}.json")
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            "graph_stats": {
                "nodes": G.number_of_nodes(),
                "edges": G.number_of_edges(),
                "density": nx.density(G),
                "connected_components": nx.number_connected_components(G)
            },
            "timestamp": timestamp,
            "performance_metrics": performance_df.to_dict(orient='records')
        }, f, indent=2)
    
    logger.info(f"Resultados salvos em {results_path}")
    return {
        "graph": G,
        "geometric_data": geometric_data,
        "performance": performance_df,
        "reports": reports,
        "spatial_data": spatial_data
    }

if __name__ == "__main__":
    main()
