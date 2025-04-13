#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Funções para enriquecimento de dados ferroviários com processamento paralelo e análise de rede.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import json
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, unary_union, split, snap
import warnings
from concurrent.futures import ProcessPoolExecutor
import psutil
from tqdm import tqdm
import time
import networkx as nx
import matplotlib.pyplot as plt
import contextily as ctx
import logging
from matplotlib.colors import LinearSegmentedColormap
import folium
from folium.plugins import MarkerCluster, MeasureControl
import math
import traceback

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('enriched_railways')

# Configurar processamento paralelo
N_WORKERS = min(psutil.cpu_count(logical=False), 8)  # Usar número físico de cores, máximo 8
PARTITION_SIZE = 1000  # Tamanho do chunk para processamento em paralelo

# Mostrar configuração do sistema
logger.info(f"Configuração do sistema:")
logger.info(f"- Número de workers: {N_WORKERS}")
logger.info(f"- Memória disponível: {psutil.virtual_memory().available / (1024*1024*1024):.2f} GB")
logger.info(f"- Tamanho dos chunks: {PARTITION_SIZE}")

# Classe personalizada para serialização JSON de tipos NumPy
class NpEncoder(json.JSONEncoder):
    """Encoder JSON personalizado para tipos NumPy."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

# Configurar diretórios
WORKSPACE_DIR = r"F:\TESE_MESTRADO\geoprocessing"
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
RAW_DIR = os.path.join(WORKSPACE_DIR, 'data', 'raw')
ENRICHED_DATA_DIR = os.path.join(WORKSPACE_DIR, 'data', 'enriched_data')
OUTPUT_DIR = ENRICHED_DATA_DIR
REPORT_DIR = r"F:\TESE_MESTRADO\geoprocessing\src\enriched_data\railways\quality_report"
VISUALIZATION_DIR = r"F:\TESE_MESTRADO\geoprocessing\outputs\visualize_enriched_data\railways"

# Garantir que os diretórios de saída existam
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Arquivo de entrada das ferrovias
RAILWAYS_FILE = os.path.join(INPUT_DIR, 'railways_processed.gpkg')
DEM_FILE = r"F:\TESE_MESTRADO\geoprocessing\data\raw\dem.tif"

# Arquivo de saída
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'railways_enriched.gpkg')

# Metadados para as ferrovias
RAILWAY_COLUMNS = {
    'osm_id': {'type': 'str', 'description': 'ID único da ferrovia no OpenStreetMap'},
    'name': {'type': 'str', 'description': 'Nome da ferrovia'},
    'railway': {'type': 'str', 'description': 'Tipo/classificação da ferrovia'},
    'z_order': {'type': 'int32', 'description': 'Ordem de sobreposição'},
    'length_km': {'type': 'float64', 'description': 'Comprimento da ferrovia em quilômetros', 'validation': {'min': 0}},
    'railway_class': {'type': 'str', 'description': 'Classificação hierárquica da ferrovia'},
    'connectivity': {'type': 'int32', 'description': 'Número de conexões da ferrovia'},
    'sinuosity': {'type': 'float64', 'description': 'Índice de sinuosidade da ferrovia', 'validation': {'min': 1}},
    'gauge_mm': {'type': 'int32', 'description': 'Bitola da ferrovia em milímetros'},
    'electrified': {'type': 'bool', 'description': 'Indica se a ferrovia é eletrificada'},
    'operator': {'type': 'str', 'description': 'Operador da ferrovia'},
    'owner': {'type': 'str', 'description': 'Proprietário da ferrovia'},
    'traffic_mode': {'type': 'str', 'description': 'Modo de tráfego (passageiros, carga)'},
    'is_bridge': {'type': 'bool', 'description': 'Indica se a ferrovia está em uma ponte/viaduto'},
    'is_tunnel': {'type': 'bool', 'description': 'Indica se a ferrovia está em um túnel'},
    'passenger_lines': {'type': 'int32', 'description': 'Número de linhas para passageiros'},
    'layer': {'type': 'int32', 'description': 'Nível da ferrovia em relação ao solo'}
}

# Mapeamento de classificação de ferrovias
RAILWAY_CLASS_MAPPING = {
    'main': 'principal',
    'branch': 'ramal',
    'service': 'serviço',
    'yard': 'pátio',
    'siding': 'desvio',
    'industrial': 'industrial',
    'preserved': 'preservada',
    'disused': 'desativada',
    'abandoned': 'abandonada'
}

def load_data():
    """
    Carrega os dados de ferrovias do arquivo de entrada.
    
    Returns:
        gpd.GeoDataFrame: Dados de ferrovias carregados.
    """
    try:
        logger.info(f"Carregando dados ferroviários de {RAILWAYS_FILE}")
        
        # Verificar se o arquivo existe
        if not os.path.exists(RAILWAYS_FILE):
            logger.error(f"Arquivo de ferrovias não encontrado: {RAILWAYS_FILE}")
            return None
        
        # Carregar o GeoDataFrame
        gdf = gpd.read_file(RAILWAYS_FILE)
        logger.info(f"Carregadas {len(gdf)} feições ferroviárias")
        logger.info(f"CRS: {gdf.crs}")
        
        return gdf
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def clean_column_names(gdf):
    """Limpa e padroniza nomes de colunas."""
    unnamed_cols = [col for col in gdf.columns if 'Unnamed' in col]
    if unnamed_cols:
        gdf = gdf.drop(columns=unnamed_cols)
    gdf.columns = gdf.columns.str.strip().str.lower()
    return gdf

def extract_tags_from_other_tags(gdf):
    """
    Extrai informações relevantes da coluna other_tags e cria colunas específicas.
    
    Args:
        gdf (gpd.GeoDataFrame): Dados de ferrovias
        
    Returns:
        gpd.GeoDataFrame: Dados com novas colunas extraídas de other_tags
    """
    logger.info("Extraindo informações da coluna other_tags")
    
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    # Inicializar novas colunas
    result['gauge_mm'] = np.nan
    result['electrified'] = False
    result['operator'] = None
    result['owner'] = None
    result['traffic_mode'] = None
    result['is_bridge'] = False
    result['is_tunnel'] = False
    result['passenger_lines'] = np.nan
    result['branch'] = None
    result['layer'] = 0
    result['usage'] = None
    result['ref'] = None
    
    # Processar each entrada em other_tags
    for idx, row in result.iterrows():
        if pd.isna(row['other_tags']):
            continue
            
        # Extrair gauge (bitola) - normalmente 1000mm para ferrovias brasileiras
        if '"gauge"=>' in row['other_tags']:
            try:
                gauge_str = row['other_tags'].split('"gauge"=>"')[1].split('"')[0]
                result.at[idx, 'gauge_mm'] = int(gauge_str)
            except:
                pass
        
        # Extrair eletrificação
        if '"electrified"=>' in row['other_tags']:
            electrified_str = row['other_tags'].split('"electrified"=>"')[1].split('"')[0]
            result.at[idx, 'electrified'] = (electrified_str.lower() != 'no')
        
        # Extrair operador
        if '"operator"=>' in row['other_tags']:
            try:
                operator_str = row['other_tags'].split('"operator"=>"')[1].split('"')[0]
                result.at[idx, 'operator'] = operator_str
            except:
                pass
        
        # Extrair proprietário
        if '"owner"=>' in row['other_tags']:
            try:
                owner_str = row['other_tags'].split('"owner"=>"')[1].split('"')[0]
                result.at[idx, 'owner'] = owner_str
            except:
                pass
        
        # Extrair modo de tráfego
        if '"railway:traffic_mode"=>' in row['other_tags']:
            try:
                traffic_str = row['other_tags'].split('"railway:traffic_mode"=>"')[1].split('"')[0]
                result.at[idx, 'traffic_mode'] = traffic_str
            except:
                pass
        
        # Verificar se é ponte/viaduto
        if '"bridge"=>' in row['other_tags']:
            result.at[idx, 'is_bridge'] = True
        
        # Verificar se é túnel
        if '"tunnel"=>' in row['other_tags']:
            result.at[idx, 'is_tunnel'] = True
        
        # Extrair número de linhas para passageiros
        if '"passenger_lines"=>' in row['other_tags']:
            try:
                lines_str = row['other_tags'].split('"passenger_lines"=>"')[1].split('"')[0]
                result.at[idx, 'passenger_lines'] = int(lines_str)
            except:
                pass
        
        # Extrair branch (ramal)
        if '"branch"=>' in row['other_tags']:
            try:
                branch_str = row['other_tags'].split('"branch"=>"')[1].split('"')[0]
                result.at[idx, 'branch'] = branch_str
            except:
                pass
        
        # Extrair layer (nível)
        if '"layer"=>' in row['other_tags']:
            try:
                layer_str = row['other_tags'].split('"layer"=>"')[1].split('"')[0]
                result.at[idx, 'layer'] = int(layer_str)
            except:
                pass
        
        # Extrair usage (uso)
        if '"usage"=>' in row['other_tags']:
            try:
                usage_str = row['other_tags'].split('"usage"=>"')[1].split('"')[0]
                result.at[idx, 'usage'] = usage_str
            except:
                pass
        
        # Extrair ref (referência)
        if '"ref"=>' in row['other_tags']:
            try:
                ref_str = row['other_tags'].split('"ref"=>"')[1].split('"')[0]
                result.at[idx, 'ref'] = ref_str
            except:
                pass
                
        # Verificar service (serviço)
        if '"service"=>' in row['other_tags']:
            try:
                service_str = row['other_tags'].split('"service"=>"')[1].split('"')[0]
                # Usar o service para classificação de railway_class
                if service_str == 'yard':
                    result.at[idx, 'railway_class'] = 'pátio'
                elif service_str == 'siding':
                    result.at[idx, 'railway_class'] = 'desvio'
                else:
                    result.at[idx, 'railway_class'] = 'serviço'
            except:
                pass
    
    # Classificar ferrovias baseado no usage e service extraídos
    result['railway_class'] = result.apply(
        lambda row: 'principal' if row['usage'] == 'main' else 
                    'ramal' if row['usage'] == 'branch' else
                    row.get('railway_class', 'outro'),
        axis=1
    )
    
    # Remover coluna other_tags (opcional)
    # result = result.drop(columns=['other_tags'])
    
    return result

def calculate_geometric_attributes(gdf):
    """
    Calcula atributos geométricos como comprimento e sinuosidade.
    
    Args:
        gdf (gpd.GeoDataFrame): Dados de ferrovias
        
    Returns:
        gpd.GeoDataFrame: Dados com atributos geométricos calculados
    """
    logger.info("Calculando atributos geométricos")
    
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    # Reprojetar para um sistema de coordenadas projetado para cálculos de distância
    if result.crs and result.crs.is_geographic:
        logger.info(f"Reprojetando de {result.crs} para SIRGAS 2000 / UTM zone 23S")
        gdf_projected = result.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S para região de Sorocaba
        
        # Calcular comprimento em quilômetros
        result['length_km'] = gdf_projected.geometry.length / 1000
    else:
        logger.warning("CRS não reconhecido ou já está em um sistema projetado")
        # Assumir que já está em um sistema projetado com unidades em metros
        result['length_km'] = result.geometry.length / 1000
    
    # Calcular sinuosidade
    sinuosities = []
    
    for geom in result.geometry:
        if isinstance(geom, LineString):
            if len(geom.coords) < 2:
                sinuosity = 1.0  # Valor padrão para linhas muito curtas
            else:
                # Comprimento real
                real_length = geom.length
                
                # Distância em linha reta entre pontos inicial e final
                start_point = Point(geom.coords[0])
                end_point = Point(geom.coords[-1])
                straight_length = start_point.distance(end_point)
                
                # Calcular sinuosidade (evitar divisão por zero)
                if straight_length > 0:
                    sinuosity = real_length / straight_length
                else:
                    sinuosity = 1.0
                    
        elif isinstance(geom, MultiLineString):
            # Para MultiLineString, tentar converter para LineString
            try:
                line = linemerge(geom)
                if isinstance(line, LineString):
                    if len(line.coords) < 2:
                        sinuosity = 1.0
                    else:
                        real_length = line.length
                        start_point = Point(line.coords[0])
                        end_point = Point(line.coords[-1])
                        straight_length = start_point.distance(end_point)
                        
                        if straight_length > 0:
                            sinuosity = real_length / straight_length
                        else:
                            sinuosity = 1.0
                else:
                    # Se não for possível converter, usar um valor médio
                    sinuosity = 1.0
            except:
                sinuosity = 1.0
        else:
            sinuosity = 1.0
            
        sinuosities.append(sinuosity)
    
    result['sinuosity'] = sinuosities
    
    return result

def build_network_topology(gdf):
    """
    Constrói a topologia da rede ferroviária e calcula métricas de conectividade.
    
    Args:
        gdf (gpd.GeoDataFrame): Dados de ferrovias
        
    Returns:
        tuple: (gdf atualizado, grafo da rede)
    """
    logger.info("Construindo topologia da rede ferroviária")
    
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    # Criar um grafo vazio
    G = nx.Graph()
    
    # Adicionar vértices e arestas ao grafo
    logger.info("Adicionando segmentos ao grafo")
    
    # Tolerância para snap (juntar pontos próximos)
    tolerance = 0.00001  # Ajuste conforme necessário
    
    # Primeiro passo: adicionar todos os pontos como nós
    node_count = 0
    endpoint_to_node = {}  # Dicionário para mapear coordenadas para IDs de nós
    
    for idx, row in result.iterrows():
        geom = row.geometry
        
        if isinstance(geom, LineString):
            start_point = geom.coords[0]
            end_point = geom.coords[-1]
            
            # Verificar se já existem nós para estes pontos
            start_node = None
            end_node = None
            
            # Verificar pontos próximos para snapping
            for point, node_id in endpoint_to_node.items():
                # Calcular distância
                dist_to_start = math.sqrt((point[0] - start_point[0])**2 + (point[1] - start_point[1])**2)
                dist_to_end = math.sqrt((point[0] - end_point[0])**2 + (point[1] - end_point[1])**2)
                
                if dist_to_start < tolerance and start_node is None:
                    start_node = node_id
                
                if dist_to_end < tolerance and end_node is None:
                    end_node = node_id
            
            # Se não encontrou nós existentes, criar novos
            if start_node is None:
                start_node = f"node_{node_count}"
                endpoint_to_node[start_point] = start_node
                G.add_node(start_node, x=start_point[0], y=start_point[1])
                node_count += 1
            
            if end_node is None:
                end_node = f"node_{node_count}"
                endpoint_to_node[end_point] = end_node
                G.add_node(end_node, x=end_point[0], y=end_point[1])
                node_count += 1
            
            # Adicionar aresta
            G.add_edge(
                start_node, 
                end_node, 
                id=row.get('osm_id', str(idx)), 
                name=row.get('name', 'Desconhecido'),
                length_km=row.get('length_km', 0),
                railway_class=row.get('railway_class', 'desconhecido'),
                weight=row.get('length_km', 0)  # Usar comprimento como peso
            )
        
        elif isinstance(geom, MultiLineString):
            # Processar cada parte do MultiLineString separadamente
            for line in geom.geoms:
                start_point = line.coords[0]
                end_point = line.coords[-1]
                
                # Verificar se já existem nós para estes pontos
                start_node = None
                end_node = None
                
                # Verificar pontos próximos para snapping
                for point, node_id in endpoint_to_node.items():
                    # Calcular distância
                    dist_to_start = math.sqrt((point[0] - start_point[0])**2 + (point[1] - start_point[1])**2)
                    dist_to_end = math.sqrt((point[0] - end_point[0])**2 + (point[1] - end_point[1])**2)
                    
                    if dist_to_start < tolerance and start_node is None:
                        start_node = node_id
                    
                    if dist_to_end < tolerance and end_node is None:
                        end_node = node_id
                
                # Se não encontrou nós existentes, criar novos
                if start_node is None:
                    start_node = f"node_{node_count}"
                    endpoint_to_node[start_point] = start_node
                    G.add_node(start_node, x=start_point[0], y=start_point[1])
                    node_count += 1
                
                if end_node is None:
                    end_node = f"node_{node_count}"
                    endpoint_to_node[end_point] = end_node
                    G.add_node(end_node, x=end_point[0], y=end_point[1])
                    node_count += 1
                
                # Calcular comprimento da linha
                line_length = line.length / 1000  # km
                
                # Adicionar aresta
                G.add_edge(
                    start_node, 
                    end_node, 
                    id=row.get('osm_id', str(idx)), 
                    name=row.get('name', 'Desconhecido'),
                    length_km=line_length,
                    railway_class=row.get('railway_class', 'desconhecido'),
                    weight=line_length  # Usar comprimento como peso
                )
    
    logger.info(f"Grafo criado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas")
    
    # Segundo passo: identificar componentes conectados
    components = list(nx.connected_components(G))
    logger.info(f"Identificados {len(components)} componentes conectados")
    
    # Calcular a conectividade de cada segmento
    node_connectivity = {}
    for node in G.nodes():
        node_connectivity[node] = len(list(G.neighbors(node)))
    
    # Mapear conectividade para segmentos
    result['connectivity'] = 0
    
    for idx, row in result.iterrows():
        geom = row.geometry
        
        if isinstance(geom, LineString):
            start_point = geom.coords[0]
            end_point = geom.coords[-1]
            
            # Encontrar os nós correspondentes
            start_node = None
            end_node = None
            
            for point, node_id in endpoint_to_node.items():
                dist_to_start = math.sqrt((point[0] - start_point[0])**2 + (point[1] - start_point[1])**2)
                dist_to_end = math.sqrt((point[0] - end_point[0])**2 + (point[1] - end_point[1])**2)
                
                if dist_to_start < tolerance and start_node is None:
                    start_node = node_id
                
                if dist_to_end < tolerance and end_node is None:
                    end_node = node_id
            
            if start_node and end_node:
                # Usar a soma da conectividade dos extremos como medida
                connectivity = node_connectivity.get(start_node, 0) + node_connectivity.get(end_node, 0)
                result.at[idx, 'connectivity'] = connectivity
    
    # Calcular betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight')
        edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')
        
        # Adicionar betweenness como atributo
        result['betweenness'] = 0.0
        for idx, row in result.iterrows():
            geom = row.geometry
            
            if isinstance(geom, LineString):
                start_point = geom.coords[0]
                end_point = geom.coords[-1]
                
                # Encontrar os nós correspondentes
                start_node = None
                end_node = None
                
                for point, node_id in endpoint_to_node.items():
                    dist_to_start = math.sqrt((point[0] - start_point[0])**2 + (point[1] - start_point[1])**2)
                    dist_to_end = math.sqrt((point[0] - end_point[0])**2 + (point[1] - end_point[1])**2)
                    
                    if dist_to_start < tolerance and start_node is None:
                        start_node = node_id
                    
                    if dist_to_end < tolerance and end_node is None:
                        end_node = node_id
                
                if start_node and end_node:
                    # Usar o valor de edge_betweenness se disponível
                    edge_key = (start_node, end_node)
                    rev_edge_key = (end_node, start_node)
                    
                    if edge_key in edge_betweenness:
                        result.at[idx, 'betweenness'] = edge_betweenness[edge_key]
                    elif rev_edge_key in edge_betweenness:
                        result.at[idx, 'betweenness'] = edge_betweenness[rev_edge_key]
                    else:
                        # Usar a média dos valores de nós como fallback
                        result.at[idx, 'betweenness'] = (betweenness.get(start_node, 0) + 
                                                         betweenness.get(end_node, 0)) / 2
    except Exception as e:
        logger.warning(f"Erro ao calcular betweenness centrality: {str(e)}")
    
    # Calcular closeness centrality
    try:
        closeness = nx.closeness_centrality(G, distance='weight')
        
        # Adicionar closeness como atributo
        result['closeness'] = 0.0
        for idx, row in result.iterrows():
            geom = row.geometry
            
            if isinstance(geom, LineString):
                start_point = geom.coords[0]
                end_point = geom.coords[-1]
                
                # Encontrar os nós correspondentes
                start_node = None
                end_node = None
                
                for point, node_id in endpoint_to_node.items():
                    dist_to_start = math.sqrt((point[0] - start_point[0])**2 + (point[1] - start_point[1])**2)
                    dist_to_end = math.sqrt((point[0] - end_point[0])**2 + (point[1] - end_point[1])**2)
                    
                    if dist_to_start < tolerance and start_node is None:
                        start_node = node_id
                    
                    if dist_to_end < tolerance and end_node is None:
                        end_node = node_id
                
                if start_node and end_node:
                    # Usar a média dos valores de closeness dos nós
                    result.at[idx, 'closeness'] = (closeness.get(start_node, 0) + 
                                                   closeness.get(end_node, 0)) / 2
    except Exception as e:
        logger.warning(f"Erro ao calcular closeness centrality: {str(e)}")
    
    # Adicionar informações sobre componentes conectados
    result['component_id'] = -1
    
    for comp_idx, component in enumerate(components):
        for idx, row in result.iterrows():
            geom = row.geometry
            
            if isinstance(geom, LineString):
                start_point = geom.coords[0]
                end_point = geom.coords[-1]
                
                # Encontrar os nós correspondentes
                start_node = None
                end_node = None
                
                for point, node_id in endpoint_to_node.items():
                    dist_to_start = math.sqrt((point[0] - start_point[0])**2 + (point[1] - start_point[1])**2)
                    dist_to_end = math.sqrt((point[0] - end_point[0])**2 + (point[1] - end_point[1])**2)
                    
                    if dist_to_start < tolerance and start_node is None:
                        start_node = node_id
                    
                    if dist_to_end < tolerance and end_node is None:
                        end_node = node_id
                
                # Se pelo menos um dos nós está no componente, considerar o segmento como parte
                if start_node in component or end_node in component:
                    result.at[idx, 'component_id'] = comp_idx
    
    # Armazenar métricas do grafo como atributos
    result.attrs['network_metrics'] = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'num_components': len(components),
        'density': nx.density(G),
        'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        'diameter': max([nx.diameter(G.subgraph(c)) for c in components]) if components else 0
    }
    
    return result, G

def identify_key_points(gdf, G=None):
    """
    Identifica pontos-chave na rede ferroviária: estações, cruzamentos, terminais.
    
    Args:
        gdf (gpd.GeoDataFrame): Dados de ferrovias
        G (nx.Graph, opcional): Grafo da rede
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame com pontos-chave identificados
    """
    logger.info("Identificando pontos-chave na rede ferroviária")
    
    if G is None:
        logger.warning("Grafo não fornecido, criando a partir dos dados de ferrovia")
        gdf, G = build_network_topology(gdf)
    
    # Identificar tipos de pontos
    key_points = []
    
    # 1. Identificar cruzamentos (nós com grau > 2)
    crossings = [
        node for node, degree in G.degree() 
        if degree > 2
    ]
    
    # 2. Identificar terminais (nós com grau 1)
    terminals = [
        node for node, degree in G.degree() 
        if degree == 1
    ]
    
    # 3. Extrair coordenadas dos pontos-chave
    for point_type, points in [
        ('cruzamento', crossings), 
        ('terminal', terminals)
    ]:
        for node in points:
            try:
                # Obter coordenadas do nó
                x = G.nodes[node].get('x')
                y = G.nodes[node].get('y')
                
                if x is not None and y is not None:
                    point = Point(x, y)
                    
                    # Coletar informações adicionais
                    node_info = {
                        'geometry': point,
                        'tipo': point_type,
                        'grau_conexao': G.degree(node)
                    }
                    
                    # Identificar ferrovias conectadas
                    connected_edges = list(G.edges(node))
                    node_info['ferrovias_conectadas'] = len(connected_edges)
                    
                    # Tentar obter nomes das ferrovias conectadas
                    railway_names = set()
                    for u, v in connected_edges:
                        edge_name = G.edges[u, v].get('name', 'Desconhecido')
                        railway_names.add(edge_name)
                    
                    node_info['nomes_ferrovias'] = list(railway_names)
                    
                    key_points.append(node_info)
            except Exception as e:
                logger.warning(f"Erro ao processar nó {node}: {e}")
    
    # Criar GeoDataFrame de pontos-chave
    if key_points:
        key_points_gdf = gpd.GeoDataFrame(key_points, crs=gdf.crs)
        
        # Adicionar colunas para facilitar análise
        key_points_gdf['index_ponto'] = range(len(key_points_gdf))
        
        logger.info(f"Identificados {len(key_points_gdf)} pontos-chave")
        logger.info("Distribuição de pontos:")
        logger.info(key_points_gdf['tipo'].value_counts())
        
        return key_points_gdf
    else:
        logger.warning("Nenhum ponto-chave identificado")
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

def classify_railway_importance(gdf, graph=None):
    """
    Classifica a importância das ferrovias com base em métricas de rede.
    
    Args:
        gdf (gpd.GeoDataFrame): Dados de ferrovias
        graph (nx.Graph, opcional): Grafo da rede ferroviária
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame com níveis de importância
    """
    logger.info("Classificando importância das ferrovias")
    
    # Criar cópia dos dados
    result = gdf.copy()
    
    if graph is None:
        logger.warning("Nenhum grafo fornecido. Classificação será baseada em atributos disponíveis.")
        result['nivel_importancia'] = 'Não classificado'
        return result
    
    # Métricas para classificação
    betweenness = nx.betweenness_centrality(graph, weight='length_km')
    degree_centrality = nx.degree_centrality(graph)
    
    # Calcular pontuação de importância
    importance_scores = []
    
    for idx, row in result.iterrows():
        try:
            # Encontrar nós para este segmento
            start_point = row.geometry.coords[0]
            end_point = row.geometry.coords[-1]
            
            # Encontrar nós do grafo mais próximos
            start_node = None
            end_node = None
            
            for node, data in graph.nodes(data=True):
                node_point = Point(data['x'], data['y'])
                
                # Verificar proximidade (usar uma tolerância pequena)
                if start_point.distance(node_point) < 0.001:
                    start_node = node
                if end_point.distance(node_point) < 0.001:
                    end_node = node
            
            # Calcular pontuação de importância
            if start_node and end_node:
                betweenness_score = (betweenness.get(start_node, 0) + betweenness.get(end_node, 0)) / 2
                degree_score = (degree_centrality.get(start_node, 0) + degree_centrality.get(end_node, 0)) / 2
                length_score = row['length_km'] / result['length_km'].max()
                
                # Combinar métricas com pesos diferentes
                importance_score = (
                    0.4 * betweenness_score + 
                    0.3 * degree_score + 
                    0.3 * length_score
                )
                importance_scores.append(importance_score)
            else:
                importance_scores.append(0)
        except Exception as e:
            logger.warning(f"Erro ao calcular importância para segmento {idx}: {e}")
            importance_scores.append(0)
    
    # Adicionar pontuação de importância
    result['pontuacao_importancia'] = importance_scores
    
    # Classificar níveis de importância
    def classificar_importancia(score):
        if score < 0.2:
            return 'Baixa'
        elif score < 0.4:
            return 'Média-Baixa'
        elif score < 0.6:
            return 'Média'
        elif score < 0.8:
            return 'Alta'
        else:
            return 'Crítica'
    
    result['nivel_importancia'] = result['pontuacao_importancia'].apply(classificar_importancia)
    
    # Estatísticas de importância
    logger.info("Distribuição de níveis de importância:")
    logger.info(result['nivel_importancia'].value_counts())
    
    return result

def save_enriched_data(railways_gdf, key_points_gdf=None):
    """
    Salva os dados enriquecidos em um GeoPackage com camadas.
    
    Args:
        railways_gdf (gpd.GeoDataFrame): GeoDataFrame de ferrovias enriquecido
        key_points_gdf (gpd.GeoDataFrame, opcional): GeoDataFrame com pontos-chave
        
    Returns:
        str: Caminho do arquivo salvo
    """
    logger.info("Salvando dados enriquecidos")
    
    # Criar timestamp para nome do arquivo
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"railways_enriched_{timestamp}.gpkg")
    
    # Salvar ferrovias
    railways_gdf.to_file(output_file, layer='railways', driver='GPKG')
    logger.info(f"Ferrovias enriquecidas salvas em {output_file}, camada 'railways'")
    
    # Salvar pontos-chave se disponíveis
    if key_points_gdf is not None and len(key_points_gdf) > 0:
        key_points_gdf.to_file(output_file, layer='key_points', driver='GPKG')
        logger.info(f"Pontos-chave salvos em {output_file}, camada 'key_points'")
    
    # Salvar metadados e atributos como JSON
    metadata = {
        'timestamp': timestamp,
        'feature_count': len(railways_gdf),
        'crs': str(railways_gdf.crs),
        'attributes': railways_gdf.attrs,
        'columns': list(railways_gdf.columns)
    }
    
    metadata_file = os.path.join(OUTPUT_DIR, f"railways_enriched_{timestamp}_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, cls=NpEncoder)
    
    logger.info(f"Metadados salvos em {metadata_file}")
    
    return output_file

def generate_quality_report(original_gdf, enriched_gdf, key_points_gdf=None, network_graph=None):
    """
    Gera um relatório de qualidade para o processo de enriquecimento.
    
    Args:
        original_gdf (gpd.GeoDataFrame): GeoDataFrame original de ferrovias
        enriched_gdf (gpd.GeoDataFrame): GeoDataFrame enriquecido de ferrovias
        key_points_gdf (gpd.GeoDataFrame, opcional): GeoDataFrame com pontos-chave
        network_graph (nx.Graph, opcional): Grafo da rede ferroviária
        
    Returns:
        str: Caminho do arquivo de relatório
    """
    logger.info("Gerando relatório de qualidade")
    
    # Criar timestamp para nome do arquivo
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORT_DIR, f"railways_quality_report_{timestamp}.json")
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    # Inicializar relatório
    report = {
        'timestamp': timestamp,
        'data_summary': {
            'original_count': len(original_gdf),
            'enriched_count': len(enriched_gdf),
            'key_points_count': len(key_points_gdf) if key_points_gdf is not None else 0,
            'crs': str(enriched_gdf.crs)
        },
        'columns': {
            'original': list(original_gdf.columns),
            'added': [col for col in enriched_gdf.columns if col not in original_gdf.columns]
        },
        'geometry_types': {
            'types': [str(geom_type) for geom_type in enriched_gdf.geometry.geom_type.unique()],
            'multilinestring_count': int((enriched_gdf.geometry.geom_type == 'MultiLineString').sum()),
            'linestring_count': int((enriched_gdf.geometry.geom_type == 'LineString').sum())
        }
    }
    
    # Adicionar estatísticas de comprimento
    if 'length_km' in enriched_gdf.columns:
        report['length_statistics'] = {
            'total_km': float(enriched_gdf['length_km'].sum()),
            'min_km': float(enriched_gdf['length_km'].min()),
            'max_km': float(enriched_gdf['length_km'].max()),
            'mean_km': float(enriched_gdf['length_km'].mean()),
            'median_km': float(enriched_gdf['length_km'].median()),
            'std_km': float(enriched_gdf['length_km'].std())
        }
    
    # Adicionar estatísticas de sinuosidade
    if 'sinuosity' in enriched_gdf.columns:
        report['sinuosity_statistics'] = {
            'min': float(enriched_gdf['sinuosity'].min()),
            'max': float(enriched_gdf['sinuosity'].max()),
            'mean': float(enriched_gdf['sinuosity'].mean()),
            'median': float(enriched_gdf['sinuosity'].median()),
            'std': float(enriched_gdf['sinuosity'].std())
        }
    
    # Adicionar contagem por classes
    if 'railway_class' in enriched_gdf.columns:
        class_counts = enriched_gdf['railway_class'].value_counts().to_dict()
        report['railway_class_counts'] = {k: int(v) for k, v in class_counts.items()}
    
    # Adicionar estatísticas de rede se disponíveis
    if network_graph is not None:
        report['network_statistics'] = {
            'nodes': network_graph.number_of_nodes(),
            'edges': network_graph.number_of_edges(),
            'connected_components': nx.number_connected_components(network_graph),
            'density': nx.density(network_graph),
            'diameter': max([nx.diameter(network_graph.subgraph(c)) 
                             for c in nx.connected_components(network_graph)])
                             if nx.number_connected_components(network_graph) > 0 else 0
        }
    
    # Adicionar estatísticas de pontos-chave
    if key_points_gdf is not None and len(key_points_gdf) > 0:
        report['key_points'] = {
            'total': len(key_points_gdf),
            'type_counts': key_points_gdf['tipo'].value_counts().to_dict()
        }
    
    # Salvar relatório como JSON
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, cls=NpEncoder)
    
    logger.info(f"Relatório de qualidade salvo em {report_file}")
    return report_file

def main():
    """
    Função principal para executar o pipeline de enriquecimento de dados ferroviários.
    """
    start_time = time.time()
    logger.info("Iniciando processamento de enriquecimento de dados ferroviários")
    
    # 1. Carregar dados
    original_gdf = load_data()
    if original_gdf is None:
        logger.error("Falha ao carregar dados. Abortando processamento.")
        return
    
    # 2. Limpar nomes de colunas
    original_gdf = clean_column_names(original_gdf)
    
    # 3. Extrair informações de tags
    enriched_gdf = extract_tags_from_other_tags(original_gdf)
    
    # 4. Calcular atributos geométricos
    enriched_gdf = calculate_geometric_attributes(enriched_gdf)
    logger.info(f"Comprimento total da rede: {enriched_gdf['length_km'].sum():.2f} km")
    
    # 5. Construir topologia de rede
    enriched_gdf, network_graph = build_network_topology(enriched_gdf)
    
    # 6. Identificar pontos-chave
    key_points_gdf = identify_key_points(enriched_gdf, network_graph)
    
    # 7. Classificar importância das ferrovias
    enriched_gdf = classify_railway_importance(enriched_gdf, network_graph)
    
    # 8. Salvar dados enriquecidos
    output_file = save_enriched_data(enriched_gdf, key_points_gdf)
    
    # 9. Gerar relatório de qualidade
    report_file = generate_quality_report(original_gdf, enriched_gdf, key_points_gdf, network_graph)
    
    # Calcular tempo de processamento
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Processamento concluído em {processing_time:.2f} segundos")
    
    # Resumo do processamento
    logger.info(f"Resumo do processamento:")
    logger.info(f"- Feições processadas: {len(enriched_gdf)}")
    logger.info(f"- Pontos-chave identificados: {len(key_points_gdf)}")
    logger.info(f"- Dados enriquecidos salvos em: {output_file}")
    logger.info(f"- Relatório de qualidade salvo em: {report_file}")
    
    return {
        'enriched_data': enriched_gdf,
        'key_points': key_points_gdf,
        'network_graph': network_graph,
        'output_file': output_file,
        'report_file': report_file
    }

if __name__ == "__main__":
    main()