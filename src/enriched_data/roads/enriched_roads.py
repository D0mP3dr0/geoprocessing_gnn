#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Funções para enriquecimento de dados de estradas com métricas de rede e análises morfométricas.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiLineString
from shapely.ops import unary_union, linemerge, nearest_points
import matplotlib.pyplot as plt
import networkx as nx
import json
import datetime
import time
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from concurrent.futures import ProcessPoolExecutor
import psutil
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap
import folium
from folium.plugins import MarkerCluster
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
import math
from shapely.geometry import mapping
import traceback
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
from scipy.interpolate import griddata
import pyproj
import numba

# Configurar logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configurar processamento paralelo
N_WORKERS = min(psutil.cpu_count(logical=False), 8)  # Usar número físico de cores, máximo 8
PARTITION_SIZE = 1000  # Tamanho do chunk para processamento em paralelo

# Adicionar a opção global no início do arquivo, após os imports
SKIP_GRAPH_ANALYSIS = True  # Definir como True para pular análise de grafo

# Função para configurar estilo visual consistente
def setup_visualization_style():
    """
    Configura um estilo visual consistente e profissional para todas as visualizações.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Configurações gerais de estilo
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    mpl.rcParams['font.size'] = 11
    mpl.rcParams['axes.titlesize'] = 18
    mpl.rcParams['axes.titleweight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['xtick.labelsize'] = 12
    mpl.rcParams['ytick.labelsize'] = 12
    mpl.rcParams['legend.fontsize'] = 12
    mpl.rcParams['legend.frameon'] = True
    mpl.rcParams['legend.framealpha'] = 0.9
    mpl.rcParams['legend.edgecolor'] = 'gray'
    mpl.rcParams['figure.titlesize'] = 20
    mpl.rcParams['figure.titleweight'] = 'bold'
    mpl.rcParams['figure.figsize'] = (12, 8)
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.2
    
    # Criar paletas de cores personalizadas para uso nas visualizações
    
    # Paleta para estradas (vermelhos)
    reds = ['#7f0000', '#b30000', '#d7301f', '#ef6548', '#fc8d59', '#fdbb84', '#fdd49e', '#fee8c8', '#fff7ec']
    roads_cmap = LinearSegmentedColormap.from_list('roads_reds', reds)
    
    # Paleta para sinuosidade (vermelho-verde)
    sinuosity_colors = ['#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
    sinuosity_cmap = LinearSegmentedColormap.from_list('sinuosity', sinuosity_colors)
    
    # Paleta para tipo de estrada (tons de marrom)
    road_type_colors = ['#8c510a', '#bf812d', '#dfc27d', '#f6e8c3', '#c7eae5', '#80cdc1', '#35978f', '#01665e']
    road_type_cmap = LinearSegmentedColormap.from_list('road_types', road_type_colors)
    
    # Paleta para betweenness (centrality)
    betweenness_colors = ['#edf8fb', '#b3cde3', '#8c96c6', '#8856a7', '#810f7c']
    betweenness_cmap = LinearSegmentedColormap.from_list('betweenness', betweenness_colors)
    
    # Registrar as paletas para uso
    try:
        plt.colormaps.register(cmap=roads_cmap)
        plt.colormaps.register(cmap=sinuosity_cmap)
        plt.colormaps.register(cmap=road_type_cmap)
        plt.colormaps.register(cmap=betweenness_cmap)
    except:
        # Fallback para versões mais antigas do matplotlib
        try:
            plt.cm.register_cmap(name='roads_reds', cmap=roads_cmap)
            plt.cm.register_cmap(name='sinuosity', cmap=sinuosity_cmap)
            plt.cm.register_cmap(name='road_types', cmap=road_type_cmap)
            plt.cm.register_cmap(name='betweenness', cmap=betweenness_cmap)
        except Exception as e:
            logger.warning(f"Erro ao registrar colormaps: {str(e)}")
            logger.warning("Usando colormaps padrão em vez disso.")

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

# Get the absolute path to the project directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Define input and output directories
INPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'processed')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'enriched')
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'quality_reports', 'roads')
VISUALIZATION_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'outputs', 'visualizations', 'roads')

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Definir arquivos de entrada e saída
ROADS_FILE = os.path.join(INPUT_DIR, 'roads_processed.gpkg')

# Arquivo de dados altimétricos
DEM_FILE = r"F:\TESE_MESTRADO\geoprocessing\data\raw\dem.tif"

# Arquivo de saída
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'roads_enriched.gpkg')

def load_data():
    """
    Carrega os dados processados de estradas a partir do diretório de dados processados.
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame com os dados de estradas carregados.
    """
    logger.info("Carregando dados processados de estradas...")
    
    try:
        if os.path.exists(ROADS_FILE):
            gdf = gpd.read_file(ROADS_FILE)
            logger.info(f"Carregadas {len(gdf)} feições de estradas de {ROADS_FILE}")
            return gdf
        else:
            logger.error(f"Arquivo não encontrado: {ROADS_FILE}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Erro ao carregar os dados de estradas: {str(e)}")
        sys.exit(1)

def load_dem():
    """
    Carrega o Modelo Digital de Elevação (DEM) e retorna o objeto raster.
    
    Returns:
        rasterio.DatasetReader: Objeto raster do DEM carregado
    """
    logger.info("Carregando dados altimétricos (DEM)...")
    
    try:
        if not os.path.exists(DEM_FILE):
            logger.error(f"Arquivo DEM não encontrado: {DEM_FILE}")
            return None
        
        dem = rasterio.open(DEM_FILE)
        
        logger.info(f"DEM carregado: resolução {dem.res}, CRS {dem.crs}")
        logger.info(f"Faixa de elevação: {dem.read(1).min()} a {dem.read(1).max()} metros")
        
        return dem
    except Exception as e:
        logger.error(f"Erro ao carregar DEM: {str(e)}")
        return None

@numba.jit(nopython=True)
def calculate_sinuosity_fast(x_coords, y_coords):
    """
    Calcula sinuosidade de forma otimizada usando Numba.
    
    Args:
        x_coords (np.array): Array de coordenadas X
        y_coords (np.array): Array de coordenadas Y
        
    Returns:
        float: Índice de sinuosidade
    """
    if len(x_coords) < 2:
        return 1.0
    
    # Distância em linha reta do início ao fim
    start_x, start_y = x_coords[0], y_coords[0]
    end_x, end_y = x_coords[-1], y_coords[-1]
    straight_length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
    
    # Comprimento real da linha
    actual_length = 0.0
    for i in range(len(x_coords) - 1):
        dx = x_coords[i+1] - x_coords[i]
        dy = y_coords[i+1] - y_coords[i]
        actual_length += np.sqrt(dx*dx + dy*dy)
    
    # Evitar divisão por zero
    if straight_length > 0:
        return actual_length / straight_length
    return 1.0

def calculate_sinuosity(gdf):
    """
    Calcula o índice de sinuosidade para cada segmento de estrada.
    Sinuosidade = comprimento real / distância em linha reta
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de estradas
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com coluna de sinuosidade
    """
    logger.info("Calculando índices de sinuosidade...")
    
    # Criar uma cópia para evitar problemas de visualização com cópia
    result = gdf.copy()
    
    # Calcular sinuosidade para cada linha
    sinuosities = []
    
    for geom in result.geometry:
        if isinstance(geom, LineString):
            # Verificar se a linha tem pelo menos dois pontos
            if len(geom.coords) < 2:
                sinuosity = None
                continue
            
            # Comprimento real da linha
            actual_length = geom.length
            
            # Distância em linha reta do início ao fim
            start_point = Point(geom.coords[0])
            end_point = Point(geom.coords[-1])
            straight_length = start_point.distance(end_point)
            
            # Evitar divisão por zero
            if straight_length > 0:
                sinuosity = actual_length / straight_length
            else:
                sinuosity = 1.0
        else:
            sinuosity = None
        
        sinuosities.append(sinuosity)
    
    result['sinuosity'] = sinuosities
    
    # Estatísticas básicas
    valid_values = [s for s in sinuosities if s is not None]
    if valid_values:
        logger.info(f"Sinuosidade média: {np.mean(valid_values):.2f}")
        logger.info(f"Sinuosidade máxima: {np.max(valid_values):.2f}")
        logger.info(f"Sinuosidade mínima: {np.min(valid_values):.2f}")
    
    logger.info("Sinuosidade calculada com sucesso")
    return result

def get_sinuosity_color(sinuosity):
    """
    Retorna a cor apropriada com base no valor de sinuosidade.
    
    Args:
        sinuosity (float): Valor do índice de sinuosidade
        
    Returns:
        str: Código de cor hexadecimal
    """
    if sinuosity is None or np.isnan(sinuosity):
        return '#808080'  # Cinza para valores desconhecidos
    
    if sinuosity < 1.05:
        return '#fef0d9'  # Bege claro para estradas retas
    elif sinuosity < 1.2:
        return '#fdcc8a'  # Laranja claro
    elif sinuosity < 1.5:
        return '#fc8d59'  # Laranja médio
    elif sinuosity < 2.0:
        return '#e34a33'  # Laranja escuro
    else:
        return '#b30000'  # Vermelho para alta sinuosidade

def build_road_network(gdf, edge_key='ID', skip_analysis=False):
    """
    Constrói uma representação de rede das estradas usando NetworkX.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame contendo dados de estradas
        edge_key (str): Nome da coluna a ser usada como identificador das arestas
        skip_analysis (bool): Se True, pula a análise de centralidade
        
    Returns:
        tuple: (Grafo NetworkX, GeoDataFrame atualizado com métricas de rede)
    """
    logger.info("Construindo rede viária...")
    
    if gdf is None or gdf.empty:
        logger.error("Dados de estradas não disponíveis para construção da rede")
        return None, gdf
    
    # Criar um grafo de rede
    G = nx.Graph()
    
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    # Adicionar arestas ao grafo
    edge_count = 0
    for idx, row in result.iterrows():
        if isinstance(row.geometry, LineString):
            # Usar pontos de início e fim como nós
            start_point = row.geometry.coords[0]
            end_point = row.geometry.coords[-1]
            
            # Adicionar aresta com atributos
            edge_id = str(row[edge_key]) if edge_key in row and row[edge_key] is not None else str(idx)
            G.add_edge(start_point, end_point, 
                       length=row.geometry.length,
                       feature_id=edge_id)
            edge_count += 1
    
    logger.info(f"Rede criada com {len(G.nodes)} nós e {edge_count} arestas")
    
    # Se devemos pular a análise de centralidade
    if skip_analysis or SKIP_GRAPH_ANALYSIS:
        logger.info("Análise de grafo desativada, pulando cálculos de centralidade...")
        # Retornar o grafo sem métricas de centralidade
        return G, result
    
    # Calcular medidas de centralidade da rede
    logger.info("Calculando métricas de centralidade da rede...")
    
    # Betweenness centrality (identifica segmentos importantes de conexão)
    try:
        # Para redes muito grandes, usar amostragem
        if len(G.nodes) > 10000:
            logger.info("Rede muito grande, calculando betweenness com amostragem...")
            # Usar apenas um subconjunto de nós como fontes para reduzir o tempo de cálculo
            k = min(5000, len(G.nodes))
            sources = list(G.nodes())[:k]  # Usar os primeiros k nós
            
            edge_betweenness = nx.edge_betweenness_centrality_subset(
                G, 
                sources=sources, 
                targets=sources, 
                weight='length',
                normalized=True
            )
        else:
            edge_betweenness = nx.edge_betweenness_centrality(
                G, 
                weight='length',
                normalized=True
            )
        
        # Mapear betweenness para as features originais
        betweenness_dict = {}
        
        for (u, v), value in edge_betweenness.items():
            # Encontrar a aresta correspondente no grafo original
            if 'feature_id' in G[u][v]:
                feature_id = G[u][v]['feature_id']
                betweenness_dict[feature_id] = value
        
        # Adicionar valores ao GeoDataFrame baseado no ID
        result['betweenness'] = result[edge_key].astype(str).map(betweenness_dict)
        result['betweenness'] = result['betweenness'].fillna(0)
        
        logger.info(f"Betweenness calculado com sucesso para {len(betweenness_dict)} arestas")
    except Exception as e:
        logger.error(f"Erro ao calcular betweenness: {str(e)}")
        result['betweenness'] = 0
    
    # Identificar nós de alta centralidade (hubs)
    try:
        # Degree centrality (número de conexões)
        degree_centrality = dict(G.degree())
        
        # Encontrar nós com grau > 2 (interseções)
        intersections = {node: degree for node, degree in degree_centrality.items() if degree > 2}
        
        logger.info(f"Identificadas {len(intersections)} interseções na rede")
    except Exception as e:
        logger.error(f"Erro ao identificar interseções: {str(e)}")
        intersections = {}
    
    return G, result

def identify_intersections(gdf, crs=None):
    """
    Identifica interseções na rede viária.
    
    Args:
        gdf (gpd.GeoDataFrame ou nx.Graph): GeoDataFrame com estradas ou grafo NetworkX
        crs (pyproj.CRS, opcional): Sistema de referência de coordenadas para o resultado
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame com as interseções identificadas
    """
    logger.info("Identificando interseções na rede viária...")
    
    # Verificar se o input é um grafo ou um GeoDataFrame
    if isinstance(gdf, nx.Graph):
        G = gdf
    elif isinstance(gdf, gpd.GeoDataFrame) and not gdf.empty:
        # Construir grafo a partir do GeoDataFrame
        G = nx.Graph()
        for idx, row in gdf.iterrows():
            if isinstance(row.geometry, LineString):
                start_point = row.geometry.coords[0]
                end_point = row.geometry.coords[-1]
                G.add_edge(start_point, end_point, feature_id=idx)
    else:
        logger.warning("Não foi possível identificar interseções: dados inválidos")
        return gpd.GeoDataFrame(geometry=[], crs=crs if crs else "EPSG:4326")
    
    # Identificar nós com mais de 2 conexões (interseções)
    intersections = []
    for node, degree in G.degree():
        if degree > 2:  # Nós com mais de 2 arestas são interseções
            # Obter arestas conectadas a este nó
            connected_edges = []
            for neighbor in G.neighbors(node):
                if 'feature_id' in G[node][neighbor]:
                    connected_edges.append(str(G[node][neighbor]['feature_id']))
            
            intersections.append({
                'geometry': Point(node),
                'degree': degree,
                'connected_edges': ','.join(connected_edges),
                'type': 'crossroad' if degree == 4 else ('complex' if degree > 4 else 'junction'),
            })
    
    logger.info(f"Identificadas {len(intersections)} interseções na rede")
    
    # Criar GeoDataFrame com as interseções
    if intersections:
        intersections_gdf = gpd.GeoDataFrame(intersections, crs=crs if crs else "EPSG:4326")
        return intersections_gdf
    else:
        return gpd.GeoDataFrame(geometry=[], crs=crs if crs else "EPSG:4326")

@numba.jit(nopython=True)
def calculate_elevation_stats(elevations):
    """
    Calcula estatísticas de elevação de forma otimizada.
    
    Args:
        elevations (np.array): Array de valores de elevação
        
    Returns:
        tuple: (min, max, mean, range)
    """
    if len(elevations) == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    elev_min = np.min(elevations)
    elev_max = np.max(elevations)
    elev_mean = np.mean(elevations)
    elev_range = elev_max - elev_min
    
    return elev_min, elev_max, elev_mean, elev_range

@numba.jit(nopython=True)
def calculate_slope(elev_start, elev_end, distance):
    """
    Calcula declividade de forma otimizada.
    
    Args:
        elev_start (float): Elevação no ponto inicial
        elev_end (float): Elevação no ponto final
        distance (float): Distância entre os pontos
        
    Returns:
        tuple: (slope_pct, slope_deg)
    """
    if distance <= 0:
        return 0.0, 0.0
    
    elev_change = abs(elev_end - elev_start)
    slope_pct = (elev_change / distance) * 100
    slope_deg = np.arctan(elev_change / distance) * (180 / np.pi)
    
    return slope_pct, slope_deg

def extract_elevation_for_lines(gdf, dem):
    """
    Extrai valores de elevação para cada feição linear a partir do DEM.
    Implementação otimizada com processamento por lotes.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame com geometrias lineares
        dem (rasterio.DatasetReader): Dataset raster contendo dados de elevação
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame com colunas de elevação adicionadas
    """
    logger.info("Extraindo dados de elevação para geometrias lineares...")
    
    if dem is None:
        logger.warning("DEM não disponível. Pulando extração de elevação.")
        return gdf
    
    # Criar cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    # Verificar se o GeoDataFrame está no mesmo CRS que o DEM
    if result.crs != dem.crs:
        logger.info(f"Reprojetando dados de {result.crs} para {dem.crs}")
        result = result.to_crs(dem.crs)
    
    # Inicializar colunas para armazenar elevação
    result['elevation_min'] = np.nan
    result['elevation_max'] = np.nan
    result['elevation_mean'] = np.nan
    result['elevation_range'] = np.nan
    result['slope_pct'] = np.nan
    result['slope_deg'] = np.nan
    
    # Determinar resolução do DEM
    dem_resolution = min(abs(dem.res[0]), abs(dem.res[1]))
    sample_distance = dem_resolution * 0.8  # 80% da resolução do DEM
    
    # Carregar todo o DEM na memória para processamento mais rápido
    try:
        dem_data = dem.read(1)
        dem_bounds = dem.bounds
        dem_transform = dem.transform
        dem_nodata = dem.nodata
        logger.info("DEM carregado na memória para processamento mais rápido")
    except Exception as e:
        logger.error(f"Erro ao carregar DEM na memória: {str(e)}")
        dem_data = None
        
    # Processar em lotes para reduzir o uso de memória
    batch_size = 1000
    num_batches = (len(result) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(result))
        
        logger.info(f"Processando lote {batch_idx + 1}/{num_batches} (linhas {start_idx}-{end_idx})")
        
        for idx in range(start_idx, end_idx):
            try:
                geom = result.iloc[idx].geometry
                
                if not isinstance(geom, (LineString, MultiLineString)):
                    continue
                
                # Para MultiLineString, processar cada parte separadamente
                if isinstance(geom, MultiLineString):
                    # Converter para LineString pela união das partes
                    try:
                        geom = linemerge(geom)
                        if isinstance(geom, MultiLineString):  # Se ainda for MultiLineString após merge
                            # Usar a parte mais longa
                            geom = sorted(geom.geoms, key=lambda line: line.length, reverse=True)[0]
                    except Exception as e:
                        logger.debug(f"Erro ao mesclar MultiLineString: {str(e)}")
                        continue
                
                # Gerar pontos ao longo da linha com espaçamento adequado
                line_length = geom.length
                num_points = max(10, int(line_length / sample_distance))
                
                distances = np.linspace(0, line_length, num_points)
                elevations = []
                
                for distance in distances:
                    try:
                        # Obter ponto ao longo da linha
                        point = geom.interpolate(distance)
                        x, y = point.x, point.y
                        
                        # Verificar se o ponto está dentro da extensão do DEM
                        if (dem_bounds[0] <= x <= dem_bounds[2] and 
                            dem_bounds[1] <= y <= dem_bounds[3]):
                            
                            # Converter coordenadas para índices de pixel
                            row, col = rasterio.transform.rowcol(dem_transform, x, y)
                            
                            # Verificar limites
                            if 0 <= row < dem_data.shape[0] and 0 <= col < dem_data.shape[1]:
                                # Ler valor do pixel
                                elevation = dem_data[row, col]
                                
                                # Verificar se o valor é válido
                                if elevation != dem_nodata:
                                    elevations.append(float(elevation))
                    except Exception as e:
                        logger.debug(f"Erro ao extrair elevação para ponto: {str(e)}")
                        continue
                
                # Calcular estatísticas de elevação se temos valores válidos
                if elevations:
                    elevations_array = np.array(elevations, dtype=np.float64)
                    elev_min, elev_max, elev_mean, elev_range = calculate_elevation_stats(elevations_array)
                    
                    result.at[result.index[idx], 'elevation_min'] = elev_min
                    result.at[result.index[idx], 'elevation_max'] = elev_max
                    result.at[result.index[idx], 'elevation_mean'] = elev_mean
                    result.at[result.index[idx], 'elevation_range'] = elev_range
                    
                    # Calcular declividade
                    if len(elevations) > 1 and line_length > 0:
                        slope_pct, slope_deg = calculate_slope(elevations[0], elevations[-1], line_length)
                        result.at[result.index[idx], 'slope_pct'] = slope_pct
                        result.at[result.index[idx], 'slope_deg'] = slope_deg
            except Exception as e:
                logger.warning(f"Erro ao processar geometria {idx}: {str(e)}")
                continue
    
    # Calcular estatísticas sobre os dados de elevação
    valid_elevations = result['elevation_mean'].dropna()
    if not valid_elevations.empty:
        logger.info(f"Elevação média: {valid_elevations.mean():.2f} m")
        logger.info(f"Elevação mínima: {result['elevation_min'].min():.2f} m")
        logger.info(f"Elevação máxima: {result['elevation_max'].max():.2f} m")
        logger.info(f"Declividade média: {result['slope_pct'].mean():.2f}%")
    
    # Classificar estradas por declividade
    if not result['slope_pct'].dropna().empty:
        bins = [0, 3, 8, 15, 30, float('inf')]
        labels = ['Plana', 'Suave', 'Moderada', 'Íngreme', 'Muito Íngreme']
        
        result['slope_class'] = pd.cut(
            result['slope_pct'].fillna(0),
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Log estatísticas por classe
        slope_counts = result['slope_class'].value_counts()
        for slope_class, count in slope_counts.items():
            logger.info(f"Estradas com declividade {slope_class}: {count} ({count/len(result)*100:.1f}%)")
    
    return result

def extract_elevation_for_points(gdf, dem):
    """
    Extrai dados de elevação para features pontuais (interseções, terminais).
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame com features pontuais
        dem (rasterio.io.DatasetReader): Modelo Digital de Elevação
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame com atributos de elevação adicionados
    """
    logger.info("Extraindo dados de elevação para pontos...")
    
    # Verificar se temos dados válidos
    if gdf.empty or dem is None:
        logger.warning("Dados insuficientes para extrair elevação para pontos.")
        return gdf
    
    # Criar uma cópia para evitar avisos
    result = gdf.copy()
    
    # Reprojetar para o mesmo CRS do DEM se necessário
    if result.crs != dem.crs:
        logger.info(f"Reprojetando de {result.crs} para {dem.crs}")
        result = result.to_crs(dem.crs)
    
    # Array para armazenar elevações
    elevations = []
    
    # Para cada ponto, extrair elevação
    for idx, row in result.iterrows():
        try:
            # Obter coordenadas do ponto
            x, y = row.geometry.x, row.geometry.y
            
            # Converter coordenadas para índices de pixel
            py, px = dem.index(x, y)
            
            # Verificar se o índice está dentro dos limites
            if (0 <= py < dem.height) and (0 <= px < dem.width):
                # Obter valor do pixel
                value = dem.read(1, window=((py, py+1), (px, px+1)))
                # Adicionar elevação se não for valor nulo
                if value[0][0] != dem.nodata:
                    elevations.append(float(value[0][0]))
                else:
                    elevations.append(None)
            else:
                elevations.append(None)
        except Exception as e:
            logger.warning(f"Erro ao extrair elevação para ponto {idx}: {str(e)}")
            elevations.append(None)
    
    # Adicionar coluna ao GeoDataFrame
    result['elevation'] = elevations
    
    # Calcular estatísticas básicas para diferentes tipos de pontos, se disponível
    if 'type' in result.columns:
        for point_type in result['type'].unique():
            valid_elevs = [e for e in result[result['type'] == point_type]['elevation'] if e is not None]
            if valid_elevs:
                logger.info(f"Elevação média para {point_type}s: {np.mean(valid_elevs):.2f} m")
    
    return result

def calculate_road_quality_metrics(roads_gdf, intersections_gdf=None):
    """
    Calcula métricas de qualidade da rede viária.
    
    Args:
        roads_gdf (gpd.GeoDataFrame): GeoDataFrame com estradas
        intersections_gdf (gpd.GeoDataFrame, opcional): GeoDataFrame com interseções
        
    Returns:
        Dict: Métricas de qualidade
    """
    logger.info("Calculando métricas de qualidade da rede viária...")
    
    metrics = {}
    
    # Conectividade da rede
    if hasattr(roads_gdf, 'attrs') and 'network_info' in roads_gdf.attrs:
        network_info = roads_gdf.attrs['network_info']
        
        # Índice alfa (conectividade dos circuitos)
        alpha_index = network_info.get('alpha_index')
        if alpha_index is not None:
            metrics['alpha_connectivity_index'] = alpha_index
            
        # Densidade da rede viária
        road_density = network_info.get('road_density_km_per_sqkm')
        if road_density is not None:
            metrics['road_density_km_per_sqkm'] = road_density
    
    # Analisar distribuição da sinuosidade
    if 'sinuosity' in roads_gdf.columns:
        sinuosity = roads_gdf['sinuosity'].dropna()
        if not sinuosity.empty:
            metrics['sinuosity'] = {
                'mean': float(sinuosity.mean()),
                'max': float(sinuosity.max()),
                'min': float(sinuosity.min()),
                'std': float(sinuosity.std()),
                'pct_straight': float((sinuosity < 1.05).mean() * 100),
                'pct_winding': float((sinuosity > 1.2).mean() * 100)
            }
    
    # Interseções
    if intersections_gdf is not None and not intersections_gdf.empty:
        # Analisar densidade de interseções
        try:
            # Calcular área em km² usando bounding box dos dados
            bbox = roads_gdf.total_bounds
            bbox_area = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / 1_000_000
            
            # Calcular densidade de interseções
            intersection_density = len(intersections_gdf) / bbox_area
            metrics['intersection_density_per_sqkm'] = float(intersection_density)
            
            # Distribuição por tipo
            if 'type' in intersections_gdf.columns:
                type_counts = intersections_gdf['type'].value_counts().to_dict()
                metrics['intersection_types'] = {k: int(v) for k, v in type_counts.items()}
                
            # Estatísticas de conectividade das interseções
            if 'degree' in intersections_gdf.columns:
                metrics['avg_intersection_degree'] = float(intersections_gdf['degree'].mean())
        except Exception as e:
            logger.warning(f"Erro ao calcular métricas de interseções: {str(e)}")
    
    # Análise de declividade das estradas
    if 'slope_pct' in roads_gdf.columns:
        slope = roads_gdf['slope_pct'].dropna()
        if not slope.empty:
            metrics['slope'] = {
                'mean_pct': float(slope.mean()),
                'max_pct': float(slope.max()),
                'min_pct': float(slope.min()),
                'std_pct': float(slope.std())
            }
            
            # Calcular percentual de estradas com alta declividade
            high_slope_pct = float((slope > 8.0).mean() * 100)
            metrics['slope']['pct_high_slope'] = high_slope_pct
    
    return metrics

def enrich_road_data(data, dem=None, skip_graph_analysis=None):
    """
    Enriquece os dados de estradas com métricas adicionais.
    
    Args:
        data (Dict[str, gpd.GeoDataFrame]): Dicionário com GeoDataFrames contendo dados de estradas
        dem (rasterio.io.DatasetReader, optional): Modelo Digital de Elevação
        skip_graph_analysis (bool, optional): Se deve pular análise de grafo
        
    Returns:
        Dict[str, gpd.GeoDataFrame]: Dados enriquecidos
    """
    if skip_graph_analysis is None:
        skip_graph_analysis = SKIP_GRAPH_ANALYSIS
    
    logger.info("Iniciando enriquecimento de dados de estradas...")
    
    # Resultado que será retornado
    enriched_data = {}
    
    # Reprojetar para uma projeção métrica se necessário (importante para cálculos precisos)
    if data['roads'].crs and data['roads'].crs.is_geographic:
        logger.info(f"Reprojetando dados de {data['roads'].crs} para projeção métrica...")
        data['roads'] = data['roads'].to_crs(epsg=3857)  # Web Mercator
    
    # 1. Calcular sinuosidade
    if 'roads' in data and not data['roads'].empty:
        try:
            data['roads'] = calculate_sinuosity(data['roads'])
            logger.info("Sinuosidade calculada com sucesso")
        except Exception as e:
            logger.error(f"Erro ao calcular sinuosidade: {str(e)}")
    
    # 2. Construir rede viária para análise de centralidade
    if 'roads' in data and not data['roads'].empty:
        try:
            G, data['roads'] = build_road_network(
                data['roads'], 
                edge_key='ID' if 'ID' in data['roads'].columns else None,
                skip_analysis=skip_graph_analysis
            )
            
            # Identificar interseções se a análise não for pulada
            if not skip_graph_analysis and G is not None:
                try:
                    data['intersections'] = identify_intersections(G, data['roads'].crs)
                    if not data['intersections'].empty:
                        logger.info("Interseções identificadas com sucesso")
                    else:
                        logger.warning("Nenhuma interseção encontrada na rede viária")
                except Exception as e:
                    logger.error(f"Erro ao identificar interseções: {str(e)}")
                    # Criar GeoDataFrame vazio para interseções para evitar erros posteriores
                    data['intersections'] = gpd.GeoDataFrame(geometry=[], crs=data['roads'].crs)
            else:
                # Criar GeoDataFrame vazio para interseções quando não realizamos análise
                data['intersections'] = gpd.GeoDataFrame(geometry=[], crs=data['roads'].crs)
                
        except Exception as e:
            logger.error(f"Erro ao construir rede viária: {str(e)}")
            # Criar GeoDataFrame vazio para interseções para evitar erros posteriores
            data['intersections'] = gpd.GeoDataFrame(geometry=[], crs=data['roads'].crs if 'roads' in data else "EPSG:4326")
    
    # 3. Extrair dados de elevação e declividade se DEM disponível
    if dem is None:
        logger.warning("DEM não disponível. Pulando extração de elevação.")
    else:
        # Processar estradas
        if 'roads' in data and not data['roads'].empty:
            data['roads'] = extract_elevation_for_lines(data['roads'], dem)
            
        # Processar interseções
        if 'intersections' in data and not data['intersections'].empty:
            data['intersections'] = extract_elevation_for_points(data['intersections'], dem)
    
    # 4. Classificar estradas por importância
    if 'roads' in data and not data['roads'].empty:
        # Verificar se a coluna 'highway' existe para usar classificação OSM
        if 'highway' in data['roads'].columns:
            # Classificar baseado no tipo de via do OSM
            # Definir mapeamento de tipos OSM para classes funcionais
            highway_mapping = {
                'motorway': 'Arterial', 
                'trunk': 'Arterial',
                'primary': 'Arterial',
                'secondary': 'Coletora',
                'tertiary': 'Coletora',
                'residential': 'Local',
                'service': 'Local',
                'unclassified': 'Local',
                'living_street': 'Local'
            }
            
            # Aplicar mapeamento, com valor padrão para tipos não listados
            data['roads']['road_class'] = data['roads']['highway'].map(highway_mapping).fillna('Local')
            
            # Log estatísticas
            road_class_counts = data['roads']['road_class'].value_counts()
            for cls, count in road_class_counts.items():
                logger.info(f"Estradas classe {cls}: {count} ({count/len(data['roads'])*100:.1f}%)")
        
        # Se temos betweenness, usar para definir nível de importância
        elif 'betweenness' in data['roads'].columns:
            betweenness_values = data['roads']['betweenness'].dropna()
            if not betweenness_values.empty:
                q75 = betweenness_values.quantile(0.75)
                q50 = betweenness_values.quantile(0.50)
                q25 = betweenness_values.quantile(0.25)
                
                # Definir classes baseadas em betweenness
                conditions = [
                    (data['roads']['betweenness'] >= q75),
                    (data['roads']['betweenness'] >= q50) & (data['roads']['betweenness'] < q75),
                    (data['roads']['betweenness'] >= q25) & (data['roads']['betweenness'] < q50),
                    (data['roads']['betweenness'] < q25)
                ]
                
                classes = ['Arterial', 'Coletora', 'Secundária', 'Local']
                data['roads']['road_class'] = np.select(conditions, classes, default='Local')
                
                # Log estatísticas
                road_class_counts = data['roads']['road_class'].value_counts()
                for cls, count in road_class_counts.items():
                    logger.info(f"Estradas classe {cls}: {count} ({count/len(data['roads'])*100:.1f}%)")
            else:
                # Fallback se não temos dados de betweenness
                data['roads']['road_class'] = 'Local'
        else:
            # Não temos informação para classificação, usar comprimento como proxy
            if 'length' in data['roads'].columns:
                length_values = data['roads']['length'].dropna()
                if not length_values.empty:
                    q75 = length_values.quantile(0.75)
                    q50 = length_values.quantile(0.50)
                    
                    # Vias mais longas tendem a ser mais importantes
                    conditions = [
                        (data['roads']['length'] >= q75),
                        (data['roads']['length'] >= q50) & (data['roads']['length'] < q75),
                        (data['roads']['length'] < q50)
                    ]
                    
                    classes = ['Arterial', 'Coletora', 'Local']
                    data['roads']['road_class'] = np.select(conditions, classes, default='Local')
                else:
                    data['roads']['road_class'] = 'Local'
            else:
                # Sem dados para classificação
                data['roads']['road_class'] = 'Local'
    
    # 5. Calcular métricas de qualidade da rede viária
    try:
        # Garantir que temos a chave 'intersections' no dicionário
        if 'intersections' not in data:
            data['intersections'] = gpd.GeoDataFrame(geometry=[], crs=data['roads'].crs if 'roads' in data else "EPSG:4326")
            
        quality_metrics = calculate_road_quality_metrics(data['roads'], data['intersections'])
        
        # Adicionar métricas ao GeoDataFrame de estradas como metadados
        if not data['roads'].empty:
            data['roads'].attrs['quality_metrics'] = quality_metrics
    except Exception as e:
        logger.error(f"Erro ao calcular métricas de qualidade: {str(e)}")
    
    # 6. Adicionar aos resultados
    enriched_data['roads'] = data['roads']
    enriched_data['intersections'] = data['intersections']
    
    return enriched_data

# Add this helper function before the visualization functions
def prepare_visualization(data, feature_key, required_column=None, title=None):
    """
    Helper function to prepare data for visualization.
    
    Args:
        data (dict): Dictionary containing data to visualize
        feature_key (str): Key for the feature type in the data dictionary
        required_column (str, optional): Column that must exist in the data
        title (str, optional): Title for the visualization
        
    Returns:
        tuple: (prepared_gdf, fig, ax) or (None, None, None) if preparation fails
    """
    try:
        # Check if data exists
        if feature_key not in data or data[feature_key].empty:
            logger.error(f"Dados de {feature_key} não encontrados para visualização")
            return None, None, None
        
        # Get the GeoDataFrame
        gdf = data[feature_key].copy()
        
        # Check for required column
        if required_column and required_column not in gdf.columns:
            logger.error(f"Coluna {required_column} não encontrada nos dados")
            return None, None, None
        
        # Convert to projected CRS if needed
        if gdf.crs and gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=3857)  # Web Mercator
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(15, 12), dpi=300)
        
        # Add basemap with proper zoom level
        try:
            # Calculate appropriate zoom level
            xmin, ymin, xmax, ymax = gdf.total_bounds
            
            # Calculate appropriate zoom level based on bounding box size
            bbox_width = xmax - xmin
            if bbox_width > 20000:  # Large area (> 20km)
                zoom = 10
            elif bbox_width > 5000:  # Medium area
                zoom = 12
            else:  # Small area
                zoom = 14
                
            ctx.add_basemap(
                ax, 
                crs=gdf.crs.to_string(),
                source=ctx.providers.CartoDB.Positron,
                zoom=zoom,  # Explicitly set zoom level to prevent warnings
                alpha=0.6
            )
            logger.info("Mapa base adicionado com sucesso")
        except Exception as e:
            logger.warning(f"Não foi possível adicionar o mapa base: {str(e)}")
        
        # Add title if provided
        if title:
            ax.set_title(title, fontsize=16)
        
        return gdf, fig, ax
    except Exception as e:
        logger.error(f"Erro ao preparar visualização: {str(e)}")
        return None, None, None

def create_road_map_with_basemap(data, output_path):
    """
    Criar um mapa das estradas com um mapa base.
    Versão aprimorada para gerar visualizações de alta qualidade.
    
    Args:
        data (dict): Dicionário contendo os dados de estradas enriquecidos
        output_path (str): Caminho para salvar a visualização
    """
    logger.info(f"Criando mapa básico de estradas: {output_path}")
    
    try:
        # Verificar se temos dados de estradas
        if 'roads' not in data or data['roads'].empty:
            logger.error("Dados de estradas não encontrados para visualização")
            return False
        
        # Obter GeoDataFrame
        roads_gdf = data['roads'].copy()
        
        # Reprojetar para visualização
        if roads_gdf.crs and roads_gdf.crs.is_geographic:
            roads_gdf = roads_gdf.to_crs(epsg=3857)  # Web Mercator para melhor visualização
        
        # Criar figura com dimensões razoáveis para evitar problemas de memória
        fig, ax = plt.subplots(figsize=(12, 10), dpi=200)
        
        # Configurar estilo do mapa
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
        
        # Adicionar basemap primeiro (ficará abaixo das estradas)
        try:
            ctx.add_basemap(
                ax,
                source=ctx.providers.CartoDB.Positron,
                alpha=0.7,
                zoom=13  # Especificar o zoom para evitar inferência problemática
            )
            logger.info("Mapa base adicionado com sucesso")
        except Exception as e1:
            logger.warning(f"Erro ao adicionar basemap Positron: {str(e1)}")
            try:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.OpenStreetMap.Mapnik,
                    alpha=0.7,
                    zoom=13
                )
                logger.info("Mapa base alternativo adicionado com sucesso")
            except Exception as e2:
                logger.warning(f"Erro ao adicionar basemap OpenStreetMap: {str(e2)}")
        
        # Desenhar estradas por classe funcional
        if 'road_class' in roads_gdf.columns:
            # Definir cores e linewidth por classe funcional (similar ao visualize_roads.py)
            class_styles = {
                'Arterial': {'color': '#e41a1c', 'linewidth': 2.5, 'alpha': 0.9, 'zorder': 4},
                'Coletora': {'color': '#377eb8', 'linewidth': 1.8, 'alpha': 0.8, 'zorder': 3},
                'Secundária': {'color': '#4daf4a', 'linewidth': 1.2, 'alpha': 0.8, 'zorder': 2},
                'Local': {'color': '#984ea3', 'linewidth': 0.8, 'alpha': 0.7, 'zorder': 1}
            }
            
            # Plotar na ordem inversa de importância (menos importantes primeiro)
            for road_class in ['Local', 'Secundária', 'Coletora', 'Arterial']:
                if road_class in roads_gdf['road_class'].values:
                    subset = roads_gdf[roads_gdf['road_class'] == road_class]
                    if not subset.empty:
                        style = class_styles.get(road_class, {'color': 'grey', 'linewidth': 1.0, 'alpha': 0.7})
                        subset.plot(
                            ax=ax,
                            color=style['color'],
                            linewidth=style['linewidth'],
                            alpha=style['alpha'],
                            zorder=style['zorder'],
                            label=f"Vias {road_class}s"
                        )
        else:
            # Se não temos classificação de estradas, usar importância ou sinuosidade
            if 'importancia' in roads_gdf.columns:
                # Definir cores para cada categoria de importância
                colors = {
                    'Alta': '#e41a1c',  # Vermelho
                    'Média-Alta': '#377eb8',  # Azul
                    'Média-Baixa': '#4daf4a',  # Verde
                    'Baixa': '#984ea3',  # Roxo
                    'Indefinida': '#ff7f00'  # Laranja
                }
                
                # Ordenar por importância (menos importantes primeiro)
                for importance in ['Baixa', 'Média-Baixa', 'Média-Alta', 'Alta']:
                    subset = roads_gdf[roads_gdf['importancia'] == importance]
                    if not subset.empty:
                        subset.plot(
                            ax=ax,
                            color=colors.get(importance, '#cccccc'),
                            linewidth=1.0 if importance in ['Baixa', 'Indefinida'] else 
                                     1.5 if importance == 'Média-Baixa' else
                                     2.0 if importance == 'Média-Alta' else 2.5,
                            alpha=0.9,
                            label=f"Importância: {importance}"
                        )
            elif 'sinuosity' in roads_gdf.columns:
                # Usar sinuosidade para visualização 
                bins = [1.0, 1.05, 1.2, 1.5, float('inf')]
                labels = ['Reta', 'Pouco Sinuosa', 'Moderadamente Sinuosa', 'Muito Sinuosa']
                
                roads_gdf['sinuosity_class'] = pd.cut(
                    roads_gdf['sinuosity'], 
                    bins=bins, 
                    labels=labels,
                    include_lowest=True
                )
                
                colors = {
                    'Reta': '#fef0d9',
                    'Pouco Sinuosa': '#fdcc8a',
                    'Moderadamente Sinuosa': '#fc8d59',
                    'Muito Sinuosa': '#d7301f'
                }
                
                for sin_class in labels:
                    subset = roads_gdf[roads_gdf['sinuosity_class'] == sin_class]
                    if not subset.empty:
                        subset.plot(
                            ax=ax,
                            color=colors.get(sin_class, '#cccccc'),
                            linewidth=1.5,
                            alpha=0.9,
                            label=sin_class
                        )
            else:
                # Plotar todas as estradas com um único estilo
                roads_gdf.plot(ax=ax, color='#377eb8', linewidth=1.0, alpha=0.9)
        
        # Adicionar interseções importantes
        if 'intersections' in data and not data['intersections'].empty:
            intersections_gdf = data['intersections'].copy()
            if intersections_gdf.crs != roads_gdf.crs:
                intersections_gdf = intersections_gdf.to_crs(roads_gdf.crs)
            
            if 'degree' in intersections_gdf.columns:
                important_intersections = intersections_gdf[intersections_gdf['degree'] > 3]
                if not important_intersections.empty:
                    important_intersections.plot(
                        ax=ax, 
                        color='red',
                        markersize=25,
                        alpha=0.7,
                        zorder=5,
                        label='Interseções Principais'
                    )
        
        # Adicionar título com estilo
        plt.title("Rede de Estradas", fontsize=18, fontweight='bold', pad=15)
        
        # Remover eixos para visual mais limpo
        ax.set_axis_off()
        
        # Adicionar legenda personalizada
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(
                handles, labels,
                loc='upper right', 
                fontsize=12, 
                frameon=True, 
                framealpha=0.9,
                title="Legenda"
            )
            legend.get_title().set_fontweight('bold')
        
        # Adicionar nota de copyright
        plt.annotate('(C) OpenStreetMap contributors (C) CARTO', 
                     xy=(0.01, 0.01), 
                     xycoords='figure fraction', 
                     fontsize=8, 
                     color='#555555',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))
        
        # Salvar o mapa com boa qualidade mas tamanho gerenciável
        plt.savefig(output_path, dpi=200, bbox_inches='tight', pad_inches=0.1, 
                    facecolor='#f2f2f2', edgecolor='none')
        plt.close(fig)
        
        # Verificar se o arquivo foi gerado
        if os.path.exists(output_path):
            file_size_kb = os.path.getsize(output_path) / 1024
            logger.info(f"Mapa salvo com sucesso: {output_path} ({file_size_kb:.1f} KB)")
            return True
        else:
            logger.error(f"Falha ao salvar o mapa: arquivo não encontrado após salvamento")
            return False
    
    except Exception as e:
        logger.error(f"Erro ao criar mapa de estradas: {str(e)}")
        logger.error(traceback.format_exc())
        if 'fig' in locals() and fig is not None:
            plt.close(fig)
        return False

def visualize_road_sinuosity(data, output_path):
    """
    Visualizar a sinuosidade das estradas.
    
    Args:
        data (dict): Dicionário contendo os dados de estradas enriquecidos
        output_path (str): Caminho para salvar a visualização
    """
    # Prepare data and visualization elements
    roads_gdf, fig, ax = prepare_visualization(
        data, 'roads', required_column='sinuosity', title='Sinuosidade das Estradas'
    )
    
    if roads_gdf is None:
        return False
    
    try:
        # Classificar estradas em categorias de sinuosidade para melhor visualização
        bins = [1.0, 1.05, 1.2, 1.5, float('inf')]
        labels = ['Reta (≤1.05)', 'Pouco Sinuosa (1.05-1.2)', 'Moderadamente Sinuosa (1.2-1.5)', 'Muito Sinuosa (>1.5)']
        
        # Definir cores para cada categoria
        colors = {
            'Reta (≤1.05)': '#fef0d9',
            'Pouco Sinuosa (1.05-1.2)': '#fdcc8a',
            'Moderadamente Sinuosa (1.2-1.5)': '#fc8d59',
            'Muito Sinuosa (>1.5)': '#d7301f'
        }
        
        # Classificar estradas
        roads_gdf['sinuosity_category'] = pd.cut(
            roads_gdf['sinuosity'], 
            bins=bins, 
            labels=labels,
            include_lowest=True
        )
        
        # Plotar por categoria para maior clareza
        for category, color in colors.items():
            subset = roads_gdf[roads_gdf['sinuosity_category'] == category]
            if not subset.empty:
                subset.plot(
                    ax=ax,
                    color=color,
                    linewidth=2.5,
                    label=category
                )
        
        # Legenda personalizada
        ax.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.9)
        
        # Remover eixos
        ax.set_axis_off()
        
        # Adicionar escala (método alternativo)
        # Calcular distância em metros para 1km
        xmin, ymin, xmax, ymax = roads_gdf.total_bounds
        width = xmax - xmin
        scale_width_m = 1000  # 1 km
        scale_width_percent = scale_width_m / width
        
        # Adicionar barra de escala manualmente
        ax.plot([xmin + width*0.1, xmin + width*0.1 + scale_width_m], 
                [ymin + width*0.05, ymin + width*0.05], 
                color='black', linewidth=2)
        ax.text(xmin + width*0.1, 
                ymin + width*0.05 - width*0.01, 
                '1 km', 
                fontsize=10, 
                color='black', 
                ha='left', va='top')
        
        # Adicionar nota de copyright
        plt.annotate('(C) OpenStreetMap contributors (C) CARTO', xy=(0.01, 0.01), 
                    xycoords='figure fraction', fontsize=10, color='#555555')
        
        # Ajustar layout e salvar
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Mapa de sinuosidade das estradas salvo em {output_path}")
        return True
    except Exception as e:
        logger.error(f"Erro ao criar visualização de sinuosidade: {str(e)}")
        if fig is not None:
            plt.close(fig)
        return False

def visualize_network_centrality(data, output_path):
    """
    Visualizar a centralidade da rede viária.
    
    Args:
        data (dict): Dicionário contendo os dados de estradas enriquecidos
        output_path (str): Caminho para salvar a visualização
    """
    # Prepare data and visualization elements
    roads_gdf, fig, ax = prepare_visualization(
        data, 'roads', required_column='betweenness', title='Centralidade da Rede Viária'
    )
    
    if roads_gdf is None:
        return False
    
    try:
        # Classificar em categorias para visualização mais clara
        # Criar quartis para betweenness
        if len(roads_gdf) > 0:
            q1 = roads_gdf['betweenness'].quantile(0.25)
            q2 = roads_gdf['betweenness'].quantile(0.5)
            q3 = roads_gdf['betweenness'].quantile(0.75)
            
            # Definir categorias e cores
            bins = [0, q1, q2, q3, float('inf')]
            labels = ['Baixa', 'Média-Baixa', 'Média-Alta', 'Alta']
            
            colors = {
                'Baixa': '#edf8fb',
                'Média-Baixa': '#b2e2e2',
                'Média-Alta': '#66c2a4',
                'Alta': '#238b45'
            }
            
            # Classificar estradas
            roads_gdf['centrality_category'] = pd.cut(
                roads_gdf['betweenness'],
                bins=bins,
                labels=labels,
                include_lowest=True
            )
            
            # Plotar por categoria
            for category, color in colors.items():
                subset = roads_gdf[roads_gdf['centrality_category'] == category]
                if not subset.empty:
                    # Variar a largura da linha com base na importância
                    linewidth = 1.5
                    if category == 'Média-Alta':
                        linewidth = 2.5
                    elif category == 'Alta':
                        linewidth = 3.5
                    
                    subset.plot(
                        ax=ax,
                        color=color,
                        linewidth=linewidth,
                        label=category
                    )
            
            # Legenda personalizada
            ax.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.9)
        else:
            # Fallback se não conseguimos categorizar
            roads_gdf.plot(
                ax=ax,
                column='betweenness',
                cmap='viridis',
                linewidth=2.5,
                legend=True
            )
        
        # Adicionar interseções importantes
        if 'intersections' in data and not data['intersections'].empty:
            intersections_gdf = data['intersections'].copy()
            if intersections_gdf.crs != roads_gdf.crs:
                intersections_gdf = intersections_gdf.to_crs(roads_gdf.crs)
            
            # Filtrar e classificar interseções por importância
            if 'degree' in intersections_gdf.columns:
                # Adicionar apenas as interseções mais significativas para não sobrecarregar o mapa
                top_intersections = intersections_gdf[intersections_gdf['degree'] >= 4]
                
                if not top_intersections.empty:
                    # Tamanho variável com base no grau
                    sizes = top_intersections['degree'] * 12
                    top_intersections.plot(
                        ax=ax,
                        color='red',
                        markersize=sizes,
                        alpha=0.8,
                        zorder=10
                    )
        
        # Remover eixos
        ax.set_axis_off()
        
        # Adicionar legenda explicativa sobre centralidade
        textstr = (
            'A centralidade indica a importância de\n'
            'cada segmento na conectividade da rede.\n'
            'Valores maiores (em amarelo/verde) indicam\n'
            'estradas mais críticas para o fluxo.'
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=12,
                verticalalignment='bottom', bbox=props)
        
        # Adicionar escala (método alternativo)
        # Calcular distância em metros para 1km
        xmin, ymin, xmax, ymax = roads_gdf.total_bounds
        width = xmax - xmin
        scale_width_m = 1000  # 1 km
        scale_width_percent = scale_width_m / width
        
        # Adicionar barra de escala manualmente
        ax.plot([xmin + width*0.1, xmin + width*0.1 + scale_width_m], 
                [ymin + width*0.05, ymin + width*0.05], 
                color='black', linewidth=2)
        ax.text(xmin + width*0.1, 
                ymin + width*0.05 - width*0.01, 
                '1 km', 
                fontsize=10, 
                color='black', 
                ha='left', va='top')
        
        # Adicionar nota de copyright
        plt.annotate('(C) OpenStreetMap contributors (C) CARTO', xy=(0.01, 0.01), 
                    xycoords='figure fraction', fontsize=10, color='#555555')
        
        # Ajustar layout e salvar
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Mapa de centralidade da rede viária salvo em {output_path}")
        return True
    except Exception as e:
        logger.error(f"Erro ao criar visualização de centralidade: {str(e)}")
        if fig is not None:
            plt.close(fig)
        return False

def visualize_road_slope(data, output_path):
    """
    Visualizar a declividade das estradas.
    
    Args:
        data (dict): Dicionário contendo os dados de estradas enriquecidos
        output_path (str): Caminho para salvar a visualização
    """
    # Prepare data and visualization elements
    roads_gdf, fig, ax = prepare_visualization(
        data, 'roads', required_column='slope_pct', title='Declividade das Estradas'
    )
    
    if roads_gdf is None:
        # Informar que não é um erro crítico se os dados de elevação não estão disponíveis
        if 'roads' in data and not data['roads'].empty:
            if 'slope_pct' not in data['roads'].columns:
                logger.warning("Dados de declividade não disponíveis. O DEM provavelmente não foi encontrado.")
                # Criar um arquivo vazio para indicar que a tentativa foi feita
                with open(output_path, 'w') as f:
                    f.write("Dados de declividade não disponíveis")
        return False
    
    try:
        # Classificar em categorias para melhor visualização
        bins = [0, 3, 8, 15, float('inf')]
        labels = ['Plana (0-3%)', 'Suave (3-8%)', 'Moderada (8-15%)', 'Íngreme (>15%)']
        
        colors = {
            'Plana (0-3%)': '#ffffcc',
            'Suave (3-8%)': '#a1dab4',
            'Moderada (8-15%)': '#41b6c4',
            'Íngreme (>15%)': '#225ea8'
        }
        
        # Classificar estradas
        roads_gdf['slope_category'] = pd.cut(
            roads_gdf['slope_pct'],
            bins=bins,
            labels=labels,
            include_lowest=True
        )
        
        # Plotar por categoria
        for category, color in colors.items():
            subset = roads_gdf[roads_gdf['slope_category'] == category]
            if not subset.empty:
                subset.plot(
                    ax=ax,
                    color=color,
                    linewidth=2.5,
                    label=category
                )
        
        # Legenda personalizada
        ax.legend(loc='upper right', fontsize=12, frameon=True, framealpha=0.9)
        
        # Remover eixos
        ax.set_axis_off()
        
        # Adicionar escala (método alternativo)
        # Calcular distância em metros para 1km
        xmin, ymin, xmax, ymax = roads_gdf.total_bounds
        width = xmax - xmin
        scale_width_m = 1000  # 1 km
        scale_width_percent = scale_width_m / width
        
        # Adicionar barra de escala manualmente
        ax.plot([xmin + width*0.1, xmin + width*0.1 + scale_width_m], 
                [ymin + width*0.05, ymin + width*0.05], 
                color='black', linewidth=2)
        ax.text(xmin + width*0.1, 
                ymin + width*0.05 - width*0.01, 
                '1 km', 
                fontsize=10, 
                color='black', 
                ha='left', va='top')
        
        # Adicionar nota de copyright
        plt.annotate('(C) OpenStreetMap contributors (C) CARTO', xy=(0.01, 0.01), 
                    xycoords='figure fraction', fontsize=10, color='#555555')
        
        # Ajustar layout e salvar
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Mapa de declividade das estradas salvo em {output_path}")
        return True
    except Exception as e:
        logger.error(f"Erro ao criar visualização de declividade: {str(e)}")
        if fig is not None:
            plt.close(fig)
        return False

def create_interactive_map(data, output_path):
    """
    Cria um mapa interativo HTML das estradas usando Folium
    
    Args:
        data (dict): Dicionário contendo os dados enriquecidos
        output_path (str): Caminho para salvar o mapa interativo
    """
    logger.info(f"Criando mapa interativo HTML: {output_path}")
    
    try:
        gdf = data['roads'].copy()
        
        # Verificar CRS e converter para WGS84 se necessário
        if gdf.crs and gdf.crs != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
            
        # Calcular o centro do mapa
        center = [gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()]
        
        # Criar mapa base
        m = folium.Map(location=center, zoom_start=12, control_scale=True)
        
        # Adicionar controle de camadas
        folium.LayerControl().add_to(m)
        
        # Definir campos para exibir no tooltip
        tooltip_fields = ['name', 'highway', 'importance']
        
        # Adicionar apenas os campos que existem no GeoDataFrame
        available_fields = [field for field in tooltip_fields if field in gdf.columns]
        
        # Se nenhum campo estiver disponível, usar um conjunto padrão
        if not available_fields:
            available_fields = ['id']
        
        # Adicionar camada de estradas
        style_function = lambda x: {
            'color': '#3388ff',
            'weight': 2,
            'opacity': 0.8
        }
        
        # Verificar se existe a coluna sinuosity e usá-la para estilizar as estradas
        if 'sinuosity' in gdf.columns:
            style_function = lambda x: {
                'color': get_sinuosity_color(x['properties']['sinuosity'] if 'sinuosity' in x['properties'] else 1.0),
                'weight': 2,
                'opacity': 0.8
            }
            
            if 'sinuosity' not in available_fields:
                available_fields.append('sinuosity')
        
        folium.GeoJson(
            gdf,
            name='Estradas',
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=available_fields,
                aliases=[field.capitalize() for field in available_fields],
                localize=True,
                sticky=False,
            )
        ).add_to(m)
        
        # Adicionar interseções se disponíveis
        if 'intersections' in data and not data['intersections'].empty:
            intersection_gdf = data['intersections'].copy()
            if intersection_gdf.crs and intersection_gdf.crs != "EPSG:4326":
                intersection_gdf = intersection_gdf.to_crs("EPSG:4326")
                
            for _, row in intersection_gdf.iterrows():
                folium.CircleMarker(
                    location=[row.geometry.y, row.geometry.x],
                    radius=5,
                    color='red',
                    fill=True,
                    fill_color='red',
                    fill_opacity=0.7,
                    tooltip=f"Interseção: {row['intersection_id']}"
                ).add_to(m)
        
        # Salvar o mapa
        m.save(output_path)
        logger.info(f"Mapa interativo salvo com sucesso: {output_path}")
    except Exception as e:
        logger.error(f"Erro ao criar mapa interativo: {str(e)}")
        raise

def generate_report(data, output_path):
    """
    Gera um relatório de qualidade para os dados de estradas processados.
    
    Args:
        data (Dict[str, gpd.GeoDataFrame]): Dicionário com GeoDataFrames contendo dados de estradas
        output_path (str): Caminho para salvar o relatório em formato JSON
        
    Returns:
        Dict: Relatório de qualidade
    """
    logger.info("Gerando relatório de qualidade da rede viária...")
    
    # Verificar se temos dados
    if data is None or 'roads' not in data or data['roads'].empty:
        logger.error("Dados de estradas não disponíveis para geração de relatório")
        return None
    
    # Inicializar relatório
    report = {
        "report_type": "road_network_quality",
        "timestamp": datetime.datetime.now().isoformat(),
        "data_source": ROADS_FILE,
        "analysis_area": "N/A",  # Atualizar se tivermos informação sobre a área
        "road_network": {},
        "intersections": {},
        "topography": {},
        "quality_metrics": {}
    }
    
    # Incluir informações gerais da rede viária
    roads_gdf = data['roads']
    report["road_network"]["feature_count"] = len(roads_gdf)
    
    # Incluir extensão total em quilômetros
    if not roads_gdf.empty and roads_gdf.crs and not roads_gdf.crs.is_geographic:
        total_length_km = roads_gdf.length.sum() / 1000  # Converter para quilômetros
        report["road_network"]["total_length_km"] = float(total_length_km)
    
    # Incluir informações sobre sinuosidade
    if 'sinuosity' in roads_gdf.columns:
        sinuosity = roads_gdf['sinuosity'].dropna()
        if not sinuosity.empty:
            report["road_network"]["sinuosity"] = {
                "mean": float(sinuosity.mean()),
                "median": float(sinuosity.median()),
                "max": float(sinuosity.max()),
                "min": float(sinuosity.min()),
                "std": float(sinuosity.std())
            }
            
            # Distribuição por classes de sinuosidade
            if 'sinuosity_class' in roads_gdf.columns:
                class_counts = roads_gdf['sinuosity_class'].value_counts().to_dict()
                # Converter para formato serializável
                class_distribution = {}
                for cls, count in class_counts.items():
                    if not pd.isna(cls):
                        class_distribution[str(cls)] = int(count)
                report["road_network"]["sinuosity_class_distribution"] = class_distribution
    
    # Incluir informações sobre a rede
    if hasattr(roads_gdf, 'attrs') and 'network_info' in roads_gdf.attrs:
        network_info = roads_gdf.attrs['network_info']
        # Filtrar apenas valores serializáveis
        filtered_info = {}
        for key, value in network_info.items():
            if isinstance(value, (int, float, str, bool, list, dict)) and not isinstance(value, (np.integer, np.floating)):
                filtered_info[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                filtered_info[key] = float(value)
        report["road_network"]["network_analysis"] = filtered_info
    
    # Incluir informações sobre interseções
    if 'intersections' in data and not data['intersections'].empty:
        intersections_gdf = data['intersections']
        report["intersections"]["count"] = len(intersections_gdf)
        
        # Distribuição por tipo de interseção
        if 'type' in intersections_gdf.columns:
            type_counts = intersections_gdf['type'].value_counts().to_dict()
            report["intersections"]["type_distribution"] = {str(k): int(v) for k, v in type_counts.items()}
        
        # Adicionar estatísticas de interseções dos metadados
        if hasattr(roads_gdf, 'attrs') and 'intersection_stats' in roads_gdf.attrs:
            intersection_stats = roads_gdf.attrs['intersection_stats']
            # Filtrar valores serializáveis
            serializable_stats = {}
            for key, value in intersection_stats.items():
                if isinstance(value, (int, float, str, bool)) and not isinstance(value, (np.integer, np.floating)):
                    serializable_stats[key] = value
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_stats[key] = float(value)
                elif isinstance(value, dict):
                    serializable_stats[key] = {str(k): int(v) if isinstance(v, (int, np.integer)) else float(v) 
                                              for k, v in value.items() if not isinstance(k, (np.integer, np.floating))}
            report["intersections"]["stats"] = serializable_stats
    
    # Incluir informações topográficas
    if 'elevation_mean' in roads_gdf.columns:
        elevation = roads_gdf['elevation_mean'].dropna()
        if not elevation.empty:
            report["topography"]["elevation"] = {
                "mean_m": float(elevation.mean()),
                "max_m": float(roads_gdf['elevation_max'].max()),
                "min_m": float(roads_gdf['elevation_min'].min()),
                "range_m": float(roads_gdf['elevation_max'].max() - roads_gdf['elevation_min'].min())
            }
    
    if 'slope_pct' in roads_gdf.columns:
        slope = roads_gdf['slope_pct'].dropna()
        if not slope.empty:
            report["topography"]["slope"] = {
                "mean_pct": float(slope.mean()),
                "max_pct": float(slope.max()),
                "min_pct": float(slope.min()),
                "std_pct": float(slope.std())
            }
            
            # Distribuição por classes de declividade
            if 'slope_class' in roads_gdf.columns:
                slope_counts = roads_gdf['slope_class'].value_counts().to_dict()
                # Converter para formato serializável
                slope_distribution = {}
                for cls, count in slope_counts.items():
                    if not pd.isna(cls):
                        slope_distribution[str(cls)] = int(count)
                report["topography"]["slope_class_distribution"] = slope_distribution
    
    # Incluir métricas de qualidade se disponíveis
    if hasattr(roads_gdf, 'attrs') and 'quality_metrics' in roads_gdf.attrs:
        quality_metrics = roads_gdf.attrs['quality_metrics']
        # Filtrar valores serializáveis
        serializable_metrics = {}
        for key, value in quality_metrics.items():
            if isinstance(value, (int, float, str, bool)) and not isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = value
            elif isinstance(value, (np.integer, np.floating)):
                serializable_metrics[key] = float(value)
        report["quality_metrics"] = serializable_metrics
    
    # Salvar relatório
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
        logger.info(f"Relatório de qualidade salvo em {output_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar relatório: {str(e)}")
    
    return report

def save_enriched_data(data, output_file=None):
    """
    Salva os dados enriquecidos em um único arquivo GeoPackage.
    
    Args:
        data (Dict[str, gpd.GeoDataFrame]): Dicionário com GeoDataFrames contendo dados enriquecidos
        output_file (str, opcional): Caminho para o arquivo de saída. Se None, usa o padrão.
        
    Returns:
        str: Caminho para o arquivo salvo
    """
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"roads_enriched_{timestamp}.gpkg")
    
    logger.info(f"Salvando dados enriquecidos em: {output_file}")
    
    # Garantir que o diretório de saída existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Salvar cada layer no arquivo GPKG
    for key, gdf in data.items():
        if key in ['roads', 'intersections'] and not gdf.empty:
            try:
                # Garantir que todas as colunas sejam serializáveis
                save_gdf = gdf.copy()
                
                # Manter apenas atributos serializáveis
                for col in save_gdf.columns:
                    if col != 'geometry':
                        # Converter numpy types para Python types
                        if save_gdf[col].dtype.name.startswith(('float', 'int')):
                            save_gdf[col] = save_gdf[col].astype(float)
                
                # Salvar em GeoPackage
                save_gdf.to_file(output_file, layer=key, driver="GPKG")
                logger.info(f"Layer '{key}' salvo com {len(save_gdf)} feições")
                
                # Mostrar tamanho do arquivo
                if os.path.exists(output_file):
                    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                    logger.info(f"Tamanho do arquivo GPKG: {file_size_mb:.2f} MB")
                
            except Exception as e:
                logger.error(f"Erro ao salvar layer '{key}': {str(e)}")
                logger.error(traceback.format_exc())
    
    return output_file

def main(skip_graph_analysis=None):
    """
    Função principal que executa o fluxo de trabalho completo.
    
    Args:
        skip_graph_analysis (bool, optional): Se definido, sobrescreve a configuração global SKIP_GRAPH_ANALYSIS
    """
    logger.info("=== Iniciando processamento de enriquecimento de dados de estradas ===")
    start_time = time.time()
    
    # Configurar o parâmetro de análise de grafo
    if skip_graph_analysis is None:
        skip_graph_analysis = SKIP_GRAPH_ANALYSIS
    
    if skip_graph_analysis:
        logger.info("Análise de grafo desativada por configuração")
    
    # Dicionário para armazenar caminhos de visualizações
    viz_paths = {}
    
    try:
        # 1. Carregar dados processados
        roads_gdf = load_data()
        
        # 2. Carregar DEM (dados de elevação)
        logger.info("Carregando Modelo Digital de Elevação (DEM)...")
        dem = None
        try:
            if os.path.exists(DEM_FILE):
                dem = rasterio.open(DEM_FILE)
                logger.info(f"DEM carregado com sucesso. Dimensões: {dem.width}x{dem.height}, Resolução: {dem.res}")
            else:
                logger.warning(f"Arquivo DEM não encontrado: {DEM_FILE}")
        except Exception as e:
            logger.error(f"Erro ao carregar DEM: {str(e)}")
        
        # 3. Enriquecer dados
        logger.info("Iniciando processo de enriquecimento de dados...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            enriched_data = enrich_road_data({'roads': roads_gdf}, dem, skip_graph_analysis)
            logger.info("Enriquecimento de dados concluído com sucesso")
        except Exception as e:
            logger.error(f"Erro no processo de enriquecimento: {str(e)}")
            logger.error(traceback.format_exc())
            return
        
        # 4. Salvar dados enriquecidos
        logger.info("Salvando dados enriquecidos...")
        enriched_data_path = os.path.join(OUTPUT_DIR, f"roads_enriched_{timestamp}.gpkg")
        
        try:
            save_enriched_data(enriched_data, enriched_data_path)
            logger.info(f"Dados enriquecidos salvos em: {enriched_data_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar dados enriquecidos: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 5. Gerar visualizações de forma simplificada
        logger.info("Gerando visualizações...")
        
        try:
            # Mapa básico da rede viária
            map_path = os.path.join(VISUALIZATION_DIR, f"road_network_map_{timestamp}.png")
            create_road_map_with_basemap(enriched_data, map_path)
            viz_paths['road_network_map'] = map_path
            
            # Mapa de sinuosidade
            sinuosity_path = os.path.join(VISUALIZATION_DIR, f"road_sinuosity_{timestamp}.png")
            visualize_road_sinuosity(enriched_data, sinuosity_path)
            viz_paths['road_sinuosity'] = sinuosity_path
            
            # Mapa de centralidade
            centrality_path = os.path.join(VISUALIZATION_DIR, f"road_centrality_{timestamp}.png")
            visualize_network_centrality(enriched_data, centrality_path)
            viz_paths['road_centrality'] = centrality_path
            
            # Mapa de declividade
            slope_path = os.path.join(VISUALIZATION_DIR, f"road_slope_{timestamp}.png")
            visualize_road_slope(enriched_data, slope_path)
            viz_paths['road_slope'] = slope_path
            
            # Mapa interativo HTML (prioridade)
            interactive_map_path = os.path.join(VISUALIZATION_DIR, f"interactive_roads_map_{timestamp}.html")
            create_interactive_map(enriched_data, interactive_map_path)
            viz_paths['interactive_map'] = interactive_map_path
            
            # Mostrar estatísticas sobre visualizações
            successful_viz = sum(1 for path in viz_paths.values() if os.path.exists(path))
            logger.info(f"Visualizações geradas: {successful_viz}/{len(viz_paths)} concluídas com sucesso")
            logger.info(f"Diretório de visualizações: {VISUALIZATION_DIR}")
        except Exception as e:
            logger.error(f"Erro ao gerar visualizações: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 6. Gerar relatório de qualidade
        logger.info("Gerando relatório de qualidade...")
        report_path = os.path.join(REPORT_DIR, f"road_enrichment_report_{timestamp}.json")
        
        try:
            generate_report(enriched_data, report_path)
            logger.info(f"Relatório de qualidade gerado em: {report_path}")
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 7. Calcular tempo de execução
        elapsed_time = time.time() - start_time
        logger.info(f"=== Processamento concluído em {elapsed_time:.2f} segundos ===")
        logger.info(f"Dados enriquecidos: {enriched_data_path}")
        logger.info(f"Relatório: {report_path}")
        logger.info(f"Visualizações: {', '.join(viz_paths.values())}")
        
        return enriched_data
    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}")
        logger.error(traceback.format_exc())
        elapsed_time = time.time() - start_time
        logger.info(f"=== Processamento interrompido após {elapsed_time:.2f} segundos ===")
        return None

# Adicionar args ao script para controlar o comportamento da linha de comando
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Processar e enriquecer dados de estradas')
    parser.add_argument('--skip-graph', action='store_true', help='Pular análise de grafo (será executada na nuvem posteriormente)')
    
    args = parser.parse_args()
    
    # Executar main com a opção apropriada
    main(skip_graph_analysis=args.skip_graph) 

def create_sinuosity_map(data, output_path):
    """
    Cria um mapa das estradas colorido por sinuosidade
    
    Args:
        data (dict): Dicionário contendo os dados enriquecidos
        output_path (str): Caminho para salvar o mapa
    """
    logger.info(f"Criando mapa de sinuosidade: {output_path}")
    
    try:
        if 'roads' not in data or data['roads'].empty:
            logger.error("Dados de estradas não disponíveis para mapa de sinuosidade")
            return
            
        gdf = data['roads'].copy()
        
        # Verificar se a coluna sinuosity existe
        if 'sinuosity' not in gdf.columns:
            logger.warning("Coluna 'sinuosity' não encontrada nos dados. Calculando sinuosidade...")
            gdf = calculate_sinuosity(gdf)
        
        # Converter para projeção adequada para visualização
        if gdf.crs and gdf.crs != "EPSG:3763":
            gdf = gdf.to_crs("EPSG:3763")
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        
        # Definir uma normalização para a coloração baseada na sinuosidade
        vmin = gdf['sinuosity'].min()
        vmax = min(gdf['sinuosity'].max(), 3.0)  # Limitar para evitar que outliers distorçam a escala
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        
        # Adicionar título
        plt.title('Sinuosidade das Estradas', fontsize=14)
        
        # Plota cada segmento colorido pela sinuosidade
        for idx, row in gdf.iterrows():
            sin_value = row['sinuosity']
            color = plt.cm.viridis(norm(sin_value))
            ax.plot(*row.geometry.xy, color=color, linewidth=1.5, alpha=0.7)
        
        # Adicionar colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Índice de Sinuosidade')
        
        # Remover eixos
        ax.set_axis_off()
        
        # Ajustar o layout
        plt.tight_layout()
        
        # Salvar a figura
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Mapa de sinuosidade salvo com sucesso: {output_path}")
    except Exception as e:
        logger.error(f"Erro ao criar mapa de sinuosidade: {str(e)}")
        raise

def create_centrality_map(data, output_path):
    """
    Cria um mapa das estradas colorido por centralidade (betweenness)
    
    Args:
        data (dict): Dicionário contendo os dados enriquecidos
        output_path (str): Caminho para salvar o mapa
    """
    logger.info(f"Criando mapa de centralidade: {output_path}")
    
    try:
        if 'roads' not in data or data['roads'].empty:
            logger.error("Dados de estradas não disponíveis para mapa de centralidade")
            return
            
        gdf = data['roads'].copy()
        
        # Verificar se a coluna betweenness existe
        if 'betweenness' not in gdf.columns:
            logger.warning("Coluna 'betweenness' não encontrada nos dados. Calculando centralidade...")
            # Verificar se o G existe no dicionário, se não, construir o grafo
            if 'G' not in data or data['G'] is None:
                logger.info("Construindo grafo de estradas para análise de centralidade")
                data['G'] = build_road_network(gdf)
            
            if data['G'] is not None and len(data['G'].nodes) > 0:
                # Calcular betweenness centrality
                centrality = nx.betweenness_centrality(data['G'], weight='length')
                
                # Mapear valores de centralidade para as estradas
                edge_centrality = {}
                for u, v, data in data['G'].edges(data=True):
                    # Usar a média da centralidade dos nós de cada borda
                    edge_centrality[(u, v)] = (centrality.get(u, 0) + centrality.get(v, 0)) / 2
                
                # Adicionar valores de centralidade ao GeoDataFrame
                gdf['betweenness'] = 0.0
                
                # Associar o edge_centrality com as estradas usando o campo 'edge_id' (se existir)
                if 'edge_id' in gdf.columns:
                    for idx, row in gdf.iterrows():
                        edge_id = row['edge_id']
                        if edge_id in edge_centrality:
                            gdf.at[idx, 'betweenness'] = edge_centrality[edge_id]
                else:
                    logger.warning("Coluna 'edge_id' não encontrada, não foi possível mapear centralidade")
                    return
            else:
                logger.warning("Grafo vazio ou inexistente, pulando mapa de centralidade")
                return
        
        # Converter para projeção adequada para visualização
        if gdf.crs and gdf.crs != "EPSG:3763":
            gdf = gdf.to_crs("EPSG:3763")
        
        # Criar figura
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        
        # Normalizar a centralidade para coloração
        max_centrality = gdf['betweenness'].max()
        if max_centrality > 0:
            gdf['norm_centrality'] = gdf['betweenness'] / max_centrality
        else:
            gdf['norm_centrality'] = 0
        
        # Adicionar título
        plt.title('Centralidade das Estradas (Betweenness)', fontsize=14)
        
        # Plota todas as estradas em cinza claro primeiro (fundo)
        gdf.plot(ax=ax, color='lightgray', linewidth=0.5, alpha=0.3)
        
        # Plota cada segmento colorido pela centralidade usando uma escala de cores
        cmap = plt.cm.hot_r
        norm = plt.Normalize(vmin=0, vmax=1)
        
        # Plotar estradas com maior centralidade por cima
        for idx, row in gdf.sort_values('norm_centrality').iterrows():
            color = cmap(row['norm_centrality'])
            # Ajustar largura com base na centralidade (normalizada)
            linewidth = 0.5 + (row['norm_centrality'] * 2.5)
            ax.plot(*row.geometry.xy, color=color, linewidth=linewidth, alpha=0.7)
        
        # Adicionar colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.02)
        cbar.set_label('Centralidade Normalizada')
        
        # Remover eixos
        ax.set_axis_off()
        
        # Ajustar o layout
        plt.tight_layout()
        
        # Salvar a figura
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Mapa de centralidade salvo com sucesso: {output_path}")
    except Exception as e:
        logger.error(f"Erro ao criar mapa de centralidade: {str(e)}")
        logger.exception(e)  # Log detalhado da exceção

def generate_quality_report(data, enriched_data_path, visualization_paths=None):
    """
    Gera um relatório de qualidade para os dados de estradas.
    
    Args:
        data (dict): Dicionário com os dados enriquecidos
        enriched_data_path (str): Caminho para o arquivo de dados enriquecidos
        visualization_paths (dict): Dicionário com caminhos para visualizações
        
    Returns:
        str: Caminho para o arquivo de relatório gerado
    """
    logger.info("Gerando relatório de qualidade...")
    
    # Inicializar relatório
    report = {
        "report_type": "roads_enrichment",
        "timestamp": datetime.datetime.now().isoformat(),
        "enriched_data_path": enriched_data_path,
        "visualization_paths": visualization_paths or {},
        "summary": {},
        "metrics": {}
    }