#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Funções para enriquecimento de dados hidrográficos com métricas de rede e análises morfométricas.
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
from datetime import datetime
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
import xarray as xr
from rasterstats import zonal_stats, point_query
import matplotlib.cm as cm
from scipy.ndimage import gaussian_filter
import math
from shapely.geometry import mapping
import traceback
from numba import jit
import matplotlib as mpl

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
    
    # Paleta para hidrografia (azuis)
    blues = ['#08306b', '#0868ac', '#2b8cbe', '#4eb3d3', '#7bccc4', '#a8ddb5', '#ccebc5', '#e0f3db', '#f7fcf0']
    hidrografia_cmap = LinearSegmentedColormap.from_list('hidrografia_blues', blues)
    
    # Paleta para sinuosidade (vermelho-azul)
    sinuosity_colors = ['#d73027', '#f46d43', '#fdae61', '#fee090', '#e0f3f8', '#abd9e9', '#74add1', '#4575b4', '#313695']
    sinuosity_cmap = LinearSegmentedColormap.from_list('sinuosity', sinuosity_colors)
    
    # Paleta para ordem de Strahler (tons de laranja)
    strahler_colors = ['#fef0d9', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#990000']
    strahler_cmap = LinearSegmentedColormap.from_list('strahler', strahler_colors)
    
    # Paleta para betweenness (centrality)
    betweenness_colors = ['#edf8fb', '#b3cde3', '#8c96c6', '#8856a7', '#810f7c']
    betweenness_cmap = LinearSegmentedColormap.from_list('betweenness', betweenness_colors)
    
    # Registrar as paletas para uso
    try:
        plt.colormaps.register(cmap=hidrografia_cmap)
        plt.colormaps.register(cmap=sinuosity_cmap)
        plt.colormaps.register(cmap=strahler_cmap)
        plt.colormaps.register(cmap=betweenness_cmap)
    except:
        # Fallback para versões mais antigas do matplotlib
        try:
            plt.cm.register_cmap(name='hidrografia_blues', cmap=hidrografia_cmap)
            plt.cm.register_cmap(name='sinuosity', cmap=sinuosity_cmap)
            plt.cm.register_cmap(name='strahler', cmap=strahler_cmap)
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
INPUT_DIR = os.path.join(workspace_dir, 'data', 'processed')
OUTPUT_DIR = os.path.join(workspace_dir, 'data', 'enriched_data')
REPORT_DIR = os.path.join(workspace_dir, 'src', 'enriched_data', 'quality_reports', 'hidrografia')
VISUALIZATION_DIR = os.path.join(workspace_dir, 'outputs', 'visualize_enriched_data', 'hidrografia')

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Definir arquivos de entrada e saída
HIDROGRAFIA_FILES = {
    'trecho_drenagem': os.path.join(INPUT_DIR, 'hidrografia_trecho_drenagem_processed.gpkg'),
    'curso_dagua': os.path.join(INPUT_DIR, 'hidrografia_curso_dagua_processed.gpkg'),
    'area_drenagem': os.path.join(INPUT_DIR, 'hidrografia_area_drenagem_processed.gpkg'),
    'ponto_drenagem': os.path.join(INPUT_DIR, 'hidrografia_ponto_drenagem_processed.gpkg')
}

# Arquivos de entrada
RIVERS_FILE = os.path.join(INPUT_DIR, 'hidrografia', 'rivers_processed.gpkg')
SPRINGS_FILE = os.path.join(INPUT_DIR, 'hidrografia', 'springs_processed.gpkg')
WATER_BODIES_FILE = os.path.join(INPUT_DIR, 'hidrografia', 'water_bodies_processed.gpkg')
DEM_FILE = r"F:\TESE_MESTRADO\geoprocessing\data\raw\dem.tif"

# Arquivo de saída
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'hidrografia_enriched.gpkg')

def load_data():
    """
    Carrega os dados processados de hidrografia a partir do diretório de dados processados.
    
    Returns:
        Dict[str, gpd.GeoDataFrame]: Dicionário com os datasets hidrográficos carregados.
    """
    logger.info("Carregando dados processados de hidrografia...")
    
    data = {}
    
    for key, file_path in HIDROGRAFIA_FILES.items():
        try:
            if os.path.exists(file_path):
                gdf = gpd.read_file(file_path)
                data[key] = gdf
                logger.info(f"Carregado {key}: {len(gdf)} feições de {file_path}")
            else:
                logger.warning(f"Arquivo não encontrado: {file_path}")
    except Exception as e:
            logger.error(f"Erro ao carregar {key}: {str(e)}")
    
    if not data:
        logger.error("Nenhum dado hidrográfico pôde ser carregado. Verifique os arquivos de entrada.")
        sys.exit(1)
    
    return data

def load_dem():
    """
    Carrega o Modelo Digital de Elevação (DEM) para análise hidrográfica.
    
    Returns:
        rasterio.DatasetReader: Objeto do rasterio com o DEM carregado, ou None se não for possível carregar.
    """
    try:
        # Caminho específico para o arquivo DEM
        dem_file = "F:/TESE_MESTRADO/geoprocessing/data/raw/dem.tif"
        
        # Verificar se o arquivo existe
        if not os.path.exists(dem_file):
            logger.warning(f"Arquivo DEM não encontrado em: {dem_file}")
            return None
        
        # Carregar o DEM usando rasterio
        import rasterio
        dem = rasterio.open(dem_file)
        logger.info(f"DEM carregado com sucesso. Dimensões: {dem.width}x{dem.height}, CRS: {dem.crs}")
        
        return dem
    except Exception as e:
        logger.error(f"Erro ao carregar o DEM: {str(e)}")
        return None

def calculate_sinuosity(gdf):
    """
    Calcula o índice de sinuosidade para cada segmento de rio.
    Sinuosidade = comprimento real / distância em linha reta
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados hidrográficos (trechos de drenagem)
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com coluna de sinuosidade
    """
    logger.info("Calculando índices de sinuosidade...")
    
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    # Calcular sinuosidade para cada LineString
    sinuosities = []
    for geom in result.geometry:
        if isinstance(geom, LineString):
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
        return '#d73027'  # Vermelho para rios muito retificados
    elif sinuosity < 1.2:
        return '#fc8d59'  # Laranja para pouca sinuosidade
    elif sinuosity < 1.5:
        return '#4393c3'  # Azul médio
    elif sinuosity < 1.8:
        return '#2166ac'  # Azul mais escuro
    else:
        return '#053061'  # Azul muito escuro para alta sinuosidade

def build_stream_network(gdf, calculate_metrics=True):
    """
    Constrói uma rede hidrográfica (grafo) a partir dos trechos de drenagem.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados hidrográficos
        calculate_metrics (bool): Se True, calcula métricas de rede
        
    Returns:
        tuple: (Grafo NetworkX, GeoDataFrame atualizado com métricas de rede)
    """
    logger.info("Construindo rede hidrográfica...")
    
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
            G.add_edge(start_point, end_point, 
                       length=row.geometry.length,
                       feature_id=idx)
            edge_count += 1
    
    logger.info(f"Rede criada com {len(G.nodes)} nós e {edge_count} arestas")
    
    # Calcular medidas de centralidade da rede
    logger.info("Calculando métricas de centralidade da rede...")
    
    # Betweenness centrality (identifica segmentos importantes de conexão)
    try:
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='length')
    except Exception as e:
        logger.warning(f"Erro ao calcular betweenness centrality: {str(e)}")
        edge_betweenness = {}
    
    # Mapear edge betweenness de volta para o GeoDataFrame
    betweenness_values = []
    for idx, row in result.iterrows():
        if isinstance(row.geometry, LineString):
            start_point = row.geometry.coords[0]
            end_point = row.geometry.coords[-1]
            try:
                betweenness = edge_betweenness.get((start_point, end_point), 
                                                  edge_betweenness.get((end_point, start_point), 0))
            except:
                betweenness = 0
            betweenness_values.append(betweenness)
        else:
            betweenness_values.append(0)
    
    result['betweenness'] = betweenness_values
    
    # Adicionar outras métricas interessantes
    if len(G.nodes) > 0:
        try:
            # Calcular componentes conectados
            components = list(nx.connected_components(G))
            n_components = len(components)
            logger.info(f"Rede possui {n_components} componentes conectados")
            
            # Identificar os 3 maiores componentes
            largest_components = sorted(components, key=len, reverse=True)[:3]
            largest_sizes = [len(comp) for comp in largest_components]
            logger.info(f"Tamanho dos 3 maiores componentes: {largest_sizes}")
            
            # Calcular densidade da rede
            network_density = nx.density(G)
            logger.info(f"Densidade da rede: {network_density:.6f}")
            
            # Adicionar ao metadados da rede que serão salvos no relatório
            result.attrs['network_info'] = {
                'n_nodes': len(G.nodes),
                'n_edges': edge_count,
                'n_components': n_components,
                'largest_component_size': largest_sizes[0] if largest_sizes else 0,
                'network_density': network_density
            }
        except Exception as e:
            logger.warning(f"Erro ao calcular métricas adicionais da rede: {str(e)}")
    
    return G, result

def calculate_drainage_density(data, area_sqkm=None):
    """
    Calcula a densidade de drenagem (comprimento total dos rios / área).
    
    Args:
        data (Dict[str, gpd.GeoDataFrame]): Dados hidrográficos
        area_sqkm (float, opcional): Área em quilômetros quadrados. Se None, usa a área do bounding box.
        
    Returns:
        Dict: Densidades de drenagem por tipo
    """
    logger.info("Calculando densidade de drenagem...")
    
    result = {}
    
    # Primeiro, determinar a área de estudo
    if area_sqkm is None:
        # Se tivermos os limites da área de drenagem, usamos sua área
        if 'area_drenagem' in data and not data['area_drenagem'].empty:
            # Verificar se área está em projeção métrica
            area_gdf = data['area_drenagem']
            
            # Reprojetar para uma projeção métrica se necessário
            if not area_gdf.crs or area_gdf.crs.is_geographic:
                # Usar UTM para Brasil
                area_gdf = area_gdf.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
            
            # Calcular área total em km²
            area_sqkm = area_gdf.area.sum() / 1_000_000
            logger.info(f"Área calculada a partir dos polígonos de drenagem: {area_sqkm:.2f} km²")
        else:
            # Caso não tenhamos os polígonos, usar a união de todos os datasets
            all_geometries = []
            for key, gdf in data.items():
                if not gdf.empty:
                    all_geometries.extend(gdf.geometry.tolist())
            
            if all_geometries:
                # Calcular bounding box da união das geometrias
                all_gdf = gpd.GeoDataFrame(geometry=all_geometries)
                
                # Reprojetar para uma projeção métrica
                if not all_gdf.crs:
                    all_gdf = all_gdf.set_crs(epsg=4674)  # SIRGAS 2000
                
                if all_gdf.crs.is_geographic:
                    all_gdf = all_gdf.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
                
                # Calcular área do bounding box em km²
                bounds = all_gdf.total_bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        area_sqkm = (width * height) / 1_000_000
                logger.info(f"Área calculada a partir do bounding box: {area_sqkm:.2f} km²")
            else:
                logger.warning("Não foi possível calcular a área. Usando 1 km² como valor padrão.")
                area_sqkm = 1.0
    
    # Calcular densidade de drenagem para cada tipo de dado hidrográfico
    if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty:
        gdf = data['trecho_drenagem']
        
        # Reprojetar para projeção métrica se necessário
        if not gdf.crs:
            gdf = gdf.set_crs(epsg=4674)  # SIRGAS 2000
            
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        
        # Comprimento total em quilômetros
        total_length_km = gdf.geometry.length.sum() / 1000
        density = total_length_km / area_sqkm
        result['trecho_drenagem'] = density
        logger.info(f"Densidade de drenagem (trechos): {density:.4f} km/km²")
    
    if 'curso_dagua' in data and not data['curso_dagua'].empty:
        gdf = data['curso_dagua']
        
        # Reprojetar para projeção métrica se necessário
        if not gdf.crs:
            gdf = gdf.set_crs(epsg=4674)  # SIRGAS 2000
            
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        
        # Comprimento total em quilômetros
        total_length_km = gdf.geometry.length.sum() / 1000
        density = total_length_km / area_sqkm
        result['curso_dagua'] = density
        logger.info(f"Densidade de drenagem (cursos d'água): {density:.4f} km/km²")
    
    # Densidade geral combinando todos os tipos de linhas d'água
    total_length_km = 0
    for key in ['trecho_drenagem', 'curso_dagua']:
        if key in data and not data[key].empty:
            gdf = data[key]
            
            # Reprojetar para projeção métrica se necessário
            if not gdf.crs:
                gdf = gdf.set_crs(epsg=4674)  # SIRGAS 2000
                
            if gdf.crs.is_geographic:
                gdf = gdf.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
            
            total_length_km += gdf.geometry.length.sum() / 1000
    
    result['total'] = total_length_km / area_sqkm
    logger.info(f"Densidade de drenagem total: {result['total']:.4f} km/km²")
    
    return result

def enrich_strahler_order(gdf):
    """
    Valida e enriquece os dados de ordem de Strahler se disponíveis.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados hidrográficos
        
    Returns:
        geopandas.GeoDataFrame: Dados atualizados com ordens de Strahler validadas
    """
    logger.info("Enriquecendo informações de ordem de Strahler...")
    
    result = gdf.copy()
    
    # Verificar se a coluna de ordem de Strahler existe
    strahler_cols = [col for col in result.columns if 'strahler' in col.lower() or 'ordem' in col.lower()]
    
    if strahler_cols:
        strahler_col = strahler_cols[0]
        logger.info(f"Coluna de ordem de Strahler encontrada: {strahler_col}")
        
        # Converter para numérico se ainda não for
        if result[strahler_col].dtype not in ['int64', 'float64']:
            result[strahler_col] = pd.to_numeric(result[strahler_col], errors='coerce')
        
        # Preencher valores ausentes com 1 (ordem mais baixa)
        result[strahler_col] = result[strahler_col].fillna(1).astype(int)
        
        # Criar uma coluna padronizada de ordem de Strahler
        if strahler_col != 'strahler_order':
            result['strahler_order'] = result[strahler_col]
            
        # Verificar distribuição de ordens
        order_counts = result['strahler_order'].value_counts().sort_index()
        for order, count in order_counts.items():
            logger.info(f"Ordem de Strahler {order}: {count} segmentos")
    else:
        logger.warning("Nenhuma informação de ordem de Strahler encontrada. Criando coluna padrão.")
        result['strahler_order'] = 1
    
    return result

def generate_quality_report(data, enriched_data_path, visualization_paths=None):
    """
    Gera um relatório de qualidade a partir dos dados hidrográficos enriquecidos.
    
    Args:
        data (Dict[str, gpd.GeoDataFrame]): Dados hidrográficos enriquecidos
        enriched_data_path (str): Caminho do arquivo de dados enriquecidos
        visualization_paths (List[str], opcional): Lista de caminhos para as visualizações geradas
        
    Returns:
        Dict: Relatório de qualidade
    """
    logger.info("Gerando relatório de qualidade")
    
    # Metadados básicos
    report = {
        "name": "Relatório de Qualidade dos Dados Hidrográficos",
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "enriched_data_path": enriched_data_path,
        "visualization_paths": visualization_paths or [],
        "metrics": {}  # Inicializando a chave metrics
    }
    
    for key in data.keys():
        if key in data and not data[key].empty:
            original_gdf = data[key]
            enriched_gdf = data[key]
            
            # Dados básicos
            report["metrics"][key] = {
        "original_features": len(original_gdf),
        "enriched_features": len(enriched_gdf),
        "new_attributes": list(set(enriched_gdf.columns) - set(original_gdf.columns)),
            }
            
            # Adicionar estatísticas específicas para cada tipo
            if key in ['trecho_drenagem', 'curso_dagua']:
                # Estatísticas de sinuosidade
                if 'sinuosity' in enriched_gdf.columns:
                    sinuosity_stats = {
                "mean": float(enriched_gdf['sinuosity'].mean()),
                "median": float(enriched_gdf['sinuosity'].median()),
                "min": float(enriched_gdf['sinuosity'].min()),
                        "max": float(enriched_gdf['sinuosity'].max()),
                        "std_dev": float(enriched_gdf['sinuosity'].std())
                    }
                    report["metrics"][key]["sinuosity"] = sinuosity_stats
                
                # Estatísticas de betweenness
                if 'betweenness' in enriched_gdf.columns:
                    betweenness_stats = {
                "mean": float(enriched_gdf['betweenness'].mean()),
                "median": float(enriched_gdf['betweenness'].median()),
                "min": float(enriched_gdf['betweenness'].min()),
                        "max": float(enriched_gdf['betweenness'].max()),
                        "std_dev": float(enriched_gdf['betweenness'].std())
                    }
                    report["metrics"][key]["betweenness"] = betweenness_stats
                
                # Estatísticas de ordem de Strahler
                if 'strahler_order' in enriched_gdf.columns:
                    strahler_stats = {
                        "distribution": enriched_gdf['strahler_order'].value_counts().sort_index().to_dict(),
                        "max_order": int(enriched_gdf['strahler_order'].max())
                    }
                    report["metrics"][key]["strahler_order"] = strahler_stats
                
                # Estatísticas de elevação (dados altimétricos)
                if 'elevation_mean' in enriched_gdf.columns:
                    elevation_stats = {
                        "mean": float(enriched_gdf['elevation_mean'].mean()),
                        "min": float(enriched_gdf['elevation_min'].min()),
                        "max": float(enriched_gdf['elevation_max'].max())
                    }
                    report["metrics"][key]["elevation"] = elevation_stats
                
                # Estatísticas de declividade
                if 'slope_pct' in enriched_gdf.columns:
                    slope_stats = {
                        "mean": float(enriched_gdf['slope_pct'].mean()),
                        "median": float(enriched_gdf['slope_pct'].median()),
                        "min": float(enriched_gdf['slope_pct'].min()),
                        "max": float(enriched_gdf['slope_pct'].max())
                    }
                    report["metrics"][key]["slope"] = slope_stats
                
                # Estatísticas de potência do fluxo
                if 'stream_power_index' in enriched_gdf.columns:
                    spi_stats = {
                        "mean": float(enriched_gdf['stream_power_index'].mean()),
                        "median": float(enriched_gdf['stream_power_index'].median()),
                        "min": float(enriched_gdf['stream_power_index'].min()),
                        "max": float(enriched_gdf['stream_power_index'].max())
                    }
                    report["metrics"][key]["stream_power_index"] = spi_stats
                
                # Métricas de rede
                if hasattr(enriched_gdf, 'attrs') and 'network_info' in enriched_gdf.attrs:
                    report["metrics"][key]["network_info"] = enriched_gdf.attrs['network_info']
            
            # Estatísticas para áreas de drenagem (polígonos)
            if key == 'area_drenagem':
                # Estatísticas de elevação
                if 'elevation_mean' in enriched_gdf.columns:
                    elevation_stats = {
                        "mean": float(enriched_gdf['elevation_mean'].mean()),
                        "min": float(enriched_gdf['elevation_min'].min()),
                        "max": float(enriched_gdf['elevation_max'].max()),
                        "range": float(enriched_gdf['elevation_range'].mean())
                    }
                    report["metrics"][key]["elevation"] = elevation_stats
                
                # Integral hipsométrica
                if 'hypsometric_integral' in enriched_gdf.columns:
                    hi_stats = {
                        "mean": float(enriched_gdf['hypsometric_integral'].mean()),
                        "median": float(enriched_gdf['hypsometric_integral'].median()),
                        "min": float(enriched_gdf['hypsometric_integral'].min()),
                        "max": float(enriched_gdf['hypsometric_integral'].max())
                    }
                    report["metrics"][key]["hypsometric_integral"] = hi_stats
    
    # Adicionar informações de nós (nascentes, fozes, junções)
    if 'nodes' in data and not data['nodes'].empty:
        nodes_gdf = data['nodes']
        node_types = nodes_gdf['type'].value_counts().to_dict()
        
        report["nodes_summary"] = {
            "total_nodes": len(nodes_gdf),
            "springs": node_types.get('spring', 0),
            "outlets": node_types.get('outlet', 0),
            "junctions": node_types.get('junction', 0)
        }
        
        # Adicionar estatísticas de nós se disponíveis
        if hasattr(data['trecho_drenagem'], 'attrs') and 'node_stats' in data['trecho_drenagem'].attrs:
            report["nodes_summary"].update(data['trecho_drenagem'].attrs['node_stats'])
        
        # Adicionar estatísticas de elevação para os nós
        if 'elevation' in nodes_gdf.columns:
            node_elevation = {
                "springs_elevation_mean": float(nodes_gdf[nodes_gdf['type'] == 'spring']['elevation'].mean()),
                "outlets_elevation_mean": float(nodes_gdf[nodes_gdf['type'] == 'outlet']['elevation'].mean()),
                "elevation_drop": float(
                    nodes_gdf[nodes_gdf['type'] == 'spring']['elevation'].mean() -
                    nodes_gdf[nodes_gdf['type'] == 'outlet']['elevation'].mean()
                )
            }
            report["nodes_summary"]["elevation"] = node_elevation
    
    # Adicionar informações de densidade de drenagem
    drainage_density = calculate_drainage_density(data)
    report["drainage_density"] = drainage_density
    
    # Adicionar análise de qualidade da água
    if hasattr(data.get('trecho_drenagem', object()), 'attrs') and 'water_quality' in data['trecho_drenagem'].attrs:
        report["water_quality_analysis"] = data['trecho_drenagem'].attrs['water_quality']
    
    # Salvar relatório como JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORT_DIR, f'hidrografia_enrichment_report_{timestamp}.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, cls=NpEncoder)
    
    logger.info(f"Relatório de qualidade salvo em: {report_file}")
    
    # Gerar visualizações
    generate_visualizations(data, timestamp)
    
    return report_file

def generate_visualizations(data, timestamp):
    """
    Gera visualizações para acompanhar o relatório de qualidade.
    
    Args:
        data (Dict[str, gpd.GeoDataFrame]): Dados hidrográficos enriquecidos
        timestamp (str): Timestamp para nomear os arquivos
    """
    logger.info("Gerando visualizações...")
    
    # Configurar estilo de visualização global
    setup_visualization_style()
    
    # Verificar se temos dados válidos para visualizar
    has_valid_data = False
    for key in ['trecho_drenagem', 'curso_dagua']:
        if key in data and not data[key].empty:
            has_valid_data = True
            break
    
    if not has_valid_data:
        logger.warning("Dados insuficientes para gerar visualizações.")
        return
    
    # 1. Histograma de sinuosidade
    if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty and 'sinuosity' in data['trecho_drenagem'].columns:
        plt.figure(figsize=(12, 8))
        
        # Filtrar valores anômalos para melhor visualização
        sinuosity_data = data['trecho_drenagem']['sinuosity'].dropna()
        sinuosity_filtered = sinuosity_data[sinuosity_data < 3]  # Normalmente, sinuosidade acima de 3 é outlier
        
        # Criar histograma com estilo aprimorado
        n, bins, patches = plt.hist(sinuosity_filtered, bins=30, alpha=0.8, color='skyblue', edgecolor='white', linewidth=1.5)
        
        # Adicionar linhas de referência com anotações
        plt.axvline(x=1.0, color='#d73027', linestyle='--', linewidth=2, label='Linha Reta (1.0)')
        plt.axvline(x=1.5, color='#1a9850', linestyle='--', linewidth=2, label='Sinuosidade Moderada (1.5)')
        
        # Adicionar rótulos e título com formatação aprimorada
        plt.title('Distribuição de Sinuosidade dos Trechos de Drenagem', fontsize=22, pad=20)
        plt.xlabel('Índice de Sinuosidade', fontsize=16, labelpad=10)
        plt.ylabel('Frequência', fontsize=16, labelpad=10)
        
        # Melhorar legenda
        plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=14)
        
        # Ajustar limites e grades
        plt.xlim(0.9, min(3.0, sinuosity_filtered.max() + 0.2))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Adicionar estatísticas no gráfico
        stats_text = (f"Média: {sinuosity_filtered.mean():.2f}\n"
                      f"Mediana: {sinuosity_filtered.median():.2f}\n"
                      f"Máximo: {sinuosity_filtered.max():.2f}")
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                     ha='right', va='top', fontsize=12)
        
        sinuosity_hist_file = os.path.join(VISUALIZATION_DIR, f'sinuosity_histogram_{timestamp}.png')
        plt.savefig(sinuosity_hist_file, dpi=300, bbox_inches='tight')
    plt.close()
        logger.info(f"Histograma de sinuosidade salvo em: {sinuosity_hist_file}")
    
    # 2. Distribuição das ordens de Strahler
    if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty and 'strahler_order' in data['trecho_drenagem'].columns:
        plt.figure(figsize=(12, 8))
        
        order_counts = data['trecho_drenagem']['strahler_order'].value_counts().sort_index()
        
        # Criar gráfico de barras com cores graduais
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(order_counts)))
        bars = plt.bar(order_counts.index, order_counts.values, color=colors, edgecolor='black', linewidth=1.5, width=0.7)
        
        # Adicionar valores no topo de cada barra com formatação melhorada
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                     f'{int(height)}', 
                     ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        plt.title('Distribuição das Ordens de Strahler', fontsize=22, pad=20)
        plt.xlabel('Ordem de Strahler', fontsize=16, labelpad=10)
        plt.ylabel('Número de Trechos', fontsize=16, labelpad=10)
        
        # Ajustar eixos e grades
        plt.xticks(order_counts.index, fontsize=14)
        plt.yticks(fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adicionar bordas ao gráfico
        plt.gca().spines['left'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)
        
        plt.tight_layout()
        
        strahler_dist_file = os.path.join(VISUALIZATION_DIR, f'strahler_distribution_{timestamp}.png')
        plt.savefig(strahler_dist_file, dpi=300, bbox_inches='tight')
    plt.close()
        logger.info(f"Distribuição de ordens de Strahler salva em: {strahler_dist_file}")
    
    # 3. Histograma de betweenness centrality
    if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty and 'betweenness' in data['trecho_drenagem'].columns:
        plt.figure(figsize=(12, 8))
        
        # Preparar dados
        betweenness_data = data['trecho_drenagem']['betweenness'].replace(0, np.nan).dropna()
        
        # Criar bins personalizados para melhor visualização
        max_val = betweenness_data.max()
        bins = np.linspace(0, min(0.3, max_val*1.1), 30)
        
        # Criar histograma com escala logarítmica para frequência
        plt.hist(betweenness_data, bins=bins, log=True, color='lightgreen', 
                alpha=0.8, edgecolor='white', linewidth=1.5)
        
        plt.title('Distribuição de Betweenness Centrality (escala log)', fontsize=22, pad=20)
        plt.xlabel('Betweenness Centrality', fontsize=16, labelpad=10)
        plt.ylabel('Frequência (log)', fontsize=16, labelpad=10)
        
        # Ajustar limites e grades
        plt.xlim(0, min(0.3, max_val*1.1))
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adicionar estatísticas no gráfico
        stats_text = (f"Média: {betweenness_data.mean():.4f}\n"
                      f"Mediana: {betweenness_data.median():.4f}\n"
                      f"Máximo: {betweenness_data.max():.4f}")
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8),
                     ha='right', va='top', fontsize=12)
        
        plt.tight_layout()
        
        betweenness_hist_file = os.path.join(VISUALIZATION_DIR, f'betweenness_histogram_{timestamp}.png')
        plt.savefig(betweenness_hist_file, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Histograma de betweenness salvo em: {betweenness_hist_file}")
    
    # 4. Análise de Sinuosidade
    if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty and 'sinuosity' in data['trecho_drenagem'].columns:
        # Criar mapa temático de sinuosidade
        try:
            gdf = data['trecho_drenagem'].copy()
            
            # Reprojetar para visualização
            if not gdf.crs:
                gdf = gdf.set_crs(epsg=4674)  # SIRGAS 2000 para Brasil
                
            if gdf.crs.is_geographic:
                gdf_plot = gdf.to_crs(epsg=3857)  # Web Mercator para plotagem
            else:
                gdf_plot = gdf
            
            # Criar mapa com tamanho maior para melhor resolução
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Definir uma escala de cores personalizada para sinuosidade
            cmap = plt.cm.get_cmap('sinuosity')
            
            # Plotar trechos de drenagem coloridos por sinuosidade com visual aprimorado
            plot = gdf_plot.plot(
                column='sinuosity',
                ax=ax,
                cmap=cmap,
                linewidth=2.0,
                legend=True,
                vmin=1.0,
                vmax=max(2.5, gdf_plot['sinuosity'].quantile(0.95)),
                capstyle='round',
                alpha=0.9
            )
            
            # Configurar a legenda de cores
            cbar = plt.colorbar(plot.get_children()[0], ax=ax, fraction=0.035, pad=0.04)
            cbar.set_label('Índice de Sinuosidade', size=14, weight='bold', labelpad=10)
            cbar.ax.tick_params(labelsize=12)
            
            # Adicionar basemap com opções visuais aprimoradas
            try:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.CartoDB.Voyager,
                    zoom=12,
                    attribution_size=10
                )
            except Exception as e:
                logger.warning(f"Erro ao adicionar basemap: {str(e)}")
                try:
                    # Alternativa mais simples
                    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                except:
                    pass
            
            # Remover eixos
            plt.axis('off')
            
            # Adicionar título e nota de copyright
            plt.title('Análise de Sinuosidade da Rede Hidrográfica', fontsize=24, pad=20)
            plt.annotate('(C) OpenStreetMap contributors (C) CARTO', xy=(0.01, 0.01), 
                         xycoords='figure fraction', fontsize=10, color='#555555')
            
            plt.tight_layout()
            
            # Salvar figura com alta resolução
            sinuosity_map_file = os.path.join(VISUALIZATION_DIR, 'analise_sinuosidade.png')
            plt.savefig(sinuosity_map_file, dpi=450, bbox_inches='tight')
            plt.close()
            logger.info(f"Mapa de sinuosidade salvo em: {sinuosity_map_file}")
        except Exception as e:
            logger.error(f"Erro ao gerar mapa de sinuosidade: {str(e)}")
    
    # 5. Mapa de Áreas de Drenagem
    if 'area_drenagem' in data and not data['area_drenagem'].empty:
        try:
            areas_gdf = data['area_drenagem'].copy()
            
            # Reprojetar para visualização
            if not areas_gdf.crs:
                areas_gdf = areas_gdf.set_crs(epsg=4674)  # SIRGAS 2000 para Brasil
                
            if areas_gdf.crs.is_geographic:
                areas_plot = areas_gdf.to_crs(epsg=3857)  # Web Mercator para plotagem
            else:
                areas_plot = areas_gdf
            
            # Criar mapa com tamanho maior para melhor resolução
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Criar um colormap gradiente para áreas de drenagem
            cmap = plt.cm.get_cmap('hidrografia_blues')
            
            # Plotar áreas de drenagem com estilo aprimorado
            areas_plot.plot(
                ax=ax,
                facecolor=cmap(0.7),  # Cor de preenchimento consistente com a paleta
                edgecolor=cmap(0.2),  # Borda mais escura
                alpha=0.6,
                linewidth=1.5
            )
            
            # Plotar rede de drenagem se disponível
            if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty:
                trecho_gdf = data['trecho_drenagem'].copy()
                
                # Reprojetar para mesma projeção
                if trecho_gdf.crs != areas_plot.crs:
                    trecho_plot = trecho_gdf.to_crs(areas_plot.crs)
                else:
                    trecho_plot = trecho_gdf
                
                trecho_plot.plot(
                    ax=ax,
                    color=cmap(0.1),  # Cor escura para rios
                    linewidth=1.0,
                    alpha=0.8
                )
            
            # Adicionar basemap
            try:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.CartoDB.Positron,
                    zoom=12,
                    attribution_size=10
                )
            except Exception as e:
                logger.warning(f"Erro ao adicionar basemap: {str(e)}")
                try:
                    # Alternativa mais simples
                    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                except:
                    pass
            
            # Remover eixos
            plt.axis('off')
            
            # Adicionar título e nota de copyright
            plt.title('Áreas de Drenagem e Rede Hidrográfica', fontsize=24, pad=20)
            plt.annotate('(C) OpenStreetMap contributors (C) CARTO', xy=(0.01, 0.01), 
                         xycoords='figure fraction', fontsize=10, color='#555555')
            
            plt.tight_layout()
            
            # Salvar figura com alta resolução
            areas_map_file = os.path.join(VISUALIZATION_DIR, 'analise_areas_drenagem.png')
            plt.savefig(areas_map_file, dpi=450, bbox_inches='tight')
            plt.close()
            logger.info(f"Mapa de áreas de drenagem salvo em: {areas_map_file}")
        except Exception as e:
            logger.error(f"Erro ao gerar mapa de áreas de drenagem: {str(e)}")
    
    # 6. Mapa com Nascentes e Fozes
    if 'nodes' in data and not data['nodes'].empty:
        try:
            nodes_gdf = data['nodes'].copy()
            
            # Reprojetar para visualização
            if not nodes_gdf.crs:
                nodes_gdf = nodes_gdf.set_crs(epsg=4674)  # SIRGAS 2000 para Brasil
                
            if nodes_gdf.crs.is_geographic:
                nodes_plot = nodes_gdf.to_crs(epsg=3857)  # Web Mercator para plotagem
            else:
                nodes_plot = nodes_gdf
            
            # Criar dicionário para cores por tipo com cores mais vibrantes
            color_dict = {
                'spring': '#2ca25f',    # Verde mais vivo
                'outlet': '#d73027',    # Vermelho mais vivo
                'junction': '#feb24c'   # Laranja mais vivo
            }
            
            # Preparar figura com tamanho maior
            fig, ax = plt.subplots(figsize=(16, 12))
            
            # Plotar rede hidrográfica primeiro com estilo aprimorado
            if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty:
                trecho_gdf = data['trecho_drenagem'].copy()
                
                # Reprojetar para mesma projeção
                if trecho_gdf.crs != nodes_plot.crs:
                    trecho_plot = trecho_gdf.to_crs(nodes_plot.crs)
                else:
                    trecho_plot = trecho_gdf
                
                trecho_plot.plot(
                    ax=ax,
                    color='#4575b4',  # Azul mais atraente para rios
                    linewidth=1.0,
                    alpha=0.7
                )
            
            # Plotar cada tipo de ponto com cor específica e tamanho melhorado
            node_types = {'spring': 'Nascente', 'outlet': 'Foz', 'junction': 'Junção'}
            
            # Criar escala para tamanho dos marcadores
            min_size = 20
            max_size = 150
            
            for node_type, color in color_dict.items():
                subset = nodes_plot[nodes_plot['type'] == node_type]
                if not subset.empty:
                    # Escalar tamanho dos marcadores
                    if 'connected_features' in subset.columns:
                        markersize = subset['connected_features'].apply(
                            lambda x: min(max_size, max(min_size, x * 15))
                        )
                    else:
                        markersize = min_size + 30
                    
                    subset.plot(
                        ax=ax,
                        color=color,
                        markersize=markersize,
                        label=node_types[node_type],
                        alpha=0.8,
                        edgecolor='white',
                        linewidth=0.5
                    )
            
            # Adicionar basemap com alta qualidade
            try:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.CartoDB.Positron,
                    zoom=12,
                    attribution_size=10
                )
            except Exception as e:
                logger.warning(f"Erro ao adicionar basemap: {str(e)}")
                try:
                    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                except:
                    pass
            
            # Remover eixos
            plt.axis('off')
            
            # Adicionar título com estilo melhorado
            plt.title('Análise da Rede Hidrográfica: Nascentes, Fozes e Junções', fontsize=24, pad=20)
            
            # Melhorar a legenda
            legend = plt.legend(
                loc='lower right',
                title='Elementos da Rede',
                fontsize=14,
                frameon=True,
                framealpha=0.9,
                edgecolor='gray',
                title_fontsize=16
            )
            
            # Adicionar nota de copyright
            plt.annotate('(C) OpenStreetMap contributors (C) CARTO', xy=(0.01, 0.01), 
                         xycoords='figure fraction', fontsize=10, color='#555555')
            
            plt.tight_layout()
            
            # Salvar figura com alta resolução
            network_map_file = os.path.join(VISUALIZATION_DIR, 'analise_rede_hidrografia.png')
            plt.savefig(network_map_file, dpi=450, bbox_inches='tight')
            plt.close()
            logger.info(f"Mapa da rede hidrográfica salvo em: {network_map_file}")
        except Exception as e:
            logger.error(f"Erro ao gerar mapa da rede hidrográfica: {str(e)}")
    
    # 7. Comprimento dos trechos por ordem de Strahler
    if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty and 'strahler_order' in data['trecho_drenagem'].columns:
        try:
            # Reprojetar para sistema métrico
            trecho_gdf = data['trecho_drenagem'].copy()
            
            if not trecho_gdf.crs:
                trecho_gdf = trecho_gdf.set_crs(epsg=4674)
                
            if trecho_gdf.crs.is_geographic:
                trecho_gdf = trecho_gdf.to_crs(epsg=3857)  # Web Mercator (métrico)
            
            # Calcular comprimento em km
            trecho_gdf['length_km'] = trecho_gdf.geometry.length / 1000
            
            # Agrupar por ordem e calcular comprimento total
            length_by_order = trecho_gdf.groupby('strahler_order')['length_km'].sum().sort_index()
            
            # Criar gráfico com tamanho aprimorado
            plt.figure(figsize=(14, 8))
            
            # Criar gradiente de cores azuis para as barras
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(length_by_order)))
            
            # Criar gráfico de barras com visual aprimorado
            bars = plt.bar(
                length_by_order.index, 
                length_by_order.values, 
                color=colors,
                edgecolor='black',
                linewidth=1.5,
                width=0.7,
                alpha=0.85
            )
            
            # Adicionar valores sobre as barras com formatação melhorada
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + 2,
                    f'{height:.1f} km', 
                    ha='center', 
                    va='bottom',
                    fontsize=14,
                    fontweight='bold'
                )
            
            # Adicionar título e rótulos dos eixos com estilo aprimorado
            plt.title('Comprimento Total de Trechos por Ordem de Strahler', fontsize=22, pad=20)
            plt.xlabel('Ordem de Strahler', fontsize=16, labelpad=10)
            plt.ylabel('Comprimento Total (km)', fontsize=16, labelpad=10)
            
            # Melhorar a aparência dos eixos
            plt.xticks(length_by_order.index, fontsize=14)
            plt.yticks(fontsize=14)
            
            # Adicionar grid estilizado e ajustar limites do eixo Y
            plt.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
            plt.ylim(0, max(length_by_order.values) * 1.15)  # Espaço extra para os rótulos
            
            # Adicionar bordas aos eixos principais
            plt.gca().spines['left'].set_visible(True)
            plt.gca().spines['bottom'].set_visible(True)
            plt.gca().spines['left'].set_linewidth(1.5)
            plt.gca().spines['bottom'].set_linewidth(1.5)
            
            plt.tight_layout()
            
            # Salvar figura com alta resolução
            length_chart_file = os.path.join(VISUALIZATION_DIR, 'comprimento_por_strahler.png')
            plt.savefig(length_chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de comprimento por ordem de Strahler salvo em: {length_chart_file}")
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de comprimento por Strahler: {str(e)}")
    
    # 8. Visualização adicional: trechos de drenagem por ordem de Strahler (gráfico de pizza)
    if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty and 'strahler_order' in data['trecho_drenagem'].columns:
        try:
            # Criar gráfico de pizza com tamanho e estilo aprimorados
    plt.figure(figsize=(12, 10))
            
            order_counts = data['trecho_drenagem']['strahler_order'].value_counts().sort_index()
            
            # Definir uma paleta de cores mais atraente
            colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(order_counts)))
            
            # Calcular porcentagens para rótulos
            total = order_counts.sum()
            percentages = [f"{int(count/total*100)}%" for count in order_counts.values]
            
            # Adicionar um pequeno deslocamento para destacar cada fatia
            explode = [0.01] * len(order_counts)
            explode[np.argmax(order_counts.values)] = 0.1  # Destacar a maior fatia
            
            # Plotar gráfico de pizza com estilo aprimorado
            wedges, texts, autotexts = plt.pie(
                order_counts.values, 
                labels=order_counts.index,
                explode=explode,
                colors=colors,
                autopct='%1.1f%%',
                pctdistance=0.85,
                startangle=90,
                shadow=True,
                wedgeprops={'edgecolor': 'white', 'linewidth': 1.5, 'antialiased': True},
                textprops={'fontsize': 14, 'fontweight': 'bold'}
            )
            
            # Personalizar o texto
            plt.setp(autotexts, size=12, weight="bold", color="black")
            plt.setp(texts, size=14, weight="bold")
            
            # Adicionar um círculo central para estilo de donut
            centre_circle = plt.Circle((0,0), 0.35, fc='white', edgecolor='gray')
            plt.gca().add_patch(centre_circle)
            
            plt.axis('equal')
            
            # Adicionar título com estilo aprimorado
            plt.title('Distribuição dos Trechos por Ordem de Strahler', fontsize=22, pad=40, y=1.05)
            
            # Melhorar a legenda
            plt.tight_layout()
            
            # Salvar figura com alta resolução
            strahler_pie_file = os.path.join(VISUALIZATION_DIR, 'distribuicao_strahler.png')
            plt.savefig(strahler_pie_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Gráfico de distribuição de Strahler salvo em: {strahler_pie_file}")
        except Exception as e:
            logger.error(f"Erro ao gerar gráfico de distribuição de Strahler: {str(e)}")
    
    # 9. Mapa Estático com Basemap
    try:
        if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty:
            # Reprojetar para Web Mercator
            gdf = data['trecho_drenagem'].copy()
            
            if not gdf.crs:
                gdf = gdf.set_crs(epsg=4674)  # SIRGAS 2000
                
            if gdf.crs.is_geographic:
                gdf = gdf.to_crs(epsg=3857)  # Web Mercator
            
            # Criar mapa com tamanho maior para alta resolução
            fig, ax = plt.subplots(figsize=(18, 14), dpi=100)
            
            # Configurar estilo para mais clareza e contraste
            background_color = 'black'
            fig.patch.set_facecolor(background_color)
            ax.set_facecolor(background_color)
            
            # Plotar com cores por Strahler se disponível
            if 'strahler_order' in gdf.columns:
                # Criar um colormap personalizado para Strahler
                unique_orders = sorted(gdf['strahler_order'].unique())
                
                # Usar um colormap de alta visibilidade
                cmap = plt.cm.get_cmap('viridis')
                norm = plt.Normalize(min(unique_orders), max(unique_orders))
                
                # Plotar cada ordem com estilo diferente
                for order in unique_orders:
                    subset = gdf[gdf['strahler_order'] == order]
                    
                    # Espessura e brilho proporcionais à ordem para destaque visual
                    linewidth = order * 0.6
                    alpha = min(0.9, 0.5 + (order / 10))
                    
                    subset.plot(
                        ax=ax,
                        color=cmap(norm(order)),
                        linewidth=linewidth,
                        alpha=alpha,
                        label=f'Ordem {order}',
                        zorder=order + 10  # Ordens maiores por cima
                    )
                
                # Adicionar legenda com estilo aprimorado
                legend = plt.legend(
                    title='Ordem de Strahler',
                    loc='lower right',
                    fontsize=12,
                    title_fontsize=14,
                    frameon=True,
                    framealpha=0.8,
                    edgecolor='white',
                    facecolor='black'
                )
                plt.setp(legend.get_texts(), color='white')
                plt.setp(legend.get_title(), color='white')
            else:
                # Plotar com cor única mas estilo atraente
                gdf.plot(
                    ax=ax,
                    color='#55a8dd',
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=10
                )
            
            # Adicionar áreas de drenagem se disponíveis
            if 'area_drenagem' in data and not data['area_drenagem'].empty:
                areas_gdf = data['area_drenagem'].copy()
                
                if not areas_gdf.crs:
                    areas_gdf = areas_gdf.set_crs(epsg=4674)
                    
                if areas_gdf.crs.is_geographic:
                    areas_gdf = areas_gdf.to_crs(epsg=3857)
                
                areas_gdf.plot(
                    ax=ax,
                    facecolor='none',
                    edgecolor='white',
                    alpha=0.4,
                    linewidth=1,
                    zorder=5
                )
            
            # Adicionar pontos de interesse se disponíveis, com estilo melhorado
            if 'nodes' in data and not data['nodes'].empty:
                nodes_gdf = data['nodes'].copy()
                
                if not nodes_gdf.crs:
                    nodes_gdf = nodes_gdf.set_crs(epsg=4674)
                    
                if nodes_gdf.crs.is_geographic:
                    nodes_gdf = nodes_gdf.to_crs(epsg=3857)
                
                # Estilo aprimorado para os pontos
                node_styles = {
                    'spring': {'color': '#32CD32', 'zorder': 20, 'alpha': 0.9, 'label': 'Nascente', 'edgecolor': 'white'},
                    'outlet': {'color': '#FF4500', 'zorder': 20, 'alpha': 0.9, 'label': 'Foz', 'edgecolor': 'white'},
                    'junction': {'color': '#FFD700', 'zorder': 15, 'alpha': 0.8, 'label': 'Junção', 'edgecolor': 'white'}
                }
                
                for node_type, style in node_styles.items():
                    subset = nodes_gdf[nodes_gdf['type'] == node_type]
                    if not subset.empty:
                        subset.plot(
                            ax=ax,
                            color=style['color'],
                            markersize=40,
                            alpha=style['alpha'],
                            label=style['label'],
                            edgecolor=style['edgecolor'],
                            linewidth=0.8,
                            zorder=style['zorder']
                        )
            
            # Adicionar basemap de imagem de satélite com alta qualidade
            try:
                ctx.add_basemap(
                    ax,
                    source=ctx.providers.Esri.WorldImagery,
                    zoom=13,
                    attribution_size=10
                )
            except Exception as e:
                logger.warning(f"Erro ao adicionar basemap: {str(e)}")
                try:
                    # Alternativa
                    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)
                except:
                    pass
            
            plt.axis('off')
            
            # Adicionar título com estilo para contraste sobre a imagem de satélite
            title = plt.title('Mapa da Rede Hidrográfica', fontsize=24, pad=20, color='white')
            
            # Adicionar caixa de informação no canto
            plt.text(
                0.97, 0.03, 
                f"Trechos: {len(gdf)}\nOrdens: {'-'.join(map(str, unique_orders))}\nData: {timestamp}",
                transform=ax.transAxes,
                fontsize=12,
                color='white',
                ha='right',
                va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.7, edgecolor='white')
            )
            
            # Adicionar atribuição
            plt.annotate('Tiles (C) Esri — Fonte: Esri, Maxar, Earthstar Geographics, e GIS User Community', 
                         xy=(0.01, 0.01), xycoords='figure fraction', fontsize=8, color='white')
            
            plt.tight_layout()
            
            # Salvar mapa com alta resolução
            static_map_file = os.path.join(VISUALIZATION_DIR, 'mapa_estatico_hidrografia.png')
            plt.savefig(static_map_file, dpi=450, bbox_inches='tight', facecolor=background_color)
            plt.close()
            logger.info(f"Mapa estático salvo em: {static_map_file}")
    except Exception as e:
        logger.error(f"Erro ao gerar mapa estático: {str(e)}")
    
    # 10. Mapa Interativo
    try:
        if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty:
            # Reprojetar para WGS84 para uso com folium
            gdf = data['trecho_drenagem'].copy()
            
            if not gdf.crs:
                gdf = gdf.set_crs(epsg=4674)
            
            gdf = gdf.to_crs(epsg=4326)  # WGS84
            
            # Determinar centro do mapa
            center_lat = gdf.unary_union.centroid.y
            center_lon = gdf.unary_union.centroid.x
            
            # Criar mapa base com opções visuais melhoradas
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='CartoDB positron',
                control_scale=True,
                prefer_canvas=True
            )
            
            # Adicionar camadas base adicionais para escolha do usuário
            folium.TileLayer(
                tiles='CartoDB dark_matter',
                name='Mapa Escuro',
                attr='CartoDB'
            ).add_to(m)
            
            folium.TileLayer(
                tiles='OpenStreetMap',
                name='OpenStreetMap',
                attr='OpenStreetMap'
            ).add_to(m)
            
            # Adicionar camada de satélite com melhor resolução
            folium.TileLayer(
                tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                attr='Esri',
                name='Satélite',
                overlay=False
            ).add_to(m)
            
            # Função estilo para trechos de drenagem com visual aprimorado
            def style_function(feature):
                strahler = feature['properties'].get('strahler_order', 1)
                sinuosity = feature['properties'].get('sinuosity', 1)
                
                # Espessura baseada na ordem de Strahler
                weight = 1.5 + (strahler * 0.5)
                
                # Esquema de cores mais sofisticado baseado na sinuosidade
                if sinuosity < 1.1:
                    color = '#d73027'  # Vermelho para rios muito retificados
                elif sinuosity < 1.3:
                    color = '#fc8d59'  # Laranja para pouca sinuosidade
                elif sinuosity < 1.5:
                    color = '#4393c3'  # Azul médio
                elif sinuosity < 1.8:
                    color = '#2166ac'  # Azul mais escuro
                else:
                    color = '#053061'  # Azul muito escuro para alta sinuosidade
                
                return {
                    'color': color,
                    'weight': weight,
                    'opacity': 0.8,
                    'dashArray': '4' if strahler <= 2 else None,  # Linhas tracejadas para ordens menores
                }
            
            # Função hover para destaque
            def highlight_function(feature):
                return {
                    'color': '#ffff00',  # Amarelo brilhante
                    'weight': 5,
                    'opacity': 1,
                }
            
            # Adicionar trechos de drenagem com estilos melhorados
            if any(col in gdf.columns for col in ['strahler_order', 'sinuosity', 'betweenness']):
                # Garantir que as colunas estejam no GeoJSON
                for col in ['strahler_order', 'sinuosity', 'betweenness']:
                    if col in gdf.columns:
                        gdf[col] = gdf[col].fillna(0)
                        # Arredondar para reduzir tamanho
                        if col == 'sinuosity' or col == 'betweenness':
                            gdf[col] = gdf[col].round(4)
                
                # Criar categorias para sinuosidade para legenda
                if 'sinuosity' in gdf.columns:
                    bins = [0, 1.1, 1.3, 1.5, 1.8, float('inf')]
                    labels = ['Muito Baixa (<1.1)', 'Baixa (1.1-1.3)', 'Média (1.3-1.5)', 'Alta (1.5-1.8)', 'Muito Alta (>1.8)']
                    gdf['sinuosity_cat'] = pd.cut(gdf['sinuosity'], bins=bins, labels=labels, right=False)
                
                # Converter para GeoJSON com campos selecionados para reduzir tamanho
                if 'sinuosity_cat' in gdf.columns:
                    fields = ['strahler_order', 'sinuosity', 'sinuosity_cat', 'betweenness']
                else:
                    fields = ['strahler_order', 'sinuosity', 'betweenness']
                    
                fields = [f for f in fields if f in gdf.columns]
                
                # Converter para GeoJSON com tamanho otimizado
                gjson = gdf[['geometry'] + fields].to_json()
                
                # Adicionar como camada GeoJSON com estilo aprimorado
                network_layer = folium.GeoJson(
                    gjson,
                    name='Trechos de Drenagem',
                    style_function=style_function,
                    highlight_function=highlight_function,
                    tooltip=folium.GeoJsonTooltip(
                        fields=fields,
                        aliases=['Ordem de Strahler', 'Sinuosidade', 'Categoria de Sinuosidade', 'Betweenness'],
                        localize=True,
                        sticky=False,
                        labels=True,
                        style="""
                            background-color: rgba(255, 255, 255, 0.8);
                            border: 2px solid #444;
                            border-radius: 5px;
                            box-shadow: 3px 3px 3px rgba(0,0,0,0.3);
                            font-family: 'Arial', sans-serif;
                            font-size: 12px;
                            padding: 10px;
                        """
                    )
                ).add_to(m)
            else:
                # Adicionar como camada simples
                folium.GeoJson(
                    gdf,
                    name='Trechos de Drenagem',
                    style_function=lambda x: {'color': '#3388ff', 'weight': 2, 'opacity': 0.8}
                ).add_to(m)
            
            # Adicionar pontos de interesse se disponíveis, com estilo aprimorado
            if 'nodes' in data and not data['nodes'].empty:
                nodes_gdf = data['nodes'].copy()
                
                if not nodes_gdf.crs:
                    nodes_gdf = nodes_gdf.set_crs(epsg=4674)
                    
                nodes_gdf = nodes_gdf.to_crs(epsg=4326)  # WGS84 para folium
                
                # Log para debug
                logger.info(f"Adicionando {len(nodes_gdf)} nós ao mapa (tipos: {nodes_gdf['type'].value_counts().to_dict()})")
                
                # Criar configurações aprimoradas para os ícones
                icon_configs = {
                    'spring': {
                        'icon': 'tint', 
                        'color': 'green', 
                        'prefix': 'fa',
                        'name': 'Nascentes',
                        'tooltip': 'Nascente'
                    },
                    'outlet': {
                        'icon': 'arrow-down', 
                        'color': 'red', 
                        'prefix': 'fa',
                        'name': 'Fozes',
                        'tooltip': 'Foz'
                    },
                    'junction': {
                        'icon': 'random', 
                        'color': 'orange', 
                        'prefix': 'fa',
                        'name': 'Junções',
                        'tooltip': 'Junção'
                    }
                }
                
                # Para cada tipo de ponto, criar uma camada separada (sem clustering)
                for node_type, icon_info in icon_configs.items():
                    # Filtrar por tipo
                    subset = nodes_gdf[nodes_gdf['type'] == node_type]
                    
                    if not subset.empty:
                        # Criar grupo de camada sem clustering
                        feature_group = folium.FeatureGroup(name=icon_info['name'], overlay=True)
                        
                        # Adicionar pontos ao grupo
                        for idx, row in subset.iterrows():
                            # Coletar informações adicionais se disponíveis
                            additional_info = ""
                            if 'elevation' in row and not pd.isna(row['elevation']):
                                additional_info += f"<b>Elevação:</b> {row['elevation']:.1f} m<br>"
                            if 'bank_distance' in row and not pd.isna(row['bank_distance']):
                                additional_info += f"<b>Distância à margem:</b> {row['bank_distance']:.1f} m<br>"
                            if 'riparian_zone' in row and not pd.isna(row['riparian_zone']):
                                additional_info += f"<b>Zona ripária:</b> {row['riparian_zone']}<br>"
                            
                            popup_content = f"""
                            <div style="min-width: 200px;">
                                <h4 style="color: {icon_info['color']};">{icon_info['tooltip']}</h4>
                                <b>Tipo:</b> {row['type']}<br>
                                <b>Conexões:</b> {row['connected_features']}<br>
                                {additional_info}
                            </div>
                            """
                            
                            # Tamanho do ícone baseado na importância (número de conexões)
                            icon_size = min(6, max(3, int(row['connected_features']))) if 'connected_features' in row else 4
                            
                            marker = folium.Marker(
                                location=[row.geometry.y, row.geometry.x],
                                popup=folium.Popup(popup_content, max_width=300),
                                tooltip=f"{icon_info['tooltip']} - Conexões: {row['connected_features']}",
                                icon=folium.Icon(
                                    color=icon_info['color'],
                                    icon=icon_info['icon'],
                                    prefix=icon_info['prefix']
                                )
                            )
                            marker.add_to(feature_group)
                        
                        feature_group.add_to(m)
                        
                # Adicionar camada com todos os nós para visualização completa
                all_nodes_group = folium.FeatureGroup(name="Todos os Pontos", overlay=False)
                
                # Usar círculos coloridos para melhor visualização em modo completo
                for idx, row in nodes_gdf.iterrows():
                    node_type = row['type']
                    color = icon_configs[node_type]['color']
                    
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.7,
                        tooltip=f"{icon_configs[node_type]['tooltip']} - Conexões: {row['connected_features']}"
                    ).add_to(all_nodes_group)
                
                all_nodes_group.add_to(m)
            
            # Adicionar camada de áreas de drenagem se disponível
            if 'area_drenagem' in data and not data['area_drenagem'].empty:
                areas_gdf = data['area_drenagem'].copy()
                
                if not areas_gdf.crs:
                    areas_gdf = areas_gdf.set_crs(epsg=4674)
                    
                areas_gdf = areas_gdf.to_crs(epsg=4326)  # WGS84
                
                folium.GeoJson(
                    areas_gdf,
                    name='Áreas de Drenagem',
                    style_function=lambda x: {
                        'fillColor': 'blue',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.1
                    }
                ).add_to(m)
            
            # Adicionar controle de camadas
            folium.LayerControl().add_to(m)
            
            # Adicionar escala
            folium.plugins.MeasureControl(
                position='bottomleft',
                primary_length_unit='kilometers'
            ).add_to(m)
            
            # Adicionar minimap
            folium.plugins.MiniMap().add_to(m)
            
            # Salvar mapa
            interactive_map_file = os.path.join(VISUALIZATION_DIR, 'mapa_interativo_hidrografia.html')
            m.save(interactive_map_file)
            logger.info(f"Mapa interativo salvo em: {interactive_map_file}")
    except Exception as e:
        logger.error(f"Erro ao gerar mapa interativo: {str(e)}")

def extract_elevation_for_lines(gdf, dem):
    """
    Extrai valores de elevação para cada feição linear a partir do DEM.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame com geometrias lineares
        dem (rasterio.DatasetReader): Dataset raster contendo dados de elevação
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame com colunas de elevação adicionadas
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
    
    # Processar cada geometria
    for idx, row in result.iterrows():
        try:
            if not isinstance(row.geometry, (LineString, MultiLineString)):
                continue
                
            geom = row.geometry
            
            # Para MultiLineString, processar cada parte separadamente
            if isinstance(geom, MultiLineString):
                # Converter para LineString pela união das partes
                try:
                    geom = linemerge(geom)
                    if isinstance(geom, MultiLineString):  # Se ainda for MultiLineString após merge
                        # Usar a parte mais longa
                        geom = sorted(geom.geoms, key=lambda line: line.length, reverse=True)[0]
                except Exception as e:
                    logger.warning(f"Erro ao mesclar MultiLineString: {str(e)}")
                    continue
            
            # Gerar pontos ao longo da linha com espaçamento adequado
            # Usar uma resolução um pouco menor que a do DEM para capturar variações
            dem_resolution = min(abs(dem.res[0]), abs(dem.res[1]))
            sample_distance = dem_resolution * 0.8  # 80% da resolução do DEM
            
            # Garantir que temos pontos suficientes mesmo para linhas curtas
            line_length = geom.length
            num_points = max(10, int(line_length / sample_distance))
            
            distances = np.linspace(0, line_length, num_points)
            points = [geom.interpolate(distance) for distance in distances]
            
            # Extrair elevação para cada ponto
            elevations = []
            for point in points:
                try:
                    # Obter coordenadas do ponto
                    x, y = point.x, point.y
                    
                    # Verificar se o ponto está dentro da extensão do DEM
                    if (dem.bounds[0] <= x <= dem.bounds[2] and 
                        dem.bounds[1] <= y <= dem.bounds[3]):
                        
                        # Converter coordenadas para índices de pixel
                        row, col = dem.index(x, y)
                        
                        # Ler valor do pixel
                        elevation = dem.read(1, window=((row, row+1), (col, col+1)))
                        
                        # Verificar se o valor é válido
                        if elevation.size > 0 and elevation[0][0] != dem.nodata:
                            elevations.append(float(elevation[0][0]))
                except Exception as e:
                    logger.debug(f"Erro ao extrair elevação para ponto: {str(e)}")
                    continue
            
            # Calcular estatísticas de elevação se temos valores válidos
            if elevations:
                result.at[idx, 'elevation_min'] = min(elevations)
                result.at[idx, 'elevation_max'] = max(elevations)
                result.at[idx, 'elevation_mean'] = sum(elevations) / len(elevations)
                result.at[idx, 'elevation_range'] = max(elevations) - min(elevations)
                
                # Calcular declividade
                if len(elevations) > 1 and line_length > 0:
                    # Declividade média como porcentagem
                    elev_change = abs(elevations[0] - elevations[-1])  # Diferença de elevação (m)
                    slope_pct = (elev_change / line_length) * 100  # Declividade como porcentagem
                    slope_deg = math.degrees(math.atan(elev_change / line_length))  # Declividade em graus
                    
                    result.at[idx, 'slope_pct'] = slope_pct
                    result.at[idx, 'slope_deg'] = slope_deg
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
    
    return result

def enrich_hydrographic_data(data, dem=None):
    """
    Enriquece os dados hidrográficos com métricas e análises adicionais.
    
    Args:
        data (Dict[str, gpd.GeoDataFrame]): Dicionário com GeoDataFrames de dados hidrográficos
        dem (rasterio.DatasetReader, opcional): Modelo Digital de Elevação para análises altimétricas
        
    Returns:
        Dict[str, gpd.GeoDataFrame]: Dicionário com os dados enriquecidos
    """
    logger.info("Iniciando enriquecimento de dados hidrográficos...")
    
    enriched_data = {}
    
    # Processar cada tipo de dado hidrográfico
    if 'trecho_drenagem' in data and not data['trecho_drenagem'].empty:
        gdf = data['trecho_drenagem'].copy()
        
        # Reprojetar para uma projeção métrica para cálculos precisos
        if not gdf.crs:
            gdf = gdf.set_crs(epsg=4674)  # SIRGAS 2000
            
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        
        # 1. Calcular sinuosidade
        gdf = calculate_sinuosity(gdf)
        
        # 2. Construir rede hidrográfica
        network, gdf = build_stream_network(gdf)
        
        # 3. Enriquecer ordens de Strahler
        gdf = enrich_strahler_order(gdf)
        
        # 4. Extrair elevação e declividade
        if dem is not None:
            gdf = extract_elevation_for_lines(gdf, dem)
        
        # 5. Calcular índice de potência de fluxo (stream power index)
        if dem is not None and 'elevation_mean' in gdf.columns:
            gdf = calculate_stream_power_index({'trecho_drenagem': gdf})['trecho_drenagem']
        
        enriched_data['trecho_drenagem'] = gdf
    
    if 'curso_dagua' in data and not data['curso_dagua'].empty:
        gdf = data['curso_dagua'].copy()
        
        # Reprojetar para uma projeção métrica
        if not gdf.crs:
            gdf = gdf.set_crs(epsg=4674)  # SIRGAS 2000
            
        if gdf.crs.is_geographic:
            gdf = gdf.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        
        # Calcular sinuosidade para os cursos d'água
        gdf = calculate_sinuosity(gdf)
        
        # Extrair elevação e declividade se disponível
        if dem is not None:
            gdf = extract_elevation_for_lines(gdf, dem)
        
        enriched_data['curso_dagua'] = gdf
    
    # Processar áreas de drenagem (polígonos)
    if 'area_drenagem' in data and not data['area_drenagem'].empty:
        area_drenagem = data['area_drenagem'].copy()
        
        # Extrair estatísticas de elevação para áreas de drenagem
        if dem is not None:
            area_drenagem = extract_elevation_for_polygons(area_drenagem, dem)
        
        enriched_data['area_drenagem'] = area_drenagem
    
    # Processar pontos de drenagem
    if 'ponto_drenagem' in data and not data['ponto_drenagem'].empty:
        ponto_drenagem = data['ponto_drenagem'].copy()
        
        # Extrair elevação para pontos
        if dem is not None:
            ponto_drenagem = extract_elevation_for_points(ponto_drenagem, dem)
        
        enriched_data['ponto_drenagem'] = ponto_drenagem
    
    # 6. Calcular índice de potência do fluxo usando dados altimétricos e hidrológicos
    if dem is not None:
        enriched_data = calculate_stream_power_index(enriched_data)
    
    # 7. Realizar análise de qualidade da água baseada em parâmetros morfométricos
    water_quality = analyze_water_quality(enriched_data)
    
    # Adicionar análise de qualidade da água aos metadados
    for key in enriched_data:
        if hasattr(enriched_data[key], 'attrs'):
            enriched_data[key].attrs['water_quality'] = water_quality
        else:
            enriched_data[key].attrs = {'water_quality': water_quality}
    
    # 8. Gerar visualizações com dados altimétricos
    if dem is not None:
        visualize_hydrography_with_dem(enriched_data, dem)
    
    return enriched_data

def save_enriched_data(data, output_file=None):
    """
    Salva os dados enriquecidos em um arquivo GeoPackage.
    
    Args:
        data (Dict[str, gpd.GeoDataFrame]): Dicionário com GeoDataFrames contendo dados enriquecidos
        output_file (str, opcional): Caminho para o arquivo de saída. Se None, usa o padrão.
        
    Returns:
        str: Caminho para o arquivo salvo
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"hidrografia_enriched_{timestamp}.gpkg")
    
    logger.info(f"Salvando dados enriquecidos em: {output_file}")
    
    # Garantir que o diretório de saída existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Salvar cada layer no arquivo GPKG
    for key, gdf in data.items():
        if not gdf.empty:
            try:
                gdf.to_file(output_file, layer=key, driver='GPKG')
                logger.info(f"Layer '{key}' salvo com {len(gdf)} feições")
            except Exception as e:
                logger.error(f"Erro ao salvar layer '{key}': {str(e)}")
    
    return output_file

def identify_nodes_and_endpoints(gdf):
    """
    Identifica pontos de nascentes, fozes e junções na rede hidrográfica.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame com trechos de drenagem (linestrings)
        
    Returns:
        Tuple[gpd.GeoDataFrame, Dict]: Tuple contendo:
            - GeoDataFrame com os pontos de interesse identificados
            - Dicionário com estatísticas dos pontos
    """
    logger.info("Identificando nascentes, fozes e junções na rede hidrográfica...")
    
    # Verificar se existem dados válidos
    if gdf.empty:
        logger.warning("GeoDataFrame vazio. Impossível identificar pontos de interesse.")
        return gpd.GeoDataFrame(), {}
    
    # Extrair todos os pontos de início e fim dos trechos
    start_points = []
    end_points = []
    for idx, row in gdf.iterrows():
        if isinstance(row.geometry, LineString):
            start_points.append((row.geometry.coords[0], idx))
            end_points.append((row.geometry.coords[-1], idx))
    
    # Criar dicionários para contar ocorrências de cada ponto
    point_count = {}
    point_to_feature = {}
    
    for p, idx in start_points:
        point_count[p] = point_count.get(p, 0) + 1
        if p not in point_to_feature:
            point_to_feature[p] = []
        point_to_feature[p].append(idx)
    
    for p, idx in end_points:
        point_count[p] = point_count.get(p, 0) + 1
        if p not in point_to_feature:
            point_to_feature[p] = []
        point_to_feature[p].append(idx)
    
    # Classificar os pontos:
    # - Nascentes (springs): pontos que aparecem apenas uma vez e são início de trecho
    # - Fozes (outlets): pontos que aparecem apenas uma vez e são fim de trecho
    # - Junções (junctions): pontos que aparecem mais de uma vez
    springs = []
    outlets = []
    junctions = []
    
    for p, count in point_count.items():
        if count == 1:
            # Verificar se é início ou fim
            is_start = any(p == start_points[i][0] for i in range(len(start_points)))
            if is_start:
                springs.append(p)
            else:
                outlets.append(p)
        else:
            junctions.append(p)
    
    logger.info(f"Identificados {len(springs)} nascentes, {len(outlets)} fozes e {len(junctions)} junções")
    
    # Criar GeoDataFrame com os pontos identificados
    point_data = []
    
    for p in springs:
        point_data.append({
            'geometry': Point(p),
            'type': 'spring',
            'connected_features': len(point_to_feature.get(p, [])),
            'feature_ids': ','.join(map(str, point_to_feature.get(p, [])))
        })
    
    for p in outlets:
        point_data.append({
            'geometry': Point(p),
            'type': 'outlet',
            'connected_features': len(point_to_feature.get(p, [])),
            'feature_ids': ','.join(map(str, point_to_feature.get(p, [])))
        })
    
    for p in junctions:
        point_data.append({
            'geometry': Point(p),
            'type': 'junction',
            'connected_features': len(point_to_feature.get(p, [])),
            'feature_ids': ','.join(map(str, point_to_feature.get(p, [])))
        })
    
    # Criar GeoDataFrame
    point_gdf = gpd.GeoDataFrame(point_data, crs=gdf.crs)
    
    # Log detalhado para depuração
    logger.info(f"GeoDataFrame de nós criado com {len(point_gdf)} pontos: {point_gdf['type'].value_counts().to_dict()}")
    
    # Calcular estatísticas
    stats = {
        'n_springs': len(springs),
        'n_outlets': len(outlets),
        'n_junctions': len(junctions),
        'bifurcation_ratio': calculate_bifurcation_ratio(len(springs), len(junctions), len(outlets)),
        'drainage_pattern': identify_drainage_pattern(point_gdf, gdf),
        'connectivity_index': len(junctions) / max(1, len(springs))
    }
    
    return point_gdf, stats

def calculate_bifurcation_ratio(n_springs, n_junctions, n_outlets):
    """
    Calcula a razão de bifurcação da rede hidrográfica.
    Razão de Bifurcação = (N nascentes + N junções) / (N junções + N fozes)
    
    Args:
        n_springs (int): Número de nascentes
        n_junctions (int): Número de junções
        n_outlets (int): Número de fozes
        
    Returns:
        float: Razão de bifurcação
    """
    if n_junctions + n_outlets == 0:
        return 0
    
    return (n_springs + n_junctions) / (n_junctions + n_outlets)

def identify_drainage_pattern(point_gdf, line_gdf):
    """
    Identifica o padrão de drenagem com base na configuração da rede.
    
    Args:
        point_gdf (gpd.GeoDataFrame): GeoDataFrame com pontos de nascentes, fozes e junções
        line_gdf (gpd.GeoDataFrame): GeoDataFrame com trechos de drenagem
        
    Returns:
        Dict: Dicionário com informações sobre o padrão de drenagem
    """
    # Simplificação: usar proporções e ângulos para tentar identificar o padrão
    
    # Extrair junções
    junctions = point_gdf[point_gdf['type'] == 'junction']
    
    # Caso não haja junções suficientes para análise
    if len(junctions) < 5:
        return {'pattern': 'indefinido', 'confidence': 0.0}
    
    # Calcular estatísticas de angularidade
    angles = []
    try:
        # Reprojetar para sistema métrico se necessário
        if line_gdf.crs.is_geographic:
            metric_gdf = line_gdf.to_crs(epsg=3857)  # Web Mercator
        else:
            metric_gdf = line_gdf
            
        # Calcular orientação das linhas (azimute)
        for _, row in metric_gdf.iterrows():
            if isinstance(row.geometry, LineString) and len(row.geometry.coords) >= 2:
                start = row.geometry.coords[0]
                end = row.geometry.coords[-1]
                # Calcular azimute
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                angle = np.degrees(np.arctan2(dy, dx)) % 180
                angles.append(angle)
        
        # Analisar distribuição dos ângulos
        if not angles:
            return {'pattern': 'indefinido', 'confidence': 0.0}
            
        angles = np.array(angles)
        
        # Histograma de ângulos
        hist, bins = np.histogram(angles, bins=18, range=(0, 180))
        
        # Normalizar
        hist = hist / hist.sum()
        
        # Verificar padrões característicos
        
        # Padrão dendrítico: distribuição mais uniforme de ângulos
        dendritc_score = 1 - np.std(hist) * 10  # Alta uniformidade = baixo desvio padrão
        
        # Padrão paralelo: picos em ângulos similares
        parallel_score = np.max(hist) * 2
        
        # Padrão retangular: picos em ângulos próximos a 0, 90 e 180 graus
        rectangular_bins = np.where((bins[:-1] < 10) | (abs(bins[:-1] - 90) < 10) | (abs(bins[:-1] - 180) < 10))[0]
        rectangular_score = np.sum(hist[rectangular_bins]) * 1.5
        
        # Padrão radial: não pode ser facilmente identificado apenas com ângulos
        # Precisa verificar se convergem para um ponto central
        radial_score = 0.0
        
        # Escolher o padrão com maior pontuação
        scores = {
            'dendrítico': dendritc_score,
            'paralelo': parallel_score,
            'retangular': rectangular_score,
            'radial': radial_score
        }
        
        best_pattern = max(scores.items(), key=lambda x: x[1])
        
        return {
            'pattern': best_pattern[0],
            'confidence': min(1.0, best_pattern[1]),
            'angles_stats': {
                'mean': float(np.mean(angles)),
                'std': float(np.std(angles)),
                'dominant_angles': [float(bins[i]) for i in np.argsort(hist)[-3:]]
            }
        }
    except Exception as e:
        logger.warning(f"Erro ao calcular padrão de drenagem: {str(e)}")
        return {'pattern': 'indefinido', 'confidence': 0.0}

def analyze_water_quality(data):
    """
    Analisa parâmetros de qualidade da água com base nos dados disponíveis.
    Se dados específicos de qualidade não estiverem disponíveis, faz análises 
    morfométricas como proxy.
    
    Args:
        data (Dict[str, gpd.GeoDataFrame]): Dicionário com datasets hidrográficos
        
    Returns:
        Dict: Dicionário com métricas de qualidade da água
    """
    logger.info("Analisando qualidade da água com base em parâmetros morfométricos...")
    
    results = {}
    
    # Análise de sinuosidade como indicador de dinâmica fluvial
    if 'trecho_drenagem' in data and 'sinuosity' in data['trecho_drenagem'].columns:
        sinuosity = data['trecho_drenagem']['sinuosity'].dropna()
        if not sinuosity.empty:
            results['sinuosity_analysis'] = {
                'mean': float(sinuosity.mean()),
                'high_sinuosity_percentage': float((sinuosity > 1.5).mean() * 100),
                'sinuosity_quality_index': calculate_sinuosity_quality_index(sinuosity)
            }
    
    # Densidade de drenagem como indicador de permeabilidade e escoamento
    drainage_density = calculate_drainage_density(data)
    if drainage_density:
        results['drainage_analysis'] = {
            'drainage_density': drainage_density,
            'drainage_quality_index': calculate_drainage_quality_index(drainage_density.get('total', 0))
        }
    
    # Análise de rede e conectividade como indicador de integridade do sistema
    if 'trecho_drenagem' in data and 'betweenness' in data['trecho_drenagem'].columns:
        betweenness = data['trecho_drenagem']['betweenness'].dropna()
        if not betweenness.empty:
            results['connectivity_analysis'] = {
                'mean_betweenness': float(betweenness.mean()),
                'connectivity_quality_index': calculate_connectivity_quality_index(data['trecho_drenagem'])
            }
    
    # Análise de padrão de drenagem
    if 'nodes' in data and not data['nodes'].empty:
        try:
            drainage_pattern = identify_drainage_pattern(data['nodes'], data['trecho_drenagem'])
            results['drainage_pattern'] = drainage_pattern
        except Exception as e:
            logger.warning(f"Erro ao analisar padrão de drenagem: {str(e)}")
    
    return results

def calculate_sinuosity_quality_index(sinuosity_series):
    """
    Calcula um índice de qualidade baseado na sinuosidade.
    Maior sinuosidade geralmente indica rios mais naturais e preservados.
    
    Args:
        sinuosity_series (pd.Series): Série com valores de sinuosidade
        
    Returns:
        float: Índice de qualidade (0-1)
    """
    # Valores de referência
    # Sinuosidade < 1.05: rios retificados/canalizados (baixa qualidade)
    # Sinuosidade > 1.5: rios naturais (alta qualidade)
    
    # Calcular proporção de trechos naturais vs. retificados
    low_sinuosity = (sinuosity_series < 1.05).mean()
    high_sinuosity = (sinuosity_series > 1.5).mean()
    
    # Índice de qualidade
    quality_index = (high_sinuosity * 0.7) + ((1 - low_sinuosity) * 0.3)
    
    return min(1.0, max(0.0, float(quality_index)))

def calculate_drainage_quality_index(drainage_density):
    """
    Calcula um índice de qualidade baseado na densidade de drenagem.
    
    Args:
        drainage_density (float): Densidade de drenagem em km/km²
        
    Returns:
        float: Índice de qualidade (0-1)
    """
    # Valores de referência baseados na literatura
    # < 0.5 km/km²: densidade muito baixa
    # 0.5-1.5 km/km²: densidade baixa
    # 1.5-2.5 km/km²: densidade média
    # 2.5-3.5 km/km²: densidade alta
    # > 3.5 km/km²: densidade muito alta
    
    if drainage_density <= 0.5:
        return 0.2
    elif drainage_density <= 1.5:
        return 0.4
    elif drainage_density <= 2.5:
        return 0.6
    elif drainage_density <= 3.5:
        return 0.8
    else:
        return 1.0

def calculate_connectivity_quality_index(gdf):
    """
    Calcula um índice de qualidade baseado na conectividade da rede hidrográfica.
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame com dados de rede hidrográfica
        
    Returns:
        float: Índice de qualidade (0-1)
    """
    try:
        # Verificar se temos informações de rede
        if hasattr(gdf, 'attrs') and 'network_info' in gdf.attrs:
            network_info = gdf.attrs['network_info']
            
            # Calcular componentes conectados vs total de feições
            fragmentation = network_info.get('n_components', 1) / max(1, len(gdf))
            
            # Calcular densidade da rede
            network_density = network_info.get('network_density', 0)
            
            # Calcular tamanho do maior componente vs total
            largest_component_ratio = network_info.get('largest_component_size', 0) / max(1, len(gdf))
            
            # Combinação ponderada dos fatores
            quality_index = (
                (1 - fragmentation) * 0.4 +
                network_density * 10 * 0.3 +
                largest_component_ratio * 0.3
            )
            
            return min(1.0, max(0.0, float(quality_index)))
        else:
            return 0.5  # Valor médio caso não haja dados suficientes
    except Exception as e:
        logger.warning(f"Erro ao calcular índice de conectividade: {str(e)}")
        return 0.5

def extract_elevation_for_points(gdf, dem):
    """
    Extrai dados de elevação para features pontuais (nascentes, fozes, pontos de drenagem).
    
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
    
    # Derivar atributos topográficos adicionais
    
    # Adicionar informação de terreno local (valley bottom, hillslope, etc.)
    # Isso requer análise do entorno, que poderíamos implementar usando
    # uma janela de busca no DEM ao redor de cada ponto
    
    return result

def extract_elevation_for_polygons(gdf, dem):
    """
    Extrai dados de elevação para features poligonais (áreas de drenagem).
    
    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame com features poligonais
        dem (rasterio.io.DatasetReader): Modelo Digital de Elevação
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame com atributos de elevação adicionados
    """
    logger.info("Extraindo dados de elevação para polígonos...")
    
    # Verificar se temos dados válidos
    if gdf.empty or dem is None:
        logger.warning("Dados insuficientes para extrair elevação para polígonos.")
        return gdf
    
    # Criar uma cópia para evitar avisos
    result = gdf.copy()
    
    # Reprojetar para o mesmo CRS do DEM se necessário
    if result.crs != dem.crs:
        logger.info(f"Reprojetando de {result.crs} para {dem.crs}")
        result = result.to_crs(dem.crs)
    
    # Arrays para armazenar resultados
    elevation_min = []
    elevation_max = []
    elevation_mean = []
    elevation_range = []
    elevation_median = []
    elevation_std = []
    hypsometric_integral = []
    ruggedness_index = []
    
    # Para cada polígono, extrair estatísticas de elevação
    for idx, row in result.iterrows():
        try:
            # Converter polígono para máscara de raster
            geom = row.geometry
            if geom.is_empty:
                # Pular geometrias vazias
                elevation_min.append(None)
                elevation_max.append(None)
                elevation_mean.append(None)
                elevation_range.append(None)
                elevation_median.append(None)
                elevation_std.append(None)
                hypsometric_integral.append(None)
                ruggedness_index.append(None)
                continue
            
            # Obter os dados de elevação dentro do polígono
            geom_bounds = geom.bounds
            window = from_bounds(*geom_bounds, dem.transform)
            
            # Ler os dados do DEM na janela definida
            window_data = dem.read(1, window=window)
            window_transform = rasterio.windows.transform(window, dem.transform)
            
            # Criar máscara para o polígono
            mask_arr = geometry_mask(
                [geom], 
                out_shape=window_data.shape, 
                transform=window_transform, 
                invert=True
            )
            
            # Aplicar máscara e obter valores válidos
            masked_data = window_data[mask_arr]
            valid_data = masked_data[masked_data != dem.nodata]
            
            if len(valid_data) > 0:
                # Calcular estatísticas básicas
                min_elev = float(np.min(valid_data))
                max_elev = float(np.max(valid_data))
                mean_elev = float(np.mean(valid_data))
                median_elev = float(np.median(valid_data))
                std_elev = float(np.std(valid_data))
                range_elev = max_elev - min_elev
                
                # Calcular integral hipsométrica (volume relativo)
                # HI = (elevação média - elevação mínima) / (elevação máxima - elevação mínima)
                if range_elev > 0:
                    hi = (mean_elev - min_elev) / range_elev
                else:
                    hi = 0.5
                
                # Índice de rugosidade do terreno (variação de elevação por área)
                # Simplificação: usando desvio padrão da elevação
                tri = std_elev
                
                elevation_min.append(min_elev)
                elevation_max.append(max_elev)
                elevation_mean.append(mean_elev)
                elevation_range.append(range_elev)
                elevation_median.append(median_elev)
                elevation_std.append(std_elev)
                hypsometric_integral.append(hi)
                ruggedness_index.append(tri)
            else:
                # Sem dados válidos dentro do polígono
                elevation_min.append(None)
                elevation_max.append(None)
                elevation_mean.append(None)
                elevation_range.append(None)
                elevation_median.append(None)
                elevation_std.append(None)
                hypsometric_integral.append(None)
                ruggedness_index.append(None)
        except Exception as e:
            logger.warning(f"Erro ao extrair elevação para polígono {idx}: {str(e)}")
            elevation_min.append(None)
            elevation_max.append(None)
            elevation_mean.append(None)
            elevation_range.append(None)
            elevation_median.append(None)
            elevation_std.append(None)
            hypsometric_integral.append(None)
            ruggedness_index.append(None)
    
    # Adicionar colunas ao GeoDataFrame
    result['elevation_min'] = elevation_min
    result['elevation_max'] = elevation_max
    result['elevation_mean'] = elevation_mean
    result['elevation_range'] = elevation_range
    result['elevation_median'] = elevation_median
    result['elevation_std'] = elevation_std
    result['hypsometric_integral'] = hypsometric_integral
    result['ruggedness_index'] = ruggedness_index
    
    # Classificar bacias por relevo
    if all(col in result.columns for col in ['hypsometric_integral', 'ruggedness_index']):
        # Classificar bacias com base na integral hipsométrica
        # HI > 0.6: Bacia jovem/montanhosa
        # 0.35 <= HI <= 0.6: Bacia madura/equilibrada
        # HI < 0.35: Bacia antiga/plana
        hi_conditions = [
            (result['hypsometric_integral'] < 0.35),
            (result['hypsometric_integral'] >= 0.35) & (result['hypsometric_integral'] <= 0.6),
            (result['hypsometric_integral'] > 0.6)
        ]
        hi_classes = ['Antiga/Plana', 'Madura/Equilibrada', 'Jovem/Montanhosa']
        result['basin_age_class'] = np.select(hi_conditions, hi_classes, default='Indeterminada')
        
        # Classificar por rugosidade do terreno
        # Usando estatísticas da distribuição do índice de rugosidade na amostra
        if len([x for x in ruggedness_index if x is not None]) > 0:
            valid_tri = np.array([x for x in ruggedness_index if x is not None])
            tri_q1 = np.percentile(valid_tri, 25)
            tri_q3 = np.percentile(valid_tri, 75)
            
            tri_conditions = [
                (result['ruggedness_index'] < tri_q1),
                (result['ruggedness_index'] >= tri_q1) & (result['ruggedness_index'] <= tri_q3),
                (result['ruggedness_index'] > tri_q3)
            ]
            tri_classes = ['Baixa Rugosidade', 'Rugosidade Média', 'Alta Rugosidade']
            result['terrain_ruggedness_class'] = np.select(tri_conditions, tri_classes, default='Indeterminada')
    
    # Estatísticas básicas
    valid_elevations = [e for e in elevation_mean if e is not None]
    if valid_elevations:
        logger.info(f"Elevação média das bacias: {np.mean(valid_elevations):.2f} m")
        logger.info(f"Amplitude média de elevação das bacias: {np.mean([e for e in elevation_range if e is not None]):.2f} m")
    
    valid_hi = [h for h in hypsometric_integral if h is not None]
    if valid_hi:
        logger.info(f"Integral hipsométrica média: {np.mean(valid_hi):.2f}")
    
    return result

def calculate_stream_power_index(data):
    """
    Calcula o índice de potência do fluxo (Stream Power Index) e outros índices
    topográficos derivados do DEM.
    
    Args:
        data (Dict[str, gpd.GeoDataFrame]): Dicionário com dados hidrográficos
        
    Returns:
        Dict[str, gpd.GeoDataFrame]: Dados hidrográficos com índices adicionados
    """
    logger.info("Calculando índices topográficos avançados...")
    
    result = data.copy()
    
    # Calcular SPI para trechos de drenagem se tiverem dados de elevação e ordem de Strahler
    if 'trecho_drenagem' in result and not result['trecho_drenagem'].empty:
        trecho_gdf = result['trecho_drenagem']
        
        if 'slope_pct' in trecho_gdf.columns and 'strahler_order' in trecho_gdf.columns:
            # Verificar que temos dados válidos
            valid_rows = trecho_gdf[trecho_gdf['slope_pct'].notna() & trecho_gdf['strahler_order'].notna()]
            
            if not valid_rows.empty:
                logger.info("Calculando Stream Power Index (SPI) para trechos de drenagem...")
                
                # SPI = area * tan(slope)
                # Usando ordem de Strahler como proxy para área de contribuição
                # e convertendo slope_pct para radianos
                trecho_gdf['stream_power_index'] = np.nan
                
                for idx, row in valid_rows.iterrows():
                    # Converter declividade de porcentagem para radianos
                    slope_rad = np.arctan(row['slope_pct'] / 100)
                    # Calcular SPI usando um fator de escala para a ordem de Strahler
                    spi = (2 ** row['strahler_order']) * np.tan(slope_rad)
                    trecho_gdf.at[idx, 'stream_power_index'] = spi
                
                # Classificar trechos por potencial erosivo
                spi_values = trecho_gdf['stream_power_index'].dropna()
                if not spi_values.empty:
                    spi_q1 = spi_values.quantile(0.25)
                    spi_q3 = spi_values.quantile(0.75)
                    
                    conditions = [
                        (trecho_gdf['stream_power_index'] < spi_q1),
                        (trecho_gdf['stream_power_index'] >= spi_q1) & (trecho_gdf['stream_power_index'] <= spi_q3),
                        (trecho_gdf['stream_power_index'] > spi_q3)
                    ]
                    classes = ['Baixo Potencial Erosivo', 'Médio Potencial Erosivo', 'Alto Potencial Erosivo']
                    trecho_gdf['erosion_potential'] = np.select(conditions, classes, default='Indeterminado')
                
                result['trecho_drenagem'] = trecho_gdf
                logger.info(f"SPI médio: {trecho_gdf['stream_power_index'].mean():.2f}")
    
    # Calcular índice de proximidade das margens para pontos (caracterização da zona ripária)
    if 'nodes' in result and not result['nodes'].empty and 'trecho_drenagem' in result:
        nodes_gdf = result['nodes']
        trecho_gdf = result['trecho_drenagem']
        
        if not trecho_gdf.empty:
            # Criar buffer ao redor dos trechos para analisar a proximidade das margens
            logger.info("Calculando proximidade das margens para pontos de interesse...")
            
            # Garantir que ambos estão no mesmo CRS e é um CRS projetado (métrico)
            if nodes_gdf.crs and trecho_gdf.crs:
                if nodes_gdf.crs != trecho_gdf.crs:
                    nodes_gdf = nodes_gdf.to_crs(trecho_gdf.crs)
                
                # Se o CRS for geográfico, converter para projetado
                if nodes_gdf.crs.is_geographic:
                    # Usar UTM apropriado para a área
                    nodes_gdf = nodes_gdf.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
                    trecho_gdf = trecho_gdf.to_crs(epsg=31983)
                
                # Criar união de todos os trechos
                rios_unidos = unary_union(trecho_gdf.geometry.tolist())
                
                # Calcular distância de cada ponto para o rio mais próximo
                bank_distances = []
                for idx, row in nodes_gdf.iterrows():
                    # Calcular distância para o rio mais próximo
                    distance = row.geometry.distance(rios_unidos)
                    bank_distances.append(distance)
                
                # Adicionar como atributo
                nodes_gdf['bank_distance'] = bank_distances
                
                # Classificar por zonas ripárias
                # Zonas comuns: 30m, 50m, 100m, 200m
                riparian_conditions = [
                    (nodes_gdf['bank_distance'] <= 30),
                    (nodes_gdf['bank_distance'] > 30) & (nodes_gdf['bank_distance'] <= 100),
                    (nodes_gdf['bank_distance'] > 100) & (nodes_gdf['bank_distance'] <= 200),
                    (nodes_gdf['bank_distance'] > 200)
                ]
                riparian_classes = ['Zona Ripária Imediata', 'Zona Ripária Próxima', 
                                   'Zona Ripária Estendida', 'Fora da Zona Ripária']
                nodes_gdf['riparian_zone'] = np.select(riparian_conditions, riparian_classes, default='Indeterminado')
                
                result['nodes'] = nodes_gdf
                logger.info(f"Distância média até a margem: {nodes_gdf['bank_distance'].mean():.2f} m")
    
    # Calcular Topographic Wetness Index (TWI) para áreas de drenagem
    if 'area_drenagem' in result and not result['area_drenagem'].empty:
        area_gdf = result['area_drenagem']
        
        if all(col in area_gdf.columns for col in ['elevation_range', 'hypsometric_integral']):
            logger.info("Calculando Topographic Wetness Index (TWI) aproximado para bacias...")
            
            # Para um cálculo completo do TWI, precisaríamos da área de contribuição específica e da declividade
            # para cada célula do DEM. Aqui fazemos uma aproximação usando dados da bacia.
            
            # TWI aproximado = ln(area / tan(declividade média))
            # Onde declividade média é estimada a partir da amplitude de elevação e integral hipsométrica
            
            twi_values = []
            for idx, row in area_gdf.iterrows():
                try:
                    # Calcular área em m²
                    if area_gdf.crs.is_geographic:
                        # Converter temporariamente para projeção métrica
                        area_m2 = row.geometry.to_crs(epsg=31983).area
                    else:
                        area_m2 = row.geometry.area
                    
                    # Estimar declividade média
                    if row['elevation_range'] and row['hypsometric_integral']:
                        # Aproximação simples da declividade média usando geometria e amplitude
                        # Raio equivalente da bacia
                        radius = np.sqrt(area_m2 / np.pi)
                        # Declividade aproximada (em radianos)
                        slope_rad = np.arctan(row['elevation_range'] / (2 * radius))
                        # Ajustar com integral hipsométrica (bacias mais jovens/montanhosas têm maior declividade)
                        slope_rad = slope_rad * (1 + row['hypsometric_integral'])
                        
                        # Evitar divisão por zero ou valores muito pequenos
                        min_slope = 0.001  # 0.1% mínimo
                        if slope_rad < min_slope:
                            slope_rad = min_slope
                        
                        # Calcular TWI
                        twi = np.log(area_m2 / np.tan(slope_rad))
                        twi_values.append(twi)
                    else:
                        twi_values.append(None)
                except Exception as e:
                    logger.warning(f"Erro ao calcular TWI para bacia {idx}: {str(e)}")
                    twi_values.append(None)
            
            # Adicionar ao GeoDataFrame
            area_gdf['topographic_wetness_index'] = twi_values
            
            # Classificar áreas por potencial de umidade/saturação
            valid_twi = [t for t in twi_values if t is not None]
            if valid_twi:
                twi_q1 = np.percentile(valid_twi, 25)
                twi_q3 = np.percentile(valid_twi, 75)
                
                twi_conditions = [
                    (area_gdf['topographic_wetness_index'] < twi_q1),
                    (area_gdf['topographic_wetness_index'] >= twi_q1) & (area_gdf['topographic_wetness_index'] <= twi_q3),
                    (area_gdf['topographic_wetness_index'] > twi_q3)
                ]
                twi_classes = ['Baixo Potencial de Umidade', 'Médio Potencial de Umidade', 'Alto Potencial de Umidade']
                area_gdf['moisture_potential'] = np.select(twi_conditions, twi_classes, default='Indeterminado')
                
                result['area_drenagem'] = area_gdf
                logger.info(f"TWI médio: {np.mean(valid_twi):.2f}")
    
    return result

def visualize_hydrography_with_dem(data, dem):
    """
    Gera visualizações da hidrografia sobrepondo com dados de elevação (DEM).
    
    Args:
        data (Dict[str, gpd.GeoDataFrame]): Dicionário com GeoDataFrames contendo dados enriquecidos
        dem (rasterio.DatasetReader): Dados de elevação (DEM)
    """
    logger.info("Gerando visualizações com dados altimétricos...")
    
    try:
        import matplotlib.pyplot as plt
        import rasterio
        import rasterio.plot
        from matplotlib.colors import LightSource
        import numpy as np
        import contextily as ctx
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        # Criar diretório para visualizações
        os.makedirs(VISUALIZATION_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Verificar se temos dados
        if not data or all(gdf.empty for gdf in data.values()):
            logger.warning("Sem dados para visualização com DEM.")
            return
        
        # 1. Visualização 2D com DEM como fundo e rios sobrepostos
        plt.figure(figsize=(14, 10))
    ax = plt.subplot(111)
        
        # Obter bounds para o recorte do DEM
        bounds = None
        for key, gdf in data.items():
            if not gdf.empty:
                if bounds is None:
                    bounds = gdf.total_bounds
                else:
                    bounds = [
                        min(bounds[0], gdf.total_bounds[0]),
                        min(bounds[1], gdf.total_bounds[1]),
                        max(bounds[2], gdf.total_bounds[2]),
                        max(bounds[3], gdf.total_bounds[3])
                    ]
        
        # Adicionar margem aos bounds
        margin = 0.05
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        bounds = [
            bounds[0] - width * margin,
            bounds[1] - height * margin,
            bounds[2] + width * margin,
            bounds[3] + height * margin
        ]
        
        # Plotar o DEM com sombreamento de relevo
        dem_data = dem.read(1)
        dem_mask = dem_data != dem.nodata
        dem_data = np.ma.masked_array(dem_data, ~dem_mask)
        
        # Criar sombreamento de relevo
        ls = LightSource(azdeg=315, altdeg=45)
        dem_shade = ls.hillshade(dem_data, vert_exag=5, dx=dem.res[0], dy=dem.res[1])
        
        # Recortar ao bound de interesse
        row_start, row_end, col_start, col_end = rasterio.transform.rowcol(
            dem.transform, 
            [bounds[0], bounds[2]], 
            [bounds[1], bounds[3]]
        )
        
        # Garantir que os índices estão dentro dos limites
        row_start = max(0, min(row_start, dem_data.shape[0] - 1))
        row_end = max(0, min(row_end, dem_data.shape[0] - 1))
        col_start = max(0, min(col_start, dem_data.shape[1] - 1))
        col_end = max(0, min(col_end, dem_data.shape[1] - 1))
        
        # Recortar os dados
        dem_data_cropped = dem_data[row_start:row_end, col_start:col_end]
        dem_shade_cropped = dem_shade[row_start:row_end, col_start:col_end]
        
        # Criar transform para o recorte
        new_transform = rasterio.transform.from_origin(
            dem.transform * (col_start, row_start),
            dem.res[0],
            dem.res[1]
        )
        
        # Plotar DEM com sombreamento
        cmap = plt.cm.terrain
        rasterio.plot.show(
            dem_data_cropped, 
            transform=new_transform,
            ax=ax, 
            cmap=cmap,
            alpha=0.8
        )
        
        # Sobrepor sombreamento
        ax.imshow(
            dem_shade_cropped,
            extent=(bounds[0], bounds[2], bounds[1], bounds[3]),
            cmap='gray',
            alpha=0.3
        )
        
        # Plotar rios
        if 'rios' in data and not data['rios'].empty:
            # Ordenar por ordem de Strahler (se disponível) para plotar rios principais por último
            if 'strahler_order' in data['rios'].columns:
                rios_sorted = data['rios'].sort_values('strahler_order')
            else:
                rios_sorted = data['rios']
            
            # Definir cores e espessuras com base na sinuosidade
            if 'sinuosity' in rios_sorted.columns:
                # Usar função de cor baseada na sinuosidade
                rios_sorted['color'] = rios_sorted['sinuosity'].apply(get_sinuosity_color)
                rios_sorted.plot(ax=ax, column='color', linewidth=1.5, legend=False)
                
                # Adicionar legenda para sinuosidade
                legend_elements = [
                    Line2D([0], [0], color=get_sinuosity_color(1.0), lw=2, label='Baixa sinuosidade (<1.2)'),
                    Line2D([0], [0], color=get_sinuosity_color(1.5), lw=2, label='Média sinuosidade (1.2-1.5)'),
                    Line2D([0], [0], color=get_sinuosity_color(2.0), lw=2, label='Alta sinuosidade (>1.5)')
                ]
                ax.legend(handles=legend_elements, loc='upper right', title='Sinuosidade')
            else:
                rios_sorted.plot(ax=ax, color='blue', linewidth=1.5)
        
        # Plotar nascentes e fozes
        if 'nodes' in data and not data['nodes'].empty:
            springs = data['nodes'][data['nodes']['type'] == 'spring']
            outlets = data['nodes'][data['nodes']['type'] == 'outlet']
            junctions = data['nodes'][data['nodes']['type'] == 'junction']
            
            if not springs.empty:
                springs.plot(ax=ax, color='green', markersize=30, marker='^', label='Nascentes')
            if not outlets.empty:
                outlets.plot(ax=ax, color='red', markersize=30, marker='v', label='Fozes')
            if not junctions.empty:
                junctions.plot(ax=ax, color='orange', markersize=15, marker='o', label='Junções')
        
        # Plotar massas d'água (lagos, reservatórios)
        if 'areas' in data and not data['areas'].empty:
            data['areas'].plot(ax=ax, color='lightblue', alpha=0.6, edgecolor='blue')
        
        # Adicionar título e legendas
        plt.title('Hidrografia com Modelo Digital de Elevação (DEM)', fontsize=14)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # Adicionar escala de cores para elevação
        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array(dem_data)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
        cbar.set_label('Elevação (m)')
        
        # Salvar a figura
        output_file = os.path.join(VISUALIZATION_DIR, f'hidrografia_com_dem_{timestamp}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
        logger.info(f"Visualização 2D com DEM salva em: {output_file}")
        
        # 2. Visualização de perfil longitudinal dos rios principais
        if 'rios' in data and not data['rios'].empty and 'elevation' in data['rios'].columns:
            # Identificar rios principais (maior ordem de Strahler ou maior comprimento)
            if 'strahler_order' in data['rios'].columns:
                max_order = data['rios']['strahler_order'].max()
                main_rivers = data['rios'][data['rios']['strahler_order'] >= max(2, max_order-1)]
            else:
                # Usar os 5 maiores rios por comprimento
                data['rios']['length'] = data['rios'].geometry.length
                main_rivers = data['rios'].sort_values('length', ascending=False).head(5)
            
            if len(main_rivers) > 0:
                plt.figure(figsize=(14, 8))
                
                # Plotar perfil para cada rio principal
                for idx, river in main_rivers.iterrows():
                    if isinstance(river.geometry, LineString) and 'elevation' in river:
                        # Extrai pontos ao longo da linha
                        points = np.array(river.geometry.coords)
                        # Calcular distância acumulada
                        distances = [0]
                        for i in range(1, len(points)):
                            distances.append(distances[i-1] + ((points[i][0] - points[i-1][0])**2 + 
                                                            (points[i][1] - points[i-1][1])**2)**0.5)
                        
                        # Converter para km
                        distances = [d/1000 for d in distances]
                        
                        # Plotar perfil
                        river_name = f"Rio {idx}" if 'name' not in river else river['name']
                        plt.plot(distances, river.elevation, label=river_name)
                
                plt.title('Perfil Longitudinal dos Rios Principais', fontsize=14)
                plt.xlabel('Distância (km)')
                plt.ylabel('Elevação (m)')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Salvar a figura
                output_file = os.path.join(VISUALIZATION_DIR, f'perfil_longitudinal_{timestamp}.png')
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Visualização de perfil longitudinal salva em: {output_file}")
        
        # 3. Mapa 3D usando o DEM e a hidrografia
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Criar figura 3D
            fig = plt.figure(figsize=(14, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            # Preparar grid de elevação
            if dem_data_cropped.shape[0] > 500 or dem_data_cropped.shape[1] > 500:
                # Reduzir a resolução para melhor desempenho
                factor = max(1, min(dem_data_cropped.shape) // 500)
                dem_small = dem_data_cropped[::factor, ::factor]
            else:
                dem_small = dem_data_cropped
                
            # Criar grid x,y
            ny, nx = dem_small.shape
            x = np.linspace(bounds[0], bounds[2], nx)
            y = np.linspace(bounds[1], bounds[3], ny)
            xv, yv = np.meshgrid(x, y)
            
            # Plotar superfície 3D
            surf = ax.plot_surface(xv, yv, dem_small, cmap='terrain', alpha=0.8,
                                  linewidth=0, antialiased=True, shade=True)
            
            # Plotar rios em 3D
            if 'rios' in data and not data['rios'].empty:
                for idx, river in data['rios'].iterrows():
                    if isinstance(river.geometry, LineString):
                        # Obter coordenadas
                        xs, ys = river.geometry.xy
                        
                        # Obter elevação para cada ponto
                        if 'elevation' in river:
                            # Se já tiver elevação calculada
                            zs = river.elevation
                        else:
                            # Interpolar do DEM
                            zs = []
                            for i in range(len(xs)):
                                # Converter coordenadas para índices no raster
                                row, col = rasterio.transform.rowcol(
                                    dem.transform, [xs[i]], [ys[i]])
                                
                                # Verificar se os índices são válidos
                                if (0 <= row[0] < dem_data.shape[0] and 
                                    0 <= col[0] < dem_data.shape[1]):
                                    z = dem_data[row[0], col[0]]
                                    if np.ma.is_masked(z):
                                        z = 0  # Usar valor padrão para dados mascarados
                                else:
                                    z = 0
                                
                                zs.append(z)
                        
                        # Plotar linha 3D
                        ax.plot(xs, ys, zs, color='blue', linewidth=2)
            
            # Configurar visualização
            ax.set_title('Visualização 3D da Hidrografia com Elevação', fontsize=14)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Elevação (m)')
            
            # Ajustar ângulo de visualização
            ax.view_init(elev=30, azim=225)
            
            # Adicionar colorbar
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Elevação (m)')
            
            # Salvar a figura
            output_file = os.path.join(VISUALIZATION_DIR, f'hidrografia_3d_{timestamp}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Visualização 3D salva em: {output_file}")
            
        except Exception as e:
            logger.warning(f"Erro ao criar visualização 3D: {str(e)}")
        
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações com DEM: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("Visualizações com dados altimétricos concluídas")

def setup_logging():
    """
    Configura o sistema de logging para o módulo de hidrografia.
    """
    # Formato do log
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    # Configurar logging básico
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Definir níveis para outros loggers (opcional)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('rasterio').setLevel(logging.WARNING)
    logging.getLogger('fiona').setLevel(logging.WARNING)
    
    logger.info("Sistema de logging configurado")

def main():
    """
    Função principal que executa o fluxo de trabalho completo de enriquecimento de dados hidrográficos.
    """
    logger.info("=== Iniciando processamento de enriquecimento de dados hidrográficos ===")
    start_time = time.time()
    
    # Dicionário para armazenar caminhos de visualizações
    viz_paths = {}
    
    try:
        # 1. Carregar dados processados
        logger.info("Carregando dados hidrográficos...")
        data = load_data()
        if not data:
            logger.error("Não foi possível carregar os dados hidrográficos")
            return None
        
        # 2. Carregar DEM (dados de elevação)
        logger.info("Carregando Modelo Digital de Elevação (DEM)...")
        dem = None
        try:
            if os.path.exists(DEM_FILE):
                dem = load_dem()
                logger.info("DEM carregado com sucesso")
            else:
                logger.warning(f"Arquivo DEM não encontrado: {DEM_FILE}")
        except Exception as e:
            logger.error(f"Erro ao carregar DEM: {str(e)}")
        
        # 3. Enriquecer dados
        logger.info("Iniciando processo de enriquecimento de dados...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            enriched_data = enrich_hydrographic_data(data, dem)
            logger.info("Enriquecimento de dados concluído com sucesso")
            
            # Identificar e adicionar nós (nascentes, fozes e junções)
            nodes_gdf, node_stats = identify_nodes_and_endpoints(enriched_data['trecho_drenagem'])
            enriched_data['nodes'] = nodes_gdf
            enriched_data['trecho_drenagem'].attrs['node_stats'] = node_stats
            
            logger.info(f"Adicionados {len(nodes_gdf)} nós ao dicionário de dados. Tipos: {nodes_gdf['type'].value_counts().to_dict() if not nodes_gdf.empty else 'Nenhum'}")
        except Exception as e:
            logger.error(f"Erro no processo de enriquecimento: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
        # 4. Salvar dados enriquecidos
        logger.info("Salvando dados enriquecidos...")
        output_file = os.path.join(OUTPUT_DIR, f"hidrografia_enriched_{timestamp}.gpkg")
        
        try:
            save_enriched_data(enriched_data, output_file)
            logger.info(f"Dados enriquecidos salvos em: {output_file}")
        except Exception as e:
            logger.error(f"Erro ao salvar dados enriquecidos: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 5. Gerar visualizações
        logger.info("Gerando visualizações...")
        
        try:
            viz_paths = generate_visualizations(enriched_data, timestamp)
            
            # Mostrar estatísticas sobre visualizações
            successful_viz = sum(1 for path in viz_paths.values() if os.path.exists(path))
            logger.info(f"Visualizações geradas: {successful_viz}/{len(viz_paths)} concluídas com sucesso")
            logger.info(f"Diretório de visualizações: {VISUALIZATION_DIR}")
        except Exception as e:
            logger.error(f"Erro ao gerar visualizações: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 6. Gerar relatório de qualidade
        logger.info("Gerando relatório de qualidade...")
        report_file = os.path.join(REPORT_DIR, f"hidrografia_enrichment_report_{timestamp}.json")
        
        try:
            report = generate_quality_report(enriched_data, output_file, viz_paths)
            
            # Salvar relatório
            os.makedirs(os.path.dirname(report_file), exist_ok=True)
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, cls=NpEncoder)
            
            logger.info(f"Relatório de qualidade gerado em: {report_file}")
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 7. Calcular tempo de execução
        elapsed_time = time.time() - start_time
        logger.info(f"=== Processamento concluído em {elapsed_time:.2f} segundos ===")
        logger.info(f"Dados enriquecidos: {output_file}")
        logger.info(f"Relatório: {report_file}")
        logger.info(f"Visualizações: {', '.join(viz_paths.values())}")
        
        return {
            "enriched_data": enriched_data,
            "output_file": output_file,
            "report_file": report_file,
            "visualization_paths": viz_paths
        }
    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}")
        logger.error(traceback.format_exc())
        elapsed_time = time.time() - start_time
        logger.info(f"=== Processamento interrompido após {elapsed_time:.2f} segundos ===")
        return None

# Adicionar args ao script para controlar o comportamento da linha de comando
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Processar e enriquecer dados hidrográficos')
    parser.add_argument('--skip-analysis', action='store_true', help='Pular análises complexas (será executada na nuvem posteriormente)')
    
    args = parser.parse_args()
    
    # Executar main
    main() 