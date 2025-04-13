#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para enriquecimento avançado de dados de edificações (buildings) com análises
morfológicas, padrões de clusters, relações com o sistema viário, e modelagem 3D.

Este script estende as funcionalidades do módulo existente:
- src/preprocessing/buildings.py: Processamento básico dos dados de edificações

Autor: Pesquisador em Análise Espacial e Ciência de Dados
Data: Abril/2025
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import contextily as ctx
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, box
from shapely.ops import unary_union, nearest_points
import rasterio
from rasterio.mask import mask
from shapely.geometry import mapping
import json
import datetime
import time
import logging
import argparse
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor
import psutil
import math
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import networkx as nx
import folium
from folium.plugins import MarkerCluster, HeatMap
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure  # Para análise de morfologia

# Importar numba para aceleração
try:
    import numba
    from numba import jit, prange
    HAS_NUMBA = True
    print("Numba encontrado e habilitado para otimização.")
except ImportError:
    HAS_NUMBA = False
    warnings.warn("Numba não encontrado. Algumas otimizações estarão desabilitadas.")

# Tentar importar pacotes opcionais para análises mais avançadas
try:
    import torch
    import torchvision
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    warnings.warn("PyTorch não encontrado. Algumas funcionalidades estarão desabilitadas.")

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("enhanced_buildings_enrichment.log")
    ]
)
logger = logging.getLogger("enhanced_buildings")

# Classe para serialização JSON de tipos NumPy
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
            return obj.to_dict()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

# Definição de diretórios e arquivos
WORKSPACE_DIR = r"F:\TESE_MESTRADO\geoprocessing"
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
RAW_DIR = os.path.join(WORKSPACE_DIR, 'data', 'raw')
ENRICHED_DATA_DIR = os.path.join(WORKSPACE_DIR, 'data', 'enriched_data')
OUTPUT_DIR = ENRICHED_DATA_DIR
REPORT_DIR = r"F:\TESE_MESTRADO\geoprocessing\src\enriched_data\buildings\quality_report"
VISUALIZATION_DIR = r"F:\TESE_MESTRADO\geoprocessing\outputs\visualize_enriched_data\buildings"

# Arquivos
BUILDINGS_FILE = os.path.join(INPUT_DIR, 'buildings_processed.gpkg')
ROADS_FILE = os.path.join(INPUT_DIR, 'roads_processed.gpkg')
DEM_FILE = r"F:\TESE_MESTRADO\geoprocessing\data\raw\dem.tif"
LANDUSE_FILE = os.path.join(INPUT_DIR, 'landuse_processed.gpkg')

# Configuração de paralelização
# Usar no máximo 75% dos núcleos disponíveis para evitar sobrecarga do sistema
N_CORES = max(1, min(int(psutil.cpu_count(logical=False) * 0.75), 8))
CHUNK_SIZE = 1000

# Garantir que os diretórios de saída existam
for directory in [ENRICHED_DATA_DIR, REPORT_DIR, VISUALIZATION_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Diretório garantido: {directory}")

# Métodos para paralelização
def parallelize_dataframe(df, func, n_cores=N_CORES):
    """Processa um DataFrame em paralelo usando multiprocessing."""
    if len(df) < 1000:  # Para pequenos DataFrames, não vale a pena paralelizar
        return func(df)
        
    df_split = np.array_split(df, n_cores)
    pool = ProcessPoolExecutor(max_workers=n_cores)
    df = pd.concat(list(pool.map(func, df_split)))
    pool.shutdown()
    return df

# Funções para carregamento de dados
def load_data():
    """
    Carrega os dados de edificações processados e datasets complementares.
    
    Returns:
        dict: Dicionário contendo os dados carregados
    """
    data = {}
    
    # Carregar edificações
    logger.info(f"Carregando dados de edificações: {BUILDINGS_FILE}")
    if os.path.exists(BUILDINGS_FILE):
        data['buildings'] = gpd.read_file(BUILDINGS_FILE)
        logger.info(f"Carregadas {len(data['buildings'])} feições de edificações")
    else:
        logger.error(f"Arquivo não encontrado: {BUILDINGS_FILE}")
        sys.exit(1)
    
    # Carregar estradas se existir
    if os.path.exists(ROADS_FILE):
        logger.info(f"Carregando dados de estradas: {ROADS_FILE}")
        try:
            data['roads'] = gpd.read_file(ROADS_FILE)
            logger.info(f"Carregadas {len(data['roads'])} feições de estradas")
        except Exception as e:
            logger.error(f"Erro ao carregar estradas: {str(e)}")
            data['roads'] = None
    else:
        logger.warning(f"Arquivo de estradas não encontrado: {ROADS_FILE}")
        data['roads'] = None
    
    # Carregar DEM se existir
    if os.path.exists(DEM_FILE):
        logger.info(f"Carregando Modelo Digital de Elevação: {DEM_FILE}")
        try:
            data['dem'] = rasterio.open(DEM_FILE)
            logger.info(f"DEM carregado. Resolução: {data['dem'].res}")
        except Exception as e:
            logger.error(f"Erro ao carregar DEM: {str(e)}")
            data['dem'] = None
    else:
        logger.warning(f"Arquivo DEM não encontrado: {DEM_FILE}")
        data['dem'] = None
    
    # Carregar uso do solo se existir
    if os.path.exists(LANDUSE_FILE):
        logger.info(f"Carregando dados de uso do solo: {LANDUSE_FILE}")
        try:
            data['landuse'] = gpd.read_file(LANDUSE_FILE)
            logger.info(f"Carregadas {len(data['landuse'])} feições de uso do solo")
        except Exception as e:
            logger.error(f"Erro ao carregar uso do solo: {str(e)}")
            data['landuse'] = None
    else:
        logger.warning(f"Arquivo de uso do solo não encontrado: {LANDUSE_FILE}")
        data['landuse'] = None
        
    return data

def extract_building_height(buildings_gdf):
    """
    Extrai informações de altura dos edifícios a partir de atributos disponíveis.
    
    Args:
        buildings_gdf (gpd.GeoDataFrame): GeoDataFrame com os edifícios
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame atualizado com altura padronizada
    """
    logger.info("Extraindo e padronizando informações de altura dos edifícios...")
    
    # Criar cópia para não alterar os dados originais
    result = buildings_gdf.copy()
    
    # Verificar se 'height' já existe
    if 'height' not in result.columns:
        result['height'] = np.nan
    
    # Verificar se 'levels' já existe
    if 'levels' not in result.columns:
        result['levels'] = np.nan
    
    # Processar other_tags para extrair informações de altura e níveis
    if 'other_tags' in result.columns:
        # Expressões para buscar no campo other_tags
        height_patterns = ['height', 'building:height']
        levels_patterns = ['building:levels', 'levels']
        
        # Extrair height
        for idx, row in tqdm(result.iterrows(), total=len(result), desc="Processando alturas"):
            # Pular se já temos altura
            if pd.notna(row['height']) and row['height'] > 0:
                continue
                
            if pd.notna(row['other_tags']):
                # Procurar por padrões de altura
                for pattern in height_patterns:
                    if f'"{pattern}"=>' in row['other_tags']:
                        try:
                            # Extrair valor
                            parts = row['other_tags'].split(f'"{pattern}"=>')
                            if len(parts) > 1:
                                height_str = parts[1].split(',')[0].strip('"\'')
                                
                                # Remover unidades (m, ft, etc.)
                                height_str = ''.join(c for c in height_str if c.isdigit() or c == '.')
                                
                                # Converter para float
                                height_value = float(height_str)
                                
                                # Atualizar coluna
                                result.at[idx, 'height'] = height_value
                                break
                        except:
                            continue
                        
                # Procurar por padrões de níveis
                for pattern in levels_patterns:
                    if f'"{pattern}"=>' in row['other_tags']:
                        try:
                            # Extrair valor
                            parts = row['other_tags'].split(f'"{pattern}"=>')
                            if len(parts) > 1:
                                levels_str = parts[1].split(',')[0].strip('"\'')
                                
                                # Converter para int
                                levels_value = int(float(levels_str))
                                
                                # Atualizar coluna
                                result.at[idx, 'levels'] = levels_value
                                break
                        except:
                            continue
    
    # Estimar altura a partir de níveis quando não disponível diretamente
    missing_height = pd.isna(result['height'])
    has_levels = ~pd.isna(result['levels'])
    to_estimate = missing_height & has_levels
    
    if to_estimate.sum() > 0:
        # Usar 3 metros por nível como padrão
        result.loc[to_estimate, 'height'] = result.loc[to_estimate, 'levels'] * 3.0
        logger.info(f"Altura estimada para {to_estimate.sum()} edifícios a partir do número de pavimentos")
    
    # Estimar níveis a partir de altura quando não disponível diretamente
    missing_levels = pd.isna(result['levels'])
    has_height = ~pd.isna(result['height']) 
    to_estimate = missing_levels & has_height
    
    if to_estimate.sum() > 0:
        # Usar altura / 3 para estimar níveis
        result.loc[to_estimate, 'levels'] = np.ceil(result.loc[to_estimate, 'height'] / 3.0)
        logger.info(f"Número de pavimentos estimado para {to_estimate.sum()} edifícios a partir da altura")
    
    # Fornecer valores padrão para edifícios sem altura ou níveis
    # Baseado no tipo de edifício
    still_missing = pd.isna(result['height']) & pd.isna(result['levels'])
    
    if still_missing.sum() > 0:
        logger.info(f"Atribuindo valores padrão para {still_missing.sum()} edifícios sem informações de altura")
        
        # Definir valores padrão baseados no tipo do edifício
        default_values = {
            'residential': {'levels': 2, 'height': 6},
            'house': {'levels': 1, 'height': 4},
            'apartments': {'levels': 4, 'height': 12},
            'commercial': {'levels': 1, 'height': 4},
            'industrial': {'levels': 1, 'height': 5},
            'office': {'levels': 3, 'height': 9},
            'retail': {'levels': 1, 'height': 4},
            'warehouse': {'levels': 1, 'height': 6},
            'school': {'levels': 2, 'height': 6},
            'church': {'levels': 1, 'height': 8},
            'hospital': {'levels': 3, 'height': 9},
            'university': {'levels': 2, 'height': 6},
            'garage': {'levels': 1, 'height': 3},
            'shed': {'levels': 1, 'height': 3},
            'roof': {'levels': 1, 'height': 3},
            'construction': {'levels': 1, 'height': 3},
        }
        
        # Aplicar valores padrão
        for idx, row in result[still_missing].iterrows():
            # Obter tipo do edifício
            building_type = row.get('building', 'yes')
            if building_type in default_values:
                result.at[idx, 'levels'] = default_values[building_type]['levels']
                result.at[idx, 'height'] = default_values[building_type]['height']
            else:
                # Padrão para tipos não especificados
                result.at[idx, 'levels'] = 1
                result.at[idx, 'height'] = 3
    
    # Verificar os resultados
    logger.info(f"Altura média dos edifícios: {result['height'].mean():.2f} metros")
    logger.info(f"Número médio de pavimentos: {result['levels'].mean():.2f}")
    logger.info(f"Edifícios mais altos: {result.nlargest(5, 'height')['height'].values}")
    
    return result

def calculate_advanced_morphological_metrics(buildings_gdf):
    """
    Calcula métricas morfológicas avançadas para cada edifício.
    
    Args:
        buildings_gdf (gpd.GeoDataFrame): GeoDataFrame com os edifícios
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame atualizado com métricas morfológicas
    """
    logger.info("Calculando métricas morfológicas avançadas...")
    
    # Criar cópia para não alterar os dados originais
    result = buildings_gdf.copy()
    
    # Garantir que temos uma projeção métrica para cálculos precisos
    if result.crs and result.crs.is_geographic:
        result_proj = result.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
    else:
        result_proj = result
    
    # Calcular área se não existir
    if 'area_m2' not in result.columns:
        result['area_m2'] = result_proj.geometry.area
        
    # Calcular perímetro se não existir
    if 'perimeter_m' not in result.columns:
        result['perimeter_m'] = result_proj.geometry.length
    
    # 1. Índice de compacidade (razão entre área e perímetro ao quadrado)
    # Um círculo tem IC=1, formas mais complexas têm IC menor
    result['compactness_index'] = (4 * np.pi * result['area_m2']) / (result['perimeter_m'] ** 2)
    
    # 2. Relação de aspecto (usando o retângulo envolvente)
    result['aspect_ratio'] = result_proj.geometry.apply(
        lambda geom: calculate_aspect_ratio(geom)
    )
    
    # 3. Complexidade de forma (baseada em decomposição convexa)
    result['shape_complexity'] = result_proj.geometry.apply(
        lambda geom: calculate_shape_complexity(geom)
    )
    
    # 4. Orientação do edifício (ângulo em graus)
    result['orientation_degrees'] = result_proj.geometry.apply(
        lambda geom: calculate_orientation(geom)
    )
    
    # 5. Elongação (relação entre os eixos maior e menor do polígono)
    result['elongation'] = result_proj.geometry.apply(
        lambda geom: calculate_elongation(geom)
    )
    
    # 6. Volume aproximado (m³)
    result['volume_m3'] = result['area_m2'] * result['height']
    
    # 7. Densidade volumétrica (relação entre volume e volume do envelope)
    # Corrigido para evitar erro de índice
    result['volume_density'] = result_proj.apply(
        lambda row: calculate_volume_density(row.geometry, row['height']) 
        if not pd.isna(row['height']) else np.nan,
        axis=1
    )
    
    # 8. Fator de visibilidade (baseado na exposição da fachada)
    result['facade_exposure'] = result['perimeter_m'] * result['height']
    
    # 9. Relação entre área e altura
    result['area_height_ratio'] = result['area_m2'] / result['height']
    
    # 10. Área de cobertura (telhado)
    result['roof_area_m2'] = result['area_m2']
    
    # Calcular estatísticas básicas por tipo de edifício
    if 'building' in result.columns:
        stats_by_type = result.groupby('building').agg({
            'area_m2': ['mean', 'median', 'min', 'max', 'count'],
            'height': ['mean', 'median', 'min', 'max'],
            'volume_m3': ['mean', 'sum'],
            'compactness_index': 'mean',
            'aspect_ratio': 'mean'
        })
        
        logger.info("\nEstatísticas por tipo de edifício:")
        for building_type, stats in stats_by_type.iterrows():
            if stats[('area_m2', 'count')] > 10:  # Mostrar apenas tipos com pelo menos 10 edificações
                logger.info(f"\n{building_type}:")
                logger.info(f"  Quantidade: {stats[('area_m2', 'count')]:.0f}")
                logger.info(f"  Área média: {stats[('area_m2', 'mean')]:.1f} m²")
                logger.info(f"  Altura média: {stats[('height', 'mean')]:.1f} m")
                logger.info(f"  Volume total: {stats[('volume_m3', 'sum')]:.1f} m³")
                logger.info(f"  Compacidade média: {stats[('compactness_index', 'mean')]:.3f}")
    
    return result

def calculate_aspect_ratio(geom):
    """
    Calcula a relação de aspecto (largura/altura) do retângulo envolvente da geometria.
    
    Args:
        geom (shapely.geometry.Polygon): Geometria do edifício
        
    Returns:
        float: Relação de aspecto (sempre >= 1)
    """
    try:
        # Obter o retângulo envolvente mínimo
        minx, miny, maxx, maxy = geom.bounds
        width = maxx - minx
        height = maxy - miny
        
        # Garantir que a relação de aspecto seja >= 1
        if width > height:
            return width / max(height, 0.001)  # Evitar divisão por zero
        else:
            return height / max(width, 0.001)
    except:
        return np.nan

def calculate_shape_complexity(geom):
    """
    Calcula a complexidade da forma baseada na relação entre a geometria e sua envoltória convexa.
    
    Args:
        geom (shapely.geometry.Polygon): Geometria do edifício
        
    Returns:
        float: Índice de complexidade (1 = forma simples, valores > 1 indicam maior complexidade)
    """
    try:
        if not isinstance(geom, (Polygon, MultiPolygon)):
            return np.nan
            
        if isinstance(geom, MultiPolygon):
            # Para MultiPolygon, usar a soma das áreas
            area = sum(poly.area for poly in geom.geoms)
            hull_area = geom.convex_hull.area
        else:
            area = geom.area
            hull_area = geom.convex_hull.area
        
        if hull_area > 0:
            # Razão entre a área da envoltória convexa e a área real
            # Valores próximos a 1 indicam formas mais simples/convexas
            return hull_area / area
        else:
            return np.nan
    except:
        return np.nan

def calculate_orientation(geom):
    """
    Calcula a orientação dominante do edifício em graus.
    
    Args:
        geom (shapely.geometry.Polygon): Geometria do edifício
        
    Returns:
        float: Ângulo de orientação em graus (0-180)
    """
    try:
        if not isinstance(geom, (Polygon, MultiPolygon)):
            return np.nan
            
        # Para MultiPolygon, usar a maior parte
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)
        
        # Obter o retângulo envolvente mínimo orientado
        from shapely.geometry import box
        from shapely.affinity import rotate
        
        # Usar aproximação baseada na envoltória convexa
        hull = geom.convex_hull
        
        # Encontrar os pontos mais distantes no hull (diâmetro)
        coords = np.array(hull.exterior.coords)
        dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2, axis=2))
        i, j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
        
        # Calcular o ângulo entre esses pontos
        dx = coords[j][0] - coords[i][0]
        dy = coords[j][1] - coords[i][1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Normalizar para o intervalo 0-180
        angle = angle % 180
        
        return angle
    except:
        return np.nan

def calculate_elongation(geom):
    """
    Calcula a elongação do edifício (relação entre o eixo maior e menor).
    
    Args:
        geom (shapely.geometry.Polygon): Geometria do edifício
        
    Returns:
        float: Índice de elongação
    """
    try:
        if not isinstance(geom, (Polygon, MultiPolygon)):
            return np.nan
            
        # Para MultiPolygon, usar a maior parte
        if isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda g: g.area)
        
        # Usar a relação entre os eixos do retângulo envolvente mínimo orientado
        minx, miny, maxx, maxy = geom.bounds
        width = maxx - minx
        height = maxy - miny
        
        # Evitar divisão por zero
        if min(width, height) > 0:
            return max(width, height) / min(width, height)
        else:
            return np.nan
    except:
        return np.nan

def calculate_volume_density(geom, height):
    """
    Calcula a densidade volumétrica (relação entre o volume real e o volume do envelope).
    
    Args:
        geom (shapely.geometry.Polygon): Geometria do edifício
        height (float): Altura do edifício
        
    Returns:
        float: Densidade volumétrica (entre 0 e 1)
    """
    try:
        if not isinstance(geom, (Polygon, MultiPolygon)) or pd.isna(height):
            return np.nan
            
        # Volume do edifício
        if isinstance(geom, MultiPolygon):
            volume = sum(poly.area for poly in geom.geoms) * height
        else:
            volume = geom.area * height
        
        # Volume do envelope (caixa envolvente)
        minx, miny, maxx, maxy = geom.bounds
        envelope_volume = (maxx - minx) * (maxy - miny) * height
        
        if envelope_volume > 0:
            return volume / envelope_volume
        else:
            return np.nan
    except:
        return np.nan

def analyze_building_clusters(buildings_gdf):
    """
    Analisa padrões de agrupamento e clusters de edificações.
    
    Args:
        buildings_gdf (gpd.GeoDataFrame): GeoDataFrame com os edifícios
        
    Returns:
        tuple: (GeoDataFrame atualizado, GeoDataFrame de clusters)
    """
    logger.info("Analisando padrões de clusters de edificações...")
    
    # Criar cópia para não alterar os dados originais
    result = buildings_gdf.copy()
    
    # Garantir que temos uma projeção métrica para cálculos precisos
    if result.crs and result.crs.is_geographic:
        result_proj = result.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
    else:
        result_proj = result
    
    # Extrair centroides para análise de clusters
    centroids = result_proj.geometry.centroid
    
    # Converter para DataFrame para uso com DBSCAN
    points = pd.DataFrame({
        'x': centroids.x,
        'y': centroids.y
    })
    
    # Normalizar os dados
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    
    # Aplicar DBSCAN para identificar clusters
    # O parâmetro eps define a distância máxima entre pontos no mesmo cluster (30m)
    # min_samples define o número mínimo de pontos para formar um cluster central (5)
    dbscan = DBSCAN(eps=0.05, min_samples=5)
    clusters = dbscan.fit_predict(points_scaled)
    
    # Adicionar rótulos de cluster ao GeoDataFrame
    result['cluster_id'] = clusters
    
    # Calcular estatísticas de cada cluster
    cluster_stats = {}
    
    # Ignorar outliers (cluster_id = -1)
    unique_clusters = np.unique(clusters)
    unique_clusters = unique_clusters[unique_clusters >= 0]
    
    logger.info(f"Identificados {len(unique_clusters)} clusters de edificações")
    
    for cluster_id in unique_clusters:
        # Edificações neste cluster
        cluster_buildings = result[result['cluster_id'] == cluster_id]
        
        # Estatísticas básicas
        stats = {
            'count': len(cluster_buildings),
            'area_total': cluster_buildings['area_m2'].sum(),
            'area_mean': cluster_buildings['area_m2'].mean(),
            'height_mean': cluster_buildings['height'].mean(),
            'height_std': cluster_buildings['height'].std(),
            'volume_total': cluster_buildings['volume_m3'].sum(),
            'compactness_mean': cluster_buildings['compactness_index'].mean(),
            'centroid': [
                cluster_buildings.geometry.centroid.x.mean(),
                cluster_buildings.geometry.centroid.y.mean()
            ],
            # Tipos de edifícios no cluster
            'building_types': dict(cluster_buildings['building'].value_counts())
        }
        
        # Densidade de construção no cluster
        hull = unary_union(cluster_buildings.geometry).convex_hull
        if isinstance(hull, (Polygon, MultiPolygon)):
            hull_area = hull.area
            if hull_area > 0:
                stats['density'] = stats['area_total'] / hull_area
                stats['hull_area'] = hull_area
                stats['hull_geometry'] = hull
        
        # Adicionar estatísticas ao dicionário
        cluster_stats[int(cluster_id)] = stats
    
    # Criar GeoDataFrame para os clusters
    clusters_gdf = None
    if cluster_stats:
        # Extrair geometrias e dados dos clusters
        geometries = []
        data = []
        
        for cluster_id, stats in cluster_stats.items():
            if 'hull_geometry' in stats:
                geometries.append(stats['hull_geometry'])
                stats_copy = stats.copy()
                stats_copy.pop('hull_geometry')  # Remover a geometria do dicionário
                stats_copy['cluster_id'] = cluster_id
                data.append(stats_copy)
        
        # Criar GeoDataFrame dos clusters
        if geometries:
            clusters_gdf = gpd.GeoDataFrame(data, geometry=geometries, crs=result_proj.crs)
            
            # Adicionar tipologia dominante de cada cluster
            clusters_gdf['dominant_type'] = clusters_gdf.apply(
                lambda x: max(x['building_types'].items(), key=lambda item: item[1])[0] 
                if x['building_types'] else "unknown", 
                axis=1
            )
            
            # Calcular diversidade de tipos de edificação (índice de Shannon)
            clusters_gdf['type_diversity'] = clusters_gdf.apply(
                lambda x: calculate_shannon_diversity(x['building_types']),
                axis=1
            )
            
            # Classificar clusters
            clusters_gdf['cluster_class'] = clusters_gdf.apply(classify_cluster, axis=1)
            
            # Converter para a projeção original
            if result.crs != result_proj.crs:
                clusters_gdf = clusters_gdf.to_crs(result.crs)
    
    # Refinar rótulos de clusters para edificações
    if clusters_gdf is not None:
        # Adicionar informações do cluster à cada edificação
        for cluster_id in unique_clusters:
            mask = result['cluster_id'] == cluster_id
            if mask.any() and cluster_id in clusters_gdf['cluster_id'].values:
                cluster_data = clusters_gdf[clusters_gdf['cluster_id'] == cluster_id].iloc[0]
                result.loc[mask, 'cluster_class'] = cluster_data['cluster_class']
                result.loc[mask, 'cluster_density'] = cluster_data['density']
                result.loc[mask, 'cluster_type_diversity'] = cluster_data['type_diversity']
                result.loc[mask, 'cluster_dominant_type'] = cluster_data['dominant_type']
    
    # Calcular outliers (edifícios isolados)
    outliers = result[result['cluster_id'] == -1]
    logger.info(f"Edifícios isolados (não agrupados): {len(outliers)} ({len(outliers)/len(result)*100:.1f}%)")
    
    # Calcular estatísticas de isolamento para outliers
    if len(outliers) > 0:
        # Criar KDTree para calcular distâncias
        tree = cKDTree(points[result['cluster_id'] >= 0][['x', 'y']].values)
        
        # Para cada outlier, encontrar distância para o edifício mais próximo em um cluster
        distances = []
        nearest_idx = []
        
        for idx, row in outliers.iterrows():
            point = points.loc[idx][['x', 'y']].values
            dist, nn_idx = tree.query(point, k=1)
            distances.append(dist)
            nearest_idx.append(nn_idx)
        
        # Adicionar à tabela de outliers
        result.loc[outliers.index, 'isolation_distance'] = distances
        
        # Categorizar isolamento
        bins = [0, 50, 100, 200, 500, float('inf')]
        labels = ['Muito baixo', 'Baixo', 'Médio', 'Alto', 'Muito alto']
        result.loc[outliers.index, 'isolation_category'] = pd.cut(
            result.loc[outliers.index, 'isolation_distance'],
            bins=bins,
            labels=labels
        )
        
        logger.info("Estatísticas de isolamento para edifícios não agrupados:")
        logger.info(f"  Distância média ao cluster mais próximo: {np.mean(distances):.1f} metros")
        logger.info(f"  Distância máxima: {np.max(distances):.1f} metros")
        logger.info(f"  Distribuição por categoria de isolamento: \n{result.loc[outliers.index, 'isolation_category'].value_counts()}")
    
    return result, clusters_gdf

def calculate_shannon_diversity(counts_dict):
    """
    Calcula o índice de diversidade de Shannon para um dicionário de contagens.
    
    Args:
        counts_dict (dict): Dicionário com contagens
        
    Returns:
        float: Índice de diversidade de Shannon
    """
    if not counts_dict:
        return 0
        
    total = sum(counts_dict.values())
    if total == 0:
        return 0
    
    # Versão otimizada com numba se disponível
    if HAS_NUMBA and len(counts_dict) > 10:  # Só vale a pena para muitos tipos
        counts = np.array(list(counts_dict.values()))
        return shannon_diversity_numba(counts)
    else:
        # Calcular a proporção de cada tipo
        proportions = [count / total for count in counts_dict.values()]
        
        # Calcular o índice de Shannon
        shannon = -sum(p * np.log(p) for p in proportions if p > 0)
        
        return shannon

def classify_cluster(row):
    """
    Classifica um cluster com base em suas características.
    
    Args:
        row (pd.Series): Linha do GeoDataFrame de clusters
        
    Returns:
        str: Classificação do cluster
    """
    # Verificar tipo dominante
    dominant_type = row['dominant_type'] if 'dominant_type' in row else "unknown"
    
    # Verificar densidade
    density = row['density'] if 'density' in row else 0
    
    # Verificar diversidade
    diversity = row['type_diversity'] if 'type_diversity' in row else 0
    
    # Critérios de classificação
    if dominant_type in ['house', 'residential'] and density < 0.3:
        return 'Residencial de baixa densidade'
    elif dominant_type in ['house', 'residential'] and density >= 0.3 and density < 0.6:
        return 'Residencial de média densidade'
    elif dominant_type in ['house', 'residential'] and density >= 0.6:
        return 'Residencial de alta densidade'
    elif dominant_type in ['apartments'] and density > 0.4:
        return 'Conjunto de apartamentos'
    elif dominant_type in ['commercial', 'retail', 'office'] and diversity > 0.5:
        return 'Centro comercial misto'
    elif dominant_type in ['commercial', 'retail', 'office']:
        return 'Área comercial especializada'
    elif dominant_type == 'industrial':
        return 'Área industrial'
    elif dominant_type in ['school', 'university']:
        return 'Complexo educacional'
    elif dominant_type in ['hospital']:
        return 'Complexo de saúde'
    elif diversity > 0.7:
        return 'Área de uso misto'
    else:
        return 'Agrupamento genérico'

def analyze_building_street_relationship(buildings_gdf, roads_gdf=None):
    """
    Analisa a relação entre edificações e o sistema viário.
    
    Args:
        buildings_gdf (gpd.GeoDataFrame): GeoDataFrame com os edifícios
        roads_gdf (gpd.GeoDataFrame, optional): GeoDataFrame com as estradas
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame atualizado com informações de relação com sistema viário
    """
    logger.info("Analisando relação entre edificações e sistema viário...")
    
    # Criar cópia para não alterar os dados originais
    result = buildings_gdf.copy()
    
    # Se não temos dados de estradas, retornar apenas o resultado
    if roads_gdf is None or roads_gdf.empty:
        logger.warning("Dados de estradas não disponíveis. Pulando análise de relação com sistema viário.")
        return result
    
    # Garantir projeção uniforme e métrica
    if result.crs and result.crs.is_geographic:
        result_proj = result.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        roads_proj = roads_gdf.to_crs(epsg=31983)
    else:
        result_proj = result
        roads_proj = roads_gdf.to_crs(result.crs)
    
    # Criar buffer nas estradas para análise
    buffer_distance = 50  # metros
    roads_buffer = roads_proj.buffer(buffer_distance)
    roads_buffer_dissolved = unary_union(roads_buffer)
    
    # 1. Verificar se o edifício está próximo a uma via (dentro do buffer)
    result['near_road'] = False
    centroid_points = result_proj.geometry.centroid
    
    # Criar índice espacial para otimização
    sindex = roads_proj.sindex
    
    # Para cada edifício, calcular distância para a via mais próxima
    distances = []
    nearest_road_ids = []
    nearest_road_types = []
    
    for idx, geom in tqdm(enumerate(result_proj.geometry), 
                          total=len(result_proj), 
                          desc="Calculando distâncias para vias"):
        try:
            # Obter centroide
            centroid = geom.centroid
            
            # Verificar se está dentro do buffer
            within_buffer = centroid.within(roads_buffer_dissolved)
            result.loc[result.index[idx], 'near_road'] = within_buffer
            
            # Encontrar a estrada mais próxima usando o índice espacial
            point = centroid
            possible_matches_index = list(sindex.intersection(point.buffer(buffer_distance).bounds))
            possible_matches = roads_proj.iloc[possible_matches_index]
            
            if not possible_matches.empty:
                # Calcular distância para cada estrada possível
                dists = [point.distance(road) for road in possible_matches.geometry]
                min_dist_idx = np.argmin(dists)
                min_dist = dists[min_dist_idx]
                nearest_road = possible_matches.iloc[min_dist_idx]
                
                distances.append(min_dist)
                nearest_road_ids.append(nearest_road.name)
                
                # Guardar o tipo de via mais próxima
                if 'highway' in nearest_road:
                    nearest_road_types.append(nearest_road['highway'])
                elif 'road_class' in nearest_road:
                    nearest_road_types.append(nearest_road['road_class'])
                else:
                    nearest_road_types.append('unknown')
            else:
                distances.append(np.nan)
                nearest_road_ids.append(None)
                nearest_road_types.append('unknown')
                
        except Exception as e:
            logger.warning(f"Erro ao calcular distância para via, edifício {idx}: {e}")
            distances.append(np.nan)
            nearest_road_ids.append(None)
            nearest_road_types.append('unknown')
    
    # Adicionar informações calculadas ao GeoDataFrame
    result['distance_to_road'] = distances
    result['nearest_road_id'] = nearest_road_ids
    result['nearest_road_type'] = nearest_road_types
    
    # 2. Categorizar acessibilidade por distância à via
    bins = [0, 10, 25, 50, 100, float('inf')]
    labels = ['Excelente', 'Boa', 'Média', 'Limitada', 'Remota']
    result['road_accessibility'] = pd.cut(result['distance_to_road'], bins=bins, labels=labels)
    
    # 3. Calcular densidade de ruas no entorno de cada edifício
    result['road_density_500m'] = np.nan
    
    # Calcular comprimento total de estradas em buffer de 500m para cada edifício
    for idx, geom in tqdm(enumerate(result_proj.geometry), 
                         total=len(result_proj), 
                         desc="Calculando densidade de vias no entorno"):
        try:
            # Criar buffer de 500m ao redor do centroide do edifício
            centroid = geom.centroid
            buffer_500m = centroid.buffer(500)
            
            # Encontrar estradas que intersectam este buffer
            possible_matches_index = list(sindex.intersection(buffer_500m.bounds))
            possible_matches = roads_proj.iloc[possible_matches_index]
            
            if not possible_matches.empty:
                # Calcular comprimento total das estradas dentro do buffer
                road_length = 0
                for road in possible_matches.geometry:
                    # Recortar a estrada pelo buffer
                    if road.intersects(buffer_500m):
                        intersection = road.intersection(buffer_500m)
                        if not intersection.is_empty:
                            if hasattr(intersection, 'length'):
                                road_length += intersection.length
                            elif hasattr(intersection, 'geoms'):
                                road_length += sum(geom.length for geom in intersection.geoms if hasattr(geom, 'length'))
                
                # Área do buffer em km²
                buffer_area_km2 = np.pi * 0.5**2
                
                # Densidade de estradas em km/km²
                if buffer_area_km2 > 0:
                    road_density = road_length / 1000 / buffer_area_km2
                    result.loc[result.index[idx], 'road_density_500m'] = road_density
            
        except Exception as e:
            logger.warning(f"Erro ao calcular densidade de vias para edifício {idx}: {e}")
    
    # 4. Categorizar integração urbana com base na densidade de vias
    if not result['road_density_500m'].isna().all():
        bins = [0, 5, 10, 15, 20, float('inf')]
        labels = ['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito alta']
        result['urban_integration'] = pd.cut(result['road_density_500m'], bins=bins, labels=labels)
    
    # Estatísticas gerais da relação edificações-sistema viário
    logger.info(f"Edifícios próximos a vias (< {buffer_distance}m): {result['near_road'].sum()} ({result['near_road'].sum()/len(result)*100:.1f}%)")
    logger.info(f"Distância média à via mais próxima: {result['distance_to_road'].mean():.1f} metros")
    logger.info(f"Distribuição por categoria de acessibilidade: \n{result['road_accessibility'].value_counts()}")
    
    # Estatísticas por tipo de edifício
    if 'building' in result.columns:
        for building_type in result['building'].value_counts().nlargest(5).index:
            subset = result[result['building'] == building_type]
            if len(subset) > 10:
                logger.info(f"\nEstatísticas de acessibilidade para '{building_type}':")
                logger.info(f"  Distância média à via: {subset['distance_to_road'].mean():.1f} metros")
                logger.info(f"  % próximos a vias: {subset['near_road'].mean()*100:.1f}%")
    
    return result

def extract_elevation_data(buildings_gdf, dem=None):
    """
    Extrai informações de elevação para cada edifício usando o Modelo Digital de Elevação.
    
    Args:
        buildings_gdf (gpd.GeoDataFrame): GeoDataFrame com os edifícios
        dem (rasterio.io.DatasetReader, optional): Modelo Digital de Elevação
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame atualizado com informações de elevação
    """
    logger.info("Extraindo dados de elevação para edificações...")
    
    # Criar cópia para não alterar os dados originais
    result = buildings_gdf.copy()
    
    # Se não temos DEM, retornar apenas o resultado
    if dem is None:
        logger.warning("Dados de DEM não disponíveis. Pulando extração de elevação.")
        return result
    
    # Garantir que os dados estão no mesmo CRS do DEM
    if result.crs != dem.crs:
        result_proj = result.to_crs(dem.crs)
    else:
        result_proj = result
    
    # Inicializar colunas para elevação
    result['elevation_base'] = np.nan
    result['elevation_top'] = np.nan
    result['terrain_slope'] = np.nan
    
    # Processar em lotes para reduzir uso de memória
    chunk_size = 1000
    n_chunks = (len(result_proj) + chunk_size - 1) // chunk_size
    
    for i in tqdm(range(n_chunks), desc="Processando lotes de edifícios para elevação"):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(result_proj))
        chunk = result_proj.iloc[start_idx:end_idx]
        
        # Para cada edifício no lote, extrair elevação do DEM
        for idx, row in chunk.iterrows():
            try:
                # Obter o centroide
                centroid = row.geometry.centroid
                x, y = centroid.x, centroid.y
                
                # Extrair coordenadas de pixel
                # Correção: dem.index retorna uma tupla (row, col) e não um iterador
                py, px = dem.index(x, y)
                
                # Verificar se as coordenadas estão dentro dos limites
                if 0 <= py < dem.height and 0 <= px < dem.width:
                    # Ler valor do pixel
                    # Correção: Usar py e px como coordenadas de pixel
                    elevation = dem.read(1, window=((py, py+1), (px, px+1)))
                    if elevation[0][0] != dem.nodata:
                        # Adicionar elevação base
                        result.loc[idx, 'elevation_base'] = float(elevation[0][0])
                        
                        # Calcular elevação do topo usando a altura do edifício
                        if 'height' in row and not pd.isna(row['height']):
                            result.loc[idx, 'elevation_top'] = float(elevation[0][0]) + row['height']
                
                # Calcular declividade do terreno ao redor do edifício
                # Usar um buffer de 30m para análise do terreno
                buffer_geom = centroid.buffer(30)
                buffer_bounds = buffer_geom.bounds
                
                # Verificar se o buffer está completamente dentro dos limites do DEM
                xmin, ymin, xmax, ymax = buffer_bounds
                if (xmin >= dem.bounds[0] and ymin >= dem.bounds[1] and 
                    xmax <= dem.bounds[2] and ymax <= dem.bounds[3]):
                    
                    # Criar máscara para o buffer
                    mask_geom = [mapping(buffer_geom)]
                    masked_dem, masked_transform = mask(dem, mask_geom, crop=True, nodata=dem.nodata)
                    
                    # Calcular estatísticas no buffer
                    if masked_dem.size > 0 and not np.all(masked_dem == dem.nodata):
                        valid_data = masked_dem[masked_dem != dem.nodata]
                        if len(valid_data) > 0:
                            # Calcular declividade usando máx - mín / diâmetro do buffer
                            elev_min = np.min(valid_data)
                            elev_max = np.max(valid_data)
                            elev_range = elev_max - elev_min
                            
                            # Declividade aproximada em porcentagem
                            slope_pct = (elev_range / 60) * 100  # 60m é o diâmetro
                            result.loc[idx, 'terrain_slope'] = slope_pct
            
            except Exception as e:
                logger.warning(f"Erro ao extrair elevação para edifício {idx}: {str(e)}")
    
    # Calcular estatísticas de elevação
    if not result['elevation_base'].isna().all():
        logger.info(f"Elevação média da base: {result['elevation_base'].mean():.1f} metros")
        if not result['elevation_top'].isna().all():
            logger.info(f"Elevação média do topo: {result['elevation_top'].mean():.1f} metros")
        if not result['terrain_slope'].isna().all():
            logger.info(f"Declividade média do terreno: {result['terrain_slope'].mean():.1f}%")
    
    # Classificar declividade do terreno
    if not result['terrain_slope'].isna().all():
        bins = [0, 2, 5, 10, 15, float('inf')]
        labels = ['Plano', 'Suave', 'Moderado', 'Íngreme', 'Muito íngreme']
        result['terrain_slope_class'] = pd.cut(result['terrain_slope'], bins=bins, labels=labels)
        logger.info(f"Distribuição por classe de declividade: \n{result['terrain_slope_class'].value_counts()}")
    
    return result

def analyze_urban_morphology(buildings_gdf, landuse_gdf=None):
    """
    Realiza análises de morfologia urbana em escala de quadra/bairro.
    
    Args:
        buildings_gdf (gpd.GeoDataFrame): GeoDataFrame com os edifícios
        landuse_gdf (gpd.GeoDataFrame, optional): GeoDataFrame com uso do solo
        
    Returns:
        tuple: (GeoDataFrame de edifícios atualizado, GeoDataFrame de quadras)
    """
    logger.info("Analisando morfologia urbana...")
    
    # Criar cópia para não alterar os dados originais
    result = buildings_gdf.copy()
    
    # Garantir projeção métrica
    if result.crs and result.crs.is_geographic:
        result_proj = result.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        landuse_proj = landuse_gdf.to_crs(epsg=31983) if landuse_gdf is not None else None
    else:
        result_proj = result
        landuse_proj = landuse_gdf if landuse_gdf is not None else None
    
    # Criar grid de células para análise (100x100m)
    grid_size = 100  # metros
    
    # Obter extensão dos dados
    minx, miny, maxx, maxy = result_proj.total_bounds
    
    # Criar grade regular
    x_coords = np.arange(minx, maxx, grid_size)
    y_coords = np.arange(miny, maxy, grid_size)
    
    # Criar células do grid
    grid_cells = []
    
    for x in x_coords:
        for y in y_coords:
            cell = box(x, y, x+grid_size, y+grid_size)
            grid_cells.append(cell)
    
    # Criar GeoDataFrame da grade
    grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells}, crs=result_proj.crs)
    grid_gdf['cell_id'] = range(len(grid_gdf))
    
    # Métricas para cada célula da grade
    grid_metrics = []
    
    for idx, cell in tqdm(grid_gdf.iterrows(), total=len(grid_gdf), desc="Calculando métricas por célula"):
        # Selecionar edifícios na célula
        buildings_in_cell = result_proj[result_proj.geometry.intersects(cell.geometry)]
        
        # Se não houver edifícios, adicionar métricas vazias
        if len(buildings_in_cell) == 0:
            metrics = {
                'cell_id': cell['cell_id'],
                'building_count': 0,
                'building_coverage_ratio': 0,
                'floor_area_ratio': 0,
                'avg_height': 0,
                'height_std': 0,
                'building_density': 0,
                'avg_volume': 0,
                'dominant_type': 'none',
                'shannon_diversity': 0
            }
            grid_metrics.append(metrics)
            continue
        
        # Calcular métricas para a célula
        
        # 1. Número de edifícios
        building_count = len(buildings_in_cell)
        
        # 2. Taxa de ocupação (Building Coverage Ratio - BCR)
        building_area = buildings_in_cell.area.sum()
        cell_area = cell.geometry.area
        bcr = building_area / cell_area if cell_area > 0 else 0
        
        # 3. Coeficiente de Aproveitamento (Floor Area Ratio - FAR)
        if 'levels' in buildings_in_cell.columns and not buildings_in_cell['levels'].isna().all():
            total_floor_area = (buildings_in_cell.area * buildings_in_cell['levels']).sum()
            far = total_floor_area / cell_area if cell_area > 0 else 0
        else:
            far = 0
            
        # 4. Altura média
        avg_height = buildings_in_cell['height'].mean() if 'height' in buildings_in_cell.columns else 0
        height_std = buildings_in_cell['height'].std() if 'height' in buildings_in_cell.columns else 0
        
        # 5. Densidade de edifícios (unidades/hectare)
        density = building_count / (cell_area / 10000)
        
        # 6. Volume médio
        avg_volume = buildings_in_cell['volume_m3'].mean() if 'volume_m3' in buildings_in_cell.columns else 0
        
        # 7. Tipo de edifício dominante
        if 'building' in buildings_in_cell.columns:
            type_counts = buildings_in_cell['building'].value_counts()
            dominant_type = type_counts.index[0] if len(type_counts) > 0 else 'unknown'
            
            # 8. Diversidade de tipos (Índice de Shannon)
            type_counts_dict = {key: val for key, val in type_counts.items()}
            shannon_diversity = calculate_shannon_diversity(type_counts_dict)
        else:
            dominant_type = 'unknown'
            shannon_diversity = 0
        
        metrics = {
            'cell_id': cell['cell_id'],
            'building_count': building_count,
            'building_coverage_ratio': bcr,
            'floor_area_ratio': far,
            'avg_height': avg_height,
            'height_std': height_std,
            'building_density': density,
            'avg_volume': avg_volume,
            'dominant_type': dominant_type,
            'shannon_diversity': shannon_diversity
        }
        
        grid_metrics.append(metrics)
    
    # Adicionar métricas às células da grade
    for metrics in grid_metrics:
        cell_id = metrics['cell_id']
        for key, value in metrics.items():
            if key != 'cell_id':
                grid_gdf.loc[grid_gdf['cell_id'] == cell_id, key] = value
    
    # Reclassificar células com base nas métricas
    # 1. Classificar tipologia urbana
    conditions = [
        (grid_gdf['building_count'] == 0),
        (grid_gdf['building_coverage_ratio'] < 0.1),
        (grid_gdf['building_coverage_ratio'] < 0.3) & (grid_gdf['building_density'] < 5),
        (grid_gdf['building_coverage_ratio'] < 0.3) & (grid_gdf['building_density'] >= 5),
        (grid_gdf['building_coverage_ratio'] >= 0.3) & (grid_gdf['building_coverage_ratio'] < 0.5),
        (grid_gdf['building_coverage_ratio'] >= 0.5)
    ]
    choices = [
        'Não urbanizado',
        'Espaço aberto',
        'Suburbano de baixa densidade',
        'Residencial de média densidade',
        'Urbano de média densidade',
        'Urbano de alta densidade'
    ]
    grid_gdf['urban_typology'] = np.select(conditions, choices, default='Indeterminado')
    
    # 2. Calcular verticalização
    grid_gdf['verticalization'] = np.select(
        [
            (grid_gdf['avg_height'] < 4),
            (grid_gdf['avg_height'] < 9),
            (grid_gdf['avg_height'] < 15),
            (grid_gdf['avg_height'] < 30),
            (grid_gdf['avg_height'] >= 30)
        ],
        [
            'Muito baixa',
            'Baixa',
            'Média',
            'Alta',
            'Muito alta'
        ],
        default='Indeterminada'
    )
    
    # 3. Calcular heterogeneidade de alturas
    grid_gdf['height_heterogeneity'] = np.select(
        [
            (grid_gdf['height_std'] < 1),
            (grid_gdf['height_std'] < 3),
            (grid_gdf['height_std'] < 6),
            (grid_gdf['height_std'] < 10),
            (grid_gdf['height_std'] >= 10)
        ],
        [
            'Muito baixa',
            'Baixa',
            'Média',
            'Alta',
            'Muito alta'
        ],
        default='Indeterminada'
    )
    
    # 4. Calcular compacidade urbana (combina BCR e FAR)
    grid_gdf['urban_compactness'] = np.select(
        [
            (grid_gdf['building_coverage_ratio'] < 0.2) & (grid_gdf['floor_area_ratio'] < 0.5),
            (grid_gdf['building_coverage_ratio'] < 0.4) & (grid_gdf['floor_area_ratio'] < 1.0),
            (grid_gdf['building_coverage_ratio'] < 0.6) & (grid_gdf['floor_area_ratio'] < 2.0),
            (grid_gdf['building_coverage_ratio'] < 0.8) & (grid_gdf['floor_area_ratio'] < 4.0),
            (grid_gdf['building_coverage_ratio'] >= 0.8) | (grid_gdf['floor_area_ratio'] >= 4.0)
        ],
        [
            'Muito baixa',
            'Baixa',
            'Média', 
            'Alta',
            'Muito alta'
        ],
        default='Indeterminada'
    )
    
    # 5. Classificar diversidade de usos com base no índice de Shannon
    grid_gdf['use_mix'] = np.select(
        [
            (grid_gdf['shannon_diversity'] < 0.2),
            (grid_gdf['shannon_diversity'] < 0.5),
            (grid_gdf['shannon_diversity'] < 0.8),
            (grid_gdf['shannon_diversity'] < 1.2),
            (grid_gdf['shannon_diversity'] >= 1.2)
        ],
        [
            'Monofuncional',
            'Baixa diversidade',
            'Média diversidade',
            'Alta diversidade',
            'Muito alta diversidade'
        ],
        default='Indeterminado'
    )
    
    # Registrar estatísticas sobre as células
    logger.info("\nEstatísticas de morfologia urbana:")
    logger.info(f"Total de células: {len(grid_gdf)}")
    logger.info(f"Células urbanizadas: {len(grid_gdf[grid_gdf['building_count'] > 0])}")
    logger.info("\nDistribuição por tipologia urbana:")
    for typology, count in grid_gdf['urban_typology'].value_counts().items():
        logger.info(f"  {typology}: {count} células")
    
    logger.info("\nDistribuição por verticalização:")
    for vert, count in grid_gdf['verticalization'].value_counts().items():
        logger.info(f"  {vert}: {count} células")
    
    # Atribuir propriedades da célula a cada edifício
    result['cell_id'] = -1
    result['urban_typology'] = 'Indeterminado'
    result['cell_building_density'] = np.nan
    result['cell_bcr'] = np.nan
    result['cell_far'] = np.nan
    result['cell_verticalization'] = 'Indeterminada'
    result['cell_height_heterogeneity'] = 'Indeterminada'
    result['cell_use_mix'] = 'Indeterminado'
    
    # Usar otimização para acelerar a atribuição de métricas
    # Criar índice espacial para as células
    grid_sindex = grid_gdf.sindex
    
    for idx, building in tqdm(result_proj.iterrows(), total=len(result_proj), desc="Atribuindo métricas de células aos edifícios"):
        # Encontrar a célula que contém o centroide do edifício usando índice espacial
        centroid = building.geometry.centroid
        possible_matches_idx = list(grid_sindex.intersection(centroid.bounds))
        possible_matches = grid_gdf.iloc[possible_matches_idx]
        
        containing_cells = possible_matches[possible_matches.contains(centroid)]
        
        if not containing_cells.empty:
            cell = containing_cells.iloc[0]
            
            # Copiar métricas relevantes
            result.loc[idx, 'cell_id'] = cell['cell_id']
            result.loc[idx, 'urban_typology'] = cell['urban_typology']
            result.loc[idx, 'cell_building_density'] = cell['building_density']
            result.loc[idx, 'cell_bcr'] = cell['building_coverage_ratio']
            result.loc[idx, 'cell_far'] = cell['floor_area_ratio']
            result.loc[idx, 'cell_verticalization'] = cell['verticalization']
            result.loc[idx, 'cell_height_heterogeneity'] = cell['height_heterogeneity']
            result.loc[idx, 'cell_use_mix'] = cell['use_mix']
    
    # Calcular índice de centralidade urbana para as células
    # Baseado nas métricas de morfologia urbana (densidade, FAR, diversidade)
    if len(grid_gdf) > 0:
        # Normalizar métricas para escala 0-1
        metrics_to_normalize = ['building_density', 'floor_area_ratio', 'shannon_diversity']
        
        for metric in metrics_to_normalize:
            if metric in grid_gdf.columns:
                min_val = grid_gdf[metric].min()
                max_val = grid_gdf[metric].max()
                
                if max_val > min_val:
                    grid_gdf[f'{metric}_norm'] = (grid_gdf[metric] - min_val) / (max_val - min_val)
                else:
                    grid_gdf[f'{metric}_norm'] = 0
        
        # Calcular índice de centralidade como média ponderada
        weights = {
            'building_density_norm': 0.4,
            'floor_area_ratio_norm': 0.4,
            'shannon_diversity_norm': 0.2
        }
        
        # Inicializar coluna de centralidade
        grid_gdf['centrality_index'] = 0.0
        
        # Aplicar pesos para calcular índice de centralidade
        for metric, weight in weights.items():
            if metric in grid_gdf.columns:
                grid_gdf['centrality_index'] += grid_gdf[metric] * weight
        
        # Classificar centralidade
        # Corrigido: Usar método discreto em vez de qcut para evitar erro com valores duplicados
        # Primeiro, filtramos células não urbanizadas
        urbanized_cells = grid_gdf[grid_gdf['building_count'] > 0]
        
        if not urbanized_cells.empty:
            # Calcular quantis manualmente para evitar erro de 'bin edges must be unique'
            quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
            bin_edges = []
            
            # Garantir que os valores sejam únicos
            sorted_values = sorted(urbanized_cells['centrality_index'].unique())
            
            if len(sorted_values) >= 5:
                for q in quantiles:
                    if q == 0:
                        bin_edges.append(sorted_values[0])
                    elif q == 1.0:
                        bin_edges.append(sorted_values[-1] + 0.00001)  # Adicionar um pequeno valor para incluir o máximo
                    else:
                        idx = int(q * (len(sorted_values) - 1))
                        bin_edges.append(sorted_values[idx])
                
                # Garantir que temos 5 bins com bordas únicas
                if len(set(bin_edges)) == len(bin_edges) and len(bin_edges) >= 5:
                    # Classificar usando np.digitize
                    labels = ['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito alta']
                    centrality_bins = np.digitize(grid_gdf['centrality_index'], bin_edges[1:-1])
                    
                    # Mapear os bins para os rótulos (0-4)
                    centrality_mapping = {i: labels[i] if i < len(labels) else labels[-1] for i in range(5)}
                    grid_gdf['centrality_class'] = [centrality_mapping.get(b, 'Indeterminada') for b in centrality_bins]
                else:
                    # Fallback para discretização simples se não conseguirmos calcular quantis adequados
                    grid_gdf['centrality_class'] = np.select(
                        [
                            (grid_gdf['centrality_index'] <= 0.01),
                            (grid_gdf['centrality_index'] <= 0.1),
                            (grid_gdf['centrality_index'] <= 0.2),
                            (grid_gdf['centrality_index'] <= 0.3),
                            (grid_gdf['centrality_index'] > 0.3)
                        ],
                        [
                            'Muito baixa',
                            'Baixa',
                            'Média',
                            'Alta',
                            'Muito alta'
                        ],
                        default='Indeterminada'
                    )
            else:
                # Não há dados suficientes para criar 5 classes
                grid_gdf['centrality_class'] = 'Indeterminada'
        else:
            # Não há células urbanizadas
            grid_gdf['centrality_class'] = 'Indeterminada'
        
        # Transferir informação de centralidade para os edifícios
        for idx, building in result.iterrows():
            if building['cell_id'] >= 0:
                cell_id = building['cell_id']
                if cell_id in grid_gdf['cell_id'].values:
                    cell_row = grid_gdf[grid_gdf['cell_id'] == cell_id].iloc[0]
                    result.loc[idx, 'cell_centrality'] = cell_row['centrality_index']
                    result.loc[idx, 'cell_centrality_class'] = cell_row['centrality_class']
    
    # Converter grid para o CRS original se necessário
    if result.crs != result_proj.crs:
        grid_gdf = grid_gdf.to_crs(result.crs)
    
    return result, grid_gdf

def create_3d_model(buildings_gdf):
    """
    Cria um modelo 3D simples das edificações para visualização.
    
    Args:
        buildings_gdf (gpd.GeoDataFrame): GeoDataFrame com os edifícios
        
    Returns:
        dict: Dicionário com dados para modelo 3D
    """
    logger.info("Gerando modelo 3D das edificações...")
    
    # Verificar se temos altura e geometria
    if 'height' not in buildings_gdf.columns:
        logger.warning("Coluna 'height' não encontrada. Não é possível criar modelo 3D.")
        return None
    
    # Criar cópia para não alterar os dados originais
    result = buildings_gdf.copy()
    
    # Garantir que estamos usando uma projeção métrica
    if result.crs and result.crs.is_geographic:
        result_proj = result.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
    else:
        result_proj = result
    
    # Preparar dados para visualização 3D
    building_data = []
    
    # Extrair edifícios para visualização (limitar a 10.000 para desempenho)
    sample_size = min(len(result_proj), 10000)
    sample_buildings = result_proj.sample(sample_size) if len(result_proj) > sample_size else result_proj
    
    for idx, building in sample_buildings.iterrows():
        try:
            if pd.isna(building['height']) or building['height'] <= 0:
                continue
                
            # Extrair polígono do edifício
            polygon = building.geometry
            
            if not isinstance(polygon, (Polygon, MultiPolygon)):
                continue
                
            # Para MultiPolygon, usar o polígono de maior área
            if isinstance(polygon, MultiPolygon):
                polygon = max(polygon.geoms, key=lambda x: x.area)
            
            # Calcular centroide para referência
            centroid = polygon.centroid
            
            # Extrair coordenadas
            coords = list(polygon.exterior.coords)
            x = [c[0] - centroid.x for c in coords]
            y = [c[1] - centroid.y for c in coords]
            z = [0] * len(coords)  # Base no nível do solo
            
            # Altura do edifício
            height = float(building['height'])
            
            # Adicionar dados do edifício
            building_info = {
                'id': idx,
                'coords': {'x': x, 'y': y, 'z': z},
                'height': height,
                'centroid': [float(centroid.x), float(centroid.y)],
                'building_class': str(building.get('building', 'unknown')),
                'color': get_building_color(building)
            }
            
            building_data.append(building_info)
        
        except Exception as e:
            logger.warning(f"Erro ao processar edifício {idx} para modelo 3D: {e}")
    
    logger.info(f"Modelo 3D gerado com {len(building_data)} edifícios")
    
    return {
        'buildings': building_data,
        'crs': str(result_proj.crs),
        'count': len(building_data)
    }

def get_building_color(building):
    """
    Define uma cor para o edifício com base em seus atributos.
    
    Args:
        building (pd.Series): Linha do GeoDataFrame de edifícios
        
    Returns:
        str: Código de cor em formato hexadecimal
    """
    # Cores por tipo de edifício
    building_colors = {
        'residential': '#E8DAEF',  # Roxo claro
        'house': '#D2B4DE',        # Roxo
        'apartments': '#A569BD',   # Roxo escuro
        'commercial': '#AED6F1',   # Azul claro
        'retail': '#5DADE2',       # Azul médio
        'office': '#2E86C1',       # Azul escuro
        'industrial': '#F5CBA7',   # Laranja claro
        'warehouse': '#E67E22',    # Laranja
        'school': '#ABEBC6',       # Verde claro
        'university': '#58D68D',   # Verde
        'hospital': '#EC7063',     # Vermelho
        'church': '#F4D03F',       # Amarelo
        'hotel': '#F1948A',        # Rosa
        'garage': '#CCD1D1',       # Cinza claro
        'shed': '#95A5A6',         # Cinza
        'roof': '#7F8C8D',         # Cinza escuro
        'construction': '#F2F3F4', # Branco acinzentado
    }
    
    # Obter tipo do edifício
    building_type = building.get('building', 'unknown')
    
    # Retornar cor correspondente ou cor padrão
    return building_colors.get(building_type, '#D5DBDB')  # Cinza padrão

def generate_visualization(buildings_gdf, clusters_gdf=None, grid_gdf=None, buildings_3d=None):
    """
    Gera visualizações para os dados enriquecidos de edificações.
    
    Args:
        buildings_gdf (gpd.GeoDataFrame): GeoDataFrame com os edifícios
        clusters_gdf (gpd.GeoDataFrame, optional): GeoDataFrame com clusters
        grid_gdf (gpd.GeoDataFrame, optional): GeoDataFrame com grade de análise
        buildings_3d (dict, optional): Dados para modelo 3D
    """
    logger.info("Gerando visualizações para os dados enriquecidos...")
    
    # Criar timestamp para os arquivos
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 1. Mapa de altura das edificações
    logger.info("Gerando mapa de altura das edificações...")
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)
    
    # Plotar edifícios coloridos por altura
    buildings_gdf.plot(column='height', cmap='viridis', legend=True, ax=ax,
                       legend_kwds={'label': 'Altura (m)'})
    
    # Adicionar mapa base
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logger.warning(f"Erro ao adicionar mapa base: {e}")
    
    plt.title('Altura das Edificações', fontsize=16)
    plt.axis('off')
    
    # Salvar figura
    height_map_file = os.path.join(VISUALIZATION_DIR, f'altura_edificacoes_{timestamp}.png')
    plt.savefig(height_map_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mapa de clusters
    if clusters_gdf is not None and not clusters_gdf.empty:
        logger.info("Gerando mapa de clusters de edificações...")
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111)
        
        # Plotar clusters com cores diferentes
        clusters_gdf.plot(column='cluster_id', cmap='tab20', alpha=0.5, ax=ax,
                          legend=True, legend_kwds={'label': 'ID do Cluster'})
        
        # Plotar edifícios
        buildings_gdf.plot(ax=ax, color='darkblue', markersize=1, alpha=0.3)
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            logger.warning(f"Erro ao adicionar mapa base: {e}")
        
        plt.title('Clusters de Edificações', fontsize=16)
        plt.axis('off')
        
        # Salvar figura
        clusters_map_file = os.path.join(VISUALIZATION_DIR, f'clusters_edificacoes_{timestamp}.png')
        plt.savefig(clusters_map_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Mapa de tipologia de clusters
        logger.info("Gerando mapa de tipologia de clusters...")
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111)
        
        # Definir cores para cada classe de cluster
        cluster_classes = clusters_gdf['cluster_class'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(cluster_classes)))
        color_dict = dict(zip(cluster_classes, colors))
        
        # Plotar cada classe de cluster com uma cor diferente
        for cls, color in color_dict.items():
            subset = clusters_gdf[clusters_gdf['cluster_class'] == cls]
            subset.plot(color=color, ax=ax, label=cls)
        
        # Adicionar legenda
        ax.legend(title='Tipologia de Cluster', loc='upper right')
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            logger.warning(f"Erro ao adicionar mapa base: {e}")
        
        plt.title('Tipologia de Clusters de Edificações', fontsize=16)
        plt.axis('off')
        
        # Salvar figura
        cluster_type_map_file = os.path.join(VISUALIZATION_DIR, f'tipologia_clusters_{timestamp}.png')
        plt.savefig(cluster_type_map_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Mapa de morfologia urbana (células da grade)
    if grid_gdf is not None and not grid_gdf.empty:
        logger.info("Gerando mapa de morfologia urbana (tipologia)...")
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111)
        
        # Definir cores para cada tipologia urbana
        urban_types = grid_gdf['urban_typology'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(urban_types)))
        color_dict = dict(zip(urban_types, colors))
        
        # Plotar cada tipologia urbana com uma cor diferente
        for urban_type, color in color_dict.items():
            subset = grid_gdf[grid_gdf['urban_typology'] == urban_type]
            subset.plot(color=color, ax=ax, label=urban_type)
        
        # Adicionar legenda
        ax.legend(title='Tipologia Urbana', loc='upper right')
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            logger.warning(f"Erro ao adicionar mapa base: {e}")
        
        plt.title('Morfologia Urbana - Tipologia', fontsize=16)
        plt.axis('off')
        
        # Salvar figura
        urban_type_map_file = os.path.join(VISUALIZATION_DIR, f'tipologia_urbana_{timestamp}.png')
        plt.savefig(urban_type_map_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Mapa de verticalização
        logger.info("Gerando mapa de verticalização...")
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111)
        
        # Plotar células coloridas por verticalização
        grid_gdf.plot(column='verticalization', cmap='RdYlBu_r', ax=ax, alpha=0.7,
                      legend=True, legend_kwds={'label': 'Verticalização'})
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            logger.warning(f"Erro ao adicionar mapa base: {e}")
        
        plt.title('Verticalização Urbana', fontsize=16)
        plt.axis('off')
        
        # Salvar figura
        vertical_map_file = os.path.join(VISUALIZATION_DIR, f'verticalizacao_{timestamp}.png')
        plt.savefig(vertical_map_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Mapa de densidade construída (BCR)
        logger.info("Gerando mapa de densidade construída (BCR)...")
        plt.figure(figsize=(12, 10))
        ax = plt.subplot(111)
        
        # Plotar células coloridas por BCR
        grid_gdf.plot(column='building_coverage_ratio', cmap='YlOrRd', ax=ax, alpha=0.7,
                      legend=True, legend_kwds={'label': 'Taxa de Ocupação (BCR)'})
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            logger.warning(f"Erro ao adicionar mapa base: {e}")
        
        plt.title('Densidade Construída (BCR)', fontsize=16)
        plt.axis('off')
        
        # Salvar figura
        bcr_map_file = os.path.join(VISUALIZATION_DIR, f'densidade_bcr_{timestamp}.png')
        plt.savefig(bcr_map_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Mapa de índice de centralidade
        if 'centrality_index' in grid_gdf.columns:
            logger.info("Gerando mapa de centralidade urbana...")
            plt.figure(figsize=(12, 10))
            ax = plt.subplot(111)
            
            # Plotar células coloridas por índice de centralidade
            grid_gdf.plot(column='centrality_index', cmap='plasma', ax=ax, alpha=0.7,
                          legend=True, legend_kwds={'label': 'Índice de Centralidade'})
            
            # Adicionar mapa base
            try:
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
            except Exception as e:
                logger.warning(f"Erro ao adicionar mapa base: {e}")
            
            plt.title('Centralidade Urbana', fontsize=16)
            plt.axis('off')
            
            # Salvar figura
            central_map_file = os.path.join(VISUALIZATION_DIR, f'centralidade_{timestamp}.png')
            plt.savefig(central_map_file, dpi=300, bbox_inches='tight')
            plt.close()
    
    # 8. Visualização 3D simples
    if buildings_3d is not None and len(buildings_3d['buildings']) > 0:
        logger.info("Gerando visualização 3D simples...")
        
        # Criar figura 3D
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Desenhar cada edifício
        for building in buildings_3d['buildings'][:1000]:  # Limitar a 1000 para desempenho
            try:
                # Extrair dados
                x = building['coords']['x']
                y = building['coords']['y']
                z = building['coords']['z']
                height = building['height']
                color = building['color']
                
                # Desenhar base (face inferior)
                ax.plot(x, y, z, color=color, alpha=0.7)
                
                # Desenhar topo (face superior)
                z_top = [height] * len(z)
                ax.plot(x, y, z_top, color=color, alpha=0.7)
                
                # Conectar base e topo
                for i in range(len(x)-1):
                    ax.plot([x[i], x[i]], [y[i], y[i]], [z[i], z_top[i]], color=color, alpha=0.7)
            except Exception as e:
                logger.debug(f"Erro ao desenhar edifício em 3D: {e}")
        
        # Ajustar perspectiva e título
        ax.view_init(elev=30, azim=45)
        plt.title('Visualização 3D Simplificada das Edificações', fontsize=16)
        
        # Salvar figura
        model_3d_file = os.path.join(VISUALIZATION_DIR, f'modelo_3d_simples_{timestamp}.png')
        plt.savefig(model_3d_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 9. Mapa interativo com folium
    logger.info("Gerando mapa interativo...")
    
    # Converter para WGS84 para uso com folium
    buildings_wgs84 = buildings_gdf.to_crs(epsg=4326)
    
    # Calcular centro do mapa
    center_lat = buildings_wgs84.geometry.centroid.y.mean()
    center_lon = buildings_wgs84.geometry.centroid.x.mean()
    
    # Criar mapa base
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles='CartoDB Positron')
    
    # Criar camadas para diferentes categorias de edifícios
    if 'building' in buildings_wgs84.columns:
        building_types = buildings_wgs84['building'].unique()
        
        for building_type in building_types:
            if pd.isna(building_type) or building_type == '':
                continue
                
            # Filtrar por tipo
            subset = buildings_wgs84[buildings_wgs84['building'] == building_type]
            
            if len(subset) > 2000:
                # Para tipos com muitos edifícios, usar clusters
                marker_cluster = MarkerCluster(name=f'Edifícios: {building_type}')
                
                # Adicionar amostra de edifícios ao cluster
                sample_size = min(2000, len(subset))
                for idx, row in subset.sample(sample_size).iterrows():
                    popup_html = f"""
                    <b>Tipo:</b> {building_type}<br>
                    <b>Altura:</b> {row.get('height', 'N/A')} m<br>
                    <b>Área:</b> {row.get('area_m2', 'N/A'):.1f} m²<br>
                    """
                    
                    folium.Marker(
                        location=[row.geometry.centroid.y, row.geometry.centroid.x],
                        popup=folium.Popup(popup_html),
                        icon=folium.Icon(icon='home')
                    ).add_to(marker_cluster)
                
                marker_cluster.add_to(m)
            else:
                # Para tipos com poucos edifícios, adicionar diretamente
                for idx, row in subset.iterrows():
                    popup_html = f"""
                    <b>Tipo:</b> {building_type}<br>
                    <b>Altura:</b> {row.get('height', 'N/A')} m<br>
                    <b>Área:</b> {row.get('area_m2', 'N/A'):.1f} m²<br>
                    """
                    
                    folium.GeoJson(
                        row.geometry.__geo_interface__,
                        name=f'Edifício {idx}',
                        style_function=lambda x, color=get_building_color(row): {
                            'fillColor': color,
                            'color': 'black',
                            'weight': 1,
                            'fillOpacity': 0.7
                        },
                        popup=folium.Popup(popup_html)
                    ).add_to(m)
    
    # Adicionar clusters se disponíveis
    if clusters_gdf is not None and not clusters_gdf.empty:
        clusters_wgs84 = clusters_gdf.to_crs(epsg=4326)
        
        for idx, row in clusters_wgs84.iterrows():
            popup_html = f"""
            <b>Cluster:</b> {row['cluster_id']}<br>
            <b>Tipologia:</b> {row['cluster_class']}<br>
            <b>Edifícios:</b> {row['count']}<br>
            <b>Tipo dominante:</b> {row['dominant_type']}<br>
            <b>Altura média:</b> {row['height_mean']:.1f} m<br>
            """
            
            folium.GeoJson(
                row.geometry.__geo_interface__,
                name=f'Cluster {row["cluster_id"]}',
                style_function=lambda x: {
                    'fillColor': 'blue',
                    'color': 'blue',
                    'weight': 2,
                    'fillOpacity': 0.2
                },
                popup=folium.Popup(popup_html)
            ).add_to(m)
    
    # Adicionar grade de morfologia urbana se disponível
    if grid_gdf is not None and not grid_gdf.empty:
        grid_wgs84 = grid_gdf.to_crs(epsg=4326)
        
        # Camada para tipologia urbana
        folium.GeoJson(
            grid_wgs84[grid_wgs84['building_count'] > 0],
            name='Tipologia Urbana',
            style_function=lambda x: {
                'fillColor': 'green',
                'color': 'green',
                'weight': 1,
                'fillOpacity': 0.1
            },
            popup=folium.GeoJsonPopup(
                fields=['urban_typology', 'building_count', 'building_density', 'avg_height'],
                aliases=['Tipologia', 'Nº de Edifícios', 'Densidade', 'Altura Média'],
                localize=True
            )
        ).add_to(m)
    
    # Adicionar mapa de calor de alturas
    if 'height' in buildings_wgs84.columns:
        # Preparar dados para o mapa de calor
        heat_data = []
        for idx, row in buildings_wgs84.iterrows():
            try:
                if pd.notna(row['height']) and row['height'] > 0:
                    # Adicionar ponto com intensidade proporcional à altura
                    centroid = row.geometry.centroid
                    heat_data.append([centroid.y, centroid.x, min(row['height'], 50)])  # Limitar altura para visualização
            except Exception as e:
                logger.debug(f"Erro ao adicionar edifício ao mapa de calor: {e}")
        
        # Adicionar mapa de calor se tivermos dados suficientes
        if len(heat_data) > 0:
            HeatMap(
                heat_data,
                name='Mapa de Calor de Altura',
                radius=15,
                blur=10,
                gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}
            ).add_to(m)
    
    # Adicionar controle de camadas
    folium.LayerControl().add_to(m)
    
    # Salvar mapa interativo
    interactive_map_file = os.path.join(VISUALIZATION_DIR, f'mapa_interativo_{timestamp}.html')
    m.save(interactive_map_file)
    
    logger.info(f"Visualizações salvas em {VISUALIZATION_DIR}")

def generate_quality_report(original_gdf, enriched_gdf, grid_gdf=None, clusters_gdf=None):
    """
    Gera um relatório de qualidade dos dados enriquecidos.
    
    Args:
        original_gdf (gpd.GeoDataFrame): GeoDataFrame original
        enriched_gdf (gpd.GeoDataFrame): GeoDataFrame enriquecido
        grid_gdf (gpd.GeoDataFrame, optional): GeoDataFrame da grade urbana
        clusters_gdf (gpd.GeoDataFrame, optional): GeoDataFrame dos clusters
        
    Returns:
        dict: Relatório de qualidade
    """
    logger.info("Gerando relatório de qualidade dos dados enriquecidos...")
    
    # Criar timestamp para o arquivo
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Estrutura básica do relatório
    report = {
        "meta": {
            "titulo": "Relatório de Qualidade de Dados de Edificações Enriquecidos",
            "data_geracao": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "versao": "1.0"
        },
        "dados_originais": {
            "numero_edificacoes": len(original_gdf),
            "colunas_originais": original_gdf.columns.tolist(),
            "crs": str(original_gdf.crs)
        },
        "dados_enriquecidos": {
            "numero_edificacoes": len(enriched_gdf),
            "novas_colunas": list(set(enriched_gdf.columns) - set(original_gdf.columns)),
            "metricas": {}
        },
        "analise_morfologica": {},
        "analise_clusters": {}
    }
    
    # Métricas básicas para os dados enriquecidos
    # Altura
    if 'height' in enriched_gdf.columns:
        height_stats = {
            "media": float(enriched_gdf['height'].mean()),
            "mediana": float(enriched_gdf['height'].median()),
            "min": float(enriched_gdf['height'].min()),
            "max": float(enriched_gdf['height'].max()),
            "std": float(enriched_gdf['height'].std()),
            "edificios_sem_altura": int(enriched_gdf['height'].isna().sum())
        }
        report["dados_enriquecidos"]["metricas"]["altura"] = height_stats
    
    # Área
    if 'area_m2' in enriched_gdf.columns:
        area_stats = {
            "media": float(enriched_gdf['area_m2'].mean()),
            "mediana": float(enriched_gdf['area_m2'].median()),
            "min": float(enriched_gdf['area_m2'].min()),
            "max": float(enriched_gdf['area_m2'].max()),
            "std": float(enriched_gdf['area_m2'].std()),
            "area_total": float(enriched_gdf['area_m2'].sum())
        }
        report["dados_enriquecidos"]["metricas"]["area"] = area_stats
    
    # Volume
    if 'volume_m3' in enriched_gdf.columns:
        volume_stats = {
            "media": float(enriched_gdf['volume_m3'].mean()),
            "mediana": float(enriched_gdf['volume_m3'].median()),
            "min": float(enriched_gdf['volume_m3'].min()),
            "max": float(enriched_gdf['volume_m3'].max()),
            "std": float(enriched_gdf['volume_m3'].std()),
            "volume_total": float(enriched_gdf['volume_m3'].sum())
        }
        report["dados_enriquecidos"]["metricas"]["volume"] = volume_stats
    
    # Tipos de edifícios
    if 'building' in enriched_gdf.columns:
        building_types = enriched_gdf['building'].value_counts().to_dict()
        report["dados_enriquecidos"]["metricas"]["tipos_edificios"] = {
            str(key): int(value) for key, value in building_types.items() if pd.notna(key)
        }
    
    # Estatísticas para a grade urbana
    if grid_gdf is not None and not grid_gdf.empty:
        urban_typology = grid_gdf['urban_typology'].value_counts().to_dict()
        
        grid_stats = {
            "numero_celulas": len(grid_gdf),
            "celulas_urbanizadas": int((grid_gdf['building_count'] > 0).sum()),
            "distribuicao_tipologia": {str(key): int(value) for key, value in urban_typology.items()},
            "bcr_medio": float(grid_gdf['building_coverage_ratio'].mean()),
            "far_medio": float(grid_gdf['floor_area_ratio'].mean()),
            "densidade_media": float(grid_gdf['building_density'].mean()),
            "altura_media": float(grid_gdf['avg_height'].mean())
        }
        
        # Adicionar distribuição de verticalização se disponível
        if 'verticalization' in grid_gdf.columns:
            verticalization = grid_gdf['verticalization'].value_counts().to_dict()
            grid_stats["distribuicao_verticalizacao"] = {
                str(key): int(value) for key, value in verticalization.items()
            }
        
        report["analise_morfologica"] = grid_stats
    
    # Estatísticas para clusters
    if clusters_gdf is not None and not clusters_gdf.empty:
        cluster_classes = clusters_gdf['cluster_class'].value_counts().to_dict()
        
        cluster_stats = {
            "numero_clusters": len(clusters_gdf),
            "distribuicao_classes": {str(key): int(value) for key, value in cluster_classes.items()},
            "edificios_agrupados": int(clusters_gdf['count'].sum()),
            "edificios_isolados": len(enriched_gdf) - int(clusters_gdf['count'].sum()),
        }
        
        report["analise_clusters"] = cluster_stats
    
    # Salvar relatório como JSON
    report_file = os.path.join(REPORT_DIR, f'relatorio_qualidade_{timestamp}.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4, cls=NpEncoder)
    
    logger.info(f"Relatório de qualidade salvo em {report_file}")
    
    return report

def save_enriched_data(enriched_gdf, grid_gdf=None, clusters_gdf=None):
    """
    Salva os dados enriquecidos em arquivos GPKG.
    
    Args:
        enriched_gdf (gpd.GeoDataFrame): GeoDataFrame enriquecido
        grid_gdf (gpd.GeoDataFrame, optional): GeoDataFrame da grade urbana
        clusters_gdf (gpd.GeoDataFrame, optional): GeoDataFrame dos clusters
        
    Returns:
        str: Caminho para o arquivo principal
    """
    logger.info("Salvando dados enriquecidos...")
    
    # Criar timestamp para os arquivos
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Nome do arquivo principal
    main_file = os.path.join(ENRICHED_DATA_DIR, f'buildings_enriched_{timestamp}.gpkg')
    
    # Salvar edifícios enriquecidos
    enriched_gdf.to_file(main_file, driver='GPKG', layer='buildings')
    logger.info(f"Dados de edificações enriquecidos salvos em {main_file} (layer: buildings)")
    
    # Salvar grade urbana se disponível
    if grid_gdf is not None and not grid_gdf.empty:
        grid_gdf.to_file(main_file, driver='GPKG', layer='urban_grid')
        logger.info(f"Grade urbana salva em {main_file} (layer: urban_grid)")
    
    # Salvar clusters se disponíveis
    if clusters_gdf is not None and not clusters_gdf.empty:
        clusters_gdf.to_file(main_file, driver='GPKG', layer='building_clusters')
        logger.info(f"Clusters de edificações salvos em {main_file} (layer: building_clusters)")
    
    return main_file

def main():
    """
    Função principal que executa o fluxo de trabalho completo de enriquecimento.
    """
    logger.info("=== Iniciando processo de enriquecimento avançado de dados de edificações ===")
    
    # Registrar hora de início
    start_time = time.time()
    step_times = {}
    
    def log_step_time(step_name):
        current_time = time.time()
        elapsed = current_time - start_time
        step_times[step_name] = elapsed
        logger.info(f"[TEMPO] {step_name}: {elapsed:.2f} segundos")
        return current_time
    
    try:
        # 1. Carregar dados
        step_start = time.time()
        data = load_data()
        log_step_time("Carregamento de dados")
        
        if data is None or 'buildings' not in data or data['buildings'] is None:
            logger.error("Erro ao carregar dados de edificações. Abortando.")
            return
        
        buildings_gdf = data['buildings']
        roads_gdf = data.get('roads')
        dem = data.get('dem')
        landuse_gdf = data.get('landuse')
        
        # Guardar cópia dos dados originais
        original_buildings = buildings_gdf.copy()
        
        # 2. Extrair e padronizar alturas
        step_start = time.time()
        logger.info("Extraindo e padronizando alturas...")
        buildings_gdf = extract_building_height(buildings_gdf)
        log_step_time("Extração de alturas")
        
        # 3. Calcular métricas morfológicas avançadas
        step_start = time.time()
        logger.info("Calculando métricas morfológicas avançadas...")
        buildings_gdf = calculate_advanced_morphological_metrics(buildings_gdf)
        log_step_time("Cálculo de métricas morfológicas")
        
        # 4. Analisar clusters de edificações
        step_start = time.time()
        logger.info("Analisando clusters de edificações...")
        buildings_gdf, clusters_gdf = analyze_building_clusters(buildings_gdf)
        log_step_time("Análise de clusters")
        
        # 5. Analisar relação com o sistema viário
        if roads_gdf is not None:
            step_start = time.time()
            logger.info("Analisando relação com sistema viário...")
            buildings_gdf = analyze_building_street_relationship(buildings_gdf, roads_gdf)
            log_step_time("Análise de relação com sistema viário")
        
        # 6. Extrair dados de elevação se DEM disponível
        if dem is not None:
            step_start = time.time()
            logger.info("Extraindo dados de elevação...")
            buildings_gdf = extract_elevation_data(buildings_gdf, dem)
            log_step_time("Extração de dados de elevação")
        
        # 7. Analisar morfologia urbana
        step_start = time.time()
        logger.info("Analisando morfologia urbana...")
        buildings_gdf, grid_gdf = analyze_urban_morphology(buildings_gdf, landuse_gdf)
        log_step_time("Análise de morfologia urbana")
        
        # 8. Criar modelo 3D
        step_start = time.time()
        buildings_3d = create_3d_model(buildings_gdf)
        log_step_time("Criação de modelo 3D")
        
        # 9. Gerar visualizações
        step_start = time.time()
        logger.info("Gerando visualizações...")
        generate_visualization(buildings_gdf, clusters_gdf, grid_gdf, buildings_3d)
        log_step_time("Geração de visualizações")
        
        # 10. Gerar relatório de qualidade
        step_start = time.time()
        logger.info("Gerando relatório de qualidade...")
        quality_report = generate_quality_report(original_buildings, buildings_gdf, grid_gdf, clusters_gdf)
        log_step_time("Geração de relatório de qualidade")
        
        # 11. Salvar dados enriquecidos
        step_start = time.time()
        logger.info("Salvando dados enriquecidos...")
        output_file = save_enriched_data(buildings_gdf, grid_gdf, clusters_gdf)
        log_step_time("Salvamento de dados enriquecidos")
        
        # Registrar tempo de execução
        elapsed_time = time.time() - start_time
        logger.info(f"=== Processo concluído em {elapsed_time:.2f} segundos ===")
        
        # Mostrar sumário de tempos
        logger.info("=== Sumário de tempos de execução ===")
        sorted_steps = sorted(step_times.items(), key=lambda x: x[1])
        for step_name, step_time in sorted_steps:
            logger.info(f"{step_name}: {step_time:.2f} segundos ({(step_time/elapsed_time)*100:.1f}%)")
        
        logger.info(f"Dados enriquecidos salvos em: {output_file}")
        
        return {
            'buildings_gdf': buildings_gdf,
            'clusters_gdf': clusters_gdf,
            'grid_gdf': grid_gdf,
            'output_file': output_file
        }
        
    except Exception as e:
        logger.error(f"Erro durante o processo de enriquecimento: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Registrar tempo de execução mesmo em caso de erro
        elapsed_time = time.time() - start_time
        logger.info(f"=== Processo interrompido após {elapsed_time:.2f} segundos devido a erro ===")
        return None

if __name__ == "__main__":
    # Configurar e parsear argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Enriquecimento avançado de dados de edificações')
    
    parser.add_argument('--input', type=str, help='Caminho para arquivo de entrada (sobrescreve padrão)')
    parser.add_argument('--output', type=str, help='Diretório de saída (sobrescreve padrão)')
    parser.add_argument('--roads', type=str, help='Caminho para arquivo de estradas (sobrescreve padrão)')
    parser.add_argument('--dem', type=str, help='Caminho para arquivo DEM (sobrescreve padrão)')
    parser.add_argument('--no-viz', action='store_true', help='Desativa geração de visualizações')
    parser.add_argument('--no-3d', action='store_true', help='Desativa geração de modelo 3D')
    parser.add_argument('--cores', type=int, default=N_CORES, help=f'Número de cores para processamento paralelo (padrão: {N_CORES})')
    
    args = parser.parse_args()
    
    # Sobrescrever configurações padrão com argumentos da linha de comando
    if args.input:
        BUILDINGS_FILE = args.input
        print(f"Arquivo de entrada configurado para: {BUILDINGS_FILE}")
    
    if args.output:
        ENRICHED_DATA_DIR = args.output
        OUTPUT_DIR = ENRICHED_DATA_DIR
        print(f"Diretório de saída configurado para: {ENRICHED_DATA_DIR}")
        os.makedirs(ENRICHED_DATA_DIR, exist_ok=True)
    
    if args.roads:
        ROADS_FILE = args.roads
        print(f"Arquivo de estradas configurado para: {ROADS_FILE}")
    
    if args.dem:
        DEM_FILE = args.dem
        print(f"Arquivo DEM configurado para: {DEM_FILE}")
    
    if args.cores:
        N_CORES = args.cores
        print(f"Número de cores configurado para: {N_CORES}")
    
    # Executar a função principal
    main()