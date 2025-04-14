#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Funções para enriquecimento de dados de uso do solo (landuse) com métricas de paisagem,
altimetria e análises espaciais avançadas.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, Point, MultiPolygon, LineString
from shapely.ops import unary_union
import rasterio
from rasterio.mask import mask
import rasterio.features
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap
import json
import datetime
import time
import warnings
import logging
from tqdm import tqdm
import math
import scipy.stats as stats
from scipy.spatial import Voronoi
import folium
from folium.plugins import MarkerCluster, HeatMap
import seaborn as sns
import rasterstats
from rasterstats import zonal_stats
import matplotlib.patches as mpatches

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('enriched_landuse')

# Definir caminhos absolutos para diretórios conforme solicitado
LAND_USE_DIR = r"F:\TESE_MESTRADO\geoprocessing\src\enriched_data\land_use"
QUALITY_REPORTS_DIR = r"F:\TESE_MESTRADO\geoprocessing\src\enriched_data\quality_reports"
ENRICHED_DATA_DIR = r"F:\TESE_MESTRADO\geoprocessing\data\enriched_data"

# Definir diretórios de entrada e saída
WORKSPACE_DIR = r"F:\TESE_MESTRADO\geoprocessing"
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
RAW_DIR = os.path.join(WORKSPACE_DIR, 'data', 'raw')
OUTPUT_DIR = ENRICHED_DATA_DIR
REPORT_DIR = os.path.join(QUALITY_REPORTS_DIR, 'land_use')
VISUALIZATION_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualize_enriched_data', 'land_use')

# Definir caminhos de arquivos específicos
LANDUSE_FILE = os.path.join(INPUT_DIR, 'landuse_processed.gpkg')
DEM_FILE = os.path.join(RAW_DIR, 'dem.tif')

# Garantir que os diretórios de saída existam
for directory in [OUTPUT_DIR, REPORT_DIR, VISUALIZATION_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Diretório garantido: {directory}")

# Classe para serialização JSON de tipos numpy
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return super(NpEncoder, self).default(obj)

# Definição de categorias de uso do solo para classificação consistente
LAND_CATEGORIES = {
    'urban': ['residential', 'retail', 'commercial', 'industrial', 'construction', 
              'garages', 'parking', 'brownfield', 'landfill'],
    'green': ['grass', 'meadow', 'recreation_ground', 'village_green', 'park', 'garden',
              'greenfield', 'allotments', 'orchard', 'plant_nursery', 'greenhouse_horticulture'],
    'forest': ['forest', 'wood'],
    'agriculture': ['farmland', 'farmyard', 'farm', 'aquaculture'],
    'water': ['water', 'reservoir', 'basin'],
    'institutional': ['education', 'religious', 'government', 'governmental', 'hospital', 'institutional'],
    'extraction': ['quarry', 'mining'],
    'other': ['military', 'proposed', 'tabacaria', 'yes', 'propose', 'gaop\u00e3o', 'various']
}

# Função para mapear categorias de uso do solo de maneira consistente
def map_landuse_category(landuse_value):
    """
    Mapeia um valor de landuse para uma categoria mais geral.
    
    Args:
        landuse_value (str): Valor original de landuse
        
    Returns:
        str: Categoria geral de landuse
    """
    if not landuse_value or pd.isna(landuse_value):
        return 'unknown'
    
    landuse_lower = str(landuse_value).lower()
    
    for category, values in LAND_CATEGORIES.items():
        if any(value in landuse_lower for value in values):
            return category
            
    return 'other'

def load_data():
    """
    Carrega os dados de uso do solo do diretório de processamento.
    
    Returns:
        geopandas.GeoDataFrame: Os dados de uso do solo carregados.
    """
    try:
        logger.info(f"Carregando dados de uso do solo de {LANDUSE_FILE}")
        if not os.path.exists(LANDUSE_FILE):
            logger.error(f"Arquivo de dados não encontrado: {LANDUSE_FILE}")
            return None
            
        gdf = gpd.read_file(LANDUSE_FILE)
        logger.info(f"Carregados {len(gdf)} polígonos de uso do solo")
        return gdf
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        return None

def load_dem():
    """
    Carrega o Modelo Digital de Elevação (DEM).
    
    Returns:
        rasterio.DatasetReader: O DEM carregado ou None em caso de erro.
    """
    try:
        logger.info(f"Carregando Modelo Digital de Elevação (DEM) de {DEM_FILE}")
        if not os.path.exists(DEM_FILE):
            logger.warning(f"Arquivo DEM não encontrado: {DEM_FILE}")
            return None
            
        dem = rasterio.open(DEM_FILE)
        logger.info(f"DEM carregado com sucesso. Dimensões: {dem.width}x{dem.height}, CRS: {dem.crs}")
        return dem
    except Exception as e:
        logger.error(f"Erro ao carregar DEM: {str(e)}")
        return None

def classify_landuse(gdf):
    """
    Classifica os polígonos de uso do solo em categorias mais gerais e 
    calcula estatísticas de área para cada categoria.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de uso do solo
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame com classificações adicionadas
    """
    logger.info("Classificando polígonos de uso do solo e calculando estatísticas")
    
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    # Garantir que temos uma coluna 'landuse'
    if 'landuse' not in result.columns:
        logger.warning("Coluna 'landuse' não encontrada. Usando a coluna de geometria apenas.")
        result['landuse'] = 'unknown'
    
    # Classificar cada polígono em uma categoria
    result['land_category'] = result['landuse'].apply(map_landuse_category)
    
    # Converter para sistema de referência projetado para calcular área corretamente
    if result.crs and result.crs.is_geographic:
        # SIRGAS 2000 / UTM zone 23S para o Brasil central
        gdf_utm = result.to_crs(epsg=31983)
        
        # Calcular área em km² e perímetro em km
        result['area_km2'] = gdf_utm.geometry.area / 1_000_000
        result['perimeter_km'] = gdf_utm.geometry.length / 1_000
        
        # Calcular índice de compacidade (4π × Area/Perímetro²)
        # Valor máximo é 1 (círculo perfeito)
        result['compactness'] = (4 * np.pi * result['area_km2']) / (result['perimeter_km'] ** 2)
    else:
        logger.warning("O GeoDataFrame não possui um CRS geográfico definido. Calculando área nas unidades originais.")
        result['area_km2'] = result.geometry.area
        result['perimeter_km'] = result.geometry.length
        result['compactness'] = (4 * np.pi * result['area_km2']) / (result['perimeter_km'] ** 2)
    
    # Calcular estatísticas agrupadas por categoria
    stats_by_category = result.groupby('land_category').agg({
        'area_km2': ['sum', 'mean', 'count'],
        'compactness': 'mean'
    })
    
    # Calcular porcentagem de área por categoria
    total_area = result['area_km2'].sum()
    stats_by_category['area_percentage'] = (stats_by_category[('area_km2', 'sum')] / total_area) * 100
    
    # Armazenar estatísticas como metadados do GeoDataFrame
    result.attrs['landuse_stats'] = {
        'total_area_km2': float(total_area),
        'area_by_category_km2': {cat: float(area) for cat, area in 
                              stats_by_category[('area_km2', 'sum')].items()},
        'area_percentage': {cat: float(pct) for cat, pct in 
                         stats_by_category['area_percentage'].items()},
        'land_categories': {cat: int(count) for cat, count in 
                         stats_by_category[('area_km2', 'count')].items()}
    }
    
    # Classificar tamanho dos polígonos
    bins = [0, 0.01, 0.05, 0.25, 1.0, float('inf')]
    labels = ['Muito pequeno', 'Pequeno', 'Médio', 'Grande', 'Muito grande']
    result['size_category'] = pd.cut(result['area_km2'], bins=bins, labels=labels)
    
    # Classificar compacidade
    bins = [0, 0.4, 0.6, 0.8, 1.0]
    labels = ['Baixa', 'Média', 'Alta', 'Muito alta']
    result['compactness_category'] = pd.cut(result['compactness'], bins=bins, labels=labels)
    
    logger.info(f"Classificação concluída. {len(result)} polígonos classificados em {len(result['land_category'].unique())} categorias.")
    
    return result

def extract_elevation_data(gdf, dem):
    """
    Extrai estatísticas de elevação do DEM para cada polígono de uso do solo.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de uso do solo
        dem (rasterio.DatasetReader): Modelo Digital de Elevação
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com dados de elevação
    """
    logger.info("Extraindo dados de elevação para polígonos de uso do solo")
    
    if dem is None:
        logger.warning("DEM não fornecido. Pulando extração de dados de elevação.")
        return gdf
    
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    # Garantir que o GeoDataFrame esteja no mesmo CRS que o DEM
    if result.crs != dem.crs:
        logger.info(f"Reprojetando dados de {result.crs} para {dem.crs}")
        result = result.to_crs(dem.crs)
    
    # Extrair estatísticas de elevação para cada polígono
    # Usando rasterstats para processamento eficiente
    try:
        logger.info("Calculando estatísticas zonais com rasterstats")
        stats = rasterstats.zonal_stats(
            result.geometry,
            dem.read(1),
            affine=dem.transform,
            stats=['min', 'max', 'mean', 'median', 'std'],
            nodata=dem.nodata
        )
        
        # Adicionar estatísticas ao GeoDataFrame
        result['elevation_min'] = [s.get('min') for s in stats]
        result['elevation_max'] = [s.get('max') for s in stats]
        result['elevation_mean'] = [s.get('mean') for s in stats]
        result['elevation_median'] = [s.get('median') for s in stats]
        result['elevation_std'] = [s.get('std') for s in stats]
        
        # Calcular amplitude de elevação
        result['elevation_range'] = result['elevation_max'] - result['elevation_min']
        
        # Classificar variação de elevação
        bins = [0, 5, 15, 30, 50, float('inf')]
        labels = ['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito alta']
        result['elevation_category'] = pd.cut(result['elevation_range'], bins=bins, labels=labels)
        
        # Calcular estatísticas de elevação por categoria de uso do solo
        elevation_by_category = result.groupby('land_category').agg({
            'elevation_mean': 'mean',
            'elevation_min': 'min',
            'elevation_max': 'max',
            'elevation_range': 'mean'
        })
        
        # Armazenar como metadados
        result.attrs['elevation_stats'] = {
            'global': {
                'min': float(result['elevation_min'].min()),
                'max': float(result['elevation_max'].max()),
                'mean': float(result['elevation_mean'].mean()),
                'range_mean': float(result['elevation_range'].mean())
            },
            'by_category': {
                cat: {
                    'mean': float(row['elevation_mean']),
                    'min': float(row['elevation_min']),
                    'max': float(row['elevation_max']),
                    'range': float(row['elevation_range'])
                } for cat, row in elevation_by_category.iterrows()
            }
        }
        
        logger.info(f"Estatísticas de elevação extraídas com sucesso para {len(result)} polígonos")
        
    except Exception as e:
        logger.error(f"Erro ao extrair dados de elevação: {str(e)}")
    
    return result

def calculate_slope_aspect(gdf, dem):
    """
    Calcula declividade e orientação (aspecto) para cada polígono.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de uso do solo com elevação
        dem (rasterio.DatasetReader): Modelo Digital de Elevação
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com dados de declividade e aspecto
    """
    logger.info("Calculando declividade e aspecto para polígonos de uso do solo")
    
    if dem is None:
        logger.warning("DEM não fornecido. Pulando cálculo de declividade e aspecto.")
        return gdf
    
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    try:
        # Verificar se rasterio tem o módulo de cálculo de declividade
        try:
            from rasterio.enums import MaskFlags
            from rasterio.fill import fillnodata
            import numpy as np
            
            # Ler dados do DEM
            dem_data = dem.read(1)
            transform = dem.transform
            
            # Calcular declividade (em graus)
            dx, dy = np.gradient(dem_data)
            slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
            
            # Calcular aspecto (em graus, 0 = norte, 90 = leste, 180 = sul, 270 = oeste)
            aspect = np.degrees(np.arctan2(-dx, dy))
            # Converter para orientação geográfica (0-360, com norte = 0)
            aspect = np.where(aspect < 0, aspect + 360, aspect)
            
            # Extrair estatísticas zonais para declividade
            slope_stats = rasterstats.zonal_stats(
                result.geometry,
                slope,
                affine=transform,
                stats=['mean', 'min', 'max', 'std'],
                nodata=dem.nodata
            )
            
            # Extrair estatísticas zonais para aspecto
            aspect_stats = rasterstats.zonal_stats(
                result.geometry,
                aspect,
                affine=transform,
                stats=['mean', 'min', 'max', 'std'],
                nodata=dem.nodata
            )
            
            # Adicionar estatísticas ao GeoDataFrame
            result['slope_mean'] = [s.get('mean') for s in slope_stats]
            result['slope_min'] = [s.get('min') for s in slope_stats]
            result['slope_max'] = [s.get('max') for s in slope_stats]
            result['slope_std'] = [s.get('std') for s in slope_stats]
            
            result['aspect_mean'] = [s.get('mean') for s in aspect_stats]
            result['aspect_min'] = [s.get('min') for s in aspect_stats]
            result['aspect_max'] = [s.get('max') for s in aspect_stats]
            result['aspect_std'] = [s.get('std') for s in aspect_stats]
            
            # Classificar declividade
            bins = [0, 3, 8, 16, 30, float('inf')]
            labels = ['Plano', 'Suave', 'Moderado', 'Íngreme', 'Muito íngreme']
            result['slope_category'] = pd.cut(result['slope_mean'], bins=bins, labels=labels)
            
            # Classificar aspecto (orientação)
            def classify_aspect(aspect):
                if pd.isna(aspect):
                    return 'Indefinido'
                if aspect < 22.5 or aspect >= 337.5:
                    return 'Norte'
                elif aspect < 67.5:
                    return 'Nordeste'
                elif aspect < 112.5:
                    return 'Leste'
                elif aspect < 157.5:
                    return 'Sudeste'
                elif aspect < 202.5:
                    return 'Sul'
                elif aspect < 247.5:
                    return 'Sudoeste'
                elif aspect < 292.5:
                    return 'Oeste'
                else:
                    return 'Noroeste'
            
            result['aspect_direction'] = result['aspect_mean'].apply(classify_aspect)
            
            # Calcular estatísticas de declividade por categoria de uso do solo
            slope_by_category = result.groupby('land_category').agg({
                'slope_mean': 'mean',
                'slope_max': 'max'
            })
            
            # Armazenar como metadados
            result.attrs['terrain_stats'] = {
                'slope': {
                    'mean': float(result['slope_mean'].mean()),
                    'by_category': {cat: float(row['slope_mean']) for cat, row in slope_by_category.iterrows()}
                },
                'aspect': {
                    'distribution': {direction: int(count) for direction, count in 
                                 result['aspect_direction'].value_counts().items()}
                }
            }
            
            logger.info(f"Declividade e aspecto calculados com sucesso para {len(result)} polígonos")
            
        except ImportError as e:
            logger.warning(f"Módulo necessário não encontrado para cálculo de declividade: {e}")
            
    except Exception as e:
        logger.error(f"Erro ao calcular declividade e aspecto: {str(e)}")
    
    return result

def calculate_landscape_metrics(gdf):
    """
    Calcula métricas de paisagem para os polígonos de uso do solo.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de uso do solo classificados
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com métricas de paisagem
    """
    logger.info("Calculando métricas de paisagem para polígonos de uso do solo")
    
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    try:
        # Para cada categoria de uso do solo
        for category in result['land_category'].unique():
            # Filtrar polígonos desta categoria
            category_gdf = result[result['land_category'] == category]
            
            # Número de fragmentos (patches)
            num_patches = len(category_gdf)
            
            # Área total da categoria (km²)
            total_area = category_gdf['area_km2'].sum()
            
            # Tamanho médio de fragmento (Mean Patch Size)
            mean_patch_size = category_gdf['area_km2'].mean()
            
            # Maior fragmento (Largest Patch Index)
            largest_patch = category_gdf['area_km2'].max()
            largest_patch_index = (largest_patch / total_area) * 100 if total_area > 0 else 0
            
            # Densidade de fragmentos (número de fragmentos por km²)
            landscape_area = result['area_km2'].sum()
            patch_density = (num_patches / landscape_area) if landscape_area > 0 else 0
            
            # Armazenar resultados como atributos do dataframe
            if 'landscape_metrics' not in result.attrs:
                result.attrs['landscape_metrics'] = {}
                
            result.attrs['landscape_metrics'][category] = {
                'num_patches': num_patches,
                'total_area_km2': float(total_area),
                'mean_patch_size_km2': float(mean_patch_size),
                'largest_patch_km2': float(largest_patch),
                'largest_patch_index': float(largest_patch_index),
                'patch_density': float(patch_density)
            }
        
        # Calcular métricas de diversidade para toda a paisagem
        
        # Porcentagem de cada categoria na paisagem
        category_proportions = {cat: (result[result['land_category'] == cat]['area_km2'].sum() / 
                                  result['area_km2'].sum()) 
                             for cat in result['land_category'].unique()}
        
        # Índice de Shannon (diversidade)
        shannon_index = -sum(p * np.log(p) for p in category_proportions.values() if p > 0)
        
        # Índice de Simpson (diversidade)
        simpson_index = 1 - sum(p**2 for p in category_proportions.values())
        
        # Índice de Dominância (1 - Simpson)
        dominance_index = 1 - simpson_index
        
        # Métricas de fragmentação (Total Edge e Edge Density)
        if 'perimeter_km' in result.columns:
            total_edge = result['perimeter_km'].sum() / 2  # Dividir por 2 pois cada borda é contada duas vezes
            landscape_area = result['area_km2'].sum()
            edge_density = total_edge / landscape_area if landscape_area > 0 else 0
        else:
            total_edge = None
            edge_density = None
        
        # Armazenar métricas de paisagem gerais
        result.attrs['landscape_metrics']['overall'] = {
            'shannon_diversity_index': float(shannon_index),
            'simpson_diversity_index': float(simpson_index),
            'dominance_index': float(dominance_index),
            'total_edge_km': float(total_edge) if total_edge is not None else None,
            'edge_density_km_per_km2': float(edge_density) if edge_density is not None else None,
            'num_categories': len(result['land_category'].unique()),
            'total_patches': len(result),
            'landscape_area_km2': float(result['area_km2'].sum())
        }
        
        logger.info(f"Métricas de paisagem calculadas para {len(result['land_category'].unique())} categorias de uso do solo")
        
    except Exception as e:
        logger.error(f"Erro ao calcular métricas de paisagem: {str(e)}")
    
    return result

def create_landuse_adjacency(gdf):
    """
    Cria uma matriz de adjacência para categorias de uso do solo.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de uso do solo classificados
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com informações de adjacência
        pd.DataFrame: Matriz de adjacência
    """
    logger.info("Calculando matriz de adjacência para categorias de uso do solo")
    
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    try:
        # Inicializar matriz de adjacência
        categories = sorted(result['land_category'].unique())
        adjacency_matrix = pd.DataFrame(0, index=categories, columns=categories)
        
        # Adicionar coluna com IDs para melhor rastreamento
        result['landuse_id'] = range(len(result))
        
        # Comparar cada polígono com todos os outros (pode ser otimizado)
        # Apenas contando a adjacência entre categorias diferentes
        n_polygons = len(result)
        
        # Para grandes conjuntos de dados, podemos usar o RTTree para otimizar
        if n_polygons > 1000:
            logger.info("Conjunto de dados grande. Usando abordagem otimizada para cálculo de adjacência.")
            
            # Inicializar contagens de adjacência
            adjacency_counts = {}
            unique_borders = set()
            
            # Processar por lotes para evitar memória excessiva
            batch_size = 100
            for i in range(0, n_polygons, batch_size):
                end_idx = min(i + batch_size, n_polygons)
                batch = result.iloc[i:end_idx]
                
                # Para cada polígono no lote
                for idx1, row1 in tqdm(batch.iterrows(), total=len(batch), 
                                       desc=f"Processando lote {i//batch_size + 1}/{(n_polygons+batch_size-1)//batch_size}"):
                    poly1 = row1.geometry
                    cat1 = row1['land_category']
                    id1 = row1['landuse_id']
                    
                    # Obter possíveis vizinhos usando bounding box
                    bbox = poly1.bounds
                    potential_neighbors = result[result.geometry.intersects(poly1.buffer(0.0001))]
                    
                    # Verificar vizinhança real
                    for idx2, row2 in potential_neighbors.iterrows():
                        if row2['landuse_id'] == id1:
                            continue  # Pular o próprio polígono
                        
                        poly2 = row2.geometry
                        cat2 = row2['land_category']
                        id2 = row2['landuse_id']
                        
                        # Verificar adjacência real (compartilham borda)
                        if poly1.touches(poly2):
                            # Adicionar à matriz (contagem simétrica)
                            adjacency_matrix.at[cat1, cat2] += 1
                            adjacency_matrix.at[cat2, cat1] += 1
                            
                            # Registrar borda única
                            border_pair = tuple(sorted([id1, id2]))
                            unique_borders.add(border_pair)
                            
                            # Incrementar contagens
                            pair = tuple(sorted([cat1, cat2]))
                            if pair in adjacency_counts:
                                adjacency_counts[pair] += 1
                            else:
                                adjacency_counts[pair] = 1
            
            # Corrigir matriz para contar apenas bordas únicas
            for cat1 in categories:
                for cat2 in categories:
                    if cat1 != cat2:
                        pair = tuple(sorted([cat1, cat2]))
                        if pair in adjacency_counts:
                            adjacency_matrix.at[cat1, cat2] = adjacency_counts[pair]
                            adjacency_matrix.at[cat2, cat1] = adjacency_counts[pair]
        
        else:
            # Abordagem direta para conjuntos de dados menores
            for i, (idx1, row1) in enumerate(tqdm(result.iterrows(), total=n_polygons, desc="Calculando adjacência")):
                for idx2, row2 in result.iloc[i+1:].iterrows():
                    # Verificar se os polígonos são adjacentes (compartilham bordas)
                    if row1.geometry.touches(row2.geometry):
                        cat1 = row1['land_category']
                        cat2 = row2['land_category']
                        
                        # Incrementar contagem na matriz
                        adjacency_matrix.at[cat1, cat2] += 1
                        adjacency_matrix.at[cat2, cat1] += 1
        
        # Converter adjacency_matrix para valores inteiros
        adjacency_matrix = adjacency_matrix.astype(int)
        
        # Armazenar matriz como atributo do dataframe
        result.attrs['adjacency_matrix'] = adjacency_matrix.to_dict()
        
        # Calcular índice de entrelaçamento (Interspersion and Juxtaposition Index - IJI)
        # Baseado na proporção de bordas adjacentes entre diferentes classes
        iji_by_category = {}
        
        for cat in categories:
            # Total de bordas com outras categorias
            total_edges = adjacency_matrix.loc[cat].sum()
            
            if total_edges > 0:
                # Probabilidade de adjacência com cada outra categoria
                edge_proportions = [adjacency_matrix.at[cat, other_cat] / total_edges 
                                   for other_cat in categories if other_cat != cat and adjacency_matrix.at[cat, other_cat] > 0]
                
                # Índice de entrelaçamento (quanto maior, mais misturado com outros usos)
                if edge_proportions:
                    iji = -sum(p * np.log(p) for p in edge_proportions) * 100 / np.log(len(categories) - 1)
                    iji_by_category[cat] = float(iji)
                else:
                    iji_by_category[cat] = 0.0
            else:
                iji_by_category[cat] = 0.0
        
        # Armazenar índice de entrelaçamento como atributo
        result.attrs['interspersion_index'] = iji_by_category
        
        logger.info(f"Matriz de adjacência calculada para {len(categories)} categorias de uso do solo")
        
        return result, adjacency_matrix
        
    except Exception as e:
        logger.error(f"Erro ao calcular matriz de adjacência: {str(e)}")
        return result, None

def save_enriched_data(gdf, output_file=None):
    """
    Salva os dados enriquecidos em um arquivo GeoPackage.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de uso do solo enriquecidos
        output_file (str, optional): Caminho para o arquivo de saída
        
    Returns:
        str: Caminho do arquivo salvo
    """
    # Definir nome de arquivo padrão se não fornecido
    if output_file is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(OUTPUT_DIR, f"landuse_enriched_{timestamp}.gpkg")
    
    try:
        logger.info(f"Salvando dados enriquecidos em {output_file}")
        
        # Garantir que o diretório existe
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Fazer uma cópia para remover atributos que não podem ser salvos
        save_gdf = gdf.copy()
        
        # Converter NaN para None para evitar problemas de serialização
        for col in save_gdf.columns:
            if save_gdf[col].dtype == 'float64':
                save_gdf[col] = save_gdf[col].astype(object)
                save_gdf[col] = save_gdf[col].where(~save_gdf[col].isna(), None)
        
        # Salvar o GeoDataFrame
        save_gdf.to_file(output_file, driver='GPKG')
        
        # Salvar atributos em um arquivo JSON separado
        attrs_file = os.path.splitext(output_file)[0] + "_metadata.json"
        
        with open(attrs_file, 'w', encoding='utf-8') as f:
            json.dump(gdf.attrs, f, indent=4, cls=NpEncoder)
        
        logger.info(f"Dados enriquecidos salvos em {output_file}")
        logger.info(f"Metadados salvos em {attrs_file}")
        
        return output_file
        
    except Exception as e:
        logger.error(f"Erro ao salvar dados enriquecidos: {str(e)}")
        return None

def generate_improved_visualizations(gdf):
    """
    Gera visualizações melhoradas para os dados de uso do solo enriquecidos.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de uso do solo enriquecidos
        
    Returns:
        dict: Dicionário com caminhos das visualizações geradas
    """
    logger.info("Gerando visualizações melhoradas")
    
    # Dicionário para armazenar caminhos das visualizações
    viz_paths = {}
    
    try:
        # Definir paletas de cores consistentes para categorias de uso do solo
        land_colors = {
            'urban': '#ff0000',       # Vermelho
            'green': '#00ff00',       # Verde
            'forest': '#004400',      # Verde escuro
            'agriculture': '#ffff00', # Amarelo
            'water': '#0000ff',       # Azul
            'institutional': '#800080', # Roxo
            'extraction': '#964B00',  # Marrom
            'other': '#808080'        # Cinza
        }
        
        # 1. Mapa de categorias de uso do solo com alta resolução e melhor legibilidade
        logger.info("Gerando mapa de categorias de uso do solo em alta resolução")
        
        plt.figure(figsize=(20, 16), dpi=300)
        ax = plt.subplot(111)
        
        # Plotar cada categoria com cores específicas
        patches = []
        for category, color in land_colors.items():
            subset = gdf[gdf['land_category'] == category]
            if len(subset) > 0:
                subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.1, label=category)
                patches.append(mpatches.Patch(color=color, label=category))
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            logger.warning(f"Não foi possível adicionar o mapa base: {str(e)}")
        
        # Adicionar título e legenda melhorada
        plt.title('Categorias de Uso do Solo', fontsize=20, fontweight='bold')
        ax.legend(handles=patches, title='Categoria', loc='upper right', fontsize=12, title_fontsize=14)
        
        # Melhorar layout
        plt.tight_layout()
        
        # Remover eixos
        ax.set_axis_off()
        
        # Salvar figura em alta resolução
        categories_map_path = os.path.join(VISUALIZATION_DIR, 'landuse_categories_map.png')
        plt.savefig(categories_map_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        viz_paths['categories_map'] = categories_map_path
        
        # 2. Dashboard de estatísticas de uso do solo
        logger.info("Gerando dashboard de estatísticas de uso do solo")
        
        # Criar gráfico combinado com múltiplas informações
        fig = plt.figure(figsize=(20, 16), dpi=300)
        
        # Layout do dashboard
        gs = fig.add_gridspec(2, 2)
        
        # 2.1. Gráfico de área por categoria
        ax1 = fig.add_subplot(gs[0, 0])
        
        # Agrupar área por categoria
        area_by_category = gdf.groupby('land_category')['area_km2'].sum().sort_values(ascending=False)
        
        # Criar gráfico de barras com cores correspondentes às categorias
        bars = ax1.bar(area_by_category.index, area_by_category.values, 
                      color=[land_colors.get(cat, '#808080') for cat in area_by_category.index])
        
        # Adicionar valores acima das barras
        for i, bar in enumerate(bars):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{area_by_category.values[i]:.1f} km²',
                    ha='center', va='bottom', fontsize=10)
        
        # Configurar eixos e título
        ax1.set_ylabel('Área (km²)', fontsize=12)
        ax1.set_title('Área Total por Categoria', fontsize=16, fontweight='bold')
        ax1.set_xticklabels(area_by_category.index, rotation=45, ha='right')
        ax1.grid(axis='y', alpha=0.3)
        
        # 2.2. Distribuição de tamanho de polígonos
        ax2 = fig.add_subplot(gs[0, 1])
        
        # Contar número de polígonos por categoria
        count_by_category = gdf['land_category'].value_counts().sort_values(ascending=False)
        
        # Criar gráfico de pizza
        wedges, texts, autotexts = ax2.pie(
            count_by_category.values, 
            labels=count_by_category.index,
            autopct='%1.1f%%',
            colors=[land_colors.get(cat, '#808080') for cat in count_by_category.index],
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        
        # Melhorar aparência do texto
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax2.set_title('Distribuição do Número de Polígonos', fontsize=16, fontweight='bold')
        
        # 2.3. Estatísticas de elevação se disponíveis
        ax3 = fig.add_subplot(gs[1, 0])
        
        if 'elevation_mean' in gdf.columns:
            # Calcular elevação média por categoria
            elevation_by_category = gdf.groupby('land_category')['elevation_mean'].mean().sort_values(ascending=False)
            
            # Criar gráfico de barras horizontais
            bars = ax3.barh(elevation_by_category.index, elevation_by_category.values,
                           color=[land_colors.get(cat, '#808080') for cat in elevation_by_category.index])
            
            # Adicionar valores à direita das barras
            for i, bar in enumerate(bars):
                ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{elevation_by_category.values[i]:.1f} m',
                        va='center', fontsize=10)
            
            ax3.set_xlabel('Elevação Média (m)', fontsize=12)
            ax3.set_title('Elevação Média por Categoria', fontsize=16, fontweight='bold')
            ax3.grid(axis='x', alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Dados de elevação não disponíveis', 
                    ha='center', va='center', fontsize=14)
            ax3.set_axis_off()
        
        # 2.4. Estatísticas de declividade se disponíveis
        ax4 = fig.add_subplot(gs[1, 1])
        
        if 'slope_category' in gdf.columns:
            # Calcular distribuição de categorias de declividade
            slope_distribution = gdf['slope_category'].value_counts().sort_index()
            
            # Definir paleta de cores para declividade
            slope_colors = {
                'Plano': '#1a9850',
                'Suave': '#91cf60',
                'Moderado': '#fee08b',
                'Íngreme': '#fc8d59',
                'Muito íngreme': '#d73027'
            }
        
        # Criar gráfico de barras
            bars = ax4.bar(slope_distribution.index, slope_distribution.values,
                          color=[slope_colors.get(cat, '#808080') for cat in slope_distribution.index])
        
        # Adicionar valores acima das barras
        for i, bar in enumerate(bars):
                ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                        f'{slope_distribution.values[i]}',
                    ha='center', va='bottom', fontsize=10)
        
            ax4.set_ylabel('Número de Polígonos', fontsize=12)
            ax4.set_title('Distribuição de Classes de Declividade', fontsize=16, fontweight='bold')
            ax4.set_xticklabels(slope_distribution.index, rotation=45, ha='right')
            ax4.grid(axis='y', alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Dados de declividade não disponíveis', 
                    ha='center', va='center', fontsize=14)
            ax4.set_axis_off()
        
        # Ajustar layout
        plt.tight_layout()
        
        # Adicionar título principal ao dashboard
        fig.suptitle('Dashboard de Estatísticas do Uso do Solo', fontsize=22, fontweight='bold', y=0.98)
        plt.subplots_adjust(top=0.93)
        
        # Salvar dashboard
        dashboard_path = os.path.join(VISUALIZATION_DIR, 'landuse_dashboard.png')
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        viz_paths['dashboard'] = dashboard_path
        
        # 3. Mapa de elevação em formato 3D se disponível
        if 'elevation_mean' in gdf.columns:
            logger.info("Gerando visualização 3D de elevação")
            
            try:
                from mpl_toolkits.mplot3d import Axes3D
                
                fig = plt.figure(figsize=(20, 16), dpi=200)
                ax = fig.add_subplot(111, projection='3d')
                
                # Para cada polígono, obter o centroide e usar elevation_mean como altura
                xs = []
                ys = []
                zs = []
                colors = []
                sizes = []
                
                for idx, row in gdf.iterrows():
                    if not pd.isna(row['elevation_mean']):
                        centroid = row.geometry.centroid
                        xs.append(centroid.x)
                        ys.append(centroid.y)
                        zs.append(row['elevation_mean'])
                        colors.append(land_colors.get(row['land_category'], '#808080'))
                        sizes.append(row['area_km2'] * 10)  # Ajustar tamanho baseado na área
                
                # Plotar pontos 3D
                scatter = ax.scatter(xs, ys, zs, c=colors, s=sizes, alpha=0.6)
                
                # Configurar visualização
                ax.set_xlabel('Longitude', fontsize=12)
                ax.set_ylabel('Latitude', fontsize=12)
                ax.set_zlabel('Elevação (m)', fontsize=12)
                ax.set_title('Visualização 3D do Uso do Solo por Elevação', fontsize=20, fontweight='bold')
                
                # Adicionar legenda
                legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=color, markersize=10, label=cat)
                                  for cat, color in land_colors.items() if cat in gdf['land_category'].unique()]
                
                ax.legend(handles=legend_elements, title='Categoria', loc='upper right')
            
            # Salvar figura
                elevation_3d_path = os.path.join(VISUALIZATION_DIR, 'landuse_elevation_3d.png')
                plt.savefig(elevation_3d_path, dpi=300, bbox_inches='tight')
            plt.close()
            
                viz_paths['elevation_3d'] = elevation_3d_path
                
            except ImportError as e:
                logger.warning(f"Não foi possível gerar visualização 3D: {str(e)}")
        
        # 4. Mapa interativo melhorado usando folium
        logger.info("Gerando mapa interativo aprimorado")
        
        # Converter para WGS84 para compatibilidade com folium
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        
        # Calcular centro do mapa
        center_lat = gdf_wgs84.geometry.centroid.y.mean()
        center_lon = gdf_wgs84.geometry.centroid.x.mean()
        
        # Criar mapa com camadas de tile melhores
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12, 
                      tiles='CartoDB positron', control_scale=True)
        
        # Adicionar opções de camadas base
        folium.TileLayer('CartoDB dark_matter', name='Dark Mode').add_to(m)
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
        folium.TileLayer('Stamen Terrain', name='Terrain', attr='Map tiles by <a href="http://stamen.com">Stamen Design</a>, under <a href="http://creativecommons.org/licenses/by/3.0">CC BY 3.0</a>. Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, under <a href="http://www.openstreetmap.org/copyright">ODbL</a>.').add_to(m)
        
        # Função para escolher cor baseada na categoria
        def style_function(feature):
            category = feature['properties']['land_category']
            return {
                'fillColor': land_colors.get(category, '#808080'),
                'color': 'white',
                'weight': 1,
                'fillOpacity': 0.7
            }
        
        # Agrupar por categoria
        for category in gdf_wgs84['land_category'].unique():
            subset = gdf_wgs84[gdf_wgs84['land_category'] == category]
            
            # Criar camada para cada categoria
            category_layer = folium.FeatureGroup(name=f"Uso do Solo: {category}")
            
            # Adicionar GeoJSON para a categoria
        folium.GeoJson(
                subset.__geo_interface__,
                style_function=lambda x, cat=category: {
                    'fillColor': land_colors.get(cat, '#808080'),
                    'color': 'white',
                    'weight': 1,
                    'fillOpacity': 0.7
                },
            tooltip=folium.GeoJsonTooltip(
                fields=['land_category', 'area_km2'],
                aliases=['Categoria', 'Área (km²)'],
                localize=True
            ),
            popup=folium.GeoJsonPopup(
                fields=['land_category', 'area_km2', 'elevation_mean', 'slope_mean'],
                aliases=['Categoria', 'Área (km²)', 'Elevação Média (m)', 'Declividade Média (°)'],
                localize=True
            )
            ).add_to(category_layer)
            
            category_layer.add_to(m)
        
        # Adicionar camada com clustering de pontos por categorias
        if len(gdf_wgs84) > 0:
            marker_cluster = MarkerCluster(name="Centroides por Categoria").add_to(m)
            
            for idx, row in gdf_wgs84.iterrows():
                centroid = row.geometry.centroid
                
                # Criar popup detalhado
                popup_html = f"""
                <div style="width: 300px;">
                    <h4 style="color: {land_colors.get(row['land_category'], '#808080')};">
                        {row['land_category'].upper()}
                    </h4>
                    <table style="width: 100%;">
                        <tr><td><b>Área:</b></td><td>{row['area_km2']:.2f} km²</td></tr>
                """
                
                # Adicionar dados de elevação se disponíveis
                if 'elevation_mean' in row and not pd.isna(row['elevation_mean']):
                    popup_html += f"<tr><td><b>Elevação média:</b></td><td>{row['elevation_mean']:.1f} m</td></tr>"
                
                # Adicionar dados de declividade se disponíveis
                if 'slope_mean' in row and not pd.isna(row['slope_mean']):
                    popup_html += f"<tr><td><b>Declividade média:</b></td><td>{row['slope_mean']:.1f}°</td></tr>"
                
                # Adicionar compacidade se disponível
                if 'compactness' in row and not pd.isna(row['compactness']):
                    popup_html += f"<tr><td><b>Compacidade:</b></td><td>{row['compactness']:.2f}</td></tr>"
                
                popup_html += """
                    </table>
                </div>
                """
                
                # Adicionar marcador
                folium.Marker(
                    location=[centroid.y, centroid.x],
                    popup=folium.Popup(popup_html, max_width=350),
                    tooltip=row['land_category'],
                    icon=folium.Icon(color='lightgray', icon_color=land_colors.get(row['land_category'], '#808080'),
                                    icon='info-sign')
                ).add_to(marker_cluster)
        
        # Adicionar controle de camadas
        folium.LayerControl(collapsed=False).add_to(m)
        
        # Adicionar mini mapa
        from folium.plugins import MiniMap
        minimap = MiniMap(toggle_display=True)
        m.add_child(minimap)
        
        # Adicionar controle de tela cheia
        from folium.plugins import Fullscreen
        Fullscreen().add_to(m)
        
        # Adicionar controle de desenho
        from folium.plugins import Draw
        Draw(export=True).add_to(m)
        
        # Salvar mapa
        interactive_map_path = os.path.join(VISUALIZATION_DIR, 'landuse_interactive_map.html')
        m.save(interactive_map_path)
        
        viz_paths['interactive_map'] = interactive_map_path
        
        logger.info(f"Visualizações melhoradas geradas com sucesso. Total: {len(viz_paths)}")
        return viz_paths
        
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações melhoradas: {str(e)}")
        return viz_paths  # Retorna o que conseguiu gerar até o momento

def generate_quality_report(original_gdf, enriched_gdf, output_file, visualization_paths=None):
    """
    Gera um relatório de qualidade para os dados de uso do solo enriquecidos.
    
    Args:
        original_gdf (geopandas.GeoDataFrame): Dados originais
        enriched_gdf (geopandas.GeoDataFrame): Dados enriquecidos
        output_file (str): Caminho do arquivo de dados enriquecidos
        visualization_paths (dict, optional): Dicionário com caminhos das visualizações
        
    Returns:
        str: Caminho do arquivo de relatório
    """
    logger.info("Gerando relatório de qualidade")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(REPORT_DIR, f"landuse_quality_report_{timestamp}.json")
    
    try:
        # Criar relatório
        report = {
            "timestamp": datetime.datetime.now().isoformat(),
            "data_summary": {
                "total_features": len(enriched_gdf),
                "crs": str(enriched_gdf.crs),
                "geometry_types": [str(geom_type) for geom_type in enriched_gdf.geometry.geom_type.unique()],
                "total_columns": len(enriched_gdf.columns),
                "memory_usage_mb": enriched_gdf.memory_usage(deep=True).sum() / (1024 * 1024)
            },
            "enrichment_summary": {
                "original_columns": list(original_gdf.columns),
                "new_columns": [col for col in enriched_gdf.columns if col not in original_gdf.columns],
                "new_categories": list(enriched_gdf['land_category'].unique()) if 'land_category' in enriched_gdf.columns else [],
                "enriched_data_path": output_file,
                "visualization_paths": visualization_paths or {}
            }
        }
        
        # Adicionar estatísticas de uso do solo
        if 'landuse_stats' in enriched_gdf.attrs:
            report["landuse_statistics"] = enriched_gdf.attrs['landuse_stats']
        
        # Adicionar estatísticas numéricas
        numeric_columns = [col for col in enriched_gdf.columns if pd.api.types.is_numeric_dtype(enriched_gdf[col])]
        report["numeric_statistics"] = {}
        
        for col in numeric_columns:
            values = enriched_gdf[col].dropna()
            if len(values) > 0:
                report["numeric_statistics"][col] = {
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "median": float(values.median())
                }
        
        # Adicionar estatísticas de elevação
        if 'elevation_stats' in enriched_gdf.attrs:
            report["elevation_statistics"] = enriched_gdf.attrs['elevation_stats']
        
        # Adicionar estatísticas de terreno
        if 'terrain_stats' in enriched_gdf.attrs:
            report["terrain_statistics"] = enriched_gdf.attrs['terrain_stats']
        
        # Adicionar métricas de paisagem
        if 'landscape_metrics' in enriched_gdf.attrs:
            report["landscape_metrics"] = enriched_gdf.attrs['landscape_metrics']
        
        # Adicionar estatísticas de adjacência
        if 'adjacency_matrix' in enriched_gdf.attrs:
            report["adjacency_matrix"] = enriched_gdf.attrs['adjacency_matrix']
        
        if 'interspersion_index' in enriched_gdf.attrs:
            report["interspersion_index"] = enriched_gdf.attrs['interspersion_index']
        
        # Adicionar informações de topologia
        report["topology"] = {
            "total_features": len(enriched_gdf),
            "invalid_geometries": len(enriched_gdf[~enriched_gdf.geometry.is_valid]),
            "multipolygon_count": len(enriched_gdf[enriched_gdf.geometry.geom_type == 'MultiPolygon']),
            "polygon_count": len(enriched_gdf[enriched_gdf.geometry.geom_type == 'Polygon']),
            "self_intersections": 0,  # Placeholder, requires more complex calculation
            "potential_overlaps": 0    # Placeholder, requires more complex calculation
        }
        
        # Adicionar informações de limites (bounds)
        bounds = enriched_gdf.total_bounds
        report["bounds"] = {
            "minx": float(bounds[0]),
            "miny": float(bounds[1]),
            "maxx": float(bounds[2]),
            "maxy": float(bounds[3])
        }
        
        # Salvar relatório como JSON
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, cls=NpEncoder)
        
        logger.info(f"Relatório de qualidade salvo em {report_file}")
        return report_file
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório de qualidade: {str(e)}")
        return None

def main():
    """
    Função principal para executar todo o pipeline de enriquecimento de dados de uso do solo.
    """
    start_time = time.time()
    logger.info("Iniciando processamento de enriquecimento de dados de uso do solo")
    
    # 1. Carregar dados de uso do solo
    original_gdf = load_data()
    if original_gdf is None:
        logger.error("Falha ao carregar dados. Abortando processamento.")
        return
    
    # 2. Carregar DEM (Modelo Digital de Elevação)
    dem = load_dem()
    
    # 3. Classificar uso do solo em categorias e calcular estatísticas
    logger.info("Classificando uso do solo e calculando estatísticas básicas")
    enriched_gdf = classify_landuse(original_gdf)
    
    # 4. Extrair dados de elevação do DEM
    if dem is not None:
        logger.info("Extraindo dados de elevação do DEM")
        enriched_gdf = extract_elevation_data(enriched_gdf, dem)
        
        logger.info("Calculando declividade e aspecto")
        enriched_gdf = calculate_slope_aspect(enriched_gdf, dem)
    else:
        logger.warning("DEM não carregado. Pulando análises altimétricas.")
    
    # 5. Calcular métricas de paisagem
    logger.info("Calculando métricas de paisagem")
    enriched_gdf = calculate_landscape_metrics(enriched_gdf)
    
    # 6. Salvar dados enriquecidos
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"landuse_enriched_{timestamp}.gpkg")
    output_file = save_enriched_data(enriched_gdf, output_file)
    
    if output_file is None:
        logger.error("Falha ao salvar dados enriquecidos.")
        return
    
    # 7. Gerar visualizações melhoradas
    logger.info("Gerando visualizações melhoradas")
    visualization_paths = generate_improved_visualizations(enriched_gdf)
    
    # 8. Gerar relatório de qualidade
    logger.info("Gerando relatório de qualidade")
    report_file = generate_quality_report(original_gdf, enriched_gdf, output_file, visualization_paths)
    
    # Finalizar
    elapsed_time = time.time() - start_time
    logger.info(f"Processamento concluído em {elapsed_time:.2f} segundos")
    
    if report_file:
        logger.info(f"Relatório gerado em: {report_file}")
    
    logger.info(f"Dados enriquecidos salvos em: {output_file}")
    logger.info(f"Visualizações salvas em: {VISUALIZATION_DIR}")
    
    return {
        'enriched_data': enriched_gdf,
        'output_file': output_file,
        'report_file': report_file,
        'visualization_paths': visualization_paths
    }

if __name__ == "__main__":
    main()