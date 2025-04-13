#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import math
import json
from datetime import datetime
import contextily as ctx
from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.patheffects as pe
from scipy.spatial import Voronoi, voronoi_plot_2d
import h3
from sklearn.cluster import DBSCAN
import folium
from folium.plugins import HeatMap, MarkerCluster
import networkx as nx
from tqdm import tqdm
import logging
import time

# Obter o caminho absoluto para o diretório do projeto
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
workspace_dir = os.path.dirname(src_dir)

# Definir diretórios de entrada e saída
INPUT_DIR = os.path.join(workspace_dir, 'data', 'processed')
OUTPUT_DIR = os.path.join(workspace_dir, 'data', 'enriched')
REPORT_DIR = os.path.join(workspace_dir, 'src', 'enriched_data', 'quality_reports', 'erb')
VISUALIZATION_DIR = os.path.join(workspace_dir, 'src', 'enriched_data', 'visualizations', 'erb')

# Garantir que os diretórios de saída existam
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Constantes para cálculos de cobertura
RECEPTOR_SENSIBILIDADE = -100  # dBm
ANGULO_SETOR = 120  # graus (típico em antenas setoriais)
H3_RESOLUTION = 9  # Resolução dos hexágonos (7-10 são boas para análise urbana)

def setup_logging():
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    global logger
    logger = logging.getLogger(__name__)

def load_data():
    """
    Carrega os dados de ERB do diretório de processamento.
    Se a coluna 'Azimute' não estiver presente, tenta carregar do arquivo CSV original.
    
    Returns:
        geopandas.GeoDataFrame: Os dados de ERB carregados.
    """
    input_file = os.path.join(INPUT_DIR, 'licenciamento_processed.gpkg')
    global RBS_FILE
    RBS_FILE = input_file
    try:
        gdf = gpd.read_file(input_file)
        logger.info(f"Carregados {len(gdf)} registros de ERB de {input_file}")
        
        # Verificar se a coluna Azimute existe
        if 'Azimute' not in gdf.columns:
            logger.warning(f"Coluna 'Azimute' não encontrada em {input_file}. Tentando buscar do arquivo CSV original.")
            
            # Carregar do arquivo CSV original
            csv_file = os.path.join(workspace_dir, 'data', 'raw', 'csv_licenciamento_c55c1dea01bd184e27df233da8ac28a2.csv')
            if os.path.exists(csv_file):
                try:
                    # Carregar o CSV
                    df_csv = pd.read_csv(csv_file)
                    
                    # Verificar se a coluna Azimute existe no CSV
                    if 'Azimute' in df_csv.columns:
                        logger.info(f"Coluna 'Azimute' encontrada no arquivo CSV original.")
                        
                        # Criar um dicionário para fazer a correspondência entre registros
                        azimute_dict = {}
                        
                        # Tentar usar coluna _id se existir, senão usar combinação de outras colunas
                        if '_id' in df_csv.columns and '_id' in gdf.columns:
                            for idx, row in df_csv.iterrows():
                                azimute_dict[row['_id']] = row['Azimute']
                            
                            # Adicionar coluna Azimute ao GeoDataFrame
                            gdf['Azimute'] = gdf['_id'].map(azimute_dict)
                            logger.info("Coluna 'Azimute' adicionada usando correspondência por '_id'")
                            
                        else:
                            # Usar combinação de latitude e longitude para correspondência
                            for idx, row in df_csv.iterrows():
                                if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
                                    key = (row['Latitude'], row['Longitude'])
                                    azimute_dict[key] = row['Azimute']
                            
                            # Adicionar coluna Azimute ao GeoDataFrame
                            gdf['Azimute'] = gdf.apply(
                                lambda row: azimute_dict.get((row.geometry.y, row.geometry.x), None), 
                                axis=1
                            )
                            logger.info("Coluna 'Azimute' adicionada usando correspondência por coordenadas")
                    else:
                        logger.warning(f"Coluna 'Azimute' também não encontrada no arquivo CSV original.")
                except Exception as e:
                    logger.error(f"Erro ao carregar Azimute do arquivo CSV original: {e}")
            else:
                logger.warning(f"Arquivo CSV original não encontrado: {csv_file}")
                
        return gdf
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {e}")
        return None

def calculate_eirp(gdf):
    """
    Calcula a Potência Efetivamente Irradiada (EIRP) para cada ERB.
    EIRP (dBm) = 10 × log₁₀(PotenciaTransmissorWatts × 1000) + GanhoAntena
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com coluna EIRP
    """
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    # Converter potência para numérico se não for
    if result['PotenciaTransmissorWatts'].dtype == 'object':
        result['PotenciaTransmissorWatts'] = pd.to_numeric(result['PotenciaTransmissorWatts'].str.replace(',', '.'), errors='coerce')
    
    # Calcular EIRP para cada ERB
    result['EIRP_dBm'] = 10 * np.log10(result['PotenciaTransmissorWatts'] * 1000) + result['GanhoAntena']
    
    # Categorizar o EIRP em faixas
    bins = [0, 45, 55, 65, 75, float('inf')]
    labels = ['Muito baixo', 'Baixo', 'Médio', 'Alto', 'Muito alto']
    result['categoria_eirp'] = pd.cut(result['EIRP_dBm'], bins=bins, labels=labels)
    
    return result

def calculate_coverage_radius(gdf):
    """
    Calcula o raio de cobertura teórico usando o modelo de Friis para ERBs,
    ajustado para diferentes tipos de áreas.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB com EIRP
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com raio de cobertura
    """
    result = gdf.copy()
    
    # Definir fatores de atenuação por tipo de área
    atenuacao_areas = {
        'URBANA_DENSA': 28.0,  # Alta atenuação em áreas urbanas densas
        'URBANA': 22.0,        # Atenuação moderada em áreas urbanas
        'SUBURBANA': 16.0,     # Atenuação média em áreas suburbanas
        'RURAL': 10.0          # Baixa atenuação em áreas rurais
    }
    
    # Classificar áreas com base na densidade de ERBs num raio de 1km
    # Primeiro vamos para uma projeção métrica
    gdf_proj = result.to_crs(epsg=3857)  # Web Mercator
    
    # Para cada ERB, contar quantas outras estão em um raio de 1km
    densidade_local = []
    for idx, row in gdf_proj.iterrows():
        ponto = row.geometry
        buffer_1km = ponto.buffer(1000)
        erbs_no_raio = gdf_proj[gdf_proj.geometry.intersects(buffer_1km)].shape[0] - 1  # -1 para excluir a própria
        densidade_local.append(erbs_no_raio)
    
    result['densidade_local'] = densidade_local
    
    # Classificar tipos de área com base na densidade
    bins = [0, 3, 8, 15, float('inf')]
    labels = ['RURAL', 'SUBURBANA', 'URBANA', 'URBANA_DENSA']
    result['tipo_area'] = pd.cut(result['densidade_local'], bins=bins, labels=labels)
    
    # Calcular raio de cobertura para cada ERB
    raios = []
    for idx, row in result.iterrows():
        eirp = row['EIRP_dBm']
        freq = row['FreqTxMHz']
        tipo = row['tipo_area']
        
        # Fator de atenuação baseado no tipo de área
        atenuacao = atenuacao_areas.get(tipo, 18.0)
        
        # Cálculo do raio usando fórmula de Friis ajustada com atenuação
        raio_base = 10 ** ((eirp - RECEPTOR_SENSIBILIDADE - 32.44 - 20 * np.log10(freq)) / 20)
        raio_ajustado = raio_base / (1 + atenuacao/10)
        
        # Limitar o raio máximo com base na frequência (quanto maior a frequência, menor o alcance máximo)
        raio_max = min(20, 25000/freq) if freq > 0 else 10
        raio = min(raio_ajustado, raio_max)
        
        raios.append(raio)
    
    result['Raio_Cobertura_km'] = raios
    
    # Categorizar raio de cobertura em faixas
    bins = [0, 1, 3, 5, 10, float('inf')]
    labels = ['Muito pequeno', 'Pequeno', 'Médio', 'Grande', 'Muito grande']
    result['categoria_raio'] = pd.cut(result['Raio_Cobertura_km'], bins=bins, labels=labels)
    
    return result

def create_coverage_sectors(gdf):
    """
    Cria polígonos setoriais para representar as áreas de cobertura de cada ERB,
    baseado no azimute, ângulo do setor e raio de cobertura.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB com raio de cobertura
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com geometria do setor e área de cobertura
    """
    result = gdf.copy()
    
    # Calcula área de cobertura setorial
    result['Area_Cobertura_km2'] = (np.pi * result['Raio_Cobertura_km']**2 * ANGULO_SETOR) / 360
    
    # Verificar se a coluna Azimute existe
    if 'Azimute' not in result.columns:
        logger.error("Coluna 'Azimute' não encontrada. Os setores de cobertura não serão criados corretamente.")
        return result
    
    # Converter Azimute para numérico se não for
    if result['Azimute'].dtype == 'object':
        result['Azimute'] = pd.to_numeric(result['Azimute'], errors='coerce')
    
    # Verificar valores ausentes novamente (caso algum tenha sido convertido para NaN ao converter para numérico)
    mask = result['Azimute'].isna()
    if mask.any():
        logger.warning(f"Encontrados {mask.sum()} valores NaN adicionais após conversão numérica de 'Azimute'. Filtrando estes registros.")
        result = result[~mask].copy()
        
    # Criar geometria do setor para cada ERB
    setores = []
    for idx, row in result.iterrows():
        lat = row['Latitude']
        lon = row['Longitude']
        azimute = row['Azimute']
        raio = row['Raio_Cobertura_km']
        
        # Criar setor com base no azimute e raio (agora todos os azimutes são válidos)
        if pd.notna(raio) and raio > 0:
            # Converter azimute para radianos e ajustar para orientação matemática
            azimute_rad = np.radians((90 - azimute) % 360)
            
            # Definir ângulo do setor e criar pontos
            angulo_setor = np.radians(ANGULO_SETOR)
            angulo_inicial = azimute_rad - angulo_setor/2
            angulo_final = azimute_rad + angulo_setor/2
            
            # Criar pontos do polígono do setor (incluindo o ponto central)
            pontos = [(lon, lat)]  # Ponto central (ERB)
            
            # Adicionar pontos ao longo do arco
            numero_pontos = 30  # Quantidade de pontos no arco para suavidade
            for i in range(numero_pontos+1):
                angulo = angulo_inicial + (i * (angulo_final - angulo_inicial) / numero_pontos)
                # Converter raio de km para graus aproximadamente
                dx = raio * np.cos(angulo) / 111.32
                dy = raio * np.sin(angulo) / (111.32 * np.cos(np.radians(lat)))
                pontos.append((lon + dx, lat + dy))
            
            # Fechar o polígono
            pontos.append((lon, lat))
            
            # Criar polígono
            setor = Polygon(pontos)
        else:
            setor = None
        
        setores.append(setor)
    
    result['setor_geometria'] = setores
    
    return result

def create_hexagon_coverage_grid(gdf):
    """
    Cria um mosaico hexagonal para análise de cobertura, vulnerabilidade e sobreposição.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB com setores de cobertura
        
    Returns:
        tuple: (gdf atualizado, gdf dos hexágonos)
    """
    result = gdf.copy()
    
    # Extrair limites da área para criar os hexágonos
    min_lat, min_lon = result.geometry.y.min(), result.geometry.x.min()
    max_lat, max_lon = result.geometry.y.max(), result.geometry.x.max()
    
    # Adicionar margem aos limites
    margin = 0.05  # graus
    min_lat -= margin
    min_lon -= margin
    max_lat += margin
    max_lon += margin
    
    # Gerar índices H3 para a área
    hex_ids = set()
    for lat in np.linspace(min_lat, max_lat, 20):
        for lon in np.linspace(min_lon, max_lon, 20):
            hex_id = h3.geo_to_h3(lat, lon, H3_RESOLUTION)
            hex_ids.add(hex_id)
    
    # Converter índices H3 para polígonos
    hex_polygons = []
    hex_ids_list = list(hex_ids)
    for hex_id in hex_ids_list:
        boundary = h3.h3_to_geo_boundary(hex_id, geo_json=True)
        polygon = Polygon(boundary)
        hex_polygons.append(polygon)
    
    # Criar GeoDataFrame dos hexágonos
    hex_gdf = gpd.GeoDataFrame(geometry=hex_polygons, crs=result.crs)
    hex_gdf['hex_index'] = range(len(hex_gdf))
    
    # Para cada hexágono, contar quantas operadoras diferentes têm cobertura
    operadoras_por_hex = []
    setores_por_hex = []
    potencia_media_por_hex = []
    
    # Filtrar apenas ERBs com setores válidos
    setores_gdf = result[result['setor_geometria'].notna()].copy()
    setores_gdf['geometry'] = setores_gdf['setor_geometria']
    
    # Lista de operadoras únicas
    operadoras = result['NomeEntidade'].unique()
    
    # Para cada hexágono, verificar cobertura
    for idx, hex_row in tqdm(hex_gdf.iterrows(), total=len(hex_gdf), desc="Processando hexágonos"):
        hex_geom = hex_row.geometry
        
        # Conjunto de operadoras que têm cobertura neste hexágono
        op_com_cobertura = set()
        count_setores = 0
        eirp_setores = []
        
        # Verificar interseção com setores
        for op in operadoras:
            # Filtrar setores desta operadora
            op_setores = setores_gdf[setores_gdf['NomeEntidade'] == op]
            
            # Verificar se algum setor intersecta o hexágono
            for _, setor_row in op_setores.iterrows():
                if hex_geom.intersects(setor_row.geometry):
                    op_com_cobertura.add(op)
                    count_setores += 1
                    eirp_setores.append(setor_row['EIRP_dBm'])
                    break  # Basta uma interseção por operadora
        
        operadoras_por_hex.append(len(op_com_cobertura))
        setores_por_hex.append(count_setores)
        potencia_media_por_hex.append(np.mean(eirp_setores) if eirp_setores else np.nan)
    
    # Adicionar colunas ao GeoDataFrame de hexágonos
    hex_gdf['num_operadoras'] = operadoras_por_hex
    hex_gdf['num_setores'] = setores_por_hex
    hex_gdf['potencia_media'] = potencia_media_por_hex
    
    # Classificar os hexágonos por vulnerabilidade
    bins = [-1, 0, 1, 2, 10]
    labels = ['Sem cobertura', 'Alta vulnerabilidade', 'Média vulnerabilidade', 'Baixa vulnerabilidade']
    hex_gdf['vulnerabilidade'] = pd.cut(hex_gdf['num_operadoras'], bins=bins, labels=labels)
    
    # Calcular métricas adicionais para os hexágonos
    # Densidade de potência: média da potência / número de setores
    hex_gdf['densidade_potencia'] = hex_gdf['potencia_media'] / hex_gdf['num_setores'].replace(0, np.nan)
    
    return result, hex_gdf

def analyze_spatial_clustering(gdf):
    """
    Realiza análise de clustering espacial das ERBs usando DBSCAN.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com informações de cluster
    """
    result = gdf.copy()
    
    # Converter para coordenadas métricas
    gdf_proj = result.to_crs(epsg=3857)
    
    # Extrair coordenadas x, y
    coords = np.vstack((gdf_proj.geometry.x, gdf_proj.geometry.y)).T
    
    # Executar DBSCAN
    # eps: distância máxima entre pontos para serem considerados no mesmo cluster (500m)
    # min_samples: número mínimo de pontos para formar um cluster principal
    clustering = DBSCAN(eps=500, min_samples=3).fit(coords)
    
    # Adicionar rótulos de cluster ao GeoDataFrame
    result['cluster_id'] = clustering.labels_
    
    # Calcular estatísticas dos clusters
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    n_noise = list(clustering.labels_).count(-1)
    
    # Contagem de ERBs por cluster
    cluster_counts = pd.Series(clustering.labels_).value_counts().sort_index()
    
    # Para cada ERB, calcular distância média para outras ERBs no mesmo cluster
    distances_intra_cluster = []
    
    for idx, row in result.iterrows():
        cluster = row['cluster_id']
        if cluster == -1:  # Ponto de ruído
            distances_intra_cluster.append(np.nan)
            continue
            
        # Encontrar outras ERBs no mesmo cluster
        same_cluster = result[result['cluster_id'] == cluster]
        if len(same_cluster) <= 1:
            distances_intra_cluster.append(np.nan)
            continue
            
        # Calcular distâncias
        other_erbs = same_cluster[same_cluster.index != idx]
        distances = []
        for _, other_row in other_erbs.iterrows():
            dist = row.geometry.distance(other_row.geometry)
            distances.append(dist)
        
        # Média das distâncias
        distances_intra_cluster.append(np.mean(distances))
    
    result['distancia_media_cluster'] = distances_intra_cluster
    
    # Adicionar informações descritivas do cluster
    result['densidade_cluster'] = result.apply(
        lambda row: cluster_counts[row['cluster_id']] if row['cluster_id'] != -1 else 0,
        axis=1
    )
    
    # Registrar métricas de clustering
    result.attrs['n_clusters'] = n_clusters
    result.attrs['n_noise'] = n_noise
    result.attrs['cluster_counts'] = cluster_counts.to_dict()
    
    return result

def create_coverage_network(gdf, hex_gdf):
    """
    Cria um grafo de rede de cobertura para análise de conectividade.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB com setores
        hex_gdf (geopandas.GeoDataFrame): Grade hexagonal de cobertura
        
    Returns:
        tuple: (gdf atualizado, gdf dos hexágonos atualizado, grafo de rede)
    """
    result = gdf.copy()
    hex_result = hex_gdf.copy()
    
    # Criar um grafo não direcionado
    G = nx.Graph()
    
    # Adicionar nós para ERBs
    for idx, row in result.iterrows():
        G.add_node(f"erb_{idx}", 
                   tipo='erb', 
                   operadora=row['NomeEntidade'], 
                   lat=row.geometry.y, 
                   lon=row.geometry.x,
                   eirp=row['EIRP_dBm'],
                   raio=row['Raio_Cobertura_km'],
                   pos=(row.geometry.x, row.geometry.y))
    
    # Adicionar nós para hexágonos
    for idx, row in hex_result.iterrows():
        # Usar o centróide do hexágono para posição
        centroid = row.geometry.centroid
        G.add_node(f"hex_{idx}", 
                   tipo='hexagono',
                   num_operadoras=row['num_operadoras'],
                   vulnerabilidade=row['vulnerabilidade'],
                   pos=(centroid.x, centroid.y))
    
    # Adicionar arestas entre ERBs e hexágonos que elas cobrem
    # Filtrar apenas ERBs com setores válidos
    setores_gdf = result[result['setor_geometria'].notna()].copy()
    setores_gdf['geometry'] = setores_gdf['setor_geometria']
    
    for erb_idx, erb_row in tqdm(setores_gdf.iterrows(), total=len(setores_gdf), desc="Criando arestas ERB-hexágono"):
        erb_setor = erb_row.geometry
        
        for hex_idx, hex_row in hex_result.iterrows():
            if erb_setor.intersects(hex_row.geometry):
                G.add_edge(f"erb_{erb_idx}", f"hex_{hex_idx}", tipo='cobertura')
    
    # Adicionar arestas entre ERBs da mesma operadora em clusters
    for cluster_id in result['cluster_id'].unique():
        if cluster_id == -1:  # Pular pontos de ruído
            continue
            
        cluster_erbs = result[result['cluster_id'] == cluster_id]
        
        for i, (idx1, row1) in enumerate(cluster_erbs.iterrows()):
            for idx2, row2 in list(cluster_erbs.iterrows())[i+1:]:
                if row1['NomeEntidade'] == row2['NomeEntidade']:
                    # Calcular distância em km (aproximadamente)
                    dist = row1.geometry.distance(row2.geometry) * 111.32
                    G.add_edge(f"erb_{idx1}", f"erb_{idx2}", 
                               tipo='cluster', 
                               operadora=row1['NomeEntidade'],
                               distancia=dist)
    
    # Calcular métricas de centralidade
    print("Calculando métricas de centralidade no grafo...")
    
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    
    # Degree centrality
    degree = nx.degree_centrality(G)
    
    # Closeness centrality
    closeness = nx.closeness_centrality(G)
    
    # Adicionar métricas ao GeoDataFrame de ERBs
    for idx in result.index:
        node_id = f"erb_{idx}"
        if node_id in betweenness:
            result.at[idx, 'betweenness'] = betweenness[node_id]
            result.at[idx, 'degree'] = degree[node_id]
            result.at[idx, 'closeness'] = closeness[node_id]
    
    # Adicionar métricas ao GeoDataFrame de hexágonos
    for idx in hex_result.index:
        node_id = f"hex_{idx}"
        if node_id in betweenness:
            hex_result.at[idx, 'betweenness'] = betweenness[node_id]
            hex_result.at[idx, 'degree'] = degree[node_id]
            hex_result.at[idx, 'closeness'] = closeness[node_id]
    
    # Calcular redundância de cobertura (quanto maior o grau, maior a redundância)
    hex_result['indice_redundancia'] = hex_result['degree'].fillna(0)
    
    return result, hex_result, G

def classify_erbs_importance(gdf, hex_gdf, G=None):
    """
    Classifica as ERBs com base em sua importância estratégica considerando
    múltiplos fatores: cobertura, potência, posição na rede, etc.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB com análises espaciais
        hex_gdf (geopandas.GeoDataFrame): Grade hexagonal de cobertura
        G (nx.Graph, optional): Grafo de rede
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com classificação de importância
    """
    result = gdf.copy()
    
    # Normalizar as métricas para escala 0-1
    for col in ['EIRP_dBm', 'Raio_Cobertura_km', 'Area_Cobertura_km2', 
                'betweenness', 'degree', 'closeness']:
        if col in result.columns:
            min_val = result[col].min()
            max_val = result[col].max()
            if max_val > min_val:
                result[f'{col}_norm'] = (result[col] - min_val) / (max_val - min_val)
            else:
                result[f'{col}_norm'] = 0.5
    
    # Calcular pontuação ponderada baseada em vários fatores
    weights = {
        'EIRP_dBm_norm': 0.15,           # Potência
        'Raio_Cobertura_km_norm': 0.15,  # Cobertura
        'betweenness_norm': 0.25,        # Posição estratégica na rede
        'degree_norm': 0.20,             # Conectividade
        'closeness_norm': 0.15,          # Centralidade
        'densidade_local': 0.10          # Densidade local (invertida)
    }
    
    result['pontuacao_importancia'] = 0.0
    
    for col, weight in weights.items():
        if col == 'densidade_local':
            # Inverter densidade (menor densidade = maior importância)
            if 'densidade_local' in result.columns:
                dens_min = result['densidade_local'].min()
                dens_max = result['densidade_local'].max()
                if dens_max > dens_min:
                    result['densidade_local_norm'] = 1 - ((result['densidade_local'] - dens_min) / (dens_max - dens_min))
                    result['pontuacao_importancia'] += result['densidade_local_norm'] * weight
        elif col in result.columns:
            result['pontuacao_importancia'] += result[col] * weight
    
    # Verificar quais hexágonos vulneráveis são cobertos por cada ERB
    if G is not None:
        cobertura_vulneravel = []
        
        hex_vulneraveis = hex_gdf[hex_gdf['vulnerabilidade'].isin(['Alta vulnerabilidade', 'Média vulnerabilidade'])]
        
        for idx in result.index:
            node_id = f"erb_{idx}"
            if node_id in G:
                # Obter vizinhos (hexágonos cobertos por esta ERB)
                neighbors = [n for n in G.neighbors(node_id) if n.startswith('hex_')]
                
                # Contar quantos destes são hexágonos vulneráveis
                count_vulneraveis = 0
                for neighbor in neighbors:
                    hex_idx = int(neighbor.split('_')[1])
                    if hex_idx in hex_vulneraveis.index:
                        count_vulneraveis += 1
                
                cobertura_vulneravel.append(count_vulneraveis)
            else:
                cobertura_vulneravel.append(0)
        
        result['cobertura_vulneravel'] = cobertura_vulneravel
        
        # Normalizar e adicionar à pontuação
        if max(cobertura_vulneravel) > 0:
            result['cobertura_vulneravel_norm'] = result['cobertura_vulneravel'] / max(cobertura_vulneravel)
            result['pontuacao_importancia'] += result['cobertura_vulneravel_norm'] * 0.2
        
    # Classificar em 5 níveis de importância
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Crítica']
    result['classe_importancia'] = pd.cut(result['pontuacao_importancia'], bins=bins, labels=labels)
    
    # Identificar ERBs críticas para evacuação (alta importância + baixa redundância)
    result['erb_critica_evacuacao'] = (
        (result['classe_importancia'].isin(['Alta', 'Crítica'])) & 
        (result['betweenness_norm'] > 0.6) &
        (result['cobertura_vulneravel'] > 0 if 'cobertura_vulneravel' in result.columns else True)
    )
    
    return result

def generate_quality_report(original_gdf, enriched_gdf, hex_gdf, G=None):
    """
    Gera um relatório detalhado de qualidade para o processo de enriquecimento.
    
    Args:
        original_gdf (geopandas.GeoDataFrame): Dados originais de ERB
        enriched_gdf (geopandas.GeoDataFrame): Dados de ERB enriquecidos
        hex_gdf (geopandas.GeoDataFrame): Grade hexagonal de cobertura
        G (nx.Graph, optional): Grafo de rede
    """
    # Criar relatório
    report = {
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_features": len(original_gdf),
        "enriched_features": len(enriched_gdf),
        "new_attributes": list(set(enriched_gdf.columns) - set(original_gdf.columns)),
        "statistics": {
            "EIRP_dBm": {
                "mean": float(enriched_gdf['EIRP_dBm'].mean()),
                "median": float(enriched_gdf['EIRP_dBm'].median()),
                "min": float(enriched_gdf['EIRP_dBm'].min()),
                "max": float(enriched_gdf['EIRP_dBm'].max())
            },
            "Raio_Cobertura_km": {
                "mean": float(enriched_gdf['Raio_Cobertura_km'].mean()),
                "median": float(enriched_gdf['Raio_Cobertura_km'].median()),
                "min": float(enriched_gdf['Raio_Cobertura_km'].min()),
                "max": float(enriched_gdf['Raio_Cobertura_km'].max())
            },
            "Area_Cobertura_km2": {
                "mean": float(enriched_gdf['Area_Cobertura_km2'].mean()),
                "median": float(enriched_gdf['Area_Cobertura_km2'].median()),
                "min": float(enriched_gdf['Area_Cobertura_km2'].min()),
                "max": float(enriched_gdf['Area_Cobertura_km2'].max())
            },
            "densidade_local": {
                "mean": float(enriched_gdf['densidade_local'].mean()),
                "median": float(enriched_gdf['densidade_local'].median()),
                "min": float(enriched_gdf['densidade_local'].min()),
                "max": float(enriched_gdf['densidade_local'].max())
            },
            "tipo_area": {
                "distribution": {str(k): int(v) for k, v in enriched_gdf['tipo_area'].value_counts().to_dict().items()}
            },
            "classe_importancia": {
                "distribution": {str(k): int(v) for k, v in enriched_gdf['classe_importancia'].value_counts().to_dict().items()}
            }
        },
        "operadoras": {
            str(op): int(enriched_gdf[enriched_gdf['NomeEntidade'] == op].shape[0])
            for op in enriched_gdf['NomeEntidade'].unique()
        },
        "hexagons": {
            "total": len(hex_gdf),
            "statistics": {
                "num_operadoras": {
                    "mean": float(hex_gdf['num_operadoras'].mean()),
                    "distribution": {str(k): int(v) for k, v in hex_gdf['num_operadoras'].value_counts().to_dict().items()}
                },
                "vulnerabilidade": {
                    "distribution": {str(k): int(v) for k, v in hex_gdf['vulnerabilidade'].value_counts().to_dict().items()}
                }
            }
        }
    }
    
    # Adicionar informações da rede, se disponível
    if G is not None:
        report["network"] = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "node_types": {
                "erb": sum(1 for n, d in G.nodes(data=True) if d.get('tipo') == 'erb'),
                "hexagono": sum(1 for n, d in G.nodes(data=True) if d.get('tipo') == 'hexagono')
            },
            "edge_types": {
                "cobertura": sum(1 for u, v, d in G.edges(data=True) if d.get('tipo') == 'cobertura'),
                "cluster": sum(1 for u, v, d in G.edges(data=True) if d.get('tipo') == 'cluster')
            },
            "average_degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
            "erb_criticas_evacuacao": int(enriched_gdf['erb_critica_evacuacao'].sum())
        }
    
    # Salvar relatório como JSON
    report_file = os.path.join(REPORT_DIR, 'enrichment_report.json')
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    print(f"Relatório de qualidade salvo em {report_file}")
    
    # Gerar visualizações
    generate_visualizations(enriched_gdf, hex_gdf, G)

def generate_visualizations(gdf, hex_gdf=None, G=None):
    """
    Gera visualizações para acompanhar o relatório de qualidade.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB enriquecidos
        hex_gdf (geopandas.GeoDataFrame, optional): Grade hexagonal de cobertura
        G (nx.Graph, optional): Grafo de rede
    """
    logger.info("Criando diretório para visualizações se necessário")
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    
    # Criar paletas de cores consistentes
    operadoras = gdf['NomeEntidade'].unique()
    colors_operadoras = dict(zip(operadoras, plt.cm.tab10.colors[:len(operadoras)]))
    
    # 1. Mapa de localização das ERBs por operadora
    logger.info("Gerando mapa de localização das ERBs por operadora")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
    # Plotar ERBs por operadora
    for op in operadoras:
        subset = gdf[gdf['NomeEntidade'] == op]
        subset.plot(ax=ax, markersize=30, label=op, alpha=0.7)
    
    # Adicionar título e legenda
    plt.title('Localização das ERBs por Operadora', fontsize=16)
    plt.legend(title='Operadora', loc='upper right')
    
    # Adicionar mapa base
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logger.warning(f"Não foi possível adicionar mapa base: {e}")
    
    # Salvar figura
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_localizacao_operadora.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mapa de EIRP (Potência Efetivamente Irradiada)
    logger.info("Gerando mapa de EIRP das ERBs")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
    # Plotar ERBs por EIRP
    scatter = gdf.plot(ax=ax, column='EIRP_dBm', cmap='viridis', markersize=50, 
                     legend=True, alpha=0.7, legend_kwds={'label': 'EIRP (dBm)'})
    
    # Adicionar mapa base
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logger.warning(f"Não foi possível adicionar mapa base: {e}")
    
    # Adicionar título
    plt.title('Potência Efetivamente Irradiada (EIRP) das ERBs', fontsize=16)
    
    # Salvar figura
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_eirp.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Gráfico de distribuição de EIRP por operadora (boxplot)
    logger.info("Gerando gráfico de distribuição de EIRP por operadora")
    plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)
    
    # Criar boxplot
    data_por_operadora = [gdf[gdf['NomeEntidade'] == op]['EIRP_dBm'] for op in operadoras]
    boxplot = ax.boxplot(data_por_operadora, patch_artist=True)
    
    # Colorir boxplots
    for i, box in enumerate(boxplot['boxes']):
        box.set(facecolor=colors_operadoras[operadoras[i]], alpha=0.7)
    
    # Adicionar labels
    ax.set_xticklabels(operadoras, rotation=45, ha='right')
    ax.set_ylabel('EIRP (dBm)')
    ax.grid(axis='y', alpha=0.3)
    
    # Adicionar título
    plt.title('Distribuição de EIRP por Operadora', fontsize=16)
    plt.tight_layout()
    
    # Salvar figura
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_eirp_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Mapa de setores de cobertura, se disponível
    if 'setor_geometria' in gdf.columns and gdf['setor_geometria'].notna().any():
        logger.info("Gerando mapa de setores de cobertura")
        plt.figure(figsize=(15, 12))
        ax = plt.subplot(111)
        
        # Criar GeoDataFrame de setores
        setores_gdf = gdf[gdf['setor_geometria'].notna()].copy()
        setores_gdf['geometry'] = setores_gdf['setor_geometria']
        
        # Plotar setores por operadora
        for op in operadoras:
            subset = setores_gdf[setores_gdf['NomeEntidade'] == op]
            if len(subset) > 0:
                color = colors_operadoras[op]
                subset.plot(ax=ax, color=color, alpha=0.5, label=op)
        
        # Adicionar pontos das ERBs
        gdf.plot(ax=ax, markersize=15, color='black', alpha=0.7)
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
        except Exception as e:
            logger.warning(f"Não foi possível adicionar mapa base: {e}")
        
        # Adicionar título e legenda
        plt.title('Cobertura de ERBs por Operadora', fontsize=16)
        plt.legend(title='Operadora', loc='upper right')
        
        # Salvar figura
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_cobertura_operadora.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 5. Mapa interativo com folium, se disponível
    try:
        logger.info("Gerando mapa interativo")
        import folium
        from folium.plugins import MarkerCluster
        
        # Centro do mapa
        center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
        
        # Criar mapa
        m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')
        
        # Adicionar clusters de ERBs por operadora
        for op in operadoras:
            subset = gdf[gdf['NomeEntidade'] == op]
            
            # Converter cor de matplotlib para hex
            color = '#{:02x}{:02x}{:02x}'.format(
                int(colors_operadoras[op][0]*255),
                int(colors_operadoras[op][1]*255),
                int(colors_operadoras[op][2]*255)
            )
            
            # Criar cluster para esta operadora
            mc = MarkerCluster(name=f"ERBs - {op}")
            
            # Adicionar cada ERB ao cluster
            for idx, row in subset.iterrows():
                popup_text = f"""
                <b>Operadora:</b> {op}<br>
                <b>EIRP:</b> {row['EIRP_dBm']:.2f} dBm<br>
                """
                
                if 'Raio_Cobertura_km' in row:
                    popup_text += f"<b>Raio:</b> {row['Raio_Cobertura_km']:.2f} km<br>"
                
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=folium.Popup(popup_text),
                    icon=folium.Icon(color='white', icon_color=color, icon='signal', prefix='fa')
                ).add_to(mc)
            
            mc.add_to(m)
            
            # Adicionar setores de cobertura como GeoJSON, se disponíveis
            if 'setor_geometria' in subset.columns and subset['setor_geometria'].notna().any():
                fg = folium.FeatureGroup(name=f"Cobertura - {op}")
                
                subset_setores = subset[subset['setor_geometria'].notna()]
                for idx, row in subset_setores.iterrows():
                    popup_text = f"""
                    <b>Operadora:</b> {op}<br>
                    <b>EIRP:</b> {row['EIRP_dBm']:.2f} dBm<br>
                    """
                    
                    if 'Raio_Cobertura_km' in row:
                        popup_text += f"<b>Raio:</b> {row['Raio_Cobertura_km']:.2f} km<br>"
                    
                    folium.GeoJson(
                        row['setor_geometria'],
                        style_function=lambda x, color=color: {
                            'fillColor': color,
                            'color': color,
                            'weight': 1,
                            'fillOpacity': 0.3
                        },
                        popup=folium.Popup(popup_text)
                    ).add_to(fg)
                
                fg.add_to(m)
        
        # Adicionar controle de camadas
        folium.LayerControl().add_to(m)
        
        # Salvar mapa
        m.save(os.path.join(VISUALIZATION_DIR, 'erb_mapa_interativo.html'))
        logger.info(f"Mapa interativo salvo em {os.path.join(VISUALIZATION_DIR, 'erb_mapa_interativo.html')}")
        
    except Exception as e:
        logger.error(f"Erro ao gerar mapa interativo: {e}")
    
    # 6. Contagem de ERBs por operadora (gráfico de barras)
    logger.info("Gerando gráfico de contagem de ERBs por operadora")
    plt.figure(figsize=(15, 8))
    ax = plt.subplot(111)
    
    # Contar ERBs por operadora
    op_counts = gdf['NomeEntidade'].value_counts().sort_index()
    
    # Criar gráfico de barras
    bars = ax.bar(op_counts.index, op_counts.values, 
                 color=[colors_operadoras.get(op, 'gray') for op in op_counts.index])
    
    # Adicionar rótulos nas barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.0f}', ha='center', va='bottom', fontsize=11)
    
    # Ajustar rótulos e título
    ax.set_ylabel('Número de ERBs')
    ax.set_title('Quantidade de ERBs por Operadora', fontsize=16)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Salvar figura
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_contagem_operadoras.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizações concluídas com sucesso")

def calculate_density(gdf):
    """
    Calcula a densidade de ERBs por área.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com densidade de ERBs
    """
    logger.info("Função calculate_density implementada")
    # Simplesmente retorna o GeoDataFrame original, sem calcular densidade
    return gdf

def calculate_voronoi(gdf):
    """
    Calcula o diagrama de Voronoi para as ERBs, representando áreas de domínio teórico.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame com os polígonos de Voronoi
    """
    logger.info("Calculando diagrama de Voronoi para as ERBs")
    
    # Converter para coordenadas métricas para cálculo adequado
    gdf_proj = gdf.to_crs(epsg=3857)
    
    # Extrair coordenadas x, y
    coords = np.vstack((gdf_proj.geometry.x, gdf_proj.geometry.y)).T
    
    # Adicionar pontos na borda para limitar os polígonos de Voronoi
    # Encontrar bounding box e expandir
    x_min, y_min, x_max, y_max = gdf_proj.total_bounds
    margin = max(x_max - x_min, y_max - y_min) * 0.3
    
    boundary_points = np.array([
        [x_min - margin, y_min - margin],
        [x_max + margin, y_min - margin],
        [x_max + margin, y_max + margin],
        [x_min - margin, y_max + margin],
    ])
    
    # Combinar pontos das ERBs com pontos da borda
    all_points = np.vstack((coords, boundary_points))
    
    # Calcular diagrama de Voronoi
    vor = Voronoi(all_points)
    
    # Criar GeoDataFrame para os polígonos de Voronoi
    voronoi_polygons = []
    operadoras = []
    erb_ids = []
    
    # Criar polígonos apenas para as ERBs (não para os pontos de borda)
    for i in range(len(coords)):
        # Obter região de Voronoi
        region_idx = vor.point_region[i]
        region_vertices = vor.regions[region_idx]
        
        # Verificar se é uma região válida
        if -1 not in region_vertices and len(region_vertices) > 0:
            # Obter coordenadas dos vértices
            polygon_vertices = vor.vertices[region_vertices]
            # Criar polígono
            polygon = Polygon(polygon_vertices)
            voronoi_polygons.append(polygon)
            
            # Adicionar informações da ERB
            operadoras.append(gdf.iloc[i]['NomeEntidade'])
            erb_ids.append(i)
    
    # Criar GeoDataFrame
    voronoi_gdf = gpd.GeoDataFrame(
        {'geometry': voronoi_polygons, 'NomeEntidade': operadoras, 'erb_id': erb_ids}, 
        crs=gdf_proj.crs
    )
    
    # Voltar para CRS original
    voronoi_gdf = voronoi_gdf.to_crs(gdf.crs)
    
    # Calcular área dos polígonos em km²
    voronoi_gdf['area_voronoi_km2'] = voronoi_gdf.to_crs('+proj=utm +zone=23K +south').geometry.area / 1_000_000
    
    # Adicionar informações sobre vizinhos
    voronoi_gdf['num_vizinhos'] = 0
    
    for i, row in voronoi_gdf.iterrows():
        # Contar polígonos que compartilham fronteira
        vizinhos = voronoi_gdf[voronoi_gdf.geometry.touches(row.geometry)]
        voronoi_gdf.at[i, 'num_vizinhos'] = len(vizinhos)
    
    # Associar informações de voronoi com ERBs originais
    gdf_resultado = gdf.copy()
    gdf_resultado['voronoi_area_km2'] = np.nan
    gdf_resultado['voronoi_vizinhos'] = np.nan
    
    for i, row in voronoi_gdf.iterrows():
        erb_id = row['erb_id']
        gdf_resultado.iloc[erb_id, gdf_resultado.columns.get_loc('voronoi_area_km2')] = row['area_voronoi_km2']
        gdf_resultado.iloc[erb_id, gdf_resultado.columns.get_loc('voronoi_vizinhos')] = row['num_vizinhos']
    
    logger.info(f"Diagrama de Voronoi calculado com {len(voronoi_gdf)} polígonos")
    
    return gdf_resultado, voronoi_gdf

def plot_voronoi_map(gdf, voronoi_gdf):
    """
    Gera um mapa com o diagrama de Voronoi das ERBs.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB
        voronoi_gdf (geopandas.GeoDataFrame): Diagrama de Voronoi
    """
    logger.info("Gerando mapa de Voronoi")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
    # Criar paleta de cores para operadoras
    operadoras = gdf['NomeEntidade'].unique()
    colors_operadoras = dict(zip(operadoras, plt.cm.tab10.colors[:len(operadoras)]))
    
    # Plotar polígonos de Voronoi por operadora
    for op in operadoras:
        subset = voronoi_gdf[voronoi_gdf['NomeEntidade'] == op]
        if len(subset) > 0:
            color = colors_operadoras[op]
            subset.plot(ax=ax, color=color, alpha=0.5, label=op)
    
    # Adicionar pontos das ERBs
    gdf.plot(ax=ax, markersize=15, color='black', alpha=0.7)
    
    # Adicionar mapa base
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logger.warning(f"Não foi possível adicionar mapa base: {e}")
    
    # Adicionar título e legenda
    plt.title('Áreas de Domínio das ERBs (Diagrama de Voronoi)', fontsize=16)
    plt.legend(title='Operadora', loc='upper right')
    
    # Salvar figura
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_voronoi.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_hex_vulnerability_map(hex_gdf):
    """
    Gera um mapa de vulnerabilidade baseado na grade hexagonal.
    
    Args:
        hex_gdf (geopandas.GeoDataFrame): Grade hexagonal com análise de vulnerabilidade
    """
    logger.info("Gerando mapa de vulnerabilidade em grade hexagonal")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
    # Definir cores para categorias de vulnerabilidade
    colors_vulnerabilidade = {
        'Sem cobertura': '#d73027',
        'Alta vulnerabilidade': '#fc8d59',
        'Média vulnerabilidade': '#fee090',
        'Baixa vulnerabilidade': '#e0f3f8'
    }
    
    # Plotar hexágonos por categoria de vulnerabilidade
    for categoria, color in colors_vulnerabilidade.items():
        subset = hex_gdf[hex_gdf['vulnerabilidade'] == categoria]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, alpha=0.7, label=categoria)
    
    # Adicionar mapa base
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logger.warning(f"Não foi possível adicionar mapa base: {e}")
    
    # Adicionar título e legenda
    plt.title('Análise de Vulnerabilidade de Cobertura por Hexágonos', fontsize=16)
    plt.legend(title='Vulnerabilidade', loc='upper right')
    
    # Salvar figura
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_hex_vulnerability.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_coverage(gdf):
    """
    Gera uma visualização 3D dos setores de cobertura das ERBs.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB com setores de cobertura
    """
    try:
        logger.info("Gerando visualização 3D de cobertura")
        from mpl_toolkits.mplot3d import Axes3D
        
        # Verificar se temos dados de setor
        if 'setor_geometria' not in gdf.columns or gdf['setor_geometria'].notna().sum() == 0:
            logger.warning("Sem dados de setores de cobertura para visualização 3D")
            return
        
        # Filtrar apenas as ERBs com setores válidos
        setores_gdf = gdf[gdf['setor_geometria'].notna()].copy()
        
        # Criar paleta de cores para operadoras
        operadoras = gdf['NomeEntidade'].unique()
        colors_operadoras = dict(zip(operadoras, plt.cm.tab10.colors[:len(operadoras)]))
        
        # Criar figura 3D
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Para cada operadora
        for op in operadoras:
            subset = setores_gdf[setores_gdf['NomeEntidade'] == op]
            if len(subset) == 0:
                continue
            
            color = colors_operadoras[op]
            
            # Para cada ERB da operadora
            for idx, row in subset.iterrows():
                if not isinstance(row['setor_geometria'], Polygon):
                    continue
                
                # Extrair coordenadas do polígono do setor
                x, y = row['setor_geometria'].exterior.xy
                
                # Usar EIRP como altura
                z = np.ones(len(x)) * row['EIRP_dBm']
                
                # Plotar superfície
                ax.plot_trisurf(x, y, z, alpha=0.4, color=color)
                
                # Plotar ponto da ERB
                ax.scatter([row.geometry.x], [row.geometry.y], [row['EIRP_dBm']], 
                          color=color, s=50, edgecolor='black')
        
        # Adicionar título e labels
        ax.set_title('Visualização 3D da Cobertura das ERBs', fontsize=16)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_zlabel('EIRP (dBm)')
        
        # Adicionar legenda
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                      markerfacecolor=color, markersize=10, label=op)
                          for op, color in colors_operadoras.items()]
        ax.legend(handles=legend_elements, title='Operadora', loc='upper right')
        
        # Ajustar visão
        ax.view_init(elev=30, azim=45)
        
        # Salvar figura
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_cobertura_3d.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Erro ao gerar visualização 3D: {e}")

def create_interactive_full_map(gdf, voronoi_gdf, hex_gdf):
    """
    Cria um mapa interativo completo com todas as análises.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB
        voronoi_gdf (geopandas.GeoDataFrame): Diagrama de Voronoi
        hex_gdf (geopandas.GeoDataFrame): Grade hexagonal
    """
    try:
        logger.info("Gerando mapa interativo completo")
        import folium
        from folium.plugins import MarkerCluster
        
        # Definir cores para operadoras
        operadoras = gdf['NomeEntidade'].unique()
        colors_tab10 = plt.cm.tab10.colors[:len(operadoras)]
        colors_operadoras = dict(zip(operadoras, colors_tab10))
        
        # Definir cores para vulnerabilidade
        colors_vulnerabilidade = {
            'Sem cobertura': '#d73027',
            'Alta vulnerabilidade': '#fc8d59',
            'Média vulnerabilidade': '#fee090',
            'Baixa vulnerabilidade': '#e0f3f8'
        }
        
        # Centro do mapa
        center = [gdf.geometry.y.mean(), gdf.geometry.x.mean()]
        
        # Criar mapa
        m = folium.Map(location=center, zoom_start=11, tiles='CartoDB positron')
        
        # Adicionar hexágonos de vulnerabilidade
        fg_hex = folium.FeatureGroup(name="Análise de Vulnerabilidade", show=False)
        
        for cat, color in colors_vulnerabilidade.items():
            subset = hex_gdf[hex_gdf['vulnerabilidade'] == cat]
            
            for idx, row in subset.iterrows():
                popup_text = f"""
                <b>Vulnerabilidade:</b> {row['vulnerabilidade']}<br>
                <b>Nº de operadoras:</b> {row['num_operadoras']}<br>
                <b>Nº de setores:</b> {row['num_setores'] if 'num_setores' in row else 'N/A'}
                """
                
                folium.GeoJson(
                    row.geometry,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': color,
                        'weight': 1,
                        'fillOpacity': 0.5
                    },
                    popup=folium.Popup(popup_text)
                ).add_to(fg_hex)
        
        fg_hex.add_to(m)
        
        # Adicionar polígonos de Voronoi
        fg_voronoi = folium.FeatureGroup(name="Diagrama de Voronoi", show=False)
        
        for op in operadoras:
            subset = voronoi_gdf[voronoi_gdf['NomeEntidade'] == op]
            
            color = '#{:02x}{:02x}{:02x}'.format(
                int(colors_operadoras[op][0]*255),
                int(colors_operadoras[op][1]*255),
                int(colors_operadoras[op][2]*255)
            )
            
            for idx, row in subset.iterrows():
                popup_text = f"""
                <b>Operadora:</b> {op}<br>
                <b>Área:</b> {row['area_voronoi_km2']:.2f} km²<br>
                <b>Vizinhos:</b> {row['num_vizinhos']}
                """
                
                folium.GeoJson(
                    row.geometry,
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': color,
                        'weight': 1,
                        'fillOpacity': 0.4
                    },
                    popup=folium.Popup(popup_text)
                ).add_to(fg_voronoi)
        
        fg_voronoi.add_to(m)
        
        # Adicionar setores de cobertura
        if 'setor_geometria' in gdf.columns and gdf['setor_geometria'].notna().any():
            setores_gdf = gdf[gdf['setor_geometria'].notna()].copy()
            
            for op in operadoras:
                subset = setores_gdf[setores_gdf['NomeEntidade'] == op]
                
                if len(subset) == 0:
                    continue
                
                color = '#{:02x}{:02x}{:02x}'.format(
                    int(colors_operadoras[op][0]*255),
                    int(colors_operadoras[op][1]*255),
                    int(colors_operadoras[op][2]*255)
                )
                
                fg_setores = folium.FeatureGroup(name=f"Cobertura - {op}")
                
                for idx, row in subset.iterrows():
                    popup_text = f"""
                    <b>Operadora:</b> {op}<br>
                    <b>EIRP:</b> {row['EIRP_dBm']:.2f} dBm<br>
                    <b>Raio:</b> {row['Raio_Cobertura_km']:.2f} km<br>
                    """
                    
                    if isinstance(row['setor_geometria'], Polygon):
                        folium.GeoJson(
                            row['setor_geometria'],
                            style_function=lambda x, color=color: {
                                'fillColor': color,
                                'color': color,
                                'weight': 1,
                                'fillOpacity': 0.5
                            },
                            popup=folium.Popup(popup_text)
                        ).add_to(fg_setores)
                
                fg_setores.add_to(m)
        
        # Adicionar ERBs
        for op in operadoras:
            subset = gdf[gdf['NomeEntidade'] == op]
            
            color = '#{:02x}{:02x}{:02x}'.format(
                int(colors_operadoras[op][0]*255),
                int(colors_operadoras[op][1]*255),
                int(colors_operadoras[op][2]*255)
            )
            
            mc = MarkerCluster(name=f"ERBs - {op}")
            
            for idx, row in subset.iterrows():
                popup_text = f"""
                <b>Operadora:</b> {op}<br>
                <b>EIRP:</b> {row['EIRP_dBm']:.2f} dBm<br>
                <b>Raio:</b> {row['Raio_Cobertura_km']:.2f} km<br>
                """
                
                if 'voronoi_area_km2' in row and not pd.isna(row['voronoi_area_km2']):
                    popup_text += f"<b>Área Voronoi:</b> {row['voronoi_area_km2']:.2f} km²<br>"
                
                if 'voronoi_vizinhos' in row and not pd.isna(row['voronoi_vizinhos']):
                    popup_text += f"<b>Vizinhos:</b> {int(row['voronoi_vizinhos'])}<br>"
                
                folium.Marker(
                    location=[row.geometry.y, row.geometry.x],
                    popup=folium.Popup(popup_text),
                    icon=folium.Icon(color='white', icon_color=color, icon='signal', prefix='fa')
                ).add_to(mc)
            
            mc.add_to(m)
        
        # Adicionar controle de camadas
        folium.LayerControl().add_to(m)
        
        # Salvar mapa
        m.save(os.path.join(VISUALIZATION_DIR, 'erb_mapa_interativo_completo.html'))
        logger.info(f"Mapa interativo completo salvo em {os.path.join(VISUALIZATION_DIR, 'erb_mapa_interativo_completo.html')}")
        
    except Exception as e:
        logger.error(f"Erro ao gerar mapa interativo completo: {e}")

def main():
    """
    Função principal para executar o enriquecimento de dados de ERBs.
    """
    # Configurar o sistema de logging
    setup_logging()
    
    # Registrar início do processo
    start_time = time.time()
    logger.info("Iniciando processo de enriquecimento de dados de ERBs")
    
    # Carregar dados
    data = load_data()
    if data is None or len(data) == 0:
        logger.error("Não foi possível carregar os dados de ERB")
        return
    
    logger.info(f"Colunas disponíveis: {data.columns.tolist()}")
    
    # Verificar colunas obrigatórias
    required_columns = ['geometry', 'PotenciaTransmissorWatts', 'GanhoAntena']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logger.error(f"Colunas obrigatórias ausentes nos dados: {missing_columns}")
        logger.error("Verifique o arquivo de entrada e tente novamente.")
        return
    
    # Verificar se a coluna 'Azimute' está presente
    if 'Azimute' not in data.columns:
        logger.error("Coluna 'Azimute' não encontrada. Este campo é necessário para cálculos de cobertura.")
        return
    else:
        # Converter Azimute para numérico se não for
        if data['Azimute'].dtype == 'object':
            data['Azimute'] = pd.to_numeric(data['Azimute'], errors='coerce')
        
        # Filtrar registros com valores NaN em Azimute
        mask = data['Azimute'].isna()
        if mask.any():
            total_registros = len(data)
            registros_ausentes = mask.sum()
            logger.warning(f"Encontrados {registros_ausentes} valores ausentes em 'Azimute'. Estes registros serão excluídos.")
            
            # Excluir registros com Azimute ausente
            data = data[~mask].copy()
            logger.info(f"Registros reduzidos de {total_registros} para {len(data)} após remover valores ausentes de 'Azimute'.")
    
    # Calcular EIRP
    logger.info("Calculando EIRP")
    data = calculate_eirp(data)
    
    # Calcular densidade
    logger.info("Calculando densidade de ERBs")
    data = calculate_density(data)
    
    # Calcular raio de cobertura
    logger.info("Calculando raio de cobertura")
    data = calculate_coverage_radius(data)
    
    # Criar setores de cobertura
    logger.info("Criando setores de cobertura")
    data = create_coverage_sectors(data)
    
    # Calcular diagrama de Voronoi
    logger.info("Calculando diagrama de Voronoi")
    data, voronoi_gdf = calculate_voronoi(data)
    
    # Criar grade hexagonal para análise de vulnerabilidade
    logger.info("Criando grade hexagonal para análise de vulnerabilidade")
    data, hex_gdf = create_hexagon_coverage_grid(data)
    
    # Gerar visualizações
    logger.info("Gerando visualizações")
    try:
        # Visualizações básicas
        generate_visualizations(data)
        
        # Visualização do diagrama de Voronoi
        plot_voronoi_map(data, voronoi_gdf)
        
        # Visualização de vulnerabilidade em hexágonos
        plot_hex_vulnerability_map(hex_gdf)
        
        # Visualização 3D de cobertura
        plot_3d_coverage(data)
        
        # Mapa interativo completo com todas as análises
        create_interactive_full_map(data, voronoi_gdf, hex_gdf)
        
        logger.info("Visualizações geradas com sucesso")
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações: {e}")
    
    # Salvar dados enriquecidos
    logger.info("Salvando dados enriquecidos")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"rbs_enriched_{timestamp}.gpkg")
    
    try:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        
        # Salvar dataframe principal
        data.to_file(output_file, driver="GPKG", layer="erbs")
        
        # Salvar diagrama de Voronoi
        voronoi_gdf.to_file(output_file, driver="GPKG", layer="voronoi")
        
        # Salvar grade hexagonal
        hex_gdf.to_file(output_file, driver="GPKG", layer="hexagons")
        
        logger.info(f"Dados enriquecidos salvos em {output_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar dados enriquecidos: {e}")
    
    # Registrar fim do processo
    end_time = time.time()
    logger.info(f"Processo de enriquecimento concluído em {end_time - start_time:.2f} segundos")
    logger.info(f"Dados enriquecidos salvos em {output_file}")
    logger.info(f"Visualizações salvas em {VISUALIZATION_DIR}")

if __name__ == "__main__":
    # Importar bibliotecas adicionais que só são usadas em visualizações
    import seaborn as sns
    import matplotlib.patheffects as pe
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    
    main()