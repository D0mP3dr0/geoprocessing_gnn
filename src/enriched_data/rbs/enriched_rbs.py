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
import numba
from numba import jit, cuda, prange
from functools import lru_cache
from rtree import index
import multiprocessing as mp
import shapely
from concurrent.futures import ProcessPoolExecutor, as_completed

# Obter o caminho absoluto para o diretório do projeto
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
workspace_dir = os.path.dirname(src_dir)

# Definir diretórios de entrada e saída
INPUT_DIR = os.path.join(workspace_dir, 'data', 'processed')
OUTPUT_DIR = os.path.join(workspace_dir, 'data', 'enriched')
REPORT_DIR = os.path.join(workspace_dir, 'src', 'enriched_data', 'rbs', 'quality_reports')
VISUALIZATION_DIR = os.path.join(workspace_dir, 'outputs', 'visualize_enriched_data', 'rbs')

# CRS métrico para cálculos de distância
METRIC_CRS = 'EPSG:31983'  # SIRGAS 2000 / UTM zone 23S

# Garantir que os diretórios de saída existam
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def ensure_directories():
    """
    Garante que todos os diretórios necessários existam.
    Cria a estrutura completa de diretórios se necessário.
    """
    directories = [
        OUTPUT_DIR,
        REPORT_DIR,
        VISUALIZATION_DIR,
        os.path.join(REPORT_DIR, 'json'),
        os.path.join(REPORT_DIR, 'csv'),
        os.path.join(REPORT_DIR, 'summary')
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Diretório garantido: {directory}")

# Verificar disponibilidade de CUDA
def is_cuda_available():
    """
    Verifica se CUDA está disponível no sistema.
    
    Returns:
        bool: True se CUDA estiver disponível, False caso contrário.
    """
    try:
        # Verificar se o módulo cuda está disponível no numba
        if not hasattr(numba, 'cuda'):
            logger.warning("Módulo numba.cuda não está disponível nesta instalação")
            return False
            
        # Verificar se há dispositivos CUDA disponíveis
        cuda_devices = numba.cuda.list_devices()
        is_available = len(cuda_devices) > 0
        
        if is_available:
            logger.info(f"CUDA disponível: {len(cuda_devices)} dispositivo(s) encontrado(s)")
        else:
            logger.warning("Nenhum dispositivo CUDA encontrado")
            
        return is_available
    except AttributeError as e:
        logger.warning(f"Erro de atributo ao verificar CUDA: {e}")
        return False
    except ImportError as e:
        logger.warning(f"Erro de importação ao verificar CUDA: {e}")
        return False
    except Exception as e:
        logger.warning(f"Erro desconhecido ao verificar CUDA: {e}")
        return False

# Funções para processamento paralelo e uso de GPU
@jit(nopython=True, parallel=True)
def calculate_densities_parallel(coords, buffer_size=1000):
    """
    Calcula densidades de ERBs de forma paralela.
    
    Args:
        coords (np.ndarray): Array de coordenadas [x, y] de ERBs.
        buffer_size (float): Tamanho do buffer em metros para considerar vizinhos.
        
    Returns:
        np.ndarray: Array de densidades para cada ERB.
    """
    n = len(coords)
    densities = np.zeros(n, dtype=np.int32)
    
    for i in prange(n):
        x, y = coords[i]
        count = 0
        for j in range(n):
            if i == j:
                continue
            dist = np.sqrt((x - coords[j, 0])**2 + (y - coords[j, 1])**2)
            if dist <= buffer_size:
                count += 1
        densities[i] = count
    
    return densities

@cuda.jit
def calculate_densities_cuda(coords, results, buffer_size):
    """
    Kernel CUDA para calcular densidades de ERBs.
    
    Args:
        coords (np.ndarray): Array de coordenadas [x, y] de ERBs.
        results (np.ndarray): Array de saída para densidades.
        buffer_size (float): Tamanho do buffer em metros.
    """
    i = cuda.grid(1)
    if i < coords.shape[0]:
        count = 0
        x, y = coords[i, 0], coords[i, 1]
        
        for j in range(coords.shape[0]):
            if i == j:
                continue
            dist = math.sqrt((x - coords[j, 0])**2 + (y - coords[j, 1])**2)
            if dist <= buffer_size:
                count += 1
        
        results[i] = count

def calculate_densities_with_gpu(coords, buffer_size=1000):
    """
    Calcula densidades usando GPU se disponível, senão usa CPU paralela.
    
    Args:
        coords (np.ndarray): Array de coordenadas [x, y] de ERBs.
        buffer_size (float): Tamanho do buffer em metros.
        
    Returns:
        np.ndarray: Array de densidades para cada ERB.
    """
    if is_cuda_available():
        logger.info("Usando GPU (CUDA) para cálculo de densidades")
        results = np.zeros(len(coords), dtype=np.int32)
        threadsperblock = 256
        blockspergrid = (coords.shape[0] + (threadsperblock - 1)) // threadsperblock
        calculate_densities_cuda[blockspergrid, threadsperblock](coords, results, buffer_size)
        return results
    else:
        logger.info("GPU não disponível, usando CPU paralela para cálculo de densidades")
        return calculate_densities_parallel(coords, buffer_size)

@jit(nopython=True)
def calculate_raio_coverage(eirp, freq, atenuacao, receptor_sensibilidade=-100):
    """
    Calcula o raio de cobertura com fórmula de Friis otimizada.
    
    Args:
        eirp (float): Potência Efetivamente Irradiada em dBm.
        freq (float): Frequência de transmissão em MHz.
        atenuacao (float): Fator de atenuação baseado no tipo de área.
        receptor_sensibilidade (float): Sensibilidade do receptor em dBm.
        
    Returns:
        float: Raio de cobertura em km.
    """
    # Cálculo do raio usando fórmula de Friis ajustada com atenuação
    raio_base = 10 ** ((eirp - receptor_sensibilidade - 32.44 - 20 * np.log10(freq)) / 20)
    raio_ajustado = raio_base / (1 + atenuacao/10)
    
    # Limitar o raio máximo com base na frequência
    raio_max = min(20, 25000/freq) if freq > 0 else 10
    raio = min(raio_ajustado, raio_max)
    
    return raio

@jit(nopython=True, parallel=True)
def calculate_raios_coverage_batch(eirps, freqs, atenuacoes, receptor_sensibilidade=-100):
    """
    Calcula raios de cobertura para múltiplas ERBs em paralelo.
    
    Args:
        eirps (np.ndarray): Array de valores EIRP em dBm.
        freqs (np.ndarray): Array de frequências em MHz.
        atenuacoes (np.ndarray): Array de fatores de atenuação.
        receptor_sensibilidade (float): Sensibilidade do receptor em dBm.
        
    Returns:
        np.ndarray: Array de raios de cobertura em km.
    """
    n = len(eirps)
    raios = np.zeros(n)
    
    for i in prange(n):
        raios[i] = calculate_raio_coverage(eirps[i], freqs[i], atenuacoes[i], receptor_sensibilidade)
    
    return raios

@lru_cache(maxsize=1024)
def cached_hex_boundary(hex_id):
    """
    Função cacheada para obter os limites de um hexágono H3.
    
    Args:
        hex_id (str): ID do hexágono H3.
        
    Returns:
        list: Lista de coordenadas do limite.
    """
    return h3.cell_to_boundary(hex_id)

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
    Versão otimizada com paralelização e índice espacial.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB com setores de cobertura
        
    Returns:
        tuple: (gdf atualizado, gdf dos hexágonos)
    """
    logger.info("Criando grade hexagonal otimizada")
    start_time = time.time()
    
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
    
    logger.info(f"Gerando grade hexagonal para área {min_lat:.4f},{min_lon:.4f} até {max_lat:.4f},{max_lon:.4f}")
    
    # Reduzir o número de pontos para acelerar a geração de hexágonos
    n_points = 15  # Reduzido de 20 para 15
    
    # Gerar índices H3 para a área de forma mais eficiente
    hex_ids = set()
    for lat in np.linspace(min_lat, max_lat, n_points):
        for lon in np.linspace(min_lon, max_lon, n_points):
            hex_id = h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
            hex_ids.add(hex_id)
    
    # Usar resolução menor para reduzir o número de hexágonos se for muito grande
    if len(hex_ids) > 500:
        logger.info(f"Muitos hexágonos ({len(hex_ids)}). Reduzindo resolução.")
        # Usar uma resolução menor (menos hexágonos)
        resolution = H3_RESOLUTION - 1
        hex_ids = set()
        for lat in np.linspace(min_lat, max_lat, n_points):
            for lon in np.linspace(min_lon, max_lon, n_points):
                hex_id = h3.latlng_to_cell(lat, lon, resolution)
                hex_ids.add(hex_id)
    
    logger.info(f"Gerando {len(hex_ids)} hexágonos")
    
    # Converter índices H3 para polígonos de forma mais eficiente
    hex_polygons = []
    hex_ids_list = list(hex_ids)
    
    # Melhorar desempenho com lote
    for i in range(0, len(hex_ids_list), 100):
        batch = hex_ids_list[i:i+100]
        for hex_id in batch:
            boundary = h3.cell_to_boundary(hex_id)
            # Convert the boundary format to be compatible with Shapely
            polygon = Polygon([(lng, lat) for lat, lng in boundary])
        hex_polygons.append(polygon)
    
    # Criar GeoDataFrame dos hexágonos
    hex_gdf = gpd.GeoDataFrame(geometry=hex_polygons, crs=result.crs)
    hex_gdf['hex_index'] = range(len(hex_gdf))
    
    logger.info(f"Grade hexagonal gerada em {time.time() - start_time:.2f} segundos")
    
    # Filtrar apenas ERBs com setores válidos
    setores_gdf = result[result['setor_geometria'].notna()].copy()
    if len(setores_gdf) == 0:
        logger.warning("Nenhum setor válido encontrado")
        # Inicializar colunas vazias
        hex_gdf['num_operadoras'] = 0
        hex_gdf['num_setores'] = 0
        hex_gdf['potencia_media'] = np.nan
        hex_gdf['vulnerabilidade'] = 'Sem cobertura'
        hex_gdf['densidade_potencia'] = np.nan
        return result, hex_gdf
    
    setores_gdf['geometry'] = setores_gdf['setor_geometria']
    
    # Lista de operadoras únicas
    operadoras = result['NomeEntidade'].unique()
    
    # Converter para CRS métrico para operações espaciais mais precisas
    hex_gdf_proj = hex_gdf.to_crs(METRIC_CRS)
    setores_gdf_proj = setores_gdf.to_crs(METRIC_CRS)
    
    # Garantir que temos índices espaciais para consultas rápidas
    if not hasattr(hex_gdf_proj, 'sindex') or hex_gdf_proj.sindex is None:
        hex_gdf_proj.sindex = hex_gdf_proj.geometry.sindex
    
    if not hasattr(setores_gdf_proj, 'sindex') or setores_gdf_proj.sindex is None:
        setores_gdf_proj.sindex = setores_gdf_proj.geometry.sindex
    
    # Função para processar um lote de hexágonos em paralelo
    def process_hexagon_batch(hex_indices):
        results = []
        
        for idx in hex_indices:
            hex_geom = hex_gdf_proj.loc[idx].geometry
            
            # Conjunto de operadoras com cobertura neste hexágono
        op_com_cobertura = set()
        count_setores = 0
        eirp_setores = []
        
            # Usar índice espacial para encontrar possíveis interseções
            # Em vez de verificar todos os setores
            possible_matches_idx = list(setores_gdf_proj.sindex.intersection(hex_geom.bounds))
            candidate_setores = setores_gdf_proj.iloc[possible_matches_idx]
            
            # Verificar interseção com setores candidatos
        for op in operadoras:
                # Filtrar setores desta operadora entre os candidatos
                op_setores = candidate_setores[candidate_setores['NomeEntidade'] == op]
                
                if len(op_setores) == 0:
                    continue
            
            # Verificar se algum setor intersecta o hexágono
            for _, setor_row in op_setores.iterrows():
                if hex_geom.intersects(setor_row.geometry):
                    op_com_cobertura.add(op)
                    count_setores += 1
                    eirp_setores.append(setor_row['EIRP_dBm'])
                    break  # Basta uma interseção por operadora
        
            # Armazenar resultados para este hexágono
            results.append({
                'idx': idx,
                'num_operadoras': len(op_com_cobertura),
                'num_setores': count_setores,
                'potencia_media': np.mean(eirp_setores) if eirp_setores else np.nan
            })
            
        return results
    
    # Processar hexágonos em lotes paralelos
    logger.info("Processando cobertura dos hexágonos em paralelo")
    
    # Dividir hexágonos em lotes para processamento paralelo
    all_hex_indices = hex_gdf_proj.index.tolist()
    num_workers = min(mp.cpu_count() - 1, 8)  # Limitar a 8 workers ou CPU-1
    batch_size = max(1, len(all_hex_indices) // num_workers)
    hex_batches = [all_hex_indices[i:i+batch_size] for i in range(0, len(all_hex_indices), batch_size)]
    
    logger.info(f"Dividindo {len(all_hex_indices)} hexágonos em {len(hex_batches)} lotes para {num_workers} workers")
    
    # Processar em paralelo
    all_results = []
    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_hexagon_batch, batch) for batch in hex_batches]
            
            for i, future in enumerate(as_completed(futures)):
                batch_results = future.result()
                all_results.extend(batch_results)
                logger.info(f"Concluído lote {i+1}/{len(hex_batches)} - {len(batch_results)} hexágonos")
    except Exception as e:
        logger.error(f"Erro no processamento paralelo: {e}")
        logger.info("Continuando com processamento sequencial")
        
        # Fallback para processamento sequencial
        for batch in hex_batches:
            batch_results = process_hexagon_batch(batch)
            all_results.extend(batch_results)
    
    # Converter resultados para dicionários para atualização eficiente
    num_operadoras_dict = {item['idx']: item['num_operadoras'] for item in all_results}
    num_setores_dict = {item['idx']: item['num_setores'] for item in all_results}
    potencia_media_dict = {item['idx']: item['potencia_media'] for item in all_results}
    
    # Atualizar dataframe usando assinatura de Pandas (mais eficiente)
    hex_gdf['num_operadoras'] = pd.Series(num_operadoras_dict)
    hex_gdf['num_setores'] = pd.Series(num_setores_dict)
    hex_gdf['potencia_media'] = pd.Series(potencia_media_dict)
    
    # Preencher valores NaN
    hex_gdf['num_operadoras'] = hex_gdf['num_operadoras'].fillna(0).astype(int)
    hex_gdf['num_setores'] = hex_gdf['num_setores'].fillna(0).astype(int)
    
    logger.info("Classificando hexágonos por vulnerabilidade")
    
    # Classificar os hexágonos por vulnerabilidade
    bins = [-1, 0, 1, 2, 10]
    labels = ['Sem cobertura', 'Alta vulnerabilidade', 'Média vulnerabilidade', 'Baixa vulnerabilidade']
    hex_gdf['vulnerabilidade'] = pd.cut(hex_gdf['num_operadoras'], bins=bins, labels=labels)
    
    # Calcular métricas adicionais para os hexágonos
    # Densidade de potência: média da potência / número de setores
    hex_gdf['densidade_potencia'] = hex_gdf['potencia_media'] / hex_gdf['num_setores'].replace(0, np.nan)
    
    total_time = time.time() - start_time
    logger.info(f"Grade hexagonal de cobertura criada em {total_time:.2f} segundos")
    
    return result, hex_gdf

def analyze_spatial_clustering(gdf):
    """
    Realiza análise de clustering espacial das ERBs usando DBSCAN com otimizações.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com informações de cluster
    """
    logger.info("Iniciando análise de clustering espacial otimizada")
    start_time = time.time()
    
    result = gdf.copy()
    
    # Converter para coordenadas métricas
    gdf_proj = result.to_crs(METRIC_CRS)
    
    # Extrair coordenadas x, y
    coords = np.vstack((gdf_proj.geometry.x, gdf_proj.geometry.y)).T
    
    # Otimização 1: Usar parâmetros adequados para o tamanho do dataset
    n_samples = len(coords)
    logger.info(f"Executando DBSCAN em {n_samples} pontos")
    
    # Ajustar parâmetros conforme tamanho do dataset
    if n_samples > 10000:
        eps = 500  # 500 metros
        min_samples = 3
        algorithm = 'ball_tree'  # Mais eficiente para grandes conjuntos de dados
    else:
        eps = 500
        min_samples = 3
        algorithm = 'auto'
    
    # Executar DBSCAN com otimizações
    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm=algorithm,
        n_jobs=-1  # Usar todos os núcleos disponíveis
    ).fit(coords)
    
    logger.info(f"DBSCAN concluído em {time.time() - start_time:.2f} segundos")
    
    # Adicionar rótulos de cluster ao GeoDataFrame
    result['cluster_id'] = clustering.labels_
    
    # Calcular estatísticas dos clusters
    n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
    n_noise = list(clustering.labels_).count(-1)
    
    logger.info(f"Identificados {n_clusters} clusters e {n_noise} pontos de ruído")
    
    # Otimização 2: Calcular contagens de maneira mais eficiente
    cluster_counts = pd.Series(clustering.labels_).value_counts().sort_index()
    
    # Otimização 3: Vetorizar cálculos de distância intra-cluster
    # Pré-alocar arrays para evitar append repetitivo
    distances_intra_cluster = np.full(n_samples, np.nan)
    
    # Processar em paralelo cada cluster
    def process_cluster(cluster_id):
        if cluster_id == -1:  # Ignorar pontos de ruído
            return {}
        
        # Obter índices de pontos nesse cluster
        cluster_mask = clustering.labels_ == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        
        if len(cluster_indices) <= 1:
            return {}
        
        # Obter coordenadas de pontos nesse cluster
        cluster_coords = coords[cluster_indices]
        
        # Calcular matriz de distâncias de todos para todos no cluster
        # Usando SciPy para cálculo eficiente
        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(cluster_coords))
        
        # Para cada ponto, calcular média das distâncias para outros pontos
        # Ignorando a diagonal (distância para si mesmo = 0)
        n_points = len(cluster_indices)
        results = {}
        
        for i in range(n_points):
            # Remover distância para si mesmo (que é 0)
            other_dists = np.concatenate([dist_matrix[i, :i], dist_matrix[i, i+1:]])
            # Armazenar média das distâncias
            idx = cluster_indices[i]
            results[idx] = np.mean(other_dists) if len(other_dists) > 0 else np.nan
            
        return results
    
    # Processar clusters em paralelo se houver mais de um
    if n_clusters > 1:
        # Usar no máximo n-1 processos (deixar um núcleo livre)
        n_processes = min(n_clusters, max(1, mp.cpu_count() - 1))
        
        logger.info(f"Processando {n_clusters} clusters em paralelo com {n_processes} processos")
        
        try:
            with ProcessPoolExecutor(max_workers=n_processes) as executor:
                # Processar apenas clusters reais (não o -1)
                cluster_ids = [cid for cid in set(clustering.labels_) if cid != -1]
                results_dict = {}
                
                for cluster_id, result_dict in zip(cluster_ids, executor.map(process_cluster, cluster_ids)):
                    results_dict.update(result_dict)
                
                # Atualizar array de distâncias
                for idx, mean_dist in results_dict.items():
                    distances_intra_cluster[idx] = mean_dist
                    
        except Exception as e:
            logger.warning(f"Erro no processamento paralelo: {e}, usando processamento sequencial")
            # Fallback para processamento sequencial
            for cluster_id in set(clustering.labels_):
                if cluster_id != -1:  # Ignorar pontos de ruído
                    result_dict = process_cluster(cluster_id)
                    for idx, mean_dist in result_dict.items():
                        distances_intra_cluster[idx] = mean_dist
    else:
        # Se só tiver um cluster ou menos, processar sequencialmente
        for cluster_id in set(clustering.labels_):
            if cluster_id != -1:  # Ignorar pontos de ruído
                result_dict = process_cluster(cluster_id)
                for idx, mean_dist in result_dict.items():
                    distances_intra_cluster[idx] = mean_dist
    
    # Adicionar ao dataframe
    result['distancia_media_cluster'] = distances_intra_cluster
    
    # Otimização 4: Usar vetorização para adicionar densidade de cluster
    result['densidade_cluster'] = result['cluster_id'].map(
        lambda cid: cluster_counts.get(cid, 0) if cid != -1 else 0
    )
    
    # Registrar métricas de clustering
    result.attrs['n_clusters'] = n_clusters
    result.attrs['n_noise'] = n_noise
    result.attrs['cluster_counts'] = cluster_counts.to_dict()
    
    # Registrar tempo total
    total_time = time.time() - start_time
    logger.info(f"Análise de clustering concluída em {total_time:.2f} segundos")
    
    return result

def create_coverage_network(gdf, hex_gdf):
    """
    Cria um grafo de rede de cobertura para análise de conectividade.
    Versão otimizada usando índices espaciais e operações vetorizadas.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB com setores
        hex_gdf (geopandas.GeoDataFrame): Grade hexagonal de cobertura
        
    Returns:
        tuple: (gdf atualizado, gdf dos hexágonos atualizado, grafo de rede)
    """
    logger.info("Criando rede de cobertura otimizada")
    start_time = time.time()
    
    result = gdf.copy()
    hex_result = hex_gdf.copy()
    
    # Criar um grafo não direcionado
    G = nx.Graph()
    
    # Otimização 1: Adicionar nós em lote em vez de individualmente
    logger.info("Adicionando nós de ERBs ao grafo")
    
    # Preparar dados de nós ERB
    erb_nodes = []
    for idx, row in result.iterrows():
        erb_nodes.append((
            f"erb_{idx}", 
            {
                'tipo': 'erb', 
                'operadora': row['NomeEntidade'], 
                'lat': row.geometry.y, 
                'lon': row.geometry.x,
                'eirp': row['EIRP_dBm'],
                'raio': row['Raio_Cobertura_km'],
                'pos': (row.geometry.x, row.geometry.y)
            }
        ))
    
    # Adicionar em lote
    G.add_nodes_from(erb_nodes)
    
    # Preparar dados de nós de hexágonos
    logger.info("Adicionando nós de hexágonos ao grafo")
    hex_nodes = []
    for idx, row in hex_result.iterrows():
        centroid = row.geometry.centroid
        hex_nodes.append((
            f"hex_{idx}",
            {
                'tipo': 'hexagono',
                'num_operadoras': row['num_operadoras'],
                'vulnerabilidade': row['vulnerabilidade'],
                'pos': (centroid.x, centroid.y)
            }
        ))
    
    # Adicionar em lote
    G.add_nodes_from(hex_nodes)
    
    # Otimização 2: Usar índice espacial para arestas entre ERBs e hexágonos
    logger.info("Criando arestas entre ERBs e hexágonos usando índice espacial")
    
    # Filtrar apenas ERBs com setores válidos
    setores_gdf = result[result['setor_geometria'].notna()].copy()
    if len(setores_gdf) > 0:
    setores_gdf['geometry'] = setores_gdf['setor_geometria']
    
        # Preparar índice espacial para hexágonos
        if not hasattr(hex_result, 'sindex') or hex_result.sindex is None:
            # Criar índice espacial se não existir
            hex_result.sindex = hex_result.geometry.sindex
        
        # Criar arestas em lotes para evitar adicionar uma por uma
        erb_hex_edges = []
        
        # Para cada setor, encontrar hexágonos que intersectam
    for erb_idx, erb_row in tqdm(setores_gdf.iterrows(), total=len(setores_gdf), desc="Criando arestas ERB-hexágono"):
        erb_setor = erb_row.geometry
            # Usar o índice espacial para encontrar candidatos
            potential_matches_idx = list(hex_result.sindex.intersection(erb_setor.bounds))
            # Filtrar candidatos reais
            for hex_idx in potential_matches_idx:
                if erb_setor.intersects(hex_result.iloc[hex_idx].geometry):
                    erb_hex_edges.append((
                        f"erb_{erb_idx}", 
                        f"hex_{hex_idx}", 
                        {'tipo': 'cobertura'}
                    ))
        
        # Adicionar todas as arestas de uma vez
        logger.info(f"Adicionando {len(erb_hex_edges)} arestas entre ERBs e hexágonos")
        G.add_edges_from(erb_hex_edges)
    
    # Otimização 3: Criar arestas entre ERBs da mesma operadora em clusters
    logger.info("Criando arestas entre ERBs da mesma operadora em clusters")
    
    # Converter para CRS métrico para cálculos precisos de distância
    result_proj = result.to_crs(METRIC_CRS)
    
    # Preparar para adicionar arestas em lote
    cluster_edges = []
    
    # Processar cada cluster separadamente para reduzir comparações
    for cluster_id in tqdm(result['cluster_id'].unique(), desc="Processando clusters"):
        if cluster_id == -1:  # Pular pontos de ruído
            continue
            
        # Filtrar ERBs deste cluster
        cluster_erbs = result[result['cluster_id'] == cluster_id]
        if len(cluster_erbs) <= 1:
            continue
            
        cluster_erbs_proj = result_proj[result_proj.index.isin(cluster_erbs.index)]
        
        # Agrupar por operadora para processar apenas pares da mesma operadora
        operadoras = cluster_erbs['NomeEntidade'].unique()
        
        for operadora in operadoras:
            # Filtrar ERBs desta operadora neste cluster
            op_erbs = cluster_erbs[cluster_erbs['NomeEntidade'] == operadora]
            if len(op_erbs) <= 1:
                continue
                
            op_indices = op_erbs.index.tolist()
            op_erbs_proj = cluster_erbs_proj[cluster_erbs_proj.index.isin(op_indices)]
            
            # Criar todas as combinações de pares
            for i, idx1 in enumerate(op_indices):
                x1, y1 = op_erbs_proj.loc[idx1].geometry.x, op_erbs_proj.loc[idx1].geometry.y
                
                for idx2 in op_indices[i+1:]:
                    x2, y2 = op_erbs_proj.loc[idx2].geometry.x, op_erbs_proj.loc[idx2].geometry.y
                    
                    # Calcular distância em metros
                    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    # Converter para km
                    dist_km = dist / 1000
                    
                    cluster_edges.append((
                        f"erb_{idx1}", 
                        f"erb_{idx2}", 
                        {
                            'tipo': 'cluster', 
                            'operadora': operadora,
                            'distancia': dist_km
                        }
                    ))
    
    # Adicionar todas as arestas de cluster de uma vez
    logger.info(f"Adicionando {len(cluster_edges)} arestas entre ERBs em clusters")
    G.add_edges_from(cluster_edges)
    
    # Otimização 4: Calcular métricas de centralidade de forma mais eficiente
    logger.info("Calculando métricas de centralidade no grafo")
    
    # Verificar tamanho do grafo para ajustar método
    if len(G) > 10000:
        logger.info("Grafo grande detectado, usando aproximações para métricas de centralidade")
        # Para grafos grandes, usar amostragem
        n_samples = min(5000, len(G) // 2)
        betweenness = nx.betweenness_centrality(G, k=n_samples)
        closeness = nx.closeness_centrality(G)
    else:
        logger.info("Calculando métricas de centralidade exatas")
        betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    
    # Degree centrality é sempre rápido
    degree = nx.degree_centrality(G)
    
    # Otimização 5: Atualizar métricas usando operações vetorizadas
    # Adicionar métricas ao GeoDataFrame de ERBs
    # Preparar dicionários para atualização vetorizada
    betweenness_dict = {}
    degree_dict = {}
    closeness_dict = {}
    
    for idx in result.index:
        node_id = f"erb_{idx}"
        if node_id in betweenness:
            betweenness_dict[idx] = betweenness[node_id]
            degree_dict[idx] = degree[node_id]
            closeness_dict[idx] = closeness[node_id]
    
    # Atualizar de forma vetorizada
    result['betweenness'] = pd.Series(betweenness_dict)
    result['degree'] = pd.Series(degree_dict)
    result['closeness'] = pd.Series(closeness_dict)
    
    # Atualizar métricas do hexágono
    # Preparar dicionários para atualização vetorizada
    hex_betweenness_dict = {}
    hex_degree_dict = {}
    hex_closeness_dict = {}
    
    for idx in hex_result.index:
        node_id = f"hex_{idx}"
        if node_id in betweenness:
            hex_betweenness_dict[idx] = betweenness[node_id]
            hex_degree_dict[idx] = degree[node_id]
            hex_closeness_dict[idx] = closeness[node_id]
    
    # Atualizar de forma vetorizada
    hex_result['betweenness'] = pd.Series(hex_betweenness_dict)
    hex_result['degree'] = pd.Series(hex_degree_dict)
    hex_result['closeness'] = pd.Series(hex_closeness_dict)
    
    # Calcular redundância de cobertura 
    hex_result['indice_redundancia'] = hex_result['degree'].fillna(0)
    
    # Registrar tempo total
    total_time = time.time() - start_time
    logger.info(f"Rede de cobertura criada em {total_time:.2f} segundos")
    
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
    Gera um relatório detalhado de qualidade e estatísticas dos dados.
    
    Args:
        original_gdf (geopandas.GeoDataFrame): Dados originais
        enriched_gdf (geopandas.GeoDataFrame): Dados enriquecidos
        hex_gdf (geopandas.GeoDataFrame): Grade hexagonal de cobertura
        G (networkx.Graph, optional): Grafo de rede
        
    Returns:
        dict: Relatório de qualidade
    """
    logger.info("Gerando relatório de qualidade detalhado")
    
    # Gerar timestamp único para o relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Garantir que o diretório de relatórios exista
    os.makedirs(os.path.join(REPORT_DIR, 'json'), exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, 'summary'), exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, 'csv'), exist_ok=True)
    
    # Nomes de arquivos para o relatório
    json_file = os.path.join(REPORT_DIR, 'json', f'erb_quality_report_{timestamp}.json')
    summary_file = os.path.join(REPORT_DIR, 'summary', f'erb_report_summary_{timestamp}.txt')
    
    # Estrutura base do relatório
    report = {
        "report_id": timestamp,
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": RBS_FILE,
        "original_features": len(original_gdf),
        "enriched_features": len(enriched_gdf),
        "new_attributes": list(set(enriched_gdf.columns) - set(original_gdf.columns)),
        "statistics": {
            "EIRP_dBm": {
                "mean": float(enriched_gdf['EIRP_dBm'].mean()),
                "median": float(enriched_gdf['EIRP_dBm'].median()),
                "min": float(enriched_gdf['EIRP_dBm'].min()),
                "max": float(enriched_gdf['EIRP_dBm'].max()),
                "std": float(enriched_gdf['EIRP_dBm'].std())
            },
            "Raio_Cobertura_km": {
                "mean": float(enriched_gdf['Raio_Cobertura_km'].mean()),
                "median": float(enriched_gdf['Raio_Cobertura_km'].median()),
                "min": float(enriched_gdf['Raio_Cobertura_km'].min()),
                "max": float(enriched_gdf['Raio_Cobertura_km'].max()),
                "std": float(enriched_gdf['Raio_Cobertura_km'].std())
            },
            "Area_Cobertura_km2": {
                "mean": float(enriched_gdf['Area_Cobertura_km2'].mean()),
                "median": float(enriched_gdf['Area_Cobertura_km2'].median()),
                "min": float(enriched_gdf['Area_Cobertura_km2'].min()),
                "max": float(enriched_gdf['Area_Cobertura_km2'].max()),
                "std": float(enriched_gdf['Area_Cobertura_km2'].std())
            },
            "densidade_local": {
                "mean": float(enriched_gdf['densidade_local'].mean()),
                "median": float(enriched_gdf['densidade_local'].median()),
                "min": float(enriched_gdf['densidade_local'].min()),
                "max": float(enriched_gdf['densidade_local'].max()),
                "std": float(enriched_gdf['densidade_local'].std())
            },
            "tipo_area": {
                "distribution": {str(k): int(v) for k, v in enriched_gdf['tipo_area'].value_counts().to_dict().items()}
            }
        },
        "data_validation": {
            "valid_geometries": int(enriched_gdf.geometry.is_valid.sum()),
            "invalid_geometries": int((~enriched_gdf.geometry.is_valid).sum()),
            "null_values": {
                col: int(enriched_gdf[col].isna().sum()) 
                for col in enriched_gdf.columns if enriched_gdf[col].isna().sum() > 0
            }
        },
        "operadoras": {
            str(op): {
                "count": int(enriched_gdf[enriched_gdf['NomeEntidade'] == op].shape[0]),
                "avg_eirp": float(enriched_gdf[enriched_gdf['NomeEntidade'] == op]['EIRP_dBm'].mean()),
                "avg_raio": float(enriched_gdf[enriched_gdf['NomeEntidade'] == op]['Raio_Cobertura_km'].mean()),
                "total_coverage_km2": float(enriched_gdf[enriched_gdf['NomeEntidade'] == op]['Area_Cobertura_km2'].sum())
            } for op in enriched_gdf['NomeEntidade'].unique()
        },
        "clusters": {
            "num_clusters": int(enriched_gdf.attrs.get('n_clusters', 0)),
            "noise_points": int(enriched_gdf.attrs.get('n_noise', 0)),
            "distribution": {
                str(k): int(v) for k, v in enriched_gdf.attrs.get('cluster_counts', {}).items() 
                if k != -1
            }
        }
    }
    
    # Adicionar informações sobre classificação de importância apenas se a coluna existir
    if 'classe_importancia' in enriched_gdf.columns:
        report["statistics"]["classe_importancia"] = {
            "distribution": {str(k): int(v) for k, v in enriched_gdf['classe_importancia'].value_counts().to_dict().items()}
        }
    else:
        # Adicionar informação que a classificação será feita em nuvem
        report["statistics"]["classe_importancia"] = {
            "status": "Classificação de importância será processada em nuvem"
        }
    
    # Adicionar informações da grade hexagonal
    report["hexagons"] = {
            "total": len(hex_gdf),
        "coverage_statistics": {
            "covered": int((hex_gdf['num_operadoras'] > 0).sum()),
            "uncovered": int((hex_gdf['num_operadoras'] == 0).sum()),
            "coverage_percentage": float(((hex_gdf['num_operadoras'] > 0).sum() / len(hex_gdf)) * 100)
                },
                "vulnerabilidade": {
            "distribution": {
                str(k): int(v) for k, v in hex_gdf['vulnerabilidade'].value_counts().to_dict().items()
            }
        }
    }
    
    # Adicionar informações de rede apenas se o grafo existir
    if G is not None:
        # Análise de rede
        report["network"] = {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "node_types": {
                "erb": len([n for n, attrs in G.nodes(data=True) if attrs.get('tipo') == 'erb']),
                "hexagono": len([n for n, attrs in G.nodes(data=True) if attrs.get('tipo') == 'hexagono'])
            },
            "edge_types": {
                "cobertura": len([e for e, attrs in G.edges(data=True) if attrs.get('tipo') == 'cobertura']),
                "cluster": len([e for e, attrs in G.edges(data=True) if attrs.get('tipo') == 'cluster'])
            }
        }
        
        # Identificar ERBs críticas para evacuação (ex: alta betweenness centrality)
        if 'betweenness' in enriched_gdf.columns:
            # Definir o limiar para ser considerado crítico (ex: top 5% em betweenness)
            limiar = enriched_gdf['betweenness'].quantile(0.95)
            erbs_criticas = enriched_gdf[enriched_gdf['betweenness'] > limiar]
            
            report["network"]["erb_criticas_evacuacao"] = len(erbs_criticas)
            report["network"]["erb_criticas_percentile"] = 95  # Percentil usado
            
            # Adicionar detalhes das ERBs críticas
            report["network"]["erbs_criticas_detalhes"] = [
                {
                    "id": int(idx),
                    "operadora": str(row['NomeEntidade']),
                    "betweenness": float(row['betweenness']),
                    "lat": float(row.geometry.y),
                    "lon": float(row.geometry.x)
                }
                for idx, row in erbs_criticas.iterrows()
            ]
    else:
        # Se não temos o grafo, adicionar informação que a análise de rede será feita em nuvem
        report["network"] = {
            "status": "Análise de rede será processada em nuvem"
        }
    
    # Salvar o relatório como JSON
    try:
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Relatório de qualidade salvo em {json_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar relatório JSON: {e}")
    
    # Gerar um resumo em texto
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"RELATÓRIO DE QUALIDADE DE DADOS - ERBs - {timestamp}\n")
            f.write(f"='='='='='='='='='='='='='='='='='='='='='='='='='='='=\n\n")
            f.write(f"Data de geração: {report['report_date']}\n")
            f.write(f"Fonte de dados: {report['data_source']}\n\n")
            
            f.write(f"RESUMO GERAL:\n")
            f.write(f"  Registros originais: {report['original_features']}\n")
            f.write(f"  Registros enriquecidos: {report['enriched_features']}\n")
            f.write(f"  Novas colunas adicionadas: {len(report['new_attributes'])}\n\n")
            
            f.write(f"ESTATÍSTICAS PRINCIPAIS:\n")
            f.write(f"  EIRP média: {report['statistics']['EIRP_dBm']['mean']:.2f} dBm\n")
            f.write(f"  Raio médio de cobertura: {report['statistics']['Raio_Cobertura_km']['mean']:.2f} km\n")
            f.write(f"  Área média de cobertura: {report['statistics']['Area_Cobertura_km2']['mean']:.2f} km²\n\n")
            
            f.write(f"DISTRIBUIÇÃO POR OPERADORA:\n")
            for op, stats in report['operadoras'].items():
                f.write(f"  {op}: {stats['count']} ERBs, cobertura total: {stats['total_coverage_km2']:.2f} km²\n")
            
            f.write(f"\nANÁLISE DE COBERTURA DOS HEXÁGONOS:\n")
            f.write(f"  Hexágonos totais: {report['hexagons']['total']}\n")
            f.write(f"  Hexágonos com cobertura: {report['hexagons']['coverage_statistics']['covered']} "
                    f"({report['hexagons']['coverage_statistics']['coverage_percentage']:.2f}%)\n")
            f.write(f"  Hexágonos sem cobertura: {report['hexagons']['coverage_statistics']['uncovered']}\n\n")
            
            f.write(f"ANÁLISE DE VULNERABILIDADE:\n")
            for categoria, contagem in report['hexagons']['vulnerabilidade']['distribution'].items():
                f.write(f"  {categoria}: {contagem} hexágonos\n")
            
            if 'network' in report and 'num_nodes' in report['network']:
                f.write(f"\nANÁLISE DE REDE:\n")
                f.write(f"  Nós: {report['network']['num_nodes']}\n")
                f.write(f"  Arestas: {report['network']['num_edges']}\n")
                if 'erb_criticas_evacuacao' in report['network']:
                    f.write(f"  ERBs críticas para evacuação: {report['network']['erb_criticas_evacuacao']}\n")
            else:
                f.write(f"\nANÁLISE DE REDE: será processada em nuvem\n")
            
            f.write(f"\n='='='='='='='='='='='='='='='='='='='='='='='='='='='=\n")
            f.write(f"Fim do relatório. Para detalhes completos, consulte o arquivo JSON correspondente.")
            
        logger.info(f"Resumo do relatório de qualidade salvo em {summary_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar resumo do relatório: {e}")
    
    # Salvar dados das operadoras em CSV
    try:
        operadoras_df = pd.DataFrame.from_dict(
            {k: {'Operadora': k, 'Quantidade': v['count'], 'EIRP_Media': v['avg_eirp'], 
                'Raio_Medio': v['avg_raio'], 'Cobertura_Total_km2': v['total_coverage_km2']} 
             for k, v in report['operadoras'].items()},
            orient='index'
        )
        
        csv_op_file = os.path.join(REPORT_DIR, 'csv', f'erb_operadoras_{timestamp}.csv')
        operadoras_df.to_csv(csv_op_file, index=False)
        logger.info(f"Estatísticas por operadora salvas em {csv_op_file}")
    except Exception as e:
        logger.error(f"Erro ao salvar estatísticas das operadoras em CSV: {e}")
    
    return report

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
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
    except Exception as e:
        logger.warning(f"Não foi possível adicionar mapa base: {e}")
    
    # Adicionar título e legenda
    plt.title('Localização das ERBs por Operadora', fontsize=16)
    plt.legend(title='Operadora', loc='upper right')
    
    # Salvar figura
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_cobertura_operadora.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Mapa de setores de cobertura, se disponíveis
    if 'setor_geometria' in gdf.columns and gdf['setor_geometria'].notna().any():
        logger.info("Gerando mapa de setores de cobertura")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
        # Criar GeoDataFrame temporário para os setores
        setores_gdf = gdf[gdf['setor_geometria'].notna()].copy()
        setores_gdf['geometry'] = setores_gdf['setor_geometria']
        
        # Plotar setores por operadora
        for op in operadoras:
            subset = setores_gdf[setores_gdf['NomeEntidade'] == op]
            if len(subset) > 0:
                color = colors_operadoras[op]
                subset.plot(ax=ax, color=color, alpha=0.3, label=f"{op} (Cobertura)")
    
    # Adicionar pontos das ERBs
        gdf.plot(ax=ax, markersize=15, color='black', alpha=0.7)
    
    # Adicionar mapa base
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
        except Exception as e:
            logger.warning(f"Não foi possível adicionar mapa base: {e}")
    
    # Adicionar título e legenda
        plt.title('Setores de Cobertura das ERBs', fontsize=16)
        plt.legend(title='Operadora', loc='upper right')
    
    # Salvar figura
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_setores_cobertura.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Mapa de clusters, se disponível
    if 'cluster_id' in gdf.columns:
        logger.info("Gerando mapa de clusters de ERBs")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
        # Identificar clusters únicos (excluindo ruído)
        clusters = [c for c in gdf['cluster_id'].unique() if c != -1]
        
        # Criar paleta de cores para clusters
        colors_clusters = plt.cm.tab20(np.linspace(0, 1, len(clusters)))
        color_dict = dict(zip(clusters, colors_clusters))
        
        # Plotar pontos de ruído primeiro (em cinza)
        noise = gdf[gdf['cluster_id'] == -1]
        if len(noise) > 0:
            noise.plot(ax=ax, color='gray', alpha=0.5, label='Ruído', markersize=30)
        
        # Plotar cada cluster com cores diferentes
        for cluster_id in clusters:
            subset = gdf[gdf['cluster_id'] == cluster_id]
            color = color_dict[cluster_id]
            subset.plot(ax=ax, color=color, alpha=0.7, label=f'Cluster {cluster_id}', markersize=30)
    
    # Adicionar mapa base
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
        except Exception as e:
            logger.warning(f"Não foi possível adicionar mapa base: {e}")
    
    # Adicionar título e legenda
        plt.title('Clustering Espacial de ERBs', fontsize=16)
        plt.legend(title='Cluster', loc='upper right')
    
    # Salvar figura
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Mapa de hexágonos de vulnerabilidade, se disponível
    if hex_gdf is not None:
        plot_hex_vulnerability_map(hex_gdf)
    
    # 5. Mapa de rede, se disponível
    if G is not None:
        logger.info("Gerando visualização da rede de cobertura")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
        # Obter posições dos nós
        pos = nx.get_node_attributes(G, 'pos')
        
        # Desenhar nós por tipo
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if G.nodes[node]['tipo'] == 'erb':
                operadora = G.nodes[node]['operadora']
                node_colors.append(colors_operadoras.get(operadora, 'gray'))
                node_sizes.append(100)
            else:  # hexágono
                node_colors.append('lightgray')
                node_sizes.append(10)
        
        # Desenhar arestas
        nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
        
        # Desenhar nós
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.7)
        
        # Adicionar mapa base
        # Converter para GeoDataFrame para poder adicionar mapa base
        nodes_gdf = gpd.GeoDataFrame(
            {node: G.nodes[node] for node in G.nodes() if G.nodes[node]['tipo'] == 'erb'},
            geometry=[Point(pos[node]) for node in G.nodes() if G.nodes[node]['tipo'] == 'erb'],
            crs=gdf.crs
        ).T
        
        try:
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs)
        except Exception as e:
            logger.warning(f"Não foi possível adicionar mapa base: {e}")
        
        # Adicionar título
        plt.title('Rede de Cobertura', fontsize=16)
        
        # Configurar limites do gráfico
        x_min, x_max = min(x for x, y in pos.values()), max(x for x, y in pos.values())
        y_min, y_max = min(y for x, y in pos.values()), max(y for x, y in pos.values())
        plt.xlim(x_min - 0.05 * (x_max - x_min), x_max + 0.05 * (x_max - x_min))
        plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        
        # Desativar eixos
        plt.axis('off')
        
        # Salvar figura
        plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_rede.png'), dpi=300, bbox_inches='tight')
        plt.close()

def calculate_density(gdf):
    """
    Calcula a densidade de ERBs por área.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com densidade de ERBs
    """
    logger.info("Calculando densidade de ERBs por área")
    
    result = gdf.copy()
    
    # Converter para CRS métrico para cálculos precisos de área
    gdf_proj = result.to_crs(METRIC_CRS)
    
    # Calcular o convex hull da área total
    all_points = gdf_proj.unary_union.convex_hull
    
    # Calcular a área em km²
    total_area_km2 = all_points.area / 1_000_000
    
    # Calcular a densidade geral (ERBs por km²)
    densidade_geral = len(gdf) / total_area_km2
    
    logger.info(f"Densidade geral: {densidade_geral:.4f} ERBs/km²")
    logger.info(f"Área total analisada: {total_area_km2:.2f} km²")
    
    # Adicionar a densidade como atributo do GeoDataFrame
    result.attrs['densidade_geral'] = densidade_geral
    result.attrs['area_total_km2'] = total_area_km2
    
    # Extrair coordenadas e calcular densidade local usando CPU paralela
    logger.info("Calculando densidade local para cada ERB (otimizado para CPU)")
    try:
        # Extrair coordenadas x, y para cálculo rápido
        coords = np.vstack((gdf_proj.geometry.x, gdf_proj.geometry.y)).T
        
        # Dividir os dados em lotes para processamento em paralelo
        num_cores = max(1, mp.cpu_count() - 1)  # Deixar um núcleo livre
        batch_size = max(1, len(coords) // num_cores)
        
        logger.info(f"Usando {num_cores} núcleos de CPU para processamento paralelo")
        
        # Usar o array gerado pela função de cálculo em paralelo
        densities = calculate_densities_parallel(coords, buffer_size=1000)
        result['densidade_local'] = densities
        
        logger.info(f"Densidade local calculada para {len(result)} ERBs")
    except Exception as e:
        logger.error(f"Erro ao calcular densidade local: {e}")
        # Fallback para método mais simples
        logger.info("Usando método alternativo para cálculo de densidade")
        for idx, row in result.iterrows():
            point_buffer = row.geometry.buffer(1)
            erbs_in_buffer = sum(result.geometry.intersects(point_buffer)) - 1  # -1 para excluir o próprio ponto
            result.at[idx, 'densidade_local'] = erbs_in_buffer
    
    # Calcular densidade por cluster
    if 'cluster_id' in result.columns:
        logger.info("Calculando densidade por cluster")
        cluster_densidades = {}
        
        for cluster_id in result['cluster_id'].unique():
            if cluster_id == -1:  # Pular pontos de ruído
                continue
                
            # Filtrar apenas ERBs deste cluster
            cluster_erbs = gdf_proj[gdf_proj.index.isin(result[result['cluster_id'] == cluster_id].index)]
            
            if len(cluster_erbs) < 3:  # Precisa de pelo menos 3 pontos para um polígono válido
                continue
                
            # Calcular convex hull do cluster
            cluster_hull = cluster_erbs.unary_union.convex_hull
            
            # Calcular área em km²
            cluster_area_km2 = cluster_hull.area / 1_000_000
            
            # Calcular densidade do cluster
            cluster_densidade = len(cluster_erbs) / cluster_area_km2
            
            # Armazenar no dicionário
            cluster_densidades[cluster_id] = {
                'area_km2': cluster_area_km2,
                'densidade': cluster_densidade,
                'num_erbs': len(cluster_erbs)
            }
            
            # Atualizar o GeoDataFrame com a densidade do cluster
            result.loc[result['cluster_id'] == cluster_id, 'densidade_espacial'] = cluster_densidade
        
        # Adicionar informações ao GeoDataFrame
        result.attrs['cluster_densidades'] = cluster_densidades
        
        # Log das densidades por cluster
        for cluster_id, info in cluster_densidades.items():
            logger.info(f"Cluster {cluster_id}: {info['densidade']:.4f} ERBs/km² (área: {info['area_km2']:.2f} km², ERBs: {info['num_erbs']})")
    
    return result

def calculate_voronoi(gdf):
    """
    Calcula o diagrama de Voronoi para as ERBs.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB
        
    Returns:
        tuple: (gdf atualizado, gdf do diagrama de Voronoi)
    """
    logger.info("Calculando diagrama de Voronoi para as ERBs")
    
    # Criar uma cópia projetada para cálculos geométricos usando a constante METRIC_CRS
    gdf_proj = gdf.to_crs(METRIC_CRS)
    
    # Extrair coordenadas das ERBs
    coords = np.array([(point.x, point.y) for point in gdf_proj.geometry])
    
    # Adicionar pontos de contorno para limitar o diagrama de Voronoi
    boundary_points = np.array([
        [coords[:, 0].min() - 10000, coords[:, 1].min() - 10000],
        [coords[:, 0].min() - 10000, coords[:, 1].max() + 10000],
        [coords[:, 0].max() + 10000, coords[:, 1].min() - 10000],
        [coords[:, 0].max() + 10000, coords[:, 1].max() + 10000]
    ])
    coords_extended = np.vstack([coords, boundary_points])
    
    # Calcular diagrama de Voronoi
    vor = Voronoi(coords_extended)
    
    # Criar polígonos de Voronoi
    polygons = []
    point_indices = []
    
    for i, region_idx in enumerate(vor.point_region[:len(coords)]):
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            polygon_vertices = [vor.vertices[v] for v in region]
            if len(polygon_vertices) >= 3:
                poly = Polygon(polygon_vertices)
                polygons.append(poly)
                point_indices.append(i)
    
    # Criar GeoDataFrame para os polígonos de Voronoi
    voronoi_gdf = gpd.GeoDataFrame(
        {'erb_id': point_indices,
         'NomeEntidade': gdf.iloc[point_indices]['NomeEntidade'].values,
         'geometry': polygons},
        crs=gdf_proj.crs
    )
    
    # Calcular área dos polígonos em km² (já estamos no CRS métrico, então a divisão é simples)
    voronoi_gdf['area_voronoi_km2'] = voronoi_gdf.geometry.area / 1_000_000
    
    # Adicionar informações sobre vizinhos
    voronoi_gdf['num_vizinhos'] = 0
    
    for i, row in voronoi_gdf.iterrows():
        # Contar polígonos que compartilham fronteira
        vizinhos = voronoi_gdf[voronoi_gdf.geometry.touches(row.geometry)]
        voronoi_gdf.at[i, 'num_vizinhos'] = len(vizinhos)
    
    # Voltar para CRS original para o retorno
    voronoi_gdf = voronoi_gdf.to_crs(gdf.crs)
    
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
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=voronoi_gdf.crs)
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
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=hex_gdf.crs)
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

def process_hex_in_parallel(hex_id, voronois, data_gdf, resolution):
    """
    Processa um único hexágono para análise de vulnerabilidade, otimizado para execução paralela.
    
    Args:
        hex_id (str): ID do hexágono H3.
        voronois (GeoDataFrame): GeoDataFrame com polígonos Voronoi.
        data_gdf (GeoDataFrame): GeoDataFrame com dados de ERBs.
        resolution (int): Resolução H3.
        
    Returns:
        tuple: (hex_id, dados de vulnerabilidade)
    """
    try:
        # Usar função cacheada para obter o limite
        boundary = cached_hex_boundary(hex_id)
        hex_geom = shapely.geometry.Polygon(boundary)
        
        # Contar ERBs dentro do hexágono usando índice espacial
        hex_box = hex_geom.bounds
        possible_matches_index = list(data_gdf.sindex.intersection(hex_box))
        possible_matches = data_gdf.iloc[possible_matches_index]
        erbs_dentro = possible_matches[possible_matches.intersects(hex_geom)]
        erbs_count = len(erbs_dentro)
        
        # Calcular vulnerabilidade
        if erbs_count > 0:
            # Calcular estatísticas de cobertura
            densidades = erbs_dentro['densidade'].mean() if erbs_count > 0 else 0
            raios = erbs_dentro['raio_coverage'].mean() if erbs_count > 0 else 0
            
            # Calcular área de interseção com Voronoi usando índice espacial
            voronoi_box = hex_geom.bounds
            possible_voronoi_matches = list(voronois.sindex.intersection(voronoi_box))
            intersecting_voronois = voronois.iloc[possible_voronoi_matches]
            intersecting_voronois = intersecting_voronois[intersecting_voronois.intersects(hex_geom)]
            
            # Calcular percentual de cobertura
            area_hex = hex_geom.area
            area_intersection = sum(v.intersection(hex_geom).area for v in intersecting_voronois.geometry)
            percentual_cobertura = 100 * area_intersection / area_hex if area_hex > 0 else 0
            
            return hex_id, {
                'erbs_count': erbs_count, 
                'densidade_media': densidades,
                'raio_medio': raios,
                'percentual_cobertura': percentual_cobertura,
                'geometry': hex_geom
            }
        else:
            return hex_id, {
                'erbs_count': 0, 
                'densidade_media': 0,
                'raio_medio': 0,
                'percentual_cobertura': 0,
                'geometry': hex_geom
            }
    except Exception as e:
        logger.error(f"Erro ao processar hexágono {hex_id}: {str(e)}")
        return hex_id, None

def process_hexagons_parallel(hexagons, voronois, data_gdf, resolution):
    """
    Processa hexágonos para análise de vulnerabilidade em paralelo.
    
    Args:
        hexagons (list): Lista de IDs de hexágonos H3.
        voronois (GeoDataFrame): GeoDataFrame com polígonos Voronoi.
        data_gdf (GeoDataFrame): GeoDataFrame com dados de ERBs.
        resolution (int): Resolução H3.
        
    Returns:
        dict: Dicionário com dados de vulnerabilidade por hexágono.
    """
    # Adicionar índice espacial ao GeoDataFrame se ainda não existir
    if not hasattr(data_gdf, 'sindex') or data_gdf.sindex is None:
        data_gdf = data_gdf.copy()
        data_gdf.sindex = data_gdf.geometry.sindex
    
    if not hasattr(voronois, 'sindex') or voronois.sindex is None:
        voronois = voronois.copy()
        voronois.sindex = voronois.geometry.sindex
    
    # Determinar número de processos baseado no sistema
    num_processes = min(mp.cpu_count(), 8)  # Limite para evitar sobrecarga
    
    # Preparar para processamento paralelo
    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_hex_in_parallel,
            [(hex_id, voronois, data_gdf, resolution) for hex_id in hexagons]
        )
    
    # Filtrar resultados com erro
    valid_results = {hex_id: data for hex_id, data in results if data is not None}
    
    logger.info(f"Processados {len(valid_results)} hexágonos válidos de {len(hexagons)} total")
    return valid_results

def calculate_voronoi_optimized(data_gdf):
    """
    Calcula diagrama de Voronoi otimizado para grande volume de dados.
    
    Args:
        data_gdf (GeoDataFrame): GeoDataFrame com dados de ERBs.
        
    Returns:
        GeoDataFrame: GeoDataFrame com polígonos Voronoi.
    """
    # Extrair coordenadas para processamento vetorizado
    coords = np.array([(p.x, p.y) for p in data_gdf.geometry])
    
    # Usar SciPy para Voronoi rápido
    vor = Voronoi(coords)
    
    # Processar polígonos em lotes para melhor desempenho
    batch_size = 1000
    voronoi_polys = []
    
    total_polys = len(vor.point_region)
    num_batches = (total_polys + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_polys)
        
        batch_polys = []
        for i in range(start_idx, end_idx):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            
            if -1 not in region and len(region) > 0:
                polygon_vertices = [vor.vertices[v] for v in region]
                if len(polygon_vertices) >= 3:  # Verificar se é um polígono válido
                    poly = shapely.geometry.Polygon(polygon_vertices)
                    if poly.is_valid:
                        batch_polys.append((i, poly))
                    else:
                        # Tentar reparar polígono inválido
                        fixed_poly = shapely.geometry.Polygon(poly.buffer(0).exterior)
                        if fixed_poly.is_valid:
                            batch_polys.append((i, fixed_poly))
        
        voronoi_polys.extend(batch_polys)
        
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            logger.info(f"Processados {batch_idx + 1}/{num_batches} lotes de polígonos Voronoi")
    
    # Criar GeoDataFrame de resultados
    indices = [idx for idx, _ in voronoi_polys]
    geometries = [poly for _, poly in voronoi_polys]
    
    voronoi_gdf = gpd.GeoDataFrame(
        {'id': indices},
        geometry=geometries,
        crs=data_gdf.crs
    )
    
    # Juntar dados originais
    voronoi_gdf = voronoi_gdf.merge(data_gdf.reset_index(drop=True), left_on='id', right_index=True, how='left')
    
    return voronoi_gdf

def main():
    """
    Função principal para executar o enriquecimento de dados de ERBs.
    
    Returns:
        dict: Resultados do processamento ou None em caso de falha
    """
    # Configurar o sistema de logging
    setup_logging()
    
    # Registrar início do processo
    start_time = time.time()
    logger.info("Iniciando processo de enriquecimento de dados de ERBs")
    
    # Garantir que todos os diretórios existam
    ensure_directories()
    
    # Verificar disponibilidade de GPU
    cuda_disponivel = is_cuda_available()
    logger.info(f"GPU com CUDA disponível: {cuda_disponivel}")
    
    # Informar sobre número de CPUs disponíveis
    num_cpus = mp.cpu_count()
    logger.info(f"CPUs disponíveis para processamento paralelo: {num_cpus}")
    
    # Carregar dados
    logger.info("Carregando dados de ERB")
    original_data = load_data()
    if original_data is None or len(original_data) == 0:
        logger.error("Não foi possível carregar os dados de ERB")
        return None
    
    # Criar uma cópia dos dados originais para análise comparativa
    data = original_data.copy()
    
    logger.info(f"Colunas disponíveis: {data.columns.tolist()}")
    
    # Verificar colunas obrigatórias
    required_columns = ['geometry', 'PotenciaTransmissorWatts', 'GanhoAntena']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        logger.error(f"Colunas obrigatórias ausentes nos dados: {missing_columns}")
        logger.error("Verifique o arquivo de entrada e tente novamente.")
        return None
    
    # Verificar se a coluna 'Azimute' está presente
    if 'Azimute' not in data.columns:
        logger.error("Coluna 'Azimute' não encontrada. Este campo é necessário para cálculos de cobertura.")
        return None
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
    
    # Calcular densidade de ERBs
    logger.info("Calculando densidade de ERBs por área")
    checkpoint_time = time.time()
    data = calculate_density(data)
    logger.info(f"Densidade calculada em {time.time() - checkpoint_time:.2f} segundos")
    
    # Calcular raio de cobertura
    logger.info("Calculando raio de cobertura")
    checkpoint_time = time.time()
    data = calculate_coverage_radius(data)
    logger.info(f"Raio de cobertura calculado em {time.time() - checkpoint_time:.2f} segundos")
    
    # Criar setores de cobertura
    logger.info("Criando setores de cobertura")
    checkpoint_time = time.time()
    data = create_coverage_sectors(data)
    logger.info(f"Setores de cobertura criados em {time.time() - checkpoint_time:.2f} segundos")
    
    # Calcular diagrama de Voronoi
    logger.info("Calculando diagrama de Voronoi")
    checkpoint_time = time.time()
    data, voronoi_gdf = calculate_voronoi(data)
    logger.info(f"Diagrama de Voronoi calculado em {time.time() - checkpoint_time:.2f} segundos")
    
    # Criar grade hexagonal para análise de vulnerabilidade
    logger.info("Criando grade hexagonal para análise de vulnerabilidade")
    checkpoint_time = time.time()
    data, hex_gdf = create_hexagon_coverage_grid(data)
    logger.info(f"Grade hexagonal criada em {time.time() - checkpoint_time:.2f} segundos")
    
    # Analisar clustering espacial
    logger.info("Realizando análise de clustering espacial")
    checkpoint_time = time.time()
    data = analyze_spatial_clustering(data)
    logger.info(f"Clustering espacial realizado em {time.time() - checkpoint_time:.2f} segundos")
    
    # NOTA: Removendo criação do grafo de cobertura - será feito em nuvem
    logger.info("Pulando criação de rede de cobertura, será processado em nuvem")
    G = None  # Grafo vazio para compatibilidade com funções subsequentes
    
    # Classificar ERBs por importância (versão simplificada sem grafo)
    logger.info("Classificando ERBs por importância (versão simplificada)")
    checkpoint_time = time.time()
    # Usar uma versão adaptada sem dependência do grafo
    data = data.copy()  # Manter dados originais sem transformação adicional
    logger.info(f"Classificação de importância simplificada em {time.time() - checkpoint_time:.2f} segundos")
    
    # Gerar relatório de qualidade completo
    logger.info("Gerando relatório de qualidade detalhado")
    checkpoint_time = time.time()
    quality_report = generate_quality_report(original_data, data, hex_gdf, G)
    logger.info(f"Relatório de qualidade gerado em {time.time() - checkpoint_time:.2f} segundos")
    
    # Gerar visualizações
    logger.info("Gerando visualizações")
    try:
        checkpoint_time = time.time()
        # Visualizações básicas
        generate_visualizations(data)
        
        # Visualização do diagrama de Voronoi
        plot_voronoi_map(data, voronoi_gdf)
        
        # Visualização de vulnerabilidade em hexágonos
        plot_hex_vulnerability_map(hex_gdf)
        
        # Visualização 3D de cobertura
        plot_3d_coverage(data)
        
        # Pular criação de mapa interativo completo pois depende do grafo
        logger.info("Pulando criação de mapa interativo, será processado em nuvem")
        
        logger.info(f"Visualizações geradas com sucesso em {time.time() - checkpoint_time:.2f} segundos")
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações: {e}")
    
    # Salvar dados enriquecidos
    logger.info("Salvando dados enriquecidos")
    checkpoint_time = time.time()
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
        
        logger.info(f"Dados enriquecidos salvos em {output_file} em {time.time() - checkpoint_time:.2f} segundos")
    except Exception as e:
        logger.error(f"Erro ao salvar dados enriquecidos: {e}")
    
    # Registrar fim do processo
    end_time = time.time()
    processing_time = end_time - start_time
    logger.info(f"Processo de enriquecimento concluído em {processing_time:.2f} segundos")
    logger.info(f"Dados enriquecidos salvos em {output_file}")
    logger.info(f"Visualizações salvas em {VISUALIZATION_DIR}")
    logger.info(f"Relatórios de qualidade salvos em {REPORT_DIR}")
    
    # Retornar objeto com resultados do processamento
    return {
        "enriched_data": data,
        "voronoi": voronoi_gdf,
        "hexagons": hex_gdf,
        "network": None,  # Nenhum grafo foi criado
        "output_file": output_file,
        "quality_report": quality_report,
        "processing_time_seconds": processing_time
    }

if __name__ == "__main__":
    # Importar bibliotecas adicionais que só são usadas em visualizações
    import seaborn as sns
    import matplotlib.patheffects as pe
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    
    main()
