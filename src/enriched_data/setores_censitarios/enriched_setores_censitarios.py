#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
import matplotlib.pyplot as plt
import seaborn as sns
from shapely.geometry import Polygon, Point, box
import json
from datetime import datetime
import contextily as ctx
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde
from tqdm import tqdm
import rasterstats
import logging
import time
import traceback
import argparse
import folium
from folium.plugins import FloatImage, Fullscreen, MeasureControl, Search
from branca.colormap import linear
import fiona

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('enriched_setores_censitarios')

# Obter o caminho absoluto para o diretório do projeto
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
workspace_dir = os.path.dirname(src_dir)

# Definir diretórios de entrada e saída
INPUT_DIR = os.path.join(workspace_dir, 'data', 'processed')
RAW_DIR = os.path.join(workspace_dir, 'data', 'raw')
OUTPUT_DIR = "F:\\TESE_MESTRADO\\geoprocessing\\data\\enriched_data"
REPORT_DIR = os.path.join(workspace_dir, 'src', 'enriched_data', 'quality_reports', 'setores_censitarios')
VISUALIZATION_DIR = os.path.join(workspace_dir, 'outputs', 'visualize_enriched_data', 'setores_censitarios')

# Definir caminhos de arquivos específicos
SETORES_FILE = os.path.join(INPUT_DIR, 'setores_censitarios_processed.gpkg')
DEM_FILE = os.path.join(RAW_DIR, 'dem.tif')

# Classe auxiliar para serialização JSON de tipos numpy
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

# Garantir que os diretórios de saída existam
for directory in [OUTPUT_DIR, REPORT_DIR, VISUALIZATION_DIR]:
    os.makedirs(directory, exist_ok=True)
    logger.debug(f"Diretório garantido: {directory}")

def load_data():
    """
    Carrega os dados de setores censitários do diretório de processamento.
    
    Returns:
        geopandas.GeoDataFrame: Os dados de setores censitários carregados.
    """
    try:
        logger.info(f"Carregando dados de setores censitários de {SETORES_FILE}")
        if not os.path.exists(SETORES_FILE):
            logger.error(f"Arquivo de dados não encontrado: {SETORES_FILE}")
            return None
            
        gdf = gpd.read_file(SETORES_FILE)
        logger.info(f"Carregados {len(gdf)} setores censitários")
        return gdf
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        logger.error(traceback.format_exc())
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
        logger.error(traceback.format_exc())
        return None

def calculate_area_metrics(gdf):
    """
    Calcula métricas de área para cada setor censitário.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados dos setores censitários
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com métricas de área
    """
    # Criar uma cópia para evitar SettingWithCopyWarning
    result = gdf.copy()
    
    # Reprojetar para um sistema de coordenadas projetado para cálculos de área precisos
    if result.crs and result.crs.is_geographic:
        # SIRGAS 2000 / UTM zone 23S para o Brasil Central
        gdf_utm = result.to_crs(epsg=31983)
        
        # Calcular área em km²
        result['area_km2'] = gdf_utm.geometry.area / 1_000_000
        
        # Calcular perímetro em km
        result['perimetro_km'] = gdf_utm.geometry.length / 1_000
        
        # Calcular centróides
        centroids = gdf_utm.geometry.centroid
        centroids_wgs84 = gpd.GeoSeries(centroids, crs=gdf_utm.crs).to_crs(result.crs)
        result['centroid_x'] = centroids_wgs84.x
        result['centroid_y'] = centroids_wgs84.y
        
        # Calcular índice de compacidade (4π × Area/Perímetro²)
        # Valor máximo é 1 (círculo perfeito)
        result['compacidade'] = (4 * np.pi * result['area_km2']) / (result['perimetro_km'] ** 2)
        
        # Classificar tamanho dos setores
        bins = [0, 0.1, 0.5, 1, 5, float('inf')]
        labels = ['Muito pequeno', 'Pequeno', 'Médio', 'Grande', 'Muito grande']
        result['categoria_tamanho'] = pd.cut(result['area_km2'], bins=bins, labels=labels)
    else:
        print("AVISO: O GeoDataFrame não possui um CRS geográfico definido. Usando a geometria original para cálculos.")
        # Calcular área (unidades da geometria atual)
        result['area_unit'] = result.geometry.area
        result['perimetro_unit'] = result.geometry.length
    
    return result

def extract_elevation_data(gdf, dem):
    """
    Extrai estatísticas de elevação do DEM para cada setor censitário.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados dos setores censitários
        dem (rasterio.DatasetReader): Modelo Digital de Elevação
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com dados de elevação
    """
    result = gdf.copy()
    
    # Garantir que o GeoDataFrame esteja no mesmo CRS que o DEM
    if result.crs.to_string() != dem.crs.to_string():
        result = result.to_crs(dem.crs)
        print(f"GeoDataFrame reprojetado para coincidir com o CRS do DEM: {dem.crs}")
    
    # Extrair estatísticas de elevação para cada setor
    print("Extraindo estatísticas de elevação para cada setor...")
    
    # Usando rasterstats para processamento eficiente
    stats = rasterstats.zonal_stats(
        result.geometry,
        dem.read(1),
        affine=dem.transform,
        stats=['min', 'max', 'mean', 'median', 'std'],
        nodata=dem.nodata
    )
    
    # Adicionar estatísticas ao GeoDataFrame
    result['elevation_min'] = [s['min'] for s in stats]
    result['elevation_max'] = [s['max'] for s in stats]
    result['elevation_mean'] = [s['mean'] for s in stats]
    result['elevation_median'] = [s['median'] for s in stats]
    result['elevation_std'] = [s['std'] for s in stats]
    
    # Calcular amplitude de elevação
    result['elevation_range'] = result['elevation_max'] - result['elevation_min']
    
    # Classificar variação de elevação
    bins = [0, 10, 30, 50, 100, float('inf')]
    labels = ['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito alta']
    result['categoria_elevacao'] = pd.cut(result['elevation_range'], bins=bins, labels=labels)
    
    return result

def calculate_population_distribution(gdf):
    """
    Realiza cálculos de distribuição populacional usando as informações do setor censitário.
    Implementa distribuição por horário e estimativas de PEA conforme mencionado no arquivo MBA.txt.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados dos setores censitários
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com estimativas populacionais
    """
    result = gdf.copy()
    
    # Verificar se existem colunas de população
    # Nota: ajuste estas colunas conforme os nomes reais no seu arquivo
    pop_columns = [col for col in result.columns if 'pop' in col.lower()]
    
    if not pop_columns:
        print("AVISO: Não foram encontradas colunas de população no arquivo. Criando estimativas básicas.")
        # Criar população estimada total com base na área (apenas como exemplo)
        result['est_populacao'] = result['area_km2'] * 1000  # Assumindo densidade média de 1000 hab/km²
    else:
        # Se já existir coluna de população, use-a como base
        pop_col = pop_columns[0]
        result['est_populacao'] = result[pop_col]
    
    # Estimar População Economicamente Ativa (PEA) - tipicamente 60-70% da população total
    result['est_pop_pea'] = result['est_populacao'] * 0.65
    
    # Estimar população por horário do dia
    # Manhã (8h) - Áreas predominantemente comerciais têm mais pessoas que áreas residenciais
    # Identificar tipo de área (comercial vs residencial) com base em dados ou assumir uma distribuição
    
    # Primeiro, identificamos o tipo de área (simplificado)
    # Na falta de dados específicos, faremos uma classificação aproximada
    # Em um caso real, seria necessário cruzar com dados de uso do solo
    if 'classe' in result.columns:
        # Se já existir classificação, usar
        result['tipo_area'] = result['classe']
    else:
        # Atribuir uma classificação básica por densidade populacional
        result['densidade_pop'] = result['est_populacao'] / result['area_km2']
        
        bins = [0, 1000, 3000, 5000, 10000, float('inf')]
        labels = ['Rural', 'Suburbana', 'Residencial', 'Mista', 'Comercial']
        result['tipo_area'] = pd.cut(result['densidade_pop'], bins=bins, labels=labels)
    
    # Distribuição populacional por horário com base no tipo de área
    # Baseado nos padrões mencionados no arquivo MBA.txt
    
    # Fatores de distribuição por tipo de área e horário
    fatores_distribuicao = {
        'Rural': {'0800': 0.7, '1200': 0.6, '1500': 0.5, '1900': 0.8, '2300': 0.9},
        'Suburbana': {'0800': 0.6, '1200': 0.4, '1500': 0.5, '1900': 0.9, '2300': 0.95},
        'Residencial': {'0800': 0.4, '1200': 0.3, '1500': 0.4, '1900': 0.9, '2300': 0.95},
        'Mista': {'0800': 0.6, '1200': 0.7, '1500': 0.7, '1900': 0.8, '2300': 0.7},
        'Comercial': {'0800': 0.8, '1200': 0.9, '1500': 0.9, '1900': 0.5, '2300': 0.1}
    }
    
    # Calcular população por horário
    for horario in ['0800', '1200', '1500', '1900', '2300']:
        result[f'pop_atual_{horario}'] = result.apply(
            lambda row: row['est_populacao'] * fatores_distribuicao.get(
                row['tipo_area'], {'0800': 0.7, '1200': 0.6, '1500': 0.6, '1900': 0.8, '2300': 0.8}
            ).get(horario, 0.7),
            axis=1
        )
    
    # Calcular a variação populacional ao longo do dia
    pop_horarios = [result[f'pop_atual_{horario}'] for horario in ['0800', '1200', '1500', '1900', '2300']]
    result['pop_variacao_max'] = np.max(pop_horarios, axis=0) - np.min(pop_horarios, axis=0)
    result['pop_variacao_rel'] = result['pop_variacao_max'] / result['est_populacao']
    
    # Classificar a variação populacional
    bins = [0, 0.2, 0.4, 0.6, 0.8, float('inf')]
    labels = ['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito alta']
    result['categoria_variacao_pop'] = pd.cut(result['pop_variacao_rel'], bins=bins, labels=labels)
    
    return result

def calculate_population_vulnerability(gdf):
    """
    Calcula índices de vulnerabilidade da população com base em características 
    do setor e da população.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados dos setores censitários
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com índices de vulnerabilidade
    """
    result = gdf.copy()
    
    # Calcular densidade populacional (habitantes por km²)
    result['densidade_pop'] = result['est_populacao'] / result['area_km2']
    
    # Índice de variação populacional (quanto maior a variação, maior a vulnerabilidade)
    # Já calculado como 'pop_variacao_rel'
    
    # Índice de rugosidade do terreno (quanto maior a variação do terreno, maior a vulnerabilidade)
    # Normalizar elevation_std para um índice entre 0 e 1
    max_std = result['elevation_std'].max()
    if max_std > 0:
        result['indice_rugosidade'] = result['elevation_std'] / max_std
    else:
        result['indice_rugosidade'] = 0
    
    # Índice de acessibilidade (baseado na compacidade - formas mais compactas são mais acessíveis)
    # Inverter compacidade (1 - compacidade) para que valores mais altos indiquem maior vulnerabilidade
    result['indice_acessibilidade'] = 1 - result['compacidade']
    
    # Índice composto de vulnerabilidade (média ponderada dos índices)
    result['indice_vulnerabilidade'] = (
        result['pop_variacao_rel'] * 0.3 +
        result['indice_rugosidade'] * 0.3 +
        result['indice_acessibilidade'] * 0.4
    )
    
    # Classificar vulnerabilidade
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito alta']
    result['categoria_vulnerabilidade'] = pd.cut(result['indice_vulnerabilidade'], bins=bins, labels=labels)
    
    return result

def generate_evacuation_priority(gdf):
    """
    Gera um índice de prioridade para evacuação com base nas características 
    do setor e população.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados dos setores censitários
        
    Returns:
        geopandas.GeoDataFrame: Atualizado com índice de prioridade para evacuação
    """
    result = gdf.copy()
    
    # Normalizar as métricas usadas no cálculo
    # Densidade populacional (mais alta, mais prioritário)
    max_densidade = result['densidade_pop'].max()
    if max_densidade > 0:
        result['densidade_pop_norm'] = result['densidade_pop'] / max_densidade
    else:
        result['densidade_pop_norm'] = 0
    
    # População total (mais alta, mais prioritário)
    max_pop = result['est_populacao'].max()
    if max_pop > 0:
        result['pop_total_norm'] = result['est_populacao'] / max_pop
    else:
        result['pop_total_norm'] = 0
    
    # Vulnerabilidade (mais alta, mais prioritário)
    # Já está normalizada entre 0 e 1
    
    # Rugosidade do terreno (mais alta, mais prioritário)
    # Já está normalizada como 'indice_rugosidade'
    
    # Acessibilidade (menos acessível, mais prioritário)
    # Já está normalizada como 'indice_acessibilidade'
    
    # Calcular o índice composto de prioridade para evacuação
    result['indice_prioridade_evacuacao'] = (
        result['densidade_pop_norm'] * 0.25 +
        result['pop_total_norm'] * 0.25 +
        result['indice_vulnerabilidade'] * 0.5
    )
    
    # Classificar prioridade
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels = ['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito alta']
    result['categoria_prioridade'] = pd.cut(
        result['indice_prioridade_evacuacao'], 
        bins=bins, 
        labels=labels
    )
    
    # Identificar os top 10% de setores prioritários
    n_top = int(len(result) * 0.1)
    result['top_prioridade'] = False
    if n_top > 0:
        indices_top = result.nlargest(n_top, 'indice_prioridade_evacuacao').index
        result.loc[indices_top, 'top_prioridade'] = True
    
    return result

def save_enriched_data(enriched_gdf, output_file=None):
    """
    Salva o GeoDataFrame enriquecido em um arquivo GPKG com múltiplas camadas
    para diferentes tipos de geometria se necessário.
    
    Args:
        enriched_gdf (geopandas.GeoDataFrame): Dados enriquecidos
        output_file (str, optional): Caminho de saída. Se None, gera um nome com timestamp.
        
    Returns:
        str: Caminho do arquivo salvo
    """
    try:
        # Verificar se temos um GeoDataFrame válido
        if enriched_gdf is None or len(enriched_gdf) == 0:
            logger.error("GeoDataFrame vazio ou inválido. Nada para salvar.")
            return None
            
        # Verificar se temos uma coluna de geometria
        if 'geometry' not in enriched_gdf.columns:
            logger.error("GeoDataFrame não contém coluna 'geometry'. Impossível salvar.")
            return None
        
        # Preparar caminho de saída
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(OUTPUT_DIR, f"setores_censitarios_enriched_{timestamp}.gpkg")
        
        # Garantir que o caminho é absoluto e normalizado
        output_file = os.path.abspath(os.path.normpath(output_file))
        logger.info(f"Caminho de saída normalizado: {output_file}")
        
        # Garantir que o diretório de saída exista
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Verificar permissões de escrita
        if not os.access(os.path.dirname(output_file), os.W_OK):
            logger.error(f"Sem permissão de escrita no diretório: {os.path.dirname(output_file)}")
            return None
        
        # Criar cópia limpa do GeoDataFrame para salvar
        serializable_gdf = enriched_gdf.copy()
        
        # Remover geometrias nulas
        if serializable_gdf.geometry.isna().any():
            invalid_count = serializable_gdf.geometry.isna().sum()
            logger.warning(f"Removendo {invalid_count} geometrias nulas")
            serializable_gdf = serializable_gdf.dropna(subset=['geometry'])
        
        if len(serializable_gdf) == 0:
            logger.error("Após remover geometrias inválidas, não restaram feições para salvar.")
            return None
        
        # Corrigir geometrias inválidas
        invalid_geoms = ~serializable_gdf.geometry.is_valid
        if invalid_geoms.any():
            logger.warning(f"Corrigindo {invalid_geoms.sum()} geometrias inválidas")
            serializable_gdf.geometry = serializable_gdf.geometry.buffer(0)
        
        # Garantir que o CRS está definido
        if serializable_gdf.crs is None:
            logger.warning("CRS não definido, definindo EPSG:4326")
            serializable_gdf = serializable_gdf.set_crs(epsg=4326, allow_override=True)
        
        # Selecionar apenas colunas essenciais para minimizar problemas
        essential_columns = ['geometry', 'CD_GEOCODI', 'area_km2', 'est_populacao', 'densidade_pop',
                          'indice_vulnerabilidade', 'indice_prioridade_evacuacao', 'tipo_area']
        
        available_columns = [col for col in essential_columns if col in serializable_gdf.columns]
        
        # Criar GeoDataFrame simplificado
        slim_gdf = serializable_gdf[available_columns]
        
        # Tratar valores NaN e Infinitos
        for col in slim_gdf.columns:
            if col != 'geometry':
                if pd.api.types.is_numeric_dtype(slim_gdf[col]):
                    # Substituir infinito e NaN
                    slim_gdf[col] = slim_gdf[col].replace([np.inf, -np.inf], [1e38, -1e38])
                    slim_gdf[col] = slim_gdf[col].fillna(0)
                elif slim_gdf[col].dtype.name == 'object':
                    slim_gdf[col] = slim_gdf[col].fillna('')
        
        # Verificar se precisamos dividir por tipo de geometria
        geom_types = slim_gdf.geometry.geom_type.unique()
        
        logger.info(f"Tipos de geometria encontrados: {', '.join(geom_types)}")
        
        if len(geom_types) > 1:
            logger.info(f"Detectados múltiplos tipos de geometria. Salvando em camadas separadas do mesmo arquivo GPKG.")
            
            # Dividir por tipo de geometria e salvar em camadas separadas
            success = True
            layers_saved = []
            
            for geom_type in geom_types:
                layer_name = f"setores_{geom_type.lower()}"
                type_gdf = slim_gdf[slim_gdf.geometry.geom_type == geom_type]
                
                if len(type_gdf) > 0:
                    try:
                        # Modo 'w' para primeira camada, 'a' para as demais
                        mode = 'w' if geom_type == geom_types[0] else 'a'
                        logger.info(f"Salvando camada '{layer_name}' com {len(type_gdf)} feições")
                        type_gdf.to_file(output_file, layer=layer_name, driver='GPKG', mode=mode)
                        layers_saved.append(layer_name)
                    except Exception as e_layer:
                        logger.error(f"Erro ao salvar camada {layer_name}: {str(e_layer)}")
                        success = False
            
            if success and layers_saved:
                logger.info(f"GPKG salvo com sucesso em: {output_file} com camadas: {', '.join(layers_saved)}")
                return output_file
            else:
                logger.error("Falha ao salvar algumas camadas no GPKG")
                # Se falhou nas camadas separadas, tentar salvar como GeoJSON como último recurso
                try:
                    geojson_file = output_file.replace('.gpkg', '.geojson')
                    logger.info(f"Tentando salvar como GeoJSON: {geojson_file}")
                    slim_gdf.to_file(geojson_file, driver='GeoJSON')
                    logger.info(f"Backup em GeoJSON salvo com sucesso em: {geojson_file}")
                    return geojson_file
                except Exception as e_json:
                    logger.error(f"Também falhou ao salvar como GeoJSON: {str(e_json)}")
                    return None
        else:
            # Apenas um tipo de geometria, mas salvar em uma camada nomeada para consistência
            layer_name = f"setores_{geom_types[0].lower()}"
            
            try:
                logger.info(f"Salvando {len(slim_gdf)} feições na camada '{layer_name}'")
                slim_gdf.to_file(output_file, layer=layer_name, driver='GPKG')
                
                # Verificar se o arquivo foi criado com sucesso
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    logger.info(f"GPKG salvo com sucesso em: {output_file}")
                    return output_file
                else:
                    logger.error(f"Arquivo GPKG foi criado mas parece estar vazio ou corrompido")
                    return None
            except Exception as e_gpkg:
                logger.error(f"Erro ao salvar GPKG: {str(e_gpkg)}")
                
                # Tentar salvar como GeoJSON como último recurso
                try:
                    geojson_file = output_file.replace('.gpkg', '.geojson')
                    logger.info(f"Tentando salvar como GeoJSON: {geojson_file}")
                    slim_gdf.to_file(geojson_file, driver='GeoJSON')
                    logger.info(f"Backup em GeoJSON salvo com sucesso em: {geojson_file}")
                    return geojson_file
                except Exception as e_json:
                    logger.error(f"Também falhou ao salvar como GeoJSON: {str(e_json)}")
                    return None
        
        return None
    except Exception as e:
        logger.error(f"Erro ao salvar dados enriquecidos: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_quality_report(original_gdf, enriched_gdf, output_file, visualization_paths=None):
    """
    Gera um relatório detalhado de qualidade para o processo de enriquecimento.
    
    Args:
        original_gdf (geopandas.GeoDataFrame): Dados originais dos setores censitários
        enriched_gdf (geopandas.GeoDataFrame): Dados dos setores censitários enriquecidos
        output_file (str): Caminho do arquivo de dados enriquecidos
        visualization_paths (dict, optional): Dicionário com caminhos das visualizações
    
    Returns:
        str: Caminho do arquivo de relatório gerado
    """
    logger.info("Gerando relatório de qualidade...")
    
    # Criar timestamp para o nome do arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Verificar se o output_file existe
        if output_file is None:
            logger.warning("O caminho do arquivo de dados enriquecidos é None")
            actual_path = None
        else:
            file_exists = os.path.exists(output_file)
            actual_path = os.path.abspath(output_file) if file_exists else None
            
            if not file_exists:
                logger.warning(f"Arquivo de dados enriquecidos não encontrado em: {output_file}")
        
    # Criar relatório
    report = {
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_features": len(original_gdf),
        "enriched_features": len(enriched_gdf),
        "new_attributes": list(set(enriched_gdf.columns) - set(original_gdf.columns)),
            "enriched_data_path": actual_path,
            "visualizations": visualization_paths or {},
        "statistics": {
            "area_km2": {
                "mean": float(enriched_gdf['area_km2'].mean()),
                "median": float(enriched_gdf['area_km2'].median()),
                "min": float(enriched_gdf['area_km2'].min()),
                "max": float(enriched_gdf['area_km2'].max()),
                "total": float(enriched_gdf['area_km2'].sum())
            },
            "est_populacao": {
                "mean": float(enriched_gdf['est_populacao'].mean()),
                "median": float(enriched_gdf['est_populacao'].median()),
                "min": float(enriched_gdf['est_populacao'].min()),
                "max": float(enriched_gdf['est_populacao'].max()),
                "total": float(enriched_gdf['est_populacao'].sum())
            },
            "densidade_pop": {
                "mean": float(enriched_gdf['densidade_pop'].mean()),
                "median": float(enriched_gdf['densidade_pop'].median()),
                "min": float(enriched_gdf['densidade_pop'].min()),
                "max": float(enriched_gdf['densidade_pop'].max())
            },
            "elevation_range": {
                "mean": float(enriched_gdf['elevation_range'].mean()),
                "median": float(enriched_gdf['elevation_range'].median()),
                "min": float(enriched_gdf['elevation_range'].min()),
                "max": float(enriched_gdf['elevation_range'].max())
            },
            "indice_vulnerabilidade": {
                "mean": float(enriched_gdf['indice_vulnerabilidade'].mean()),
                "median": float(enriched_gdf['indice_vulnerabilidade'].median()),
                "min": float(enriched_gdf['indice_vulnerabilidade'].min()),
                "max": float(enriched_gdf['indice_vulnerabilidade'].max())
            },
            "categoria_vulnerabilidade": {
                "distribution": {str(k): int(v) for k, v in enriched_gdf['categoria_vulnerabilidade'].value_counts().to_dict().items()}
            },
            "indice_prioridade_evacuacao": {
                "mean": float(enriched_gdf['indice_prioridade_evacuacao'].mean()),
                "median": float(enriched_gdf['indice_prioridade_evacuacao'].median()),
                "min": float(enriched_gdf['indice_prioridade_evacuacao'].min()),
                "max": float(enriched_gdf['indice_prioridade_evacuacao'].max())
            },
            "categoria_prioridade": {
                "distribution": {str(k): int(v) for k, v in enriched_gdf['categoria_prioridade'].value_counts().to_dict().items()}
            },
            "top_prioridade": {
                "count": int(enriched_gdf['top_prioridade'].sum()),
                "percentage": float((enriched_gdf['top_prioridade'].sum() / len(enriched_gdf)) * 100)
            }
        },
        "tipo_area": {
            "distribution": {str(k): int(v) for k, v in enriched_gdf['tipo_area'].value_counts().to_dict().items()}
        }
    }
    
    # Salvar relatório como JSON
        report_file = os.path.join(REPORT_DIR, f'setores_censitarios_enrichment_report_{timestamp}.json')
        os.makedirs(os.path.dirname(report_file), exist_ok=True)
        
    with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False, cls=NpEncoder)
        
        logger.info(f"Relatório de qualidade salvo em {report_file}")
        return report_file
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório de qualidade: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def create_interactive_map(gdf, output_path):
    """
    Cria um mapa interativo em HTML com camadas de população e altimetria.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados dos setores censitários enriquecidos
        output_path (str): Caminho para salvar o arquivo HTML
        
    Returns:
        str: Caminho do arquivo HTML gerado
    """
    try:
        logger.info("Criando mapa interativo com camadas de população e altimetria...")
        
        # Converter para WGS84 para compatibilidade com folium
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        
        # Obter o centróide médio para centralizar o mapa
        center_lat = gdf_wgs84.geometry.centroid.y.mean()
        center_lon = gdf_wgs84.geometry.centroid.x.mean()
        
        # Criar o mapa base
        m = folium.Map(location=[center_lat, center_lon], 
                       zoom_start=11, 
                       tiles='cartodbpositron')
        
        # Adicionar controle de tela cheia e medição
        Fullscreen().add_to(m)
        MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)
        
        # Função para estilizar os polígonos
        def style_function(feature):
            return {
                'fillColor': '#ffffff',
                'color': '#000000',
                'weight': 1,
                'fillOpacity': 0.1
            }
        
        # Adicionar camada base de setores censitários
        folium.GeoJson(
            gdf_wgs84,
            name='Setores Censitários',
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['tipo_area', 'area_km2', 'est_populacao', 'categoria_tamanho'],
                aliases=['Tipo de Área', 'Área (km²)', 'População Estimada', 'Categoria de Tamanho'],
                localize=True
            )
        ).add_to(m)
        
        # Camada de densidade populacional
        colormap_densidade = linear.YlOrRd_09.scale(
            gdf_wgs84['densidade_pop'].min(),
            gdf_wgs84['densidade_pop'].max()
        )
        
        def style_densidade(feature):
            densidade = feature['properties']['densidade_pop']
            return {
                'fillColor': colormap_densidade(densidade),
                'color': '#000000',
                'weight': 0.5,
                'fillOpacity': 0.7
            }
        
        pop_layer = folium.GeoJson(
            gdf_wgs84.__geo_interface__,
            name='Densidade Populacional',
            style_function=style_densidade,
            tooltip=folium.GeoJsonTooltip(
                fields=['densidade_pop', 'est_populacao', 'tipo_area'],
                aliases=['Densidade (hab/km²)', 'População Estimada', 'Tipo de Área'],
                localize=True
            )
        )
        
        # Adicionar legenda para densidade populacional
        colormap_densidade.caption = 'Densidade Populacional (hab/km²)'
        colormap_densidade.add_to(m)
        pop_layer.add_to(m)
        
        # Camada de elevação
        colormap_elevacao = linear.terrain.scale(
            gdf_wgs84['elevation_mean'].min(),
            gdf_wgs84['elevation_mean'].max()
        )
        
        def style_elevacao(feature):
            elevacao = feature['properties']['elevation_mean']
            return {
                'fillColor': colormap_elevacao(elevacao),
                'color': '#000000',
                'weight': 0.5,
                'fillOpacity': 0.7
            }
        
        elev_layer = folium.GeoJson(
            gdf_wgs84.__geo_interface__,
            name='Elevação Média',
            style_function=style_elevacao,
            tooltip=folium.GeoJsonTooltip(
                fields=['elevation_mean', 'elevation_min', 'elevation_max', 'elevation_range'],
                aliases=['Elevação Média (m)', 'Elevação Mínima (m)', 'Elevação Máxima (m)', 'Amplitude (m)'],
                localize=True
            )
        )
        
        # Adicionar legenda para elevação
        colormap_elevacao.caption = 'Elevação Média (m)'
        m.add_child(elev_layer)
        
        # Camada de vulnerabilidade
        colormap_vuln = linear.RdYlGn_09.scale(
            0, 1, reverse=True
        )
        
        def style_vulnerabilidade(feature):
            vuln = feature['properties']['indice_vulnerabilidade']
            return {
                'fillColor': colormap_vuln(vuln),
                'color': '#000000',
                'weight': 0.5,
                'fillOpacity': 0.7
            }
        
        vuln_layer = folium.GeoJson(
            gdf_wgs84.__geo_interface__,
            name='Índice de Vulnerabilidade',
            style_function=style_vulnerabilidade,
            tooltip=folium.GeoJsonTooltip(
                fields=['indice_vulnerabilidade', 'categoria_vulnerabilidade'],
                aliases=['Índice de Vulnerabilidade', 'Categoria'],
                localize=True
            )
        )
        
        m.add_child(vuln_layer)
        
        # Camada de prioridade para evacuação
        colormap_prio = linear.YlGn_09.scale(
            0, 1
        )
        
        def style_prioridade(feature):
            prio = feature['properties']['indice_prioridade_evacuacao']
            top_prio = feature['properties']['top_prioridade'] == 'True'
            
            return {
                'fillColor': colormap_prio(prio),
                'color': '#ff0000' if top_prio else '#000000',
                'weight': 2 if top_prio else 0.5,
                'fillOpacity': 0.7
            }
        
        prio_layer = folium.GeoJson(
            gdf_wgs84.__geo_interface__,
            name='Prioridade para Evacuação',
            style_function=style_prioridade,
            tooltip=folium.GeoJsonTooltip(
                fields=['indice_prioridade_evacuacao', 'categoria_prioridade', 'top_prioridade'],
                aliases=['Índice de Prioridade', 'Categoria', 'Top 10% Prioridade'],
                localize=True
            )
        )
        
        m.add_child(prio_layer)
        
        # Camada de ocupação pela manhã (8h)
        colormap_pop_manha = linear.YlOrRd_09.scale(
            gdf_wgs84['pop_atual_0800'].min(),
            gdf_wgs84['pop_atual_0800'].max()
        )
        
        def style_pop_manha(feature):
            pop = feature['properties']['pop_atual_0800']
            return {
                'fillColor': colormap_pop_manha(pop),
                'color': '#000000',
                'weight': 0.5,
                'fillOpacity': 0.7
            }
        
        pop_manha_layer = folium.GeoJson(
            gdf_wgs84.__geo_interface__,
            name='População - Manhã (8h)',
            style_function=style_pop_manha,
            tooltip=folium.GeoJsonTooltip(
                fields=['pop_atual_0800', 'tipo_area'],
                aliases=['População (8h)', 'Tipo de Área'],
                localize=True
            )
        )
        
        m.add_child(pop_manha_layer)
        
        # Camada de ocupação pela noite (23h)
        colormap_pop_noite = linear.YlOrRd_09.scale(
            gdf_wgs84['pop_atual_2300'].min(),
            gdf_wgs84['pop_atual_2300'].max()
        )
        
        def style_pop_noite(feature):
            pop = feature['properties']['pop_atual_2300']
            return {
                'fillColor': colormap_pop_noite(pop),
                'color': '#000000',
                'weight': 0.5,
                'fillOpacity': 0.7
            }
        
        pop_noite_layer = folium.GeoJson(
            gdf_wgs84.__geo_interface__,
            name='População - Noite (23h)',
            style_function=style_pop_noite,
            tooltip=folium.GeoJsonTooltip(
                fields=['pop_atual_2300', 'tipo_area'],
                aliases=['População (23h)', 'Tipo de Área'],
                localize=True
            )
        )
        
        m.add_child(pop_noite_layer)
        
        # Adicionar controle de camadas
        folium.LayerControl().add_to(m)
        
        # Salvar o mapa
        m.save(output_path)
        logger.info(f"Mapa interativo salvo em {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Erro ao criar mapa interativo: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def generate_visualizations(gdf, timestamp=None):
    """
    Gera visualizações para acompanhar o relatório de qualidade.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados dos setores censitários enriquecidos
        timestamp (str, optional): Timestamp para usar nos nomes dos arquivos
        
    Returns:
        dict: Dicionário com caminhos das visualizações geradas
    """
    logger.info("Gerando visualizações...")
    
    # Criar timestamp para o nome dos arquivos se não fornecido
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Criar diretório para as visualizações com timestamp
    viz_dir = os.path.join(VISUALIZATION_DIR, timestamp)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Dicionário para armazenar caminhos das visualizações
    viz_paths = {}
    
    try:
    # Criar paletas de cores consistentes
    colors_vulnerabilidade = {
        'Muito baixa': '#1a9850', 
        'Baixa': '#91cf60', 
        'Média': '#ffffbf', 
        'Alta': '#fc8d59', 
        'Muito alta': '#d73027'
    }
    
    colors_prioridade = {
        'Muito baixa': '#edf8fb',
        'Baixa': '#b2e2e2',
        'Média': '#66c2a4',
        'Alta': '#2ca25f',
        'Muito alta': '#006d2c'
    }
    
    # 1. Mapa de densidade populacional
        logger.info("Gerando mapa de densidade populacional...")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
    # Plotar setores coloridos por densidade populacional
    gdf.plot(column='densidade_pop', cmap='viridis', legend=True,
             ax=ax, alpha=0.7, edgecolor='white', linewidth=0.2,
             legend_kwds={'label': 'Densidade Populacional (hab/km²)'})
    
    # Adicionar mapa base
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Adicionar título
    plt.title('Densidade Populacional por Setor Censitário', fontsize=16)
    
    # Adicionar escala
    scale_bar = AnchoredSizeBar(ax.transData, 0.05, '5 km', 'lower right', 
                              pad=0.5, color='black', frameon=True, size_vertical=0.01)
    ax.add_artist(scale_bar)
    
    # Salvar figura
        density_path = os.path.join(viz_dir, 'densidade_populacional.png')
        plt.savefig(density_path, dpi=300, bbox_inches='tight')
    plt.close()
        viz_paths['densidade_populacional'] = density_path
    
    # 2. Mapa de vulnerabilidade
        logger.info("Gerando mapa de vulnerabilidade...")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
    # Criar uma colormap personalizada
    cmap_vulnerabilidade = LinearSegmentedColormap.from_list(
        'vulnerabilidade',
        [colors_vulnerabilidade[cat] for cat in ['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito alta']],
        N=256
    )
    
    # Plotar setores coloridos por índice de vulnerabilidade
    gdf.plot(column='indice_vulnerabilidade', cmap=cmap_vulnerabilidade, legend=True,
             ax=ax, alpha=0.7, edgecolor='white', linewidth=0.2,
             legend_kwds={'label': 'Índice de Vulnerabilidade'})
    
    # Destacar os setores de alta vulnerabilidade
    gdf_alta_vuln = gdf[gdf['categoria_vulnerabilidade'].isin(['Alta', 'Muito alta'])]
    gdf_alta_vuln.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=0.8)
    
    # Adicionar mapa base
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Adicionar título
    plt.title('Vulnerabilidade por Setor Censitário', fontsize=16)
    
    # Adicionar escala
    scale_bar = AnchoredSizeBar(ax.transData, 0.05, '5 km', 'lower right', 
                              pad=0.5, color='black', frameon=True, size_vertical=0.01)
    ax.add_artist(scale_bar)
    
    # Salvar figura
        vuln_path = os.path.join(viz_dir, 'vulnerabilidade.png')
        plt.savefig(vuln_path, dpi=300, bbox_inches='tight')
    plt.close()
        viz_paths['vulnerabilidade'] = vuln_path
    
    # 3. Mapa de prioridade para evacuação
        logger.info("Gerando mapa de prioridade para evacuação...")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
    # Criar uma colormap personalizada para prioridade
    cmap_prioridade = LinearSegmentedColormap.from_list(
        'prioridade',
        [colors_prioridade[cat] for cat in ['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito alta']],
        N=256
    )
    
    # Plotar setores coloridos por índice de prioridade
    gdf.plot(column='indice_prioridade_evacuacao', cmap=cmap_prioridade, legend=True,
             ax=ax, alpha=0.7, edgecolor='white', linewidth=0.2,
             legend_kwds={'label': 'Prioridade para Evacuação'})
    
    # Destacar os setores de alta prioridade
    gdf_alta_prioridade = gdf[gdf['top_prioridade']]
    gdf_alta_prioridade.plot(ax=ax, facecolor='none', edgecolor='red', linewidth=1.5,
                           hatch='///', alpha=0.7, label='Top 10% Prioridade')
    
    # Adicionar mapa base
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Adicionar título e legenda
    plt.title('Prioridade para Evacuação por Setor Censitário', fontsize=16)
    plt.legend(loc='upper right')
    
    # Adicionar escala
    scale_bar = AnchoredSizeBar(ax.transData, 0.05, '5 km', 'lower right', 
                              pad=0.5, color='black', frameon=True, size_vertical=0.01)
    ax.add_artist(scale_bar)
    
    # Salvar figura
        prio_path = os.path.join(viz_dir, 'prioridade_evacuacao.png')
        plt.savefig(prio_path, dpi=300, bbox_inches='tight')
    plt.close()
        viz_paths['prioridade_evacuacao'] = prio_path
    
    # 4. Mapa de elevação
        logger.info("Gerando mapa de elevação...")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
    # Plotar setores coloridos por elevação média
    gdf.plot(column='elevation_mean', cmap='terrain', legend=True,
             ax=ax, alpha=0.7, edgecolor='white', linewidth=0.2,
             legend_kwds={'label': 'Elevação Média (m)'})
    
    # Adicionar mapa base
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Adicionar título
    plt.title('Elevação Média por Setor Censitário', fontsize=16)
    
    # Adicionar escala
    scale_bar = AnchoredSizeBar(ax.transData, 0.05, '5 km', 'lower right', 
                              pad=0.5, color='black', frameon=True, size_vertical=0.01)
    ax.add_artist(scale_bar)
    
    # Salvar figura
        elev_path = os.path.join(viz_dir, 'elevacao_media.png')
        plt.savefig(elev_path, dpi=300, bbox_inches='tight')
    plt.close()
        viz_paths['elevacao_media'] = elev_path
    
    # 5. Mapa de variação temporal da população
        logger.info("Gerando mapa de variação populacional...")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)
    
    # Plotar setores coloridos por variação populacional
    gdf.plot(column='pop_variacao_rel', cmap='plasma', legend=True,
             ax=ax, alpha=0.7, edgecolor='white', linewidth=0.2,
             legend_kwds={'label': 'Variação Populacional (%)'})
    
    # Adicionar mapa base
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Adicionar título
    plt.title('Variação Populacional Diária por Setor Censitário', fontsize=16)
    
    # Adicionar escala
    scale_bar = AnchoredSizeBar(ax.transData, 0.05, '5 km', 'lower right', 
                              pad=0.5, color='black', frameon=True, size_vertical=0.01)
    ax.add_artist(scale_bar)
    
    # Salvar figura
        var_pop_path = os.path.join(viz_dir, 'variacao_populacional.png')
        plt.savefig(var_pop_path, dpi=300, bbox_inches='tight')
    plt.close()
        viz_paths['variacao_populacional'] = var_pop_path
    
    # 6. Distribuição dos tipos de área
        logger.info("Gerando gráfico de distribuição de tipos de área...")
    plt.figure(figsize=(12, 8))
    tipo_area_counts = gdf['tipo_area'].value_counts()
    tipo_area_counts.plot(kind='bar', color=sns.color_palette("Set3"))
    plt.title('Distribuição de Tipos de Área', fontsize=16)
    plt.xlabel('Tipo de Área')
    plt.ylabel('Número de Setores')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for i, v in enumerate(tipo_area_counts):
        plt.text(i, v + 0.5, str(v), ha='center')
    
    plt.tight_layout()
        dist_path = os.path.join(viz_dir, 'distribuicao_tipos_area.png')
        plt.savefig(dist_path, dpi=300)
    plt.close()
        viz_paths['distribuicao_tipos_area'] = dist_path
    
    # 7. Relação entre densidade populacional e índice de vulnerabilidade
        logger.info("Gerando gráfico de relação densidade/vulnerabilidade...")
    plt.figure(figsize=(10, 8))
    
    # Criar scatter plot com densidade de pontos
    plt.scatter(gdf['densidade_pop'], gdf['indice_vulnerabilidade'], 
                alpha=0.5, c=gdf['indice_prioridade_evacuacao'], cmap='viridis', 
                s=20, edgecolor='none')
    
    plt.colorbar(label='Índice de Prioridade para Evacuação')
    plt.title('Relação entre Densidade Populacional e Vulnerabilidade', fontsize=16)
    plt.xlabel('Densidade Populacional (hab/km²)')
    plt.ylabel('Índice de Vulnerabilidade')
    plt.grid(alpha=0.3)
    
    # Adicionar linha de tendência
    z = np.polyfit(gdf['densidade_pop'], gdf['indice_vulnerabilidade'], 1)
    p = np.poly1d(z)
    plt.plot(sorted(gdf['densidade_pop']), p(sorted(gdf['densidade_pop'])), 
             "r--", alpha=0.8, label=f'Tendência: y={z[0]:.5f}x+{z[1]:.5f}')
    
    plt.legend()
    plt.tight_layout()
        dens_vuln_path = os.path.join(viz_dir, 'densidade_vs_vulnerabilidade.png')
        plt.savefig(dens_vuln_path, dpi=300)
    plt.close()
        viz_paths['densidade_vs_vulnerabilidade'] = dens_vuln_path
    
    # 8. Mapa comparativo de horários de ocupação (manhã vs. noite)
        logger.info("Gerando mapa comparativo de ocupação por horário...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Mapa de população pela manhã
    gdf.plot(column='pop_atual_0800', cmap='YlOrRd', legend=True,
             ax=axes[0], alpha=0.7, edgecolor='white', linewidth=0.2,
                legend_kwds={'label': 'População Estimada - 8h'})
    ctx.add_basemap(axes[0], source=ctx.providers.CartoDB.Positron)
    axes[0].set_title('População Estimada - 8h', fontsize=14)
    
    # Mapa de população à noite
    gdf.plot(column='pop_atual_2300', cmap='YlOrRd', legend=True,
             ax=axes[1], alpha=0.7, edgecolor='white', linewidth=0.2,
                legend_kwds={'label': 'População Estimada - 23h'})
    ctx.add_basemap(axes[1], source=ctx.providers.CartoDB.Positron)
    axes[1].set_title('População Estimada - 23h', fontsize=14)
    
    # Adicionar escala em ambos os mapas
    for ax in axes:
        scale_bar = AnchoredSizeBar(ax.transData, 0.05, '5 km', 'lower right', 
                                  pad=0.5, color='black', frameon=True, size_vertical=0.01)
        ax.add_artist(scale_bar)
    
    plt.tight_layout()
        pop_horario_path = os.path.join(viz_dir, 'populacao_manhã_vs_noite.png')
        plt.savefig(pop_horario_path, dpi=300, bbox_inches='tight')
    plt.close()
        viz_paths['populacao_manhã_vs_noite'] = pop_horario_path
    
    # 9. Dashboard das categorias de vulnerabilidade e prioridade
        logger.info("Gerando dashboard de categorias...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 16))
    
    # Gráfico de barras para categorias de vulnerabilidade
    vuln_counts = gdf['categoria_vulnerabilidade'].value_counts().reindex(['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito alta'])
    vuln_colors = [colors_vulnerabilidade[cat] for cat in vuln_counts.index]
    vuln_counts.plot(kind='barh', ax=axes[0], color=vuln_colors)
    axes[0].set_title('Distribuição das Categorias de Vulnerabilidade', fontsize=14)
    axes[0].set_xlabel('Número de Setores')
    axes[0].grid(axis='x', alpha=0.3)
    
    # Adicionar valores nas barras
    for i, v in enumerate(vuln_counts):
        axes[0].text(v + 0.5, i, str(v), va='center')
    
    # Gráfico de barras para categorias de prioridade
    prior_counts = gdf['categoria_prioridade'].value_counts().reindex(['Muito baixa', 'Baixa', 'Média', 'Alta', 'Muito alta'])
    prior_colors = [colors_prioridade[cat] for cat in prior_counts.index]
    prior_counts.plot(kind='barh', ax=axes[1], color=prior_colors)
    axes[1].set_title('Distribuição das Categorias de Prioridade para Evacuação', fontsize=14)
    axes[1].set_xlabel('Número de Setores')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Adicionar valores nas barras
    for i, v in enumerate(prior_counts):
        axes[1].text(v + 0.5, i, str(v), va='center')
    
    plt.tight_layout()
        dashboard_path = os.path.join(viz_dir, 'dashboard_categorias.png')
        plt.savefig(dashboard_path, dpi=300)
    plt.close()
        viz_paths['dashboard_categorias'] = dashboard_path
        
        # Gerar mapa interativo com camadas de população e altimetria
        logger.info("Gerando mapa interativo com múltiplas camadas...")
        interactive_map_path = os.path.join(viz_dir, 'mapa_interativo_setores_censitarios.html')
        interactive_map = create_interactive_map(gdf, interactive_map_path)
        if interactive_map:
            viz_paths['mapa_interativo'] = interactive_map
        
        logger.info(f"Visualizações geradas e salvas em {viz_dir}")
        return viz_paths
        
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações: {str(e)}")
        logger.error(traceback.format_exc())
        return viz_paths  # Retorna o que conseguiu gerar até o momento do erro

def explore_gpkg(gpkg_path):
    """
    Explora as camadas disponíveis em um arquivo GeoPackage.

    Args:
        gpkg_path: Caminho para o arquivo GPKG

    Returns:
        Um DataFrame com informações sobre as camadas
    """
    if not os.path.exists(gpkg_path):
        logger.error(f"Arquivo não encontrado: {gpkg_path}")
        return None
    
    try:
        # Listar todas as camadas disponíveis no GPKG
        layers = fiona.listlayers(gpkg_path)
        logger.info(f"Encontradas {len(layers)} camadas no arquivo {os.path.basename(gpkg_path)}")
        
        # Informações a coletar sobre cada camada
        layer_info = []
        
        for layer in layers:
            try:
                # Carregar a camada como GeoDataFrame
                gdf = gpd.read_file(gpkg_path, layer=layer)
                
                # Extrair informações básicas
                info = {
                    "layer_name": layer,
                    "feature_count": len(gdf),
                    "geometry_types": ", ".join(gdf.geometry.geom_type.unique()),
                    "columns": ", ".join([col for col in gdf.columns if col != 'geometry']),
                    "crs": str(gdf.crs),
                    "bounds": str(gdf.total_bounds),
                    "memory_usage_MB": round(gdf.memory_usage(deep=True).sum() / (1024 * 1024), 2)
                }
                
                # Adicionar estatísticas para algumas colunas numéricas (se disponíveis)
                numeric_stats = {}
                for col in gdf.select_dtypes(include=['number']).columns:
                    if col != 'geometry':
                        try:
                            numeric_stats[f"{col}_min"] = gdf[col].min()
                            numeric_stats[f"{col}_max"] = gdf[col].max()
                            numeric_stats[f"{col}_mean"] = gdf[col].mean()
                        except:
                            pass
                
                # Combinar as informações
                info.update(numeric_stats)
                layer_info.append(info)
                
                logger.info(f"Camada {layer}: {len(gdf)} feições, tipos: {info['geometry_types']}")
                
            except Exception as e:
                logger.warning(f"Erro ao processar camada {layer}: {str(e)}")
                layer_info.append({
                    "layer_name": layer,
                    "error": str(e)
                })
        
        # Criar DataFrame a partir das informações coletadas
        if layer_info:
            result_df = pd.DataFrame(layer_info)
            return result_df
        else:
            logger.warning("Nenhuma informação de camada foi encontrada ou processada")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Erro ao explorar o arquivo GPKG: {str(e)}")
        return None

def main():
    """
    Função principal que executa o fluxo de trabalho completo de enriquecimento de dados dos setores censitários.
    """
    logger.info("=== Iniciando processamento de enriquecimento de dados dos setores censitários ===")
    start_time = time.time()
    
    # Dicionário para armazenar caminhos de visualizações
    viz_paths = {}
    
    try:
        # 1. Carregar dados processados
        logger.info("Carregando dados dos setores censitários...")
    original_gdf = load_data()
        if original_gdf is None:
            logger.error("Não foi possível carregar os dados dos setores censitários")
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
            logger.error(traceback.format_exc())
        
        # 3. Enriquecer dados
        logger.info("Iniciando processo de enriquecimento de dados...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
    # Aplicar etapas de enriquecimento
            logger.info("Calculando métricas de área...")
    enriched_gdf = calculate_area_metrics(original_gdf)
    
            logger.info("Extraindo dados de elevação...")
            if dem is not None:
    enriched_gdf = extract_elevation_data(enriched_gdf, dem)
            else:
                logger.warning("Pulando extração de dados de elevação pois o DEM não está disponível")
    
            logger.info("Calculando distribuição populacional...")
    enriched_gdf = calculate_population_distribution(enriched_gdf)
    
            logger.info("Calculando vulnerabilidade populacional...")
    enriched_gdf = calculate_population_vulnerability(enriched_gdf)
    
            logger.info("Gerando índice de prioridade para evacuação...")
    enriched_gdf = generate_evacuation_priority(enriched_gdf)
    
            logger.info("Enriquecimento de dados concluído com sucesso")
        except Exception as e:
            logger.error(f"Erro no processo de enriquecimento: {str(e)}")
            logger.error(traceback.format_exc())
            return None
        
        # 4. Salvar dados enriquecidos
        logger.info("Salvando dados enriquecidos...")
        
        # Criar caminho para o arquivo de saída
        output_file = None  # Inicializar a variável
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gpkg_path = os.path.join(OUTPUT_DIR, f"setores_censitarios_enriched_{timestamp}.gpkg")
        
        # Tentar salvar como GPKG
        output_file = save_enriched_data(enriched_gdf, gpkg_path)
        
        if output_file:
            logger.info(f"Dados enriquecidos salvos com sucesso em: {output_file}")
            
            # Gerar visualizações
            logger.info("Gerando visualizações...")
            
            try:
                viz_paths = generate_visualizations(enriched_gdf, timestamp)
                
                # Mostrar estatísticas sobre visualizações
                successful_viz = sum(1 for path in viz_paths.values() if os.path.exists(path))
                logger.info(f"Visualizações geradas: {successful_viz}/{len(viz_paths)} concluídas com sucesso")
                logger.info(f"Diretório de visualizações: {VISUALIZATION_DIR}/{timestamp}")
            except Exception as e:
                logger.error(f"Erro ao gerar visualizações: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Gerar relatório de qualidade
            logger.info("Gerando relatório de qualidade...")
            report_file = None
            
            try:
                report_file = generate_quality_report(original_gdf, enriched_gdf, output_file, viz_paths)
                if report_file:
                    logger.info(f"Relatório de qualidade gerado em: {report_file}")
                else:
                    logger.warning("Não foi possível gerar o relatório de qualidade")
            except Exception as e:
                logger.error(f"Erro ao gerar relatório: {str(e)}")
                logger.error(traceback.format_exc())
            
            # Calcular tempo de execução
            elapsed_time = time.time() - start_time
            logger.info(f"=== Processamento concluído em {elapsed_time:.2f} segundos ===")
            logger.info(f"Dados enriquecidos: {output_file}")
            
            if report_file:
                logger.info(f"Relatório: {report_file}")
            logger.info(f"Visualizações: {VISUALIZATION_DIR}/{timestamp}")
            
            return {
                "enriched_data": enriched_gdf,
                "output_file": output_file,
                "report_file": report_file,
                "visualization_paths": viz_paths
            }
        else:
            logger.error("Falha ao salvar os dados enriquecidos")
    except Exception as e:
        logger.error(f"Erro no processamento: {str(e)}")
        logger.error(traceback.format_exc())
        elapsed_time = time.time() - start_time
        logger.info(f"=== Processamento interrompido após {elapsed_time:.2f} segundos ===")
        return None

# Adicionar args ao script para controlar o comportamento da linha de comando
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Processar e enriquecer dados de setores censitários')
    parser.add_argument('--skip-visualization', action='store_true', help='Pular geração de visualizações (para processamento em lote)')
    
    args = parser.parse_args()
    
    # Executar main
    main()