# -*- coding: utf-8 -*-
"""
Preprocessing Functions for Road Network Data

Este módulo contém funções para preparação de dados de redes viárias para análise,
incluindo limpeza, explosão de MultilineStrings e outros passos de pré-processamento
conforme descrito no documento de instruções.
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
import logging
import time
import json
import random
import sys
from datetime import datetime
from shapely.geometry import LineString, Point
import networkx as nx

# Tentar importar fiona para manipulação de camadas
try:
    import fiona
    HAS_FIONA = True
except ImportError:
    HAS_FIONA = False
    print("Fiona não está disponível diretamente. Algumas funções podem ser limitadas.")

# Comentando a importação de PyTorch para não quebrar em ambientes sem GPU
# No Colab, essas importações serão gerenciadas separadamente
try:
import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch não está disponível. Algumas funcionalidades podem ser limitadas.")

# Configuração para detecção de ambiente e montagem do Google Drive
try:
    import google.colab
    from google.colab import drive
    IN_COLAB = True
    print("Ambiente Google Colab detectado")

    # Montar o Google Drive conforme especificado pelo usuário
    # IMPORTANTE: Certifique-se de completar o processo de autorização na janela pop-up.
    # A KeyboardInterrupt aqui geralmente significa que a autorização não foi concluída.
    drive.mount('/content/drive')
    print("Google Drive montado com sucesso em /content/drive")

    # Caminhos específicos fornecidos pelo usuário
    BASE_DIR = '/content/drive/MyDrive/geoprocessamento_gnn'
    OUTPUT_DIR = os.path.join(BASE_DIR, 'OUTPUT')
    QUALITY_REPORT_DIR = os.path.join(BASE_DIR, 'QUALITY_REPORT')
    VISUALIZACOES_DIR = os.path.join(BASE_DIR, 'VISUALIZACOES')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    ROADS_ENRICHED_PATH = os.path.join(DATA_DIR, 'roads_enriched_20250412_230707.gpkg')
    PROCESSED_DATA_DIR = DATA_DIR  # Usar o mesmo diretório de dados

    # Verificação de instalação de dependências específicas
    print("\nInstalação recomendada de dependências para o Colab:")
    print("!pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118")
    print("!pip install torch-geometric==2.3.1")
    print("!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html")
    print("!pip install geopandas==0.13.2 networkx==3.1 matplotlib==3.7.2 seaborn==0.12.2")
    print("!pip install contextily==1.3.0 folium==0.14.0 rtree==1.0.1")
    print("!pip install tqdm==4.66.1 plotly==5.15.0 scikit-learn==1.3.0 jsonschema==4.17.3")
    print("!pip install osmnx==1.5.1 momepy==0.6.0")
    print("!pip install fiona==1.9.5 numpy==1.24.3 --force-reinstall")
    print("\nReinicie o runtime do Colab após estas instalações para evitar conflitos.")

except ImportError:
    IN_COLAB = False
    print("Ambiente local detectado")
    # Configuração para ambiente local
    BASE_DIR = 'F:/TESE_MESTRADO'
    GEOPROCESSING_DIR = os.path.join(BASE_DIR, 'geoprocessing')
DATA_DIR = os.path.join(GEOPROCESSING_DIR, 'data')
ENRICHED_DATA_DIR = os.path.join(DATA_DIR, 'enriched_data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUT_DIR = os.path.join(GEOPROCESSING_DIR, 'outputs')
QUALITY_REPORT_DIR = os.path.join(OUTPUT_DIR, 'quality_reports')
ROADS_PROCESSED_PATH = os.path.join(PROCESSED_DATA_DIR, 'roads_processed.gpkg')
    # Usar arquivo mais recente no ambiente local
    if os.path.exists(ENRICHED_DATA_DIR):
road_files = [f for f in os.listdir(ENRICHED_DATA_DIR) if f.startswith('roads_enriched_') and f.endswith('.gpkg')]
if road_files:
            road_files.sort(reverse=True)
    ROADS_ENRICHED_PATH = os.path.join(ENRICHED_DATA_DIR, road_files[0])
else:
            ROADS_ENRICHED_PATH = os.path.join(ENRICHED_DATA_DIR, 'roads_enriched.gpkg') # Default path if no files found
            print(f"Warning: No road files found in {ENRICHED_DATA_DIR}. Using default path: {ROADS_ENRICHED_PATH}")
    else:
         ROADS_ENRICHED_PATH = os.path.join(DATA_DIR, 'roads_enriched.gpkg') # Fallback if enriched dir doesn't exist
         print(f"Warning: Directory {ENRICHED_DATA_DIR} not found. Using path: {ROADS_ENRICHED_PATH}")


# Configuração de timestamp para arquivos de saída como especificado no documento
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Garantir que os diretórios existam para a estrutura do projeto
# Adjusted to handle potential non-existence in local setup before path definition
dirs_to_create = [OUTPUT_DIR, QUALITY_REPORT_DIR]
if IN_COLAB:
    dirs_to_create.extend([BASE_DIR, DATA_DIR, VISUALIZACOES_DIR])
else:
    dirs_to_create.extend([GEOPROCESSING_DIR, DATA_DIR, ENRICHED_DATA_DIR, PROCESSED_DATA_DIR])

for directory in dirs_to_create:
    try:
    os.makedirs(directory, exist_ok=True)
        print(f"Diretório verificado/criado: {directory}")
    except OSError as e:
        print(f"Erro ao criar diretório {directory}: {e}")
        # Decide how to handle this - maybe exit or log a critical error
        # For now, just print and continue, but paths might fail later
        pass


# Configuração de logging conforme especificado no documento
log_file_path = os.path.join(OUTPUT_DIR, f"pipeline_gnn_road_{timestamp}.log")
try:
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
            logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
except Exception as e:
    print(f"Erro ao configurar logging para {log_file_path}: {e}. Logging apenas para console.")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )


logger = logging.getLogger("TESE_MESTRADO.road_network_gnn")
logger.info("Inicializando pipeline GNN para análise de redes viárias")

# Configurar sementes para reprodutibilidade conforme especificado no documento
seed = 42
np.random.seed(seed)
random.seed(seed)

# Configurar sementes PyTorch se disponível
if HAS_TORCH:
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)
    logger.info("Sementes PyTorch configuradas para reprodutibilidade")

# Parâmetros específicos do modelo GNN conforme documento
model_params = {
    'input_dim': 2,           # Dimensão de entrada do modelo (Ex: Coordenadas X, Y normalizadas)
    'hidden_dim': 64,         # Dimensão oculta usada nos experimentos
    'output_dim': 6,          # 6 classes de vias: residential, secondary, tertiary, primary, trunk, motorway
    'dropout': 0.3,           # Taxa de dropout específica do modelo
    'learning_rate': 0.01,    # Taxa de aprendizado utilizada
    'weight_decay': 0.0005,   # Regularização L2
    'early_stopping_patience': 20 # Parâmetro usado nos experimentos
}
logger.info(f"Parâmetros do modelo GNN configurados: {model_params}")

def load_road_data(file_path=None, crs="EPSG:31983"):
    """
    Carrega dados de rede viária do caminho exato especificado pelo usuário.
    
    Args:
        file_path (str, optional): Caminho para o arquivo de estradas enriquecido.
                                  Se None, usa o caminho exato fornecido pelo usuário.
        crs (str): Sistema de coordenadas (EPSG:31983 - UTM Zone 23S, CRS padrão específico do projeto)

    Returns:
        geopandas.GeoDataFrame: GeoDataFrame com os dados carregados e reprojetados.

    Raises:
        FileNotFoundError: Se o arquivo especificado não for encontrado.
        Exception: Para outros erros durante o carregamento ou processamento.
    """
    if file_path is None:
        # Usar o caminho exato fornecido pelo usuário
        file_path = '/content/drive/MyDrive/geoprocessamento_gnn/data/roads_enriched_20250412_230707.gpkg'

    logger.info(f"Carregando dados de estradas de: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"ERRO: Arquivo não encontrado: {file_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    # Carregar o arquivo GPKG
    try:
        layers = None
        fiona_available = False
        
        # Verificar se a função check_fiona_available está disponível (definida no __main__)
        if 'check_fiona_available' in globals():
            fiona_available = check_fiona_available()
        elif HAS_FIONA:
            fiona_available = True

        if fiona_available:
            try:
                layers = fiona.listlayers(file_path)
        logger.info(f"Camadas encontradas no arquivo: {layers}")
            except Exception as e:
                logger.warning(f"Erro ao listar camadas com fiona: {str(e)}. Tentando ler sem especificar camada.")
                layers = None

        # Se conseguiu listar camadas, escolher a adequada
        if layers:
            # Priorize camadas contendo 'road' ou 'via' (case-insensitive)
            preferred_layer = next((l for l in layers if 'road' in l.lower() or 'via' in l.lower()), None)
            if preferred_layer:
                layer_name = preferred_layer
            else:
                layer_name = layers[0]  # Fallback to the first layer
            logger.info(f"Usando camada: {layer_name}")
        gdf = gpd.read_file(file_path, layer=layer_name)
        else:
            # Se não conseguiu listar camadas ou fiona não está disponível, tentar ler diretamente
            logger.warning("Não foi possível listar camadas ou Fiona indisponível. Tentando ler o arquivo diretamente.")
            gdf = gpd.read_file(file_path)
            logger.info("Arquivo carregado com sucesso usando leitura direta (sem especificar camada).")
        
    except Exception as e:
        logger.error(f"Erro fatal ao carregar arquivo GPKG: {str(e)}")
        raise
    
    if gdf is None:
        logger.error("Falha ao carregar GeoDataFrame.")
        raise ValueError("Não foi possível carregar o GeoDataFrame do arquivo especificado.")
    
    # Verificar sistema de coordenadas
    if gdf.crs is None:
        logger.warning(f"CRS não definido no arquivo. Definindo como {crs}")
        gdf.crs = crs
    elif gdf.crs.to_string() != crs:
        logger.info(f"Reprojetando de {gdf.crs.to_string()} para {crs}")
        gdf = gdf.to_crs(crs)
    
    # Verificar colunas essenciais
    essential_cols = ['geometry', 'highway']
    missing_cols = [col for col in essential_cols if col not in gdf.columns]
    if missing_cols:
        logger.error(f"Colunas essenciais ausentes: {missing_cols}. O processamento pode falhar.")
    
    # Analisar tipos de estradas
    if 'highway' in gdf.columns:
        gdf['highway'] = gdf['highway'].fillna('unknown')
        highway_counts = gdf['highway'].value_counts()
        logger.info(f"Distribuição inicial de tipos de vias ('highway'):\n{highway_counts.head(10)}")
    else:
        logger.warning("Coluna 'highway' não encontrada. A classificação de vias será limitada.")
    
    # Criar índice espacial para acesso eficiente
    if not gdf.has_sindex:
        logger.info("Criando índice espacial (sindex)...")
        try:
            gdf.sindex  # Acessar sindex para criá-lo
            logger.info("Índice espacial criado com sucesso.")
        except Exception as e:
            logger.error(f"Erro ao criar índice espacial: {e}")

    # Informações básicas sobre os dados
    logger.info(f"Dados carregados: {len(gdf)} feições (segmentos)")
    if len(gdf) > 0 and 'geometry' in gdf.columns and gdf['geometry'].notna().all():
        try:
            total_length_km = gdf.geometry.length.sum() / 1000
            logger.info(f"Extensão total da rede: {total_length_km:.2f} km")
        except Exception as e:
            logger.warning(f"Não foi possível calcular o comprimento total: {e}")
    
    return gdf


def load_contextual_data():
    """
    Carrega dados contextuais específicos com os caminhos exatos fornecidos pelo usuário.
    
    ATENÇÃO: Usa exatamente os caminhos definidos pelo usuário sem tentativas alternativas.
    
    Returns:
        dict: Dicionário contendo GeoDataFrames para 'setores', 'landuse', 'buildings'
    """
    logger = logging.getLogger(__name__)
    logger.info("Carregando dados contextuais usando caminhos específicos fornecidos...")
    
    result = {}
    
    # Caminhos exatos fornecidos pelo usuário
    exact_paths = {
        'setores': '/content/drive/MyDrive/geoprocessamento_gnn/data/setores_censitarios_enriched_20250413_175729.gpkg',
        'landuse': '/content/drive/MyDrive/geoprocessamento_gnn/data/landuse_enriched_20250413_105344.gpkg',
        'buildings': '/content/drive/MyDrive/geoprocessamento_gnn/data/buildings_enriched_20250413_131208.gpkg'
    }
    
    # Tentar carregar cada arquivo pelo caminho exato
    for key, file_path in exact_paths.items():
        if os.path.exists(file_path):
            try:
                logger.info(f"Carregando {key} de: {file_path}")
                result[key] = gpd.read_file(file_path)
                logger.info(f"Carregados {len(result[key])} elementos de {key}")
            except Exception as e:
                logger.error(f"Erro ao carregar {key} de {file_path}: {str(e)}")
        else:
            logger.warning(f"Arquivo de {key} não encontrado em: {file_path}")
    
    # Resumo do carregamento
    if result:
        logger.info(f"Carregamento de dados contextuais concluído. {len(result)} conjuntos de dados disponíveis: {list(result.keys())}")
    else:
        logger.warning("Nenhum dado contextual foi carregado. Verifique os caminhos e arquivos.")
    
    return result


def explode_multilines_improved(gdf):
    """
    Versão otimizada para processamento de MultiLineStrings que preserva
    atributos de forma mais eficiente e usa o método explode nativo do GeoPandas.
    
    Args:
        gdf: GeoDataFrame contendo dados de estradas
        
    Returns:
        GeoDataFrame com LineStrings individuais
    """
    if gdf.empty:
        logger.warning("Tentando explodir MultiLineStrings em um GeoDataFrame vazio.")
        return gdf

    # Verificar tipos de geometria
    geometry_types = gdf.geometry.type.unique()
    logger.info(f"Tipos de geometria encontrados antes da explosão: {geometry_types}")
    
    # Verificar se há MultiLineStrings
    if 'MultiLineString' not in geometry_types:
        logger.info("Não foram encontradas MultiLineStrings. Nenhuma explosão necessária.")
        return gdf
    
    # Filtrar geometrias
    multi_mask = gdf.geometry.type == "MultiLineString"
    multi_count = multi_mask.sum()

    if multi_count == 0:
         logger.info("Nenhuma MultiLineString encontrada após verificação inicial.") # Should not happen if check above passed
         return gdf

    logger.info(f"Processando {multi_count} MultiLineStrings usando método explode")

    # Separar MultiLineStrings e outras geometrias (preservando LineStrings)
    multilines = gdf[multi_mask]
    other_geoms = gdf[~multi_mask] # Includes LineStrings and potentially others if present

    # Usar método explode nativo do GeoPandas (mais eficiente)
    try:
        # index_parts=True preserves the original index for tracking, if needed
        # reset_index is often useful after explode
        exploded = multilines.explode(index_parts=False).reset_index(drop=True)
        logger.info("Utilizando explode.")
    except Exception as e:
        logger.error(f"Erro durante o método explode: {e}")
        logger.warning("Retornando GeoDataFrame original devido a erro na explosão.")
        return gdf # Return original GDF if explode fails

    # Verificar se a explosão resultou apenas em LineStrings
    exploded_types = exploded.geometry.type.unique()
    if not all(t == 'LineString' for t in exploded_types):
         logger.warning(f"Explosão resultou em tipos inesperados: {exploded_types}. Filtrando para manter apenas LineStrings.")
         exploded = exploded[exploded.geometry.type == 'LineString']


    # Concatenar com as geometrias não-MultiLineString originais
    result = pd.concat([other_geoms, exploded], ignore_index=True)

    # Verificar número de geometrias resultantes
    logger.info(f"Explosão concluída: {multi_count} MultiLineStrings processadas.")
    logger.info(f"Total de feições após explosão: {len(result)}")
    final_types = result.geometry.type.unique()
    logger.info(f"Tipos de geometria finais: {final_types}")


        return result


def calculate_sinuosity(gdf):
    """
    Calcula sinuosidade para cada segmento viário LineString.

    A sinuosidade é a razão entre o comprimento real da linha e a distância euclidiana
    entre seus pontos extremos. Um valor de 1.0 indica uma linha reta perfeita.

    Args:
        gdf: GeoDataFrame contendo dados de estradas

    Returns:
        pandas.Series: Series com valores de sinuosidade para cada feature LineString.
                       Retorna NaN para geometrias não-LineString ou inválidas.
    """
    logger.info("Calculando valores de sinuosidade para segmentos viários")
    sinuosity_values = []

    if gdf.empty:
        logger.warning("Tentando calcular sinuosidade em um GeoDataFrame vazio.")
        return pd.Series(dtype=float)


    for index, row in gdf.iterrows():
        geom = row.geometry
        sin_value = np.nan # Default to NaN
            
        if geom is not None and geom.geom_type == 'LineString' and len(geom.coords) >= 2:
        # Comprimento real da linha
        line_length = geom.length
        
        # Distância euclidiana entre extremidades
            start_point = Point(geom.coords[0])
            end_point = Point(geom.coords[-1])
            straight_distance = start_point.distance(end_point)

            # Evitar divisão por zero e lidar com segmentos muito curtos
            if straight_distance < 1e-6:  # Limiar pequeno para evitar divisão por zero
                # Se o comprimento da linha também for muito pequeno, é um ponto, sinuosidade indefinida (ou 1?)
                # Se o comprimento for maior, mas a distância reta for zero (loop), sinuosidade é infinita.
                # Vamos definir como 1.0 para pontos ou linhas muito curtas e retas.
                 sin_value = 1.0
            elif line_length < 1e-6: # Linha com comprimento quase zero
            sin_value = 1.0
        else:
            sin_value = line_length / straight_distance
                # Cap sinuosity at a reasonable upper bound if needed, e.g., 10 or 20,
                # as extremely high values might indicate data issues.
                # sin_value = min(sin_value, 20.0)

        elif geom is not None:
             logger.debug(f"Geometria no índice {index} não é LineString ou tem < 2 pontos ({geom.geom_type}). Sinuosidade será NaN.")
        else:
             logger.debug(f"Geometria nula no índice {index}. Sinuosidade será NaN.")


        sinuosity_values.append(sin_value)

    sinuosity_series = pd.Series(sinuosity_values, index=gdf.index)

    # Estatísticas sobre a sinuosidade calculada (ignorando NaNs)
    valid_sinuosity = sinuosity_series.dropna()
    if not valid_sinuosity.empty:
        logger.info(f"Sinuosidade calculada para {len(valid_sinuosity)} segmentos.")
        logger.info(f"Estatísticas de Sinuosidade: Média={valid_sinuosity.mean():.3f}, "
                    f"Min={valid_sinuosity.min():.3f}, Max={valid_sinuosity.max():.3f}, "
                    f"Mediana={valid_sinuosity.median():.3f}")
    else:
        logger.warning("Não foi possível calcular sinuosidade para nenhum segmento válido.")

    return sinuosity_series


def clean_road_data(gdf):
    """
    Limpa e valida os dados de estradas específicos do projeto.

    Operações:
    1. Remove geometrias nulas ou vazias.
    2. Corrige geometrias inválidas usando buffer(0). Remove as que não puderam ser corrigidas.
    3. Remove duplicatas geométricas exatas.
    4. Padroniza valores da coluna 'highway' e cria 'road_category'.
    5. Calcula atributos geométricos básicos ('length_m', 'sinuosity').
    6. Garante um ID único ('edge_id').
    
    Args:
        gdf: GeoDataFrame contendo dados de estradas
        
    Returns:
        GeoDataFrame limpo e validado
    """
    if gdf.empty:
        logger.warning("Tentando limpar um GeoDataFrame vazio.")
        return gdf

    initial_count = len(gdf)
    logger.info(f"Iniciando limpeza com {initial_count} feições")

    # --- 1. Remover geometrias nulas ou vazias ---
    original_len = len(gdf)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    removed_null = original_len - len(gdf)
    if removed_null > 0:
        logger.warning(f"Removidas {removed_null} geometrias nulas ou vazias.")

    if gdf.empty:
        logger.warning("GeoDataFrame vazio após remover geometrias nulas/vazias.")
        return gdf

    # --- 2. Corrigir geometrias inválidas ---
    invalid_mask = ~gdf.geometry.is_valid
    invalid_count = invalid_mask.sum()
    if invalid_count > 0:
        logger.warning(f"Encontradas {invalid_count} geometrias inválidas. Tentando corrigir com buffer(0)...")
        # Apply buffer(0) only to invalid geometries
        gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)
        
        # Verificar novamente após correção
        still_invalid_mask = ~gdf.geometry.is_valid
        still_invalid_count = still_invalid_mask.sum()
        if still_invalid_count > 0:
            logger.warning(f"Removendo {still_invalid_count} geometrias que permaneceram inválidas após correção.")
            gdf = gdf[~still_invalid_mask].copy()

    if gdf.empty:
        logger.warning("GeoDataFrame vazio após remover geometrias inválidas.")
        return gdf

    # --- 3. Remover duplicatas geométricas ---
    # Note: drop_duplicates based on geometry can be slow. Consider alternatives if performance is critical.
    # Using WKT (Well-Known Text) representation for comparison might be faster sometimes.
    logger.info("Verificando duplicatas geométricas (pode levar tempo)...")
    before_dedup = len(gdf)
    try:
        # Use WKT for potentially faster duplicate check
        gdf['wkt'] = gdf.geometry.to_wkt()
        gdf = gdf.drop_duplicates(subset=['wkt']).drop(columns=['wkt']).copy()
    except Exception as e:
         logger.warning(f"Erro ao usar WKT para drop_duplicates ({e}). Tentando com objeto geometry...")
         gdf = gdf.drop_duplicates(subset=['geometry']).copy() # Fallback

    after_dedup = len(gdf)
    if before_dedup > after_dedup:
        logger.info(f"Removidas {before_dedup - after_dedup} geometrias duplicadas.")

    if gdf.empty:
        logger.warning("GeoDataFrame vazio após remover duplicatas.")
        return gdf

    # --- 4. Padronizar valores de 'highway' ---
    if 'highway' in gdf.columns:
        logger.info("Padronizando valores do campo 'highway' e criando 'road_category'")
        
        # Converter para string, minúsculas e remover espaços extras
        gdf['highway'] = gdf['highway'].astype(str).str.lower().str.strip()
        
        # Tratar valores nulos/vazios/específicos que se tornaram strings
        null_equivalents = ['nan', 'none', '', '<na>', 'null']
        gdf['highway'] = gdf['highway'].replace(null_equivalents, 'unclassified')
        gdf.loc[gdf['highway'].isna(), 'highway'] = 'unclassified' # Catch any remaining NAs

        # Mapeamento para as 6 categorias principais + 'unclassified'
        # Ajuste o mapeamento conforme a necessidade do seu projeto e os valores encontrados
        highway_mapping = {
            'motorway': 'motorway', 'motorway_link': 'motorway',
            'trunk': 'trunk', 'trunk_link': 'trunk',
            'primary': 'primary', 'primary_link': 'primary',
            'secondary': 'secondary', 'secondary_link': 'secondary',
            'tertiary': 'tertiary', 'tertiary_link': 'tertiary',
            'residential': 'residential',
            'living_street': 'residential',
            'service': 'residential', # Agrupar service com residential (ou criar categoria 'service')
            'unclassified': 'residential', # Mapear unclassified para a categoria mais comum ou uma específica
            'road': 'residential', # Mapear 'road' genérico
            # Adicione outros mapeamentos conforme necessário (e.g., track, path, cycleway -> other?)
            'track': 'other',
            'path': 'other',
            'footway': 'other',
            'cycleway': 'other',
            'steps': 'other',
            'pedestrian': 'other',
            'construction': 'other', # Ignorar vias em construção ou mapear para 'other'
        }

        # Aplicar mapeamento, usando 'other' como padrão para não mapeados
        gdf['road_category'] = gdf['highway'].map(lambda x: highway_mapping.get(x, 'other'))
        
        # Verificar distribuição após padronização
        category_counts = gdf['road_category'].value_counts()
        logger.info(f"Distribuição de 'road_category' após padronização:\n{category_counts}")

        # Calcular percentagem para categoria predominante
        if not category_counts.empty:
            predominant_category = category_counts.idxmax()
            predominant_percentage = (category_counts.max() / len(gdf)) * 100
            logger.info(f"Categoria predominante ('road_category'): '{predominant_category}' ({predominant_percentage:.2f}%)")
        else:
            logger.warning("Não foi possível determinar a categoria predominante.")

    else:
        logger.warning("Coluna 'highway' não encontrada. 'road_category' não será criada.")

    # --- 5. Calcular atributos geométricos ---
    logger.info("Calculando atributos geométricos ('length_m', 'sinuosity')")

    # Comprimento (essencial para o projeto)
    try:
    gdf['length_m'] = gdf.geometry.length
    except Exception as e:
        logger.error(f"Erro ao calcular comprimento: {e}. Coluna 'length_m' pode conter NaNs.")
        gdf['length_m'] = np.nan
    
    # Sinuosidade (importante para análise)
    try:
    gdf['sinuosity'] = calculate_sinuosity(gdf)
    except Exception as e:
        logger.error(f"Erro ao calcular sinuosidade: {e}. Coluna 'sinuosity' pode conter NaNs.")
        gdf['sinuosity'] = np.nan


    # --- 6. Garantir ID único para arestas ---
    # Usar o índice resetado como ID único para as arestas (segmentos)
    gdf = gdf.reset_index(drop=True)
    gdf['edge_id'] = gdf.index
    logger.info("Coluna 'edge_id' criada usando o índice resetado.")

    
    # Registrar estatísticas finais
    final_count = len(gdf)
    retention = (final_count / initial_count * 100) if initial_count > 0 else 0
    logger.info(f"Limpeza concluída. Feições finais: {final_count} ({retention:.1f}% das {initial_count} iniciais)")

    # Resumo das estatísticas básicas
    if 'length_m' in gdf.columns and gdf['length_m'].notna().any():
        valid_lengths = gdf['length_m'].dropna()
        logger.info(f"Comprimento total da rede limpa: {valid_lengths.sum()/1000:.2f} km")
        logger.info(f"Comprimento médio dos segmentos: {valid_lengths.mean():.2f} m")
        logger.info(f"Comprimento (min, max): ({valid_lengths.min():.2f} m, {valid_lengths.max():.2f} m)")
    else:
        logger.warning("Não foi possível calcular estatísticas de comprimento.")

    
    return gdf


def check_connectivity(gdf):
    """
    Verifica a conectividade básica da rede viária construindo um grafo simples.
    Conecta segmentos apenas se seus endpoints exatos coincidirem.

    Args:
        gdf: GeoDataFrame contendo dados de estradas limpos (com LineStrings)

    Returns:
        dict: Dicionário com métricas básicas de conectividade do grafo.
    """
    logger.info("Analisando conectividade básica da rede viária (endpoints exatos)")

    if gdf.empty or 'geometry' not in gdf.columns:
        logger.warning("GeoDataFrame vazio ou sem coluna 'geometry'. Não é possível analisar conectividade.")
        return {
            'is_connected': False, 'num_components': 0, 'num_nodes': 0, 'num_edges': 0,
            'largest_component_size': 0, 'largest_component_percentage': 0,
            'avg_degree': 0, 'density': 0
        }
    
    # Construir grafo básico para análise de conectividade
    G = nx.Graph()
    node_map = {} # Mapeia coordenadas (tupla) para ID de nó inteiro
    next_node_id = 0
    edges_added = 0

    # Adicionar arestas baseadas nos segmentos
    for index, row in gdf.iterrows():
        geom = row.geometry
        if geom is not None and geom.geom_type == 'LineString' and len(geom.coords) >= 2:
            start_coord = tuple(geom.coords[0])
            end_coord = tuple(geom.coords[-1])

            # Mapear coordenadas para IDs de nós
            if start_coord not in node_map:
                node_map[start_coord] = next_node_id
                next_node_id += 1
            if end_coord not in node_map:
                node_map[end_coord] = next_node_id
                next_node_id += 1

            start_node_id = node_map[start_coord]
            end_node_id = node_map[end_coord]

            # Adicionar aresta se os nós forem diferentes
            if start_node_id != end_node_id:
                 # Usar 'edge_id' se existir, senão o índice original
                segment_id = row['edge_id'] if 'edge_id' in row else index
                length = row['length_m'] if 'length_m' in row else geom.length
                G.add_edge(start_node_id, end_node_id, segment_id=segment_id, length=length)
                edges_added += 1
            else:
                 logger.debug(f"Segmento {index} é um loop (início=fim). Ignorando para grafo simples.")

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges() # Should match edges_added if no loops
    logger.info(f"Grafo básico construído com {num_nodes} nós e {num_edges} arestas.")

    if num_nodes == 0:
        logger.warning("Grafo não contém nós. Conectividade não pode ser analisada.")
        return {
            'is_connected': False, 'num_components': 0, 'num_nodes': 0, 'num_edges': 0,
            'largest_component_size': 0, 'largest_component_percentage': 0,
            'avg_degree': 0, 'density': 0
        }
    
    # Contar componentes conectados
    is_connected_flag = nx.is_connected(G)
    num_components = nx.number_connected_components(G)

    largest_cc_size = 0
    largest_cc_percentage = 0.0

    if num_components > 1:
    largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_size = len(largest_cc)
        largest_cc_percentage = largest_cc_size / num_nodes
        logger.warning(f"Rede viária NÃO é totalmente conectada: {num_components} componentes distintos.")
        logger.info(f"Maior componente conectado: {largest_cc_size} nós ({largest_cc_percentage:.1%} do total)")

        all_components = list(nx.connected_components(G))
        avg_component_size = sum(len(c) for c in all_components) / num_components
        logger.info(f"Tamanho médio dos componentes: {avg_component_size:.1f} nós")

        if len(all_components) > 1:
            second_cc = sorted(all_components, key=len, reverse=True)[1]
            second_cc_perc = len(second_cc) / num_nodes
            logger.info(f"Segundo maior componente: {len(second_cc)} nós ({second_cc_perc:.1%} do total)")
    elif num_components == 1:
        logger.info("Rede viária é totalmente conectada (1 componente conectado).")
        largest_cc_size = num_nodes
        largest_cc_percentage = 1.0
    else: # num_components == 0 (only if num_nodes == 0)
         logger.warning("Grafo não possui componentes.")


    # Métricas adicionais do grafo
    avg_degree = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0
    density = nx.density(G)

    logger.info(f"Grau médio dos nós: {avg_degree:.2f}")
    logger.info(f"Densidade do grafo: {density:.6f}")

    # Retornar métricas completas
    return {
        'is_connected': is_connected_flag,
        'num_components': num_components,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'largest_component_size': largest_cc_size,
        'largest_component_percentage': largest_cc_percentage,
        'avg_degree': avg_degree,
        'density': density
    }


def prepare_node_features(gdf):
    """
    Extrai e prepara características de nós (interseções/endpoints) da rede viária.
    Atualmente, extrai apenas coordenadas X, Y e atribui um ID.
    
    Args:
        gdf: GeoDataFrame contendo dados de estradas limpos (com LineStrings)
        
    Returns:
        pandas.DataFrame: DataFrame com características dos nós ('node_id', 'x', 'y').
                          Retorna DataFrame vazio se o input for vazio.
    """
    logger.info("Preparando características dos nós da rede viária (interseções/endpoints)")

    if gdf.empty or 'geometry' not in gdf.columns:
         logger.warning("GeoDataFrame vazio ou sem coluna 'geometry'. Não é possível preparar nós.")
         return pd.DataFrame(columns=['node_id', 'x', 'y'])


    all_points_coords = set() # Use a set for efficient uniqueness check of coordinate tuples

    for index, row in gdf.iterrows():
        geom = row.geometry
        if geom is not None and geom.geom_type == 'LineString' and len(geom.coords) >= 2:
            start_coord = tuple(geom.coords[0]) # Use tuple for hashability
            end_coord = tuple(geom.coords[-1])
            all_points_coords.add(start_coord)
            all_points_coords.add(end_coord)

    if not all_points_coords:
         logger.warning("Nenhum endpoint válido encontrado nos segmentos LineString.")
         return pd.DataFrame(columns=['node_id', 'x', 'y'])

    
    # Criar um DataFrame com os pontos únicos e atribuir IDs
    unique_points_list = list(all_points_coords)
    nodes_df = pd.DataFrame({
        'node_id': range(len(unique_points_list)),
        'x': [p[0] for p in unique_points_list],
        'y': [p[1] for p in unique_points_list]
    })

    # Análise básica dos nós
    num_nodes = len(nodes_df)
    logger.info(f"Extraídos {num_nodes} nós únicos (endpoints) da rede viária.")
    if num_nodes > 0:
        min_x, max_x = nodes_df['x'].min(), nodes_df['x'].max()
        min_y, max_y = nodes_df['y'].min(), nodes_df['y'].max()
        logger.info(f"Extensão espacial dos nós: X[{min_x:.1f}, {max_x:.1f}], Y[{min_y:.1f}, {max_y:.1f}]")

        # Calcular densidade espacial aproximada (nós por km²)
        # Ensure area is positive
        delta_x = max_x - min_x
        delta_y = max_y - min_y
        if delta_x > 0 and delta_y > 0:
             area_approx_m2 = delta_x * delta_y
             area_approx_km2 = area_approx_m2 / 1e6
             density = num_nodes / area_approx_km2 if area_approx_km2 > 0 else 0
             logger.info(f"Densidade espacial aproximada: {density:.2f} nós/km² (baseado na bounding box)")
        else:
             logger.warning("Não foi possível calcular a densidade espacial (área da bounding box não positiva).")

    else:
         logger.warning("Nenhum nó foi extraído.")

    
    return nodes_df


def prepare_edge_features(gdf):
    """
    Extrai e prepara características de arestas (segmentos) a partir dos dados de estradas.

    Extrai atributos como ID, comprimento, sinuosidade, categoria da via,
    e outras colunas potencialmente úteis presentes no GeoDataFrame.
    
    Args:
        gdf: GeoDataFrame contendo dados de estradas limpos e enriquecidos.
             Espera-se que colunas como 'edge_id', 'length_m', 'sinuosity',
             'road_category' já existam (calculadas em `clean_road_data`).
        
    Returns:
        pandas.DataFrame: DataFrame com características das arestas.
                          Retorna DataFrame vazio se o input for vazio.
    """
    logger.info("Preparando características das arestas (segmentos) para o grafo")

    if gdf.empty:
        logger.warning("GeoDataFrame de entrada vazio. Retornando DataFrame de arestas vazio.")
        return pd.DataFrame()

    # Colunas a serem extraídas diretamente (se existirem)
    direct_features = [
        'edge_id', 'length_m', 'sinuosity', 'road_category',
        'highway', # Manter a coluna original também pode ser útil
        'curvature', 'point_density', 'bearing', 'length_category', 'sinuosity_category', # Advanced features
        'predominant_landuse', 'building_count', 'building_density', # Contextual features
        'slope_pct', 'slope_category', # Elevation features
        'osm_id', 'name', 'z_order' # Optional OSM/metadata features
    ]

    # Selecionar apenas as colunas existentes no GDF
    cols_to_extract = [col for col in direct_features if col in gdf.columns]
    logger.info(f"Extraindo as seguintes colunas para características de arestas: {cols_to_extract}")

    # Criar DataFrame de características das arestas
    try:
        edges_df = gdf[cols_to_extract].copy()
    except KeyError as e:
        logger.error(f"Erro ao selecionar colunas para arestas: {e}. Verifique se as colunas esperadas existem.")
        # Fallback: extract only geometry-derived features if others fail
        cols_to_extract = ['edge_id', 'length_m', 'sinuosity']
        cols_to_extract = [col for col in cols_to_extract if col in gdf.columns]
        if not cols_to_extract:
             logger.error("Nenhuma coluna básica ('edge_id', 'length_m', 'sinuosity') encontrada.")
             return pd.DataFrame()
        logger.warning(f"Extraindo apenas colunas básicas: {cols_to_extract}")
        edges_df = gdf[cols_to_extract].copy()


    # --- Adicionar características calculadas adicionais (ex: tempo de viagem) ---
    if 'road_category' in edges_df.columns and 'length_m' in edges_df.columns:
        logger.info("Calculando velocidade estimada e tempo de viagem por segmento.")
        # Velocidades estimadas em km/h (ajuste conforme necessário)
        speed_mapping = {
            'motorway': 100, 'trunk': 80, 'primary': 60, 'secondary': 50,
            'tertiary': 40, 'residential': 30, 'other': 20, 'unclassified': 30
        }
        # Usar 'residential' como default se a categoria não estiver no map
        edges_df['speed_kmh'] = edges_df['road_category'].map(lambda x: speed_mapping.get(x, 30))

        # Tempo estimado de viagem em minutos (distância em km / velocidade em km/min)
        # Avoid division by zero for speed
        speed_kpm = edges_df['speed_kmh'] / 60.0
        length_km = edges_df['length_m'] / 1000.0
        # Handle cases where speed is zero or near zero
        edges_df['travel_time_min'] = np.where(
            speed_kpm > 1e-6,
            length_km / speed_kpm,
            np.inf # Assign infinity or a very large number for zero speed segments
        )
        # Replace potential inf values if needed, e.g., with a large constant or based on length
        max_reasonable_time = 60 # e.g., 1 hour max for a single segment? Adjust as needed.
        edges_df['travel_time_min'] = edges_df['travel_time_min'].replace([np.inf, -np.inf], max_reasonable_time).fillna(max_reasonable_time)

    # Análise básica das características das arestas
    num_edges = len(edges_df)
    logger.info(f"Extraídas características para {num_edges} arestas.")

    if num_edges > 0:
        # Estatísticas para comprimento
        if 'length_m' in edges_df.columns and edges_df['length_m'].notna().any():
            valid_lengths = edges_df['length_m'].dropna()
            logger.info(f"Comprimento total (arestas): {valid_lengths.sum()/1000:.2f} km")
            logger.info(f"Estatísticas de comprimento (m): "
                        f"Min={valid_lengths.min():.1f}, Média={valid_lengths.mean():.1f}, "
                        f"Max={valid_lengths.max():.1f}")

        # Estatísticas para sinuosidade
        if 'sinuosity' in edges_df.columns and edges_df['sinuosity'].notna().any():
             valid_sinuosity = edges_df['sinuosity'].dropna()
             logger.info(f"Estatísticas de sinuosidade: "
                         f"Min={valid_sinuosity.min():.2f}, Média={valid_sinuosity.mean():.2f}, "
                         f"Max={valid_sinuosity.max():.2f}")

        # Estatísticas para categorias de vias
        if 'road_category' in edges_df.columns:
            cat_counts = edges_df['road_category'].value_counts()
            logger.info(f"Distribuição de 'road_category' nas arestas:\n{cat_counts}")
    
    return edges_df


def normalize_features(features_df, id_col_suffix='_id'):
    """
    Normaliza características numéricas de um DataFrame usando Min-Max scaling [0, 1].
    
    Args:
        features_df (pd.DataFrame): DataFrame com características a normalizar.
        id_col_suffix (str): Sufixo para identificar colunas de ID a serem ignoradas.
        
    Returns:
        tuple:
            pd.DataFrame: DataFrame com características normalizadas.
            dict: Dicionário com os parâmetros de normalização (min, max) por coluna.
                  Retorna DataFrame original e dicionário vazio se a entrada for vazia.
    """
    logger.info("Normalizando características numéricas (Min-Max Scaling)")

    if features_df.empty:
        logger.warning("DataFrame de entrada vazio. Nenhuma normalização aplicada.")
        return features_df.copy(), {}
    
    # Criar cópia para não modificar o original
    normalized_df = features_df.copy()
    normalization_factors = {}

    # Identificar colunas numéricas, excluindo IDs
    numeric_cols = normalized_df.select_dtypes(include=np.number).columns
    cols_to_normalize = [
        col for col in numeric_cols
        if not col.endswith(id_col_suffix) and col != id_col_suffix # Handle exact match too
    ]

    if not cols_to_normalize:
        logger.warning("Nenhuma coluna numérica encontrada para normalização (excluindo IDs).")
        return normalized_df, normalization_factors

    logger.info(f"Colunas a serem normalizadas: {', '.join(cols_to_normalize)}")
    
    # Normalizar cada coluna numérica
    for col in cols_to_normalize:
        # Handle potential NaNs before calculating min/max
        col_data = normalized_df[col].dropna()
        if col_data.empty:
             logger.warning(f"Coluna '{col}' contém apenas NaNs. Normalização ignorada, preenchida com 0.")
             normalized_df[col] = normalized_df[col].fillna(0) # Fill original NaNs with 0
             normalization_factors[col] = {'min': 0, 'max': 0} # Store dummy factors
             continue


        col_min = col_data.min()
        col_max = col_data.max()

        # Armazenar fatores de normalização
        normalization_factors[col] = {'min': col_min, 'max': col_max}

        # Evitar divisão por zero se todos os valores forem iguais
        if abs(col_max - col_min) > 1e-9: # Use a small tolerance for float comparison
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            # logger.debug(f"Coluna '{col}' normalizada: min={col_min}, max={col_max}")
        else:
            # Se todos os valores são (quase) iguais, definir normalizado como 0.5
            # Mantém NaNs como NaNs aqui, serão tratados depois se necessário
            normalized_df[col] = 0.5
            logger.warning(f"Coluna '{col}' tem valores (quase) constantes ({col_min}). "
                           f"Valor normalizado definido como 0.5 (NaNs permanecem NaN).")

    logger.info(f"Normalizadas {len(cols_to_normalize)} colunas numéricas.")

    # Verificar os resultados da normalização (opcional, pode ser lento)
    # for col in cols_to_normalize:
    #     # Check non-NaN values
    #     valid_normalized = normalized_df[col].dropna()
    #     if not valid_normalized.empty:
    #          min_norm, max_norm = valid_normalized.min(), valid_normalized.max()
    #          if not (0 <= min_norm <= 1 + 1e-9 and 0 <= max_norm <= 1 + 1e-9): # Allow for float precision issues
    #               logger.warning(f"Normalização pode ter resultado fora de [0,1] para coluna '{col}': "
    #                              f"min={min_norm}, max={max_norm}")

    return normalized_df, normalization_factors


def preprocess_road_data(input_path=None, output_path=None):
    """
    Executa o pipeline completo de pré-processamento para os dados de estradas.

    Coordena carregamento, limpeza, enriquecimento, análise de conectividade,
    e salvamento dos resultados e relatório de qualidade.
    
    Args:
        input_path (str, optional): Caminho para o arquivo de entrada GPKG.
                                    Se None, usa ROADS_ENRICHED_PATH global.
        output_path (str, optional): Caminho para salvar o GPKG pré-processado.
                                     Se None, não salva o arquivo GPKG.
        
    Returns:
        tuple:
            geopandas.GeoDataFrame: GeoDataFrame pré-processado.
            dict: Relatório de qualidade do processamento.
            Retorna (None, {}) se o carregamento inicial falhar.
    """
    start_time_step = time.time()
    logger.info(f"--- Iniciando Pipeline de Pré-processamento ---")
    if input_path is None:
        input_path = ROADS_ENRICHED_PATH
    logger.info(f"Arquivo de entrada: {input_path}")
    if output_path:
        logger.info(f"Arquivo de saída GPKG: {output_path}")
    else:
        logger.info("Arquivo de saída GPKG não será salvo (output_path=None).")


    # --- 1. Carregar dados ---
    try:
        gdf_raw = load_road_data(input_path)
        if gdf_raw.empty:
             logger.error("Arquivo carregado resultou em GeoDataFrame vazio. Abortando.")
             return None, {}
    except (FileNotFoundError, ValueError, Exception) as e:
        logger.error(f"Falha crítica ao carregar dados: {e}. Abortando pré-processamento.")
        return None, {} # Return None if loading fails
    logger.info(f"Carregamento concluído em {time.time() - start_time_step:.2f}s")

    # --- 2. Explodir MultiLineStrings ---
    start_time_step = time.time()
    logger.info("Processando geometrias MultiLineString...")
    gdf_exploded = explode_multilines_improved(gdf_raw)
    logger.info(f"Explosão de MultiLineStrings concluída em {time.time() - start_time_step:.2f}s")

    # --- 3. Limpar e validar dados ---
    start_time_step = time.time()
    logger.info("Executando limpeza e validação de dados...")
    gdf_cleaned = clean_road_data(gdf_exploded)
    if gdf_cleaned.empty:
         logger.error("GeoDataFrame ficou vazio após limpeza. Abortando.")
         return None, {}
    logger.info(f"Limpeza e validação concluídas em {time.time() - start_time_step:.2f}s")

    # --- 4. Análise de conectividade inicial ---
    start_time_step = time.time()
    logger.info("Analisando conectividade inicial da rede (pós-limpeza)...")
    # Usar análise avançada aqui para obter métricas mais detalhadas desde o início
    initial_connectivity_metrics, _ = advanced_connectivity_analysis(gdf_cleaned, tolerance=1.0)
    logger.info(f"Análise de conectividade inicial concluída em {time.time() - start_time_step:.2f}s")


    # --- 5. Correção Topológica (Opcional mas recomendado se desconectado) ---
    gdf_topology = gdf_cleaned # Start with cleaned data
    if initial_connectivity_metrics.get('num_components', 1) > 1:
        start_time_step = time.time()
        logger.info("Aplicando correções topológicas para tentar melhorar conectividade...")
        try:
            # Ajuste a tolerância conforme necessário (e.g., 0.5, 1.0, 2.0 metros)
            gdf_topology = improve_topology(gdf_cleaned, tolerance=1.0)
            logger.info(f"Correção topológica concluída em {time.time() - start_time_step:.2f}s")
        except Exception as e:
             logger.error(f"Erro durante a correção topológica: {e}. Usando dados pré-correção.")
             gdf_topology = gdf_cleaned # Fallback to pre-topology data
    else:
        logger.info("Rede inicial já conectada ou sem componentes > 1. Correção topológica ignorada.")

    # --- 6. Carregar e Integrar Dados Contextuais ---
    start_time_step = time.time()
    gdf_context = gdf_topology # Start with topology-corrected (or cleaned) data
    try:
        logger.info("Carregando e integrando dados contextuais...")
        context_data = load_contextual_data()
        if context_data:
            gdf_context = integrate_contextual_data(gdf_topology, context_data)
            logger.info(f"Integração de dados contextuais concluída em {time.time() - start_time_step:.2f}s")
        else:
            logger.info("Nenhum dado contextual encontrado para integração.")
    except Exception as e:
        logger.warning(f"Erro durante integração de dados contextuais: {str(e)}. Continuando sem eles.")
        gdf_context = gdf_topology # Fallback

    # --- 7. Enriquecer Características das Arestas ---
    start_time_step = time.time()
    logger.info("Enriquecendo características dos segmentos viários (arestas)...")
    try:
        gdf_enriched = enrich_edge_features(gdf_context)
        logger.info(f"Enriquecimento de características concluído em {time.time() - start_time_step:.2f}s")
    except Exception as e:
         logger.error(f"Erro durante enriquecimento de características: {e}. Usando dados pré-enriquecimento.")
         gdf_enriched = gdf_context # Fallback


    # --- 8. Análise de conectividade final ---
    start_time_step = time.time()
    logger.info("Analisando conectividade final da rede (pós-processamento)...")
    final_connectivity_metrics, final_graph = advanced_connectivity_analysis(gdf_enriched, tolerance=1.0)
    logger.info(f"Análise de conectividade final concluída em {time.time() - start_time_step:.2f}s")

    # --- 9. Salvar resultado ---
    if output_path:
        start_time_step = time.time()
        try:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                 os.makedirs(output_dir, exist_ok=True)
                 logger.info(f"Diretório de saída criado: {output_dir}")

            logger.info(f"Salvando dados pré-processados em {output_path}...")
            # Ensure data types are compatible with GPKG driver before saving
            # Convert object columns that might cause issues (like lists or dicts if any were added)
            for col in gdf_enriched.select_dtypes(include=['object']).columns:
                 if col != gdf_enriched.geometry.name: # Don't convert geometry column
                      try:
                           # Attempt conversion to string; handle potential errors
                           gdf_enriched[col] = gdf_enriched[col].astype(str)
                      except Exception as e:
                           logger.warning(f"Não foi possível converter a coluna '{col}' para string antes de salvar: {e}. Tentando remover...")
                           try:
                                gdf_enriched = gdf_enriched.drop(columns=[col])
                           except Exception as drop_e:
                                logger.error(f"Não foi possível remover a coluna '{col}': {drop_e}")


            gdf_enriched.to_file(output_path, driver="GPKG")
            logger.info(f"Dados pré-processados salvos com sucesso em {time.time() - start_time_step:.2f}s")
        except Exception as e:
            logger.error(f"Erro ao salvar arquivo GPKG em {output_path}: {e}")
            output_path = None # Indicate saving failed

    # --- 10. Gerar Relatório de Qualidade ---
    logger.info("Gerando relatório de qualidade...")
    quality_report = generate_quality_report(
        input_path=input_path,
        output_path=output_path, # Use potentially updated path
        gdf_initial=gdf_raw,
        gdf_final=gdf_enriched,
        initial_connectivity=initial_connectivity_metrics,
        final_connectivity=final_connectivity_metrics
    )

    # Salvar relatório de qualidade
    quality_path = os.path.join(QUALITY_REPORT_DIR, f"road_preprocessing_quality_{timestamp}.json")
    try:
        os.makedirs(QUALITY_REPORT_DIR, exist_ok=True) # Ensure dir exists
        with open(quality_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False, default=str) # Use default=str for non-serializable types
        logger.info(f"Relatório de qualidade salvo em {quality_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar relatório de qualidade em {quality_path}: {e}")


    logger.info("--- Pipeline de Pré-processamento Concluído ---")
    return gdf_enriched, quality_report


def generate_quality_report(input_path, output_path, gdf_initial, gdf_final, initial_connectivity, final_connectivity):
    """Gera um dicionário com o relatório de qualidade do pré-processamento."""

    # Helper to safely get stats, returning None if column missing or empty
    def get_stats(series):
        if series is None or series.empty or series.isna().all():
            return {"min": None, "max": None, "mean": None, "median": None, "std": None}
        valid_series = series.dropna()
        if valid_series.empty:
            return {"min": None, "max": None, "mean": None, "median": None, "std": None}
        return {
            "min": float(valid_series.min()),
            "max": float(valid_series.max()),
            "mean": float(valid_series.mean()),
            "median": float(valid_series.median()),
            "std": float(valid_series.std())
        }

    # Helper to safely get value counts, returning empty dict if column missing
    def get_value_counts(series):
         if series is None or series.empty:
              return {}
         return series.astype(str).value_counts().to_dict()


    initial_count = len(gdf_initial) if gdf_initial is not None else 0
    final_count = len(gdf_final) if gdf_final is not None else 0
    retention_ratio = (final_count / initial_count) if initial_count > 0 else 0

    report = {
        "report_type": "road_preprocessing_quality",
        "report_date": datetime.now().isoformat(),
        "timestamp_run": timestamp, # Global timestamp from script start
        "original_file": input_path,
        "processed_file": output_path if output_path else "N/A",
        "counts": {
            "initial_features": initial_count,
            "final_features": final_count,
            "retention_ratio": retention_ratio
        },
        "attributes": {
            "initial_columns": list(gdf_initial.columns) if gdf_initial is not None else [],
            "final_columns": list(gdf_final.columns) if gdf_final is not None else [],
            "added_columns": list(set(gdf_final.columns) - set(gdf_initial.columns)) if gdf_initial is not None and gdf_final is not None else [],
            "removed_columns": list(set(gdf_initial.columns) - set(gdf_final.columns)) if gdf_initial is not None and gdf_final is not None else []
        },
        "road_types": {
            "initial_highway_dist": get_value_counts(gdf_initial.get('highway')),
            "final_highway_dist": get_value_counts(gdf_final.get('highway')),
            "final_road_category_dist": get_value_counts(gdf_final.get('road_category'))
        },
        "geometry": {
            "initial_types": get_value_counts(gdf_initial.geometry.type) if gdf_initial is not None else {},
            "final_types": get_value_counts(gdf_final.geometry.type) if gdf_final is not None else {},
            "final_length_stats_m": get_stats(gdf_final.get('length_m')),
            "final_total_length_km": gdf_final.get('length_m').sum() / 1000 if gdf_final is not None and 'length_m' in gdf_final.columns else None
        },
        "derived_metrics": {
            "sinuosity_stats": get_stats(gdf_final.get('sinuosity')),
            "curvature_stats": get_stats(gdf_final.get('curvature')),
            "slope_pct_stats": get_stats(gdf_final.get('slope_pct')),
            "bearing_stats": get_stats(gdf_final.get('bearing')),
            "point_density_stats": get_stats(gdf_final.get('point_density')),
        },
        "categorical_distributions": {
             "length_category": get_value_counts(gdf_final.get('length_category')),
             "sinuosity_category": get_value_counts(gdf_final.get('sinuosity_category')),
             "slope_category": get_value_counts(gdf_final.get('slope_category')),
        },
        "contextual_integration": {
            "predominant_landuse_dist": get_value_counts(gdf_final.get('predominant_landuse')),
            "building_count_stats": get_stats(gdf_final.get('building_count')),
            "building_density_stats": get_stats(gdf_final.get('building_density')),
            # Add stats for other integrated context features if needed
        },
        "connectivity": {
            # Remove non-serializable items like lists of components
            "initial": {k: v for k, v in initial_connectivity.items() if not isinstance(v, list)},
            "final": {k: v for k, v in final_connectivity.items() if not isinstance(v, list)}
        },
        "spatial_extent_final": {
            "xmin": float(gdf_final.total_bounds[0]) if gdf_final is not None and len(gdf_final.total_bounds)==4 else None,
            "ymin": float(gdf_final.total_bounds[1]) if gdf_final is not None and len(gdf_final.total_bounds)==4 else None,
            "xmax": float(gdf_final.total_bounds[2]) if gdf_final is not None and len(gdf_final.total_bounds)==4 else None,
            "ymax": float(gdf_final.total_bounds[3]) if gdf_final is not None and len(gdf_final.total_bounds)==4 else None,
            "area_km2": float((gdf_final.total_bounds[2] - gdf_final.total_bounds[0]) *
                              (gdf_final.total_bounds[3] - gdf_final.total_bounds[1]) / 1e6) if gdf_final is not None and len(gdf_final.total_bounds)==4 else None
        },
        "processing_info": {
            "crs": str(gdf_final.crs) if gdf_final is not None else None,
            "python_version": sys.version,
            "geopandas_version": gpd.__version__,
            "networkx_version": nx.__version__,
            # Add other relevant library versions
        }
    }
    return report


def run_preprocessing_pipeline():
    """
    Executa pipeline completo de pré-processamento e preparação de features GNN
    usando os caminhos exatos especificados pelo usuário para o ambiente Colab.

    Returns:
        dict: Dicionário com resultados do pipeline, incluindo GeoDataFrame processado,
              DataFrames de nós e arestas para GNN, relatório de qualidade,
              caminhos dos arquivos de saída e tempo de processamento.
    """
    pipeline_start_time = time.time()
    logger.info("--- Iniciando Execução Completa do Pipeline ---")

    try:
        # Verificação de diretórios
        logger.info("Verificando diretórios necessários...")
        
        # Garantir que os diretórios importantes existam
        colab_dirs = [
            '/content/drive/MyDrive/geoprocessamento_gnn',
            '/content/drive/MyDrive/geoprocessamento_gnn/data',
            '/content/drive/MyDrive/geoprocessamento_gnn/OUTPUT',
            '/content/drive/MyDrive/geoprocessamento_gnn/QUALITY_REPORT',
            '/content/drive/MyDrive/geoprocessamento_gnn/VISUALIZACOES'
        ]
        
        for directory in colab_dirs:
            if not os.path.exists(directory):
                logger.warning(f"Diretório não encontrado: {directory}. Criando...")
                try:
            os.makedirs(directory, exist_ok=True)
                except Exception as e:
                    logger.error(f"Erro ao criar diretório {directory}: {e}")

        # Definir caminho exato para o arquivo GPKG de saída no Colab
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gpkg_output_path = '/content/drive/MyDrive/geoprocessamento_gnn/data/roads_processed_{}.gpkg'.format(timestamp)

        # Executar pré-processamento geoespacial usando o caminho exato do arquivo de entrada
        logger.info("Iniciando pré-processamento geoespacial...")
        input_path = '/content/drive/MyDrive/geoprocessamento_gnn/data/roads_enriched_20250412_230707.gpkg'
        gdf_processed, quality_report = preprocess_road_data(
            input_path=input_path,
            output_path=gpkg_output_path
        )

        if gdf_processed is None:
            logger.error("Pré-processamento geoespacial falhou. Pipeline abortado.")
            return None

        # Preparar características de nós e arestas
        logger.info("Preparando características de nós e arestas...")
        nodes_base_df = prepare_node_features(gdf_processed)
        edges_base_df = prepare_edge_features(gdf_processed)

        if nodes_base_df.empty or edges_base_df.empty:
            logger.error("Falha ao preparar características base de nós ou arestas. Pipeline abortado.")
            return None

        # Preparar características para o modelo GNN
        logger.info("Preparando características para o modelo GNN...")
        nodes_gnn_ready, edges_gnn_ready, normalization_params = prepare_gnn_features(
            nodes_base_df, edges_base_df, categorical_encoding='one_hot'
        )

        # Salvar características preparadas para GNN usando caminhos exatos
        nodes_csv_path = '/content/drive/MyDrive/geoprocessamento_gnn/data/road_nodes_gnn_{}.csv'.format(timestamp)
        edges_csv_path = '/content/drive/MyDrive/geoprocessamento_gnn/data/road_edges_gnn_{}.csv'.format(timestamp)

        try:
            nodes_gnn_ready.to_csv(nodes_csv_path, index=False)
            edges_gnn_ready.to_csv(edges_csv_path, index=False)
            logger.info(f"Características de nós para GNN salvas em: {nodes_csv_path}")
            logger.info(f"Características de arestas para GNN salvas em: {edges_csv_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar arquivos CSV de features GNN: {e}")

        # Resumo dos resultados e atualização do relatório
        processing_time = time.time() - pipeline_start_time
        logger.info(f"Pipeline completo concluído em {processing_time:.2f} segundos")
        logger.info(f"Arquivo GPKG processado: {gpkg_output_path}")
        logger.info(f"Features iniciais: {quality_report['counts']['initial_features']} -> Finais: {quality_report['counts']['final_features']}")

        # Adicionar informações da preparação GNN ao relatório
        quality_report["gnn_preparation"] = {
            "nodes_count": len(nodes_gnn_ready),
            "edges_count": len(edges_gnn_ready),
            "nodes_features_count": nodes_gnn_ready.shape[1],
            "edges_features_count": edges_gnn_ready.shape[1],
            "nodes_output_csv": nodes_csv_path,
            "edges_output_csv": edges_csv_path,
            "normalization_params_summary": {
                "nodes_normalized_cols": list(normalization_params.get('nodes', {}).keys()),
                "edges_normalized_cols": list(normalization_params.get('edges', {}).keys())
            }
        }
        quality_report["processing_pipeline_duration_seconds"] = processing_time

        # Salvar relatório atualizado
        quality_path = '/content/drive/MyDrive/geoprocessamento_gnn/QUALITY_REPORT/road_preprocessing_quality_{}.json'.format(timestamp)
        try:
            with open(quality_path, 'w', encoding='utf-8') as f:
                json.dump(quality_report, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Relatório de qualidade final salvo em: {quality_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar relatório de qualidade final: {e}")

        # Resultado final do pipeline
        return {
            'gdf_processed': gdf_processed,
            'nodes_gnn': nodes_gnn_ready,
            'edges_gnn': edges_gnn_ready,
            'quality_report': quality_report,
            'output_gpkg_path': gpkg_output_path,
            'output_nodes_csv_path': nodes_csv_path,
            'output_edges_csv_path': edges_csv_path,
            'processing_time': processing_time,
            'normalization_params': normalization_params
        }
        
    except Exception as e:
        logger.exception(f"Erro fatal durante a execução do pipeline: {str(e)}")
        return None


def improve_topology(gdf, tolerance=1.0):
    """
    Tenta corrigir a topologia da rede conectando endpoints próximos (snapping).

    Args:
        gdf: GeoDataFrame com a rede viária (LineStrings esperados)
        tolerance (float): Tolerância em metros para considerar endpoints próximos.

    Returns:
        GeoDataFrame com topologia potencialmente melhorada.
    """
    logger.info(f"Aplicando correções topológicas (snapping) com tolerância: {tolerance}m")

    if gdf.empty or 'geometry' not in gdf.columns:
        logger.warning("GeoDataFrame vazio ou sem geometria. Nenhuma correção topológica aplicada.")
        return gdf.copy()

    gdf_corrected = gdf.copy() # Work on a copy

    # 1. Extrair todos os endpoints únicos e seus índices de segmento
    endpoint_data = [] # List of tuples: (Point_geometry, segment_index, 'start'/'end')
    point_to_segments = {} # Dict: Point_geometry -> list of (segment_index, 'start'/'end')

    for idx, row in gdf_corrected.iterrows():
        geom = row.geometry
        if geom is not None and geom.geom_type == 'LineString' and len(geom.coords) >= 2:
            start_pt = Point(geom.coords[0])
            end_pt = Point(geom.coords[-1])

            endpoint_data.append((start_pt, idx, 'start'))
            endpoint_data.append((end_pt, idx, 'end'))

            if start_pt not in point_to_segments: point_to_segments[start_pt] = []
            if end_pt not in point_to_segments: point_to_segments[end_pt] = []
            point_to_segments[start_pt].append((idx, 'start'))
            point_to_segments[end_pt].append((idx, 'end'))
        else:
             logger.debug(f"Ignorando geometria não-LineString ou inválida no índice {idx} para snapping.")


    if not endpoint_data:
         logger.warning("Nenhum endpoint válido encontrado para snapping.")
         return gdf_corrected


    # 2. Criar GeoDataFrame de endpoints para busca espacial
    endpoints_gdf = gpd.GeoDataFrame(
        [{'geometry': pt, 'segment_id': idx, 'position': pos} for pt, idx, pos in endpoint_data],
        crs=gdf_corrected.crs
    )
    if not endpoints_gdf.has_sindex:
         logger.info("Criando índice espacial para endpoints...")
         endpoints_gdf.sindex


    # 3. Identificar e aplicar snapping
    modifications = {} # Dict: segment_index -> {'start_coord': new_coord, 'end_coord': new_coord}
    snap_count = 0

    # Iterate through unique points that need potential snapping
    # A point needs snapping if it's close to another point from a *different* segment.
    processed_points = set()

    for current_pt, seg_list in point_to_segments.items():
        if current_pt in processed_points:
            continue

        # Find nearby points using spatial index
        possible_matches_index = list(endpoints_gdf.sindex.query(current_pt.buffer(tolerance), predicate='intersects'))
        nearby_points_gdf = endpoints_gdf.iloc[possible_matches_index].copy()

        # Calculate exact distances and filter
        nearby_points_gdf['distance'] = nearby_points_gdf.geometry.distance(current_pt)
        # Keep points within tolerance, excluding self, and from different segments initially
        nearby_points_gdf = nearby_points_gdf[
             (nearby_points_gdf['distance'] > 1e-9) & # Exclude self
             (nearby_points_gdf['distance'] <= tolerance)
        ]

        # Find the closest point among the valid nearby points
        if not nearby_points_gdf.empty:
            closest = nearby_points_gdf.loc[nearby_points_gdf['distance'].idxmin()]
            target_pt = closest.geometry # The point to snap *to*

            # Find all original points that were close to the current_pt and should snap to target_pt
            points_to_snap = {current_pt}
            # Also check points that were close to the target_pt initially
            possible_target_matches_index = list(endpoints_gdf.sindex.query(target_pt.buffer(tolerance), predicate='intersects'))
            for idx in possible_target_matches_index:
                 pt_geom = endpoints_gdf.iloc[idx].geometry
                 if pt_geom.distance(target_pt) <= tolerance:
                      points_to_snap.add(pt_geom)


            # Apply the snap for all segments connected to any of the points_to_snap
            for pt_to_snap in points_to_snap:
                 if pt_to_snap in point_to_segments:
                      for segment_idx, position in point_to_segments[pt_to_snap]:
                           if segment_idx not in modifications:
                                modifications[segment_idx] = {}
                           # Only update if not already snapped to the same target in this group
                           if position not in modifications[segment_idx] or modifications[segment_idx].get(position) != target_pt.coords[0]:
                                modifications[segment_idx][position] = target_pt.coords[0] # Store target coordinate tuple
                                snap_count += 1
                 processed_points.add(pt_to_snap) # Mark as processed


    # 4. Aplicar as modificações ao GeoDataFrame
    logger.info(f"Aplicando {snap_count} modificações de snapping em {len(modifications)} segmentos...")
    geom_col_idx = gdf_corrected.columns.get_loc('geometry')

    for segment_idx, mod_info in modifications.items():
         try:
              original_geom = gdf_corrected.iloc[segment_idx, geom_col_idx]
              if original_geom is None or original_geom.geom_type != 'LineString':
                   continue

              coords = list(original_geom.coords)
              modified = False
              if 'start' in mod_info and tuple(coords[0]) != mod_info['start']:
                   coords[0] = mod_info['start']
                   modified = True
              if 'end' in mod_info and tuple(coords[-1]) != mod_info['end']:
                   coords[-1] = mod_info['end']
                   modified = True

              if modified and len(coords) >= 2: # Ensure we still have a valid line
                   new_geom = LineString(coords)
                   # Use iloc for potentially faster assignment
                   gdf_corrected.iloc[segment_idx, geom_col_idx] = new_geom
              elif modified:
                   logger.warning(f"Snapping para segmento {segment_idx} resultou em < 2 coordenadas. Modificação ignorada.")

         except IndexError:
              logger.warning(f"Índice {segment_idx} fora dos limites ao aplicar snapping. Ignorando.")
         except Exception as e:
              logger.error(f"Erro ao aplicar snapping para segmento {segment_idx}: {e}")


    logger.info(f"Correções topológicas (snapping) aplicadas.")

    # Recalcular comprimento e sinuosidade após modificações? Opcional.
    # gdf_corrected['length_m'] = gdf_corrected.geometry.length
    # gdf_corrected['sinuosity'] = calculate_sinuosity(gdf_corrected)
    # logger.info("Comprimento e sinuosidade recalculados após snapping.")


    return gdf_corrected


def enrich_edge_features(gdf):
    """
    Enriquece as características dos segmentos viários (arestas) com atributos
    morfológicos e funcionais avançados. Assume que 'length_m' e 'sinuosity'
    já existem.

    Args:
        gdf: GeoDataFrame contendo dados de estradas (pós-limpeza/topologia)

    Returns:
        GeoDataFrame enriquecido com características adicionais como:
        'curvature', 'point_density', 'bearing', 'length_category', 'sinuosity_category'.
    """
    logger.info("Enriquecendo segmentos viários com características morfológicas avançadas")

    if gdf.empty:
        logger.warning("GeoDataFrame de entrada vazio. Nenhum enriquecimento aplicado.")
        return gdf.copy()

    gdf_enriched = gdf.copy()

    # --- 1. Calcular Curvatura Média ---
    # (Média da mudança angular absoluta por unidade de comprimento)
    logger.info("Calculando curvatura média dos segmentos...")
    curvatures = []
    for index, row in gdf_enriched.iterrows():
        geom = row.geometry
        total_angle_change = 0.0
        segment_length = 0.0
        curvature = 0.0 # Default

        if geom is not None and geom.geom_type == 'LineString':
            coords = list(geom.coords)
            segment_length = geom.length
            if len(coords) >= 3 and segment_length > 1e-6:
                for i in range(len(coords) - 2):
                    p1, p2, p3 = coords[i], coords[i+1], coords[i+2]
                    v1 = (p2[0] - p1[0], p2[1] - p1[1])
                    v2 = (p3[0] - p2[0], p3[1] - p2[1])
                    mag_v1 = (v1[0]**2 + v1[1]**2)**0.5
                    mag_v2 = (v2[0]**2 + v2[1]**2)**0.5

                    if mag_v1 > 1e-9 and mag_v2 > 1e-9:
                        dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                        cos_angle = max(-1.0, min(1.0, dot_product / (mag_v1 * mag_v2)))
                        angle = np.arccos(cos_angle) # Angle change in radians [0, pi]
                        total_angle_change += angle

                # Curvature = total angle change / length
                curvature = total_angle_change / segment_length

        curvatures.append(curvature)

    gdf_enriched['curvature'] = curvatures
    logger.info(f"Curvatura média calculada. Média={np.mean(curvatures):.4f}")

    # --- 2. Densidade de Pontos (vértices por metro) ---
    logger.info("Calculando densidade de pontos dos segmentos...")
    gdf_enriched['point_density'] = gdf_enriched.geometry.apply(
        lambda g: len(list(g.coords)) / g.length if g and g.geom_type == 'LineString' and g.length > 1e-6 else 0
    )
    logger.info(f"Densidade de pontos calculada. Média={gdf_enriched['point_density'].mean():.4f} pontos/m")


    # --- 3. Bearing (orientação início-fim) ---
    logger.info("Calculando orientação (bearing) principal dos segmentos...")
    bearings = []
    for index, row in gdf_enriched.iterrows():
        geom = row.geometry
        bearing = 0.0 # Default
        if geom is not None and geom.geom_type == 'LineString' and len(geom.coords) >= 2:
            start, end = geom.coords[0], geom.coords[-1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            if abs(dx) > 1e-9 or abs(dy) > 1e-9: # Avoid atan2(0,0)
                angle_rad = np.arctan2(dx, dy) # Note: dx, dy order for bearing from North
                angle_deg = np.degrees(angle_rad)
                bearing = (angle_deg + 360) % 360 # Normalize to [0, 360)
        bearings.append(bearing)

    gdf_enriched['bearing'] = bearings
    logger.info("Orientação (bearing) calculada.")


    # --- 4. Classificação por Comprimento ---
    if 'length_m' in gdf_enriched.columns:
        logger.info("Classificando segmentos por comprimento...")
        length_bins = [-np.inf, 50, 100, 250, 500, 1000, np.inf]
        length_labels = ['muito_curto', 'curto', 'medio_curto', 'medio', 'longo', 'muito_longo']
        gdf_enriched['length_category'] = pd.cut(
            gdf_enriched['length_m'],
            bins=length_bins,
            labels=length_labels,
            right=False # Intervals like [0, 50), [50, 100), ...
        ).astype(str).fillna('desconhecido') # Handle potential NaNs in length_m
        logger.info(f"Categorias de comprimento criadas:\n{gdf_enriched['length_category'].value_counts()}")
    else:
        logger.warning("Coluna 'length_m' não encontrada. 'length_category' não criada.")


    # --- 5. Classificação por Sinuosidade ---
    if 'sinuosity' in gdf_enriched.columns:
        logger.info("Classificando segmentos por sinuosidade...")
        # Ensure bins start slightly below 1 to include straight lines
        sinuosity_bins = [0.999, 1.05, 1.15, 1.3, 1.5, np.inf]
        sinuosity_labels = ['reta', 'quase_reta', 'pouco_sinuosa', 'sinuosa', 'muito_sinuosa']
        gdf_enriched['sinuosity_category'] = pd.cut(
            gdf_enriched['sinuosity'],
            bins=sinuosity_bins,
            labels=sinuosity_labels,
            right=False
        ).astype(str).fillna('desconhecido') # Handle potential NaNs in sinuosity
        logger.info(f"Categorias de sinuosidade criadas:\n{gdf_enriched['sinuosity_category'].value_counts()}")
    else:
        logger.warning("Coluna 'sinuosity' não encontrada. 'sinuosity_category' não criada.")


    logger.info("Enriquecimento de características morfológicas concluído.")
    return gdf_enriched


def advanced_connectivity_analysis(gdf, tolerance=1.0):
    """
    Realiza análise avançada de conectividade da rede viária, usando tolerância
    para conectar endpoints próximos e calculando métricas de grafo detalhadas.

    Args:
        gdf: GeoDataFrame contendo dados de estradas (LineStrings esperados)
        tolerance (float): Tolerância em metros para considerar nós conectados.

    Returns:
        tuple:
            dict: Dicionário com métricas avançadas de conectividade.
            networkx.Graph: O grafo construído para análise.
            Retorna ({}, nx.Graph()) se a entrada for vazia ou inválida.
    """
    logger.info(f"Iniciando análise avançada de conectividade (tolerância: {tolerance}m)")

    if gdf.empty or 'geometry' not in gdf.columns:
        logger.warning("GeoDataFrame vazio ou sem geometria. Não é possível analisar conectividade avançada.")
        return {}, nx.Graph()

    G = nx.Graph()
    node_coords = [] # List of coordinate tuples
    node_map = {} # Map coord tuple -> node ID
    edge_data = [] # List of tuples (start_node_id, end_node_id, {attributes})
    next_node_id = 0

    # 1. Criar nós únicos para todos os endpoints
    logger.debug("Identificando nós únicos (endpoints)...")
    for index, row in gdf.iterrows():
        geom = row.geometry
        if geom is not None and geom.geom_type == 'LineString' and len(geom.coords) >= 2:
            start_coord = tuple(geom.coords[0])
            end_coord = tuple(geom.coords[-1])

            if start_coord not in node_map:
                node_map[start_coord] = next_node_id
                node_coords.append(start_coord)
                G.add_node(next_node_id, x=start_coord[0], y=start_coord[1], original_coord=start_coord)
                next_node_id += 1
            if end_coord not in node_map:
                node_map[end_coord] = next_node_id
                node_coords.append(end_coord)
                G.add_node(next_node_id, x=end_coord[0], y=end_coord[1], original_coord=end_coord)
                next_node_id += 1

            start_node_id = node_map[start_coord]
            end_node_id = node_map[end_coord]

            if start_node_id != end_node_id:
                segment_id = row.get('edge_id', index) # Use edge_id if available
                length = row.get('length_m', geom.length)
                edge_attrs = {'segment_id': segment_id, 'length': length, 'type': 'segment'}
                edge_data.append((start_node_id, end_node_id, edge_attrs))
            else:
                 logger.debug(f"Ignorando loop no segmento {index} para grafo.")


    logger.info(f"Identificados {next_node_id} nós únicos inicialmente.")

    # 2. Conectar nós próximos dentro da tolerância (Merge nodes)
    logger.info("Mesclando nós próximos (dentro da tolerância)...")
    if next_node_id > 1 and tolerance > 0:
        try:
            from scipy.spatial import KDTree
            coords_array = np.array(node_coords)
            kdtree = KDTree(coords_array)

            # Find pairs of points within tolerance
            pairs = kdtree.query_pairs(r=tolerance)

            # Use Union-Find (Disjoint Set Union) for efficient merging
            parent = list(range(next_node_id))
            def find_set(v):
                if v == parent[v]:
                    return v
                parent[v] = find_set(parent[v])
                return parent[v]

            def unite_sets(a, b):
                a = find_set(a)
                b = find_set(b)
                if a != b:
                    parent[b] = a # Merge b into a

            merged_count = 0
            for i, j in pairs:
                if find_set(i) != find_set(j):
                     unite_sets(i, j)
                     merged_count += 1

            logger.info(f"Identificadas {merged_count} mesclagens potenciais de nós.")

            # Create mapping from old node ID to new (merged) node ID (the root of the set)
            node_id_mapping = {old_id: find_set(old_id) for old_id in range(next_node_id)}

            # Add edges to the graph using the *merged* node IDs
            edges_added_count = 0
            for u, v, attrs in edge_data:
                 new_u = node_id_mapping[u]
                 new_v = node_id_mapping[v]
                 if new_u != new_v: # Avoid self-loops after merging
                      # Check if edge already exists (can happen with merged nodes)
                      if not G.has_edge(new_u, new_v):
                           G.add_edge(new_u, new_v, **attrs)
                           edges_added_count += 1
                      else:
                           # Optionally handle parallel edges, e.g., store segment IDs in edge data
                           if 'segment_id' not in G.edges[new_u, new_v]:
                                G.edges[new_u, new_v]['segment_id'] = []
                           if isinstance(G.edges[new_u, new_v]['segment_id'], list):
                                G.edges[new_u, new_v]['segment_id'].append(attrs['segment_id'])
                           else: # Convert to list if it was single value
                                G.edges[new_u, new_v]['segment_id'] = [G.edges[new_u, new_v]['segment_id'], attrs['segment_id']]
                           # Update length? Maybe average or keep shortest? For now, keep first length.


            # Remove isolated nodes (nodes that are roots of sets but have no edges after merge)
            nodes_to_remove = [node for node in G.nodes if G.degree(node) == 0]
            G.remove_nodes_from(nodes_to_remove)
            logger.info(f"Removidos {len(nodes_to_remove)} nós isolados após mesclagem.")

        except ImportError:
             logger.warning("Scipy não encontrado. Não foi possível mesclar nós próximos eficientemente. Usando grafo não mesclado.")
             # Fallback: Add edges directly without merging
             for u, v, attrs in edge_data:
                  if u != v and not G.has_edge(u,v): # Avoid self-loops and duplicates
                       G.add_edge(u, v, **attrs)
        except Exception as e:
             logger.error(f"Erro durante a mesclagem de nós: {e}. Usando grafo não mesclado.")
             # Fallback: Add edges directly without merging
             G = nx.Graph() # Reset graph
             for node_id, coord in enumerate(node_coords):
                  G.add_node(node_id, x=coord[0], y=coord[1], original_coord=coord)
             for u, v, attrs in edge_data:
                  if u != v and not G.has_edge(u,v):
                       G.add_edge(u, v, **attrs)


    else: # No merging needed or possible
         logger.info("Nenhuma mesclagem de nós realizada (poucos nós ou tolerância zero).")
         # Add edges directly
         for u, v, attrs in edge_data:
              if u != v and not G.has_edge(u,v):
                   G.add_edge(u, v, **attrs)


    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    logger.info(f"Grafo final construído com {num_nodes} nós e {num_edges} arestas (após mesclagem/conexão).")

    if num_nodes == 0:
        logger.warning("Grafo final não contém nós.")
        return {}, G # Return empty results


    # 4. Análise de componentes conectados
    is_connected_flag = nx.is_connected(G)
    components = list(nx.connected_components(G))
    num_components = len(components)

    component_sizes = sorted([len(c) for c in components], reverse=True)
    component_percentages = [size / num_nodes for size in component_sizes]

    # 5. Métricas avançadas de conectividade
    metrics = {
        'is_connected': is_connected_flag,
        'num_components': num_components,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'largest_component_size': component_sizes[0] if component_sizes else 0,
        'largest_component_percentage': component_percentages[0] if component_percentages else 0,
        'avg_degree': sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0,
        'density': nx.density(G),
        'component_sizes_top10': component_sizes[:10],
        'component_percentages_top10': component_percentages[:10],
    }

    # 6. Métricas específicas para o maior componente (se existir)
    if component_sizes:
        largest_component_nodes = max(components, key=len)
        largest_component_subgraph = G.subgraph(largest_component_nodes)

        # Check if subgraph is non-empty before calculating metrics
        if largest_component_subgraph.number_of_nodes() > 0:
            # Pontos de articulação
            try:
                articulation_points = list(nx.articulation_points(largest_component_subgraph))
                metrics['articulation_points_count'] = len(articulation_points)
            except Exception as e:
                logger.warning(f"Não foi possível calcular pontos de articulação: {e}")
                metrics['articulation_points_count'] = None

            # Pontes
            try:
                # Bridge finding can be slow on large graphs
                # bridges = list(nx.bridges(largest_component_subgraph))
                # metrics['bridges_count'] = len(bridges)
                metrics['bridges_count'] = "Calculation skipped for performance" # Placeholder
            except Exception as e:
                logger.warning(f"Não foi possível calcular pontes: {e}")
                metrics['bridges_count'] = None

            # Diâmetro e Raio (podem ser muito lentos ou inviáveis para grafos grandes e desconectados)
            if is_connected_flag: # Only calculate for fully connected graph
                 try:
                      # metrics['diameter'] = nx.diameter(largest_component_subgraph)
                      metrics['diameter'] = "Calculation skipped for performance"
                 except Exception as e:
                      logger.warning(f"Não foi possível calcular diâmetro: {e}")
                      metrics['diameter'] = None
                 try:
                      # metrics['radius'] = nx.radius(largest_component_subgraph)
                       metrics['radius'] = "Calculation skipped for performance"
                 except Exception as e:
                      logger.warning(f"Não foi possível calcular raio: {e}")
                      metrics['radius'] = None
            else:
                 metrics['diameter'] = None
                 metrics['radius'] = None


            # Centralidade de Intermediação (Betweenness) - pode ser lenta
            try:
                # Calculate on a sample or skip if too slow
                # betweenness = nx.betweenness_centrality(largest_component_subgraph, weight='length', normalized=True, k=min(1000, num_nodes // 10)) # Sample if large
                # metrics['avg_betweenness'] = np.mean(list(betweenness.values())) if betweenness else None
                # metrics['max_betweenness'] = np.max(list(betweenness.values())) if betweenness else None
                 metrics['avg_betweenness'] = "Calculation skipped for performance"
                 metrics['max_betweenness'] = "Calculation skipped for performance"
            except Exception as e:
                logger.warning(f"Não foi possível calcular betweenness centrality: {e}")
                metrics['avg_betweenness'] = None
                metrics['max_betweenness'] = None

            # Eficiência Global (também pode ser lenta)
            try:
                 if is_connected_flag: # Meaningful only for connected graphs
                     # metrics['global_efficiency'] = nx.global_efficiency(largest_component_subgraph)
                      metrics['global_efficiency'] = "Calculation skipped for performance"
                 else:
                      metrics['global_efficiency'] = None
            except Exception as e:
                logger.warning(f"Não foi possível calcular eficiência global: {e}")
                metrics['global_efficiency'] = None
        else:
             logger.warning("Maior componente está vazio após processamento do grafo.")


    # 7. Índice de fragmentação
    if num_components > 1 and num_nodes > 0:
        fragmentation = 1.0 - sum(p**2 for p in component_percentages)
        metrics['fragmentation_index'] = fragmentation
        metrics['effective_components'] = 1.0 / sum(p**2 for p in component_percentages)
    else:
        metrics['fragmentation_index'] = 0.0
        metrics['effective_components'] = 1.0 if num_nodes > 0 else 0.0

    # 8. Estatísticas de isolamento (nós com grau 0)
    isolated_nodes = [node for node, degree in G.degree() if degree == 0] # Should be empty after node removal step
    metrics['isolated_nodes_count'] = len(isolated_nodes)
    metrics['isolated_nodes_percentage'] = len(isolated_nodes) / num_nodes if num_nodes > 0 else 0

    logger.info(f"Análise avançada de conectividade concluída: {num_components} componentes.")
    logger.info(f"Maior componente: {metrics['largest_component_percentage']:.2%} da rede.")
    logger.info(f"Índice de fragmentação: {metrics['fragmentation_index']:.4f}")

    return metrics, G


def prepare_gnn_features(nodes_df, edges_df, categorical_encoding='one_hot'):
    """
    Prepara DataFrames de nós e arestas para entrada em modelos GNN.
    Inclui normalização de features numéricas e codificação de categóricas.

    Args:
        nodes_df (pd.DataFrame): DataFrame com características base dos nós (output de prepare_node_features).
        edges_df (pd.DataFrame): DataFrame com características base das arestas (output de prepare_edge_features).
        categorical_encoding (str): Método para codificar ('one_hot' ou 'ordinal').

    Returns:
        tuple:
            pd.DataFrame: Nós com features prontas para GNN.
            pd.DataFrame: Arestas com features prontas para GNN.
            dict: Dicionário com parâmetros de normalização usados.
            Retorna (None, None, {}) se a entrada for inválida.
    """
    logger.info(f"Preparando características para GNN (Encoding: {categorical_encoding})")

    if nodes_df is None or edges_df is None or nodes_df.empty or edges_df.empty:
        logger.error("DataFrames de nós ou arestas inválidos ou vazios. Não é possível preparar features GNN.")
        return None, None, {}

    nodes_prepared = nodes_df.copy()
    edges_prepared = edges_df.copy()

    # --- 1. Normalização de Features Numéricas ---
    logger.info("Normalizando features numéricas de nós...")
    nodes_prepared, node_norm_params = normalize_features(nodes_prepared, id_col_suffix='node_id')

    logger.info("Normalizando features numéricas de arestas...")
    edges_prepared, edge_norm_params = normalize_features(edges_prepared, id_col_suffix='edge_id')

    normalization_params = {'nodes': node_norm_params, 'edges': edge_norm_params}

    # --- 2. Codificação de Features Categóricas ---
    node_cat_cols = nodes_prepared.select_dtypes(include=['object', 'category']).columns.tolist()
    edge_cat_cols = edges_prepared.select_dtypes(include=['object', 'category']).columns.tolist()

    # Colunas a serem potencialmente ignoradas no encoding (ex: IDs textuais, nomes)
    ignore_cols = ['name', 'highway'] # Keep 'highway' original, use 'road_category' encoded

    node_cat_cols = [col for col in node_cat_cols if col not in ignore_cols and col != 'node_id']
    edge_cat_cols = [col for col in edge_cat_cols if col not in ignore_cols and col != 'edge_id']


    logger.info(f"Codificando features categóricas de nós: {node_cat_cols}")
    logger.info(f"Codificando features categóricas de arestas: {edge_cat_cols}")

    if categorical_encoding == 'one_hot':
        # One-hot para Nós
        if node_cat_cols:
            try:
                nodes_prepared = pd.get_dummies(nodes_prepared, columns=node_cat_cols, dummy_na=False) # dummy_na=False is usually preferred
                logger.info("One-hot encoding aplicado aos nós.")
            except Exception as e:
                 logger.error(f"Erro no one-hot encoding dos nós: {e}")
        # One-hot para Arestas
        if edge_cat_cols:
             try:
                 edges_prepared = pd.get_dummies(edges_prepared, columns=edge_cat_cols, dummy_na=False)
                 logger.info("One-hot encoding aplicado às arestas.")
             except Exception as e:
                  logger.error(f"Erro no one-hot encoding das arestas: {e}")

    elif categorical_encoding == 'ordinal':
        from sklearn.preprocessing import OrdinalEncoder
        # Ordinal para Nós
        if node_cat_cols:
            try:
                encoder_nodes = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                # Fit and transform, then scale to [0, 1] if desired
                nodes_prepared[node_cat_cols] = encoder_nodes.fit_transform(nodes_prepared[node_cat_cols].astype(str)) # Convert to str first
                # Optional scaling:
                # for col in node_cat_cols:
                #     max_val = nodes_prepared[col].max()
                #     if max_val > 0: nodes_prepared[col] /= max_val
                logger.info("Ordinal encoding aplicado aos nós.")
            except Exception as e:
                 logger.error(f"Erro no ordinal encoding dos nós: {e}")

        # Ordinal para Arestas
        if edge_cat_cols:
             try:
                 encoder_edges = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                 edges_prepared[edge_cat_cols] = encoder_edges.fit_transform(edges_prepared[edge_cat_cols].astype(str))
                 # Optional scaling
                 logger.info("Ordinal encoding aplicado às arestas.")
             except Exception as e:
                  logger.error(f"Erro no ordinal encoding das arestas: {e}")
    else:
        logger.warning(f"Método de encoding categórico não reconhecido: {categorical_encoding}. Nenhuma codificação aplicada.")

    # --- 3. Lidar com Valores Ausentes Remanescentes (após normalização/encoding) ---
    # Preencher com 0 pode ser uma estratégia simples, mas considere outras (média, mediana) se apropriado.
    nodes_final_cols = nodes_prepared.select_dtypes(include=np.number).columns
    edges_final_cols = edges_prepared.select_dtypes(include=np.number).columns

    nodes_nan_counts = nodes_prepared[nodes_final_cols].isna().sum()
    edges_nan_counts = edges_prepared[edges_final_cols].isna().sum()

    if nodes_nan_counts.sum() > 0:
         logger.warning(f"Preenchendo {nodes_nan_counts.sum()} NaNs restantes nas features numéricas dos nós com 0.")
         # print(nodes_nan_counts[nodes_nan_counts > 0]) # Log columns with NaNs
         nodes_prepared[nodes_final_cols] = nodes_prepared[nodes_final_cols].fillna(0)

    if edges_nan_counts.sum() > 0:
         logger.warning(f"Preenchendo {edges_nan_counts.sum()} NaNs restantes nas features numéricas das arestas com 0.")
         # print(edges_nan_counts[edges_nan_counts > 0]) # Log columns with NaNs
         edges_prepared[edges_final_cols] = edges_prepared[edges_final_cols].fillna(0)

    # --- 4. Selecionar apenas colunas numéricas finais para GNN ---
    # Modelos GNN geralmente esperam tensores numéricos.
    nodes_gnn_numeric = nodes_prepared.select_dtypes(include=np.number)
    edges_gnn_numeric = edges_prepared.select_dtypes(include=np.number)

    # Garantir que as colunas de ID estejam presentes se forem necessárias depois
    if 'node_id' in nodes_prepared.columns:
         nodes_gnn_numeric['node_id'] = nodes_prepared['node_id']
    if 'edge_id' in edges_prepared.columns:
         edges_gnn_numeric['edge_id'] = edges_prepared['edge_id']


    logger.info(f"Features finais dos nós prontas para GNN: {nodes_gnn_numeric.shape[1]} colunas numéricas.")
    logger.info(f"Features finais das arestas prontas para GNN: {edges_gnn_numeric.shape[1]} colunas numéricas.")
    # logger.debug(f"Colunas finais dos nós: {list(nodes_gnn_numeric.columns)}")
    # logger.debug(f"Colunas finais das arestas: {list(edges_gnn_numeric.columns)}")


    return nodes_gnn_numeric, edges_gnn_numeric, normalization_params


def integrate_contextual_data(road_gdf, context_data):
    """
    Integra dados contextuais (setores, uso do solo, edificações) aos segmentos de estrada.

    Usa junção espacial (sjoin) baseada em buffer para atribuir características
    contextuais aos segmentos de estrada.

    Args:
        road_gdf: GeoDataFrame com a rede viária (pós-limpeza/topologia).
        context_data (dict): Dicionário contendo GeoDataFrames contextuais.

    Returns:
        GeoDataFrame: GeoDataFrame da rede viária enriquecido com dados contextuais.
    """
    logger.info("Iniciando integração de dados contextuais à rede viária...")

    if road_gdf.empty:
        logger.warning("GeoDataFrame de estradas vazio. Nenhuma integração contextual realizada.")
        return road_gdf.copy()
    if not context_data:
         logger.warning("Dicionário de dados contextuais vazio. Nenhuma integração realizada.")
         return road_gdf.copy()


    enriched_gdf = road_gdf.copy()
    original_cols = set(enriched_gdf.columns)

    # --- 1. Integração com Setores Censitários ---
    if 'setores' in context_data and not context_data['setores'].empty:
        logger.info("Integrando dados de setores censitários...")
        setores_gdf = context_data['setores']
        # Garantir que o CRS corresponde ou reprojetar
        if setores_gdf.crs != enriched_gdf.crs:
             logger.warning(f"Reprojetando setores de {setores_gdf.crs} para {enriched_gdf.crs}")
             try:
                  setores_gdf = setores_gdf.to_crs(enriched_gdf.crs)
             except Exception as e:
                  logger.error(f"Falha ao reprojetar setores: {e}. Integração de setores abortada.")
                  setores_gdf = None # Skip integration


        if setores_gdf is not None:
             # Colunas demográficas/socioeconômicas de interesse
             demographic_cols = [
                 'densidade_pop', 'indice_vulnerabilidade', 'est_populacao',
                 'categoria_vulnerabilidade', 'indice_prioridade_evacuacao',
                 # Adicione outras colunas relevantes do seu GDF de setores
                 'renda_media', 'pop_idosa_perc'
             ]
             available_cols = [col for col in demographic_cols if col in setores_gdf.columns]

             if available_cols:
                 logger.info(f"Colunas de setores disponíveis para integração: {', '.join(available_cols)}")

                 # Usar sjoin com 'within' (centroide do segmento dentro do setor) pode ser mais rápido
                 # ou sjoin com 'intersects' (segmento cruza o setor)
                 # Usaremos 'intersects' que é mais geral
                 try:
                     # Renomear colunas de setores para evitar conflitos
                     setores_gdf_renamed = setores_gdf[['geometry'] + available_cols].rename(
                         columns={col: f'setor_{col}' for col in available_cols}
                     )

                     # Spatial join: associa cada estrada ao(s) setor(es) que ela intersecta
                     # 'op' foi depreciado, usar 'predicate'
                     spatial_join = gpd.sjoin(enriched_gdf, setores_gdf_renamed, how='left', predicate='intersects')

                     # Agregação: Se uma estrada intersecta múltiplos setores, como agregar?
                     # - Média para numéricos, Moda para categóricos é uma abordagem.
                     # - Poderia também pegar o setor com maior sobreposição (mais complexo).
                     # Usaremos a abordagem Média/Moda agrupando pelo índice original da estrada.
                     agg_funcs = {}
                     for col in available_cols:
                         renamed_col = f'setor_{col}'
                         if pd.api.types.is_numeric_dtype(spatial_join[renamed_col]):
                             agg_funcs[renamed_col] = 'mean'
                         else:
                             # Moda pode retornar múltiplas se houver empate, pegar a primeira
                             agg_funcs[renamed_col] = lambda x: pd.Series.mode(x)[0] if not x.empty and not pd.Series.mode(x).empty else None


                     # Agrupar pelo índice original da estrada e aplicar funções de agregação
                     aggregated_data = spatial_join.groupby(spatial_join.index).agg(agg_funcs)

                     # Juntar os dados agregados de volta ao GDF original
                     enriched_gdf = enriched_gdf.join(aggregated_data)
                     logger.info(f"Adicionadas {len(available_cols)} colunas agregadas de dados censitários.")

                 except Exception as e:
                      logger.error(f"Erro durante a junção espacial com setores: {e}")
             else:
                  logger.warning("Nenhuma coluna demográfica relevante encontrada nos dados de setores.")
    else:
         logger.info("Dados de setores não disponíveis ou vazios.")


    # --- 2. Integração com Uso do Solo ---
    if 'landuse' in context_data and not context_data['landuse'].empty:
        logger.info("Integrando dados de uso do solo...")
        landuse_gdf = context_data['landuse']
        if landuse_gdf.crs != enriched_gdf.crs:
             logger.warning(f"Reprojetando uso do solo de {landuse_gdf.crs} para {enriched_gdf.crs}")
             try:
                  landuse_gdf = landuse_gdf.to_crs(enriched_gdf.crs)
             except Exception as e:
                  logger.error(f"Falha ao reprojetar uso do solo: {e}. Integração abortada.")
                  landuse_gdf = None

        if landuse_gdf is not None and 'land_category' in landuse_gdf.columns:
             try:
                 # Usar buffer para encontrar usos do solo próximos
                 buffer_distance = 25 # metros
                 road_buffers = enriched_gdf.copy()
                 road_buffers['geometry'] = enriched_gdf.geometry.buffer(buffer_distance)

                 # Junção espacial: buffer da estrada intersecta uso do solo
                 # Renomear coluna de categoria para evitar conflito
                 landuse_renamed = landuse_gdf[['geometry', 'land_category']].rename(columns={'land_category': 'lu_cat'})
                 spatial_join_lu = gpd.sjoin(road_buffers, landuse_renamed, how='left', predicate='intersects')

                 # Agregação: Categoria de uso do solo mais frequente (moda) no buffer
                 predominant_landuse = spatial_join_lu.groupby(spatial_join_lu.index)['lu_cat'].agg(
                      lambda x: pd.Series.mode(x)[0] if not x.empty and not pd.Series.mode(x).empty else 'unknown'
                 )
                 enriched_gdf['predominant_landuse'] = predominant_landuse.fillna('unknown')
                 logger.info("Adicionada coluna 'predominant_landuse' (uso do solo predominante no entorno).")

                 # Opcional: Calcular proporção de cada categoria no buffer (pode ser lento)
                 # ... (código similar ao anterior para calcular proporções se necessário) ...

             except Exception as e:
                  logger.error(f"Erro durante a junção espacial com uso do solo: {e}")
        elif landuse_gdf is not None:
             logger.warning("Coluna 'land_category' não encontrada nos dados de uso do solo.")
    else:
         logger.info("Dados de uso do solo não disponíveis ou vazios.")


    # --- 3. Integração com Edificações ---
    if 'buildings' in context_data and not context_data['buildings'].empty:
        logger.info("Integrando dados de edificações...")
        buildings_gdf = context_data['buildings']
        if buildings_gdf.crs != enriched_gdf.crs:
             logger.warning(f"Reprojetando edificações de {buildings_gdf.crs} para {enriched_gdf.crs}")
             try:
                  buildings_gdf = buildings_gdf.to_crs(enriched_gdf.crs)
             except Exception as e:
                  logger.error(f"Falha ao reprojetar edificações: {e}. Integração abortada.")
                  buildings_gdf = None

        if buildings_gdf is not None:
             try:
                 # Criar buffer ao redor dos segmentos
                 buffer_distance = 50 # metros
                 road_buffers = enriched_gdf[['geometry']].copy() # Only need geometry for join
                 road_buffers['geometry'] = road_buffers.geometry.buffer(buffer_distance)
                 road_buffers['buffer_area_ha'] = road_buffers.geometry.area / 10000.0

                 # Colunas de interesse nas edificações
                 building_cols = ['area_m2', 'height'] # Adicione outras se tiver
                 available_building_cols = [col for col in building_cols if col in buildings_gdf.columns]

                 # Junção espacial
                 buildings_to_join = buildings_gdf[['geometry'] + available_building_cols]
                 spatial_join_bldg = gpd.sjoin(road_buffers, buildings_to_join, how='left', predicate='intersects')

                 # Agregação
                 agg_funcs_bldg = {
                     # Contagem de edificações (usando o índice da direita que não seja nulo)
                     'index_right': 'count',
                     'buffer_area_ha': 'first' # Manter a área do buffer
                 }
                 # Média para colunas numéricas disponíveis
                 for col in available_building_cols:
                     agg_funcs_bldg[col] = 'mean'

                 aggregated_bldg_data = spatial_join_bldg.groupby(spatial_join_bldg.index).agg(agg_funcs_bldg)

                 # Renomear contagem e calcular densidade
                 aggregated_bldg_data = aggregated_bldg_data.rename(columns={'index_right': 'building_count'})
                 aggregated_bldg_data['building_density_per_ha'] = aggregated_bldg_data['building_count'] / aggregated_bldg_data['buffer_area_ha']
                 # Lidar com divisão por zero se buffer_area for 0
                 aggregated_bldg_data['building_density_per_ha'] = aggregated_bldg_data['building_density_per_ha'].replace([np.inf, -np.inf], 0).fillna(0)

                 # Renomear colunas de média
                 for col in available_building_cols:
                      aggregated_bldg_data = aggregated_bldg_data.rename(columns={col: f'avg_building_{col}'})


                 # Juntar ao GDF principal
                 cols_to_join = ['building_count', 'building_density_per_ha'] + [f'avg_building_{col}' for col in available_building_cols]
                 enriched_gdf = enriched_gdf.join(aggregated_bldg_data[cols_to_join])

                 # Preencher NaNs resultantes da junção (estradas sem edificações próximas) com 0
                 for col in cols_to_join:
                      if col in enriched_gdf.columns:
                           enriched_gdf[col] = enriched_gdf[col].fillna(0)

                 logger.info(f"Adicionadas {len(cols_to_join)} métricas relacionadas a edificações.")

             except Exception as e:
                  logger.error(f"Erro durante a junção espacial com edificações: {e}")
    else:
         logger.info("Dados de edificações não disponíveis ou vazios.")


    # --- 4. Integração com Elevação (se já presente no GDF original) ---
    # (O cálculo de slope_pct e slope_category já foi movido para enrich_edge_features
    #  se as colunas de elevação existirem. Aqui apenas logamos se elas foram usadas.)
    if 'slope_pct' in enriched_gdf.columns:
         logger.info("Métricas de inclinação (slope_pct, slope_category) presentes (calculadas anteriormente).")
    elif all(c in enriched_gdf.columns for c in ['elevation_min', 'elevation_max', 'length_m']):
         logger.warning("Colunas de elevação existem, mas slope_pct não foi calculado. Verifique enrich_edge_features.")
    else:
         logger.info("Dados de elevação não disponíveis ou insuficientes para cálculo de inclinação.")


    # --- 5. Resumo das integrações ---
    final_cols = set(enriched_gdf.columns)
    new_cols = final_cols - original_cols
    if new_cols:
        logger.info(f"Integração contextual concluída. Adicionadas {len(new_cols)} novas colunas: {', '.join(sorted(list(new_cols)))}")
    else:
        logger.info("Integração contextual concluída, mas nenhuma nova coluna foi adicionada (ou dados não estavam disponíveis).")

    return enriched_gdf


# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    logger.info("============================================================")
    logger.info("INICIANDO PIPELINE DE PRÉ-PROCESSAMENTO DE REDES VIÁRIAS")
    logger.info("============================================================")

    # --- Verificação de Ambiente e Dependências ---
    if IN_COLAB:
        logger.info("Executando no ambiente Google Colab")
        # (Mensagens de diretório e solução de problemas já impressas no início)

        # Função helper para verificar Fiona (definida aqui para escopo)
        def check_fiona_available():
            try:
                import fiona
                return True
            except ImportError:
                return False

        # Verificar e tentar instalar Fiona se necessário (apenas no Colab)
        if not check_fiona_available():
            logger.warning("Fiona não encontrado no Colab. Tentando instalar...")
            try:
                import subprocess
                # Instalar na ordem correta pode ajudar a evitar conflitos
                print("Executando: !pip uninstall -y geopandas fiona pyproj")
                subprocess.run(["pip", "uninstall", "-y", "geopandas", "fiona", "pyproj"], check=True, capture_output=True)
                print("Executando: !pip install fiona==1.9.5 pyproj==3.6.1")
                subprocess.run(["pip", "install", "fiona==1.9.5", "pyproj==3.6.1"], check=True, capture_output=True)
                print("Executando: !pip install geopandas==0.13.2")
                subprocess.run(["pip", "install", "geopandas==0.13.2"], check=True, capture_output=True)

                # Tentar importar novamente
                import fiona
                logger.info("Fiona e GeoPandas reinstalados com sucesso!")
                logger.warning("IMPORTANTE: Pode ser necessário reiniciar o runtime do Colab (Runtime > Restart runtime) para que as mudanças tenham efeito completo.")
                HAS_FIONA = True # Update global flag
            except ImportError:
                logger.error("Fiona ainda não disponível após tentativa de instalação.")
                logger.warning("Funcionalidades que dependem de Fiona (como leitura de camadas específicas) podem falhar.")
                HAS_FIONA = False
            except Exception as e:
                logger.error(f"Falha ao instalar/reinstalar dependências: {str(e)}")
                logger.info("Continuando com funcionalidade potencialmente limitada...")
                HAS_FIONA = False
        else:
             logger.info("Fiona já está disponível.")


        # Verificar PyTorch e CUDA
        if HAS_TORCH:
             if torch.cuda.is_available():
                  logger.info(f"PyTorch detectado. CUDA disponível: {torch.cuda.get_device_name(0)} ({torch.cuda.device_count()} GPU(s))")
             else:
                  logger.warning("PyTorch detectado, mas CUDA não disponível. Use CPU ou habilite GPU no Colab.")
        else:
             logger.warning("PyTorch não detectado. Funcionalidades GNN não estarão disponíveis.")

    else: # Ambiente Local
        logger.info("Executando em ambiente local")
        # (Mensagens de diretório já impressas no início)
        # Definir check_fiona_available para ambiente local
        def check_fiona_available():
            return HAS_FIONA
        if not HAS_FIONA:
            logger.warning("Fiona não encontrado no ambiente local. Instale-o (e.g., via conda) para funcionalidade completa.")


    # --- Executar Pipeline Completo ---
    pipeline_result = run_preprocessing_pipeline()

    # --- Resumo Final ---
    if pipeline_result:
        logger.info("Pipeline concluído com sucesso!")
        logger.info(f"Tempo total de execução: {pipeline_result['processing_time']:.2f} segundos")
        logger.info("Arquivos gerados (verifique os caminhos exatos nos logs):")
        logger.info(f"- Dados Geoespaciais Processados (GPKG): {pipeline_result['output_gpkg_path']}")
        logger.info(f"- Features de Nós para GNN (CSV): {pipeline_result['output_nodes_csv_path']}")
        logger.info(f"- Features de Arestas para GNN (CSV): {pipeline_result['output_edges_csv_path']}")
        # logger.info(f"- Relatório de Qualidade (JSON): [Ver caminho nos logs acima]") # Path já logado

        # Próximos passos sugeridos
        logger.info("\nPróximos passos sugeridos:")
        logger.info("1. Carregar os arquivos CSV ('nodes_gnn', 'edges_gnn') e o GPKG processado.")
        logger.info("2. Construir o grafo no formato PyTorch Geometric (ou DGL).")
        logger.info("3. Realizar visualizações exploratórias dos dados processados e do grafo.")
        logger.info("4. Treinar e avaliar o modelo GNN.")
    else:
        logger.error("A execução do pipeline falhou. Verifique os logs de erro acima.")
        sys.exit(1) # Exit with error code

    logger.info("============================================================")
    logger.info("EXECUÇÃO DO SCRIPT CONCLUÍDA")
    logger.info("============================================================")