# -*- coding: utf-8 -*-
"""
Preprocessing and Graph Construction Functions for Road Network Data

Este módulo contém funções para preparação de dados de redes viárias para análise GNN,
incluindo limpeza geoespacial, construção de grafo NetworkX, cálculo de métricas,
e conversão para formato PyTorch Geometric, conforme descrito no documento de instruções.
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
from shapely.geometry import LineString, MultiLineString, Point
import networkx as nx
from scipy.spatial import KDTree # Necessário para advanced_connectivity_analysis
import itertools

# Tentar importar fiona para manipulação de camadas
try:
    import fiona
    HAS_FIONA = True
except ImportError:
    HAS_FIONA = False
    print("Fiona não está disponível diretamente. Algumas funções podem ser limitadas.")

# Tentar importar PyTorch e PyTorch Geometric
try:
    import torch
    from torch_geometric.data import Data
    HAS_TORCH = True
    print(f"PyTorch version: {torch.__version__}")
    # Tentar importar PyG para verificar a instalação
    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("PyTorch Geometric não encontrado. Instale com: pip install torch-geometric")
        # Poderia definir HAS_PYG = False aqui se necessário para controle de fluxo
except ImportError:
    HAS_TORCH = False
    print("PyTorch ou PyTorch Geometric não estão disponíveis. Funcionalidades GNN serão limitadas.")

# Configuração para detecção de ambiente e montagem do Google Drive
try:
    import google.colab
    from google.colab import drive
    IN_COLAB = True
    print("Ambiente Google Colab detectado")

    # Montar o Google Drive conforme especificado pelo usuário
    drive.mount('/content/drive')
    print("Google Drive montado com sucesso em /content/drive")

    # Caminhos específicos fornecidos pelo usuário
    BASE_DIR = '/content/drive/MyDrive/geoprocessamento_gnn'
    OUTPUT_DIR = os.path.join(BASE_DIR, 'OUTPUT')
    QUALITY_REPORT_DIR = os.path.join(BASE_DIR, 'QUALITY_REPORT')
    VISUALIZACOES_DIR = os.path.join(BASE_DIR, 'VISUALIZACOES')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    # Usar o caminho fornecido originalmente, assumindo que ele existe:
    ROADS_ENRICHED_PATH = os.path.join(DATA_DIR, 'roads_enriched_20250412_230707.gpkg')
    PROCESSED_DATA_DIR = DATA_DIR  # Usar o mesmo diretório de dados no Colab

    # Verificação de instalação de dependências específicas
    print("\nInstalação recomendada de dependências para o Colab:")
    print("!pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/cu118") # Ajustar versão do CUDA se necessário
    print("!pip install torch-geometric==2.3.1")
    print("!pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html") # Ajustar versão do CUDA
    print("!pip install geopandas==0.13.2 networkx==3.1 matplotlib==3.7.2 seaborn==0.12.2")
    print("!pip install contextily==1.3.0 folium==0.14.0 rtree==1.0.1 scipy") # Adicionado scipy
    print("!pip install tqdm==4.66.1 plotly==5.15.0 scikit-learn==1.3.0 jsonschema==4.17.3")
    print("!pip install osmnx==1.5.1 momepy==0.6.0")
    print("!pip install fiona==1.9.5 numpy==1.24.3 --force-reinstall") # Reinstalar Fiona/Numpy pode ser necessário
    print("\nReinicie o runtime do Colab após estas instalações para evitar conflitos.")

except ImportError:
    IN_COLAB = False
    print("Ambiente local detectado")
    # Configuração para ambiente local (AJUSTE ESTES CAMINHOS)
    BASE_DIR = 'F:/TESE_MESTRADO' # Exemplo
    GEOPROCESSING_DIR = os.path.join(BASE_DIR, 'geoprocessing')
    DATA_DIR = os.path.join(GEOPROCESSING_DIR, 'data')
    ENRICHED_DATA_DIR = os.path.join(DATA_DIR, 'enriched_data')
    PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
    OUTPUT_DIR = os.path.join(GEOPROCESSING_DIR, 'outputs')
    QUALITY_REPORT_DIR = os.path.join(OUTPUT_DIR, 'quality_reports')

    # Usar arquivo mais recente no ambiente local
    if os.path.exists(ENRICHED_DATA_DIR):
        try:
            road_files = [f for f in os.listdir(ENRICHED_DATA_DIR) if f.startswith('roads_enriched_') and f.endswith('.gpkg')]
            if road_files:
                road_files.sort(reverse=True)
                ROADS_ENRICHED_PATH = os.path.join(ENRICHED_DATA_DIR, road_files[0])
                print(f"Usando arquivo de estradas enriquecido mais recente: {ROADS_ENRICHED_PATH}")
            else:
                ROADS_ENRICHED_PATH = os.path.join(DATA_DIR, 'roads_enriched.gpkg') # Fallback
                print(f"Warning: Nenhum arquivo 'roads_enriched_*.gpkg' encontrado em {ENRICHED_DATA_DIR}. Usando caminho padrão: {ROADS_ENRICHED_PATH}")
        except Exception as e:
             ROADS_ENRICHED_PATH = os.path.join(DATA_DIR, 'roads_enriched.gpkg') # Fallback geral
             print(f"Erro ao buscar arquivos em {ENRICHED_DATA_DIR}: {e}. Usando caminho padrão: {ROADS_ENRICHED_PATH}")
    else:
        ROADS_ENRICHED_PATH = os.path.join(DATA_DIR, 'roads_enriched.gpkg') # Fallback
        print(f"Warning: Diretório {ENRICHED_DATA_DIR} não encontrado. Usando caminho padrão: {ROADS_ENRICHED_PATH}")


# Configuração de timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Garantir que os diretórios existam
dirs_to_create = []
if IN_COLAB:
    dirs_to_create.extend([BASE_DIR, OUTPUT_DIR, QUALITY_REPORT_DIR, VISUALIZACOES_DIR, DATA_DIR])
else:
    dirs_to_create.extend([BASE_DIR, GEOPROCESSING_DIR, DATA_DIR, ENRICHED_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR, QUALITY_REPORT_DIR])
    dirs_to_create = sorted(list(set(dirs_to_create)))

for directory in dirs_to_create:
    if directory:
        try:
            os.makedirs(directory, exist_ok=True)
            # print(f"Diretório verificado/criado: {directory}") # Reduzido verbosity
        except OSError as e:
            print(f"Erro ao criar diretório {directory}: {e}")
            pass # Continuar, mas pode falhar depois

# Configuração de logging
log_handlers = [logging.StreamHandler()] # Sempre logar no console
if os.path.exists(OUTPUT_DIR):
    log_file_path = os.path.join(OUTPUT_DIR, f"pipeline_gnn_road_{timestamp}.log")
    log_handlers.insert(0, logging.FileHandler(log_file_path, encoding='utf-8')) # Adiciona log em arquivo se possível
else:
    print(f"Aviso: Diretório de saída {OUTPUT_DIR} não existe. Log será direcionado apenas para o console.")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=log_handlers
)
logger = logging.getLogger("TESE_MESTRADO.road_network_gnn")
logger.info("="*50)
logger.info("Inicializando pipeline GNN para análise de redes viárias (v2)")
logger.info(f"Ambiente: {'Google Colab' if IN_COLAB else 'Local'}")
logger.info(f"Arquivo de entrada principal: {ROADS_ENRICHED_PATH}")
logger.info("="*50)

# Configurar sementes para reprodutibilidade
seed = 42
np.random.seed(seed)
random.seed(seed)
logger.info(f"Sementes NumPy e Random configuradas: {seed}")
if HAS_TORCH:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        logger.info(f"CUDA disponível ({torch.cuda.get_device_name(0)}). Semente CUDA configurada.")
        # Para reprodutibilidade estrita (pode impactar performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    else:
        logger.info("CUDA não disponível.")
    logger.info("Semente PyTorch configurada.")

# Parâmetros base do modelo GNN (serão usados posteriormente)
model_params = {
    'input_dim': None,      # Será definido após preparação das features
    'hidden_dim': 64,
    'output_dim': None,     # Será definido com base nas classes dos nós
    'dropout': 0.3,
    'learning_rate': 0.01,
    'weight_decay': 0.0005,
    'early_stopping_patience': 20
}
logger.info(f"Parâmetros base do modelo GNN definidos: {model_params}")

# Mapeamento de highway para índice (importância: menor índice = mais importante)
# Ajuste conforme necessário para sua classificação
HIGHWAY_TO_IDX = {
    'motorway': 0, 'trunk': 1, 'primary': 2, 'secondary': 3, 'tertiary': 4,
    'residential': 5, 'unclassified': 5, # Agrupando residenciais e não classificadas
    'service': 6, 'living_street': 6, # Agrupando serviços
    'other': 7, 'track': 7, 'path': 7, 'footway': 7, 'cycleway': 7, 'steps': 7, 'pedestrian': 7, 'construction': 7,
    'unknown': 8 # Classe para tipos desconhecidos ou não mapeados
}
# Classe padrão (se nenhum tipo for encontrado ou para nós isolados antes da limpeza)
DEFAULT_NODE_CLASS = max(HIGHWAY_TO_IDX.values()) + 1
logger.info(f"Mapeamento Highway->Índice definido (usado para classe de nó): {HIGHWAY_TO_IDX}")
logger.info(f"Classe de nó padrão: {DEFAULT_NODE_CLASS}")

# --- Funções de Pré-processamento Geoespacial ---

def load_road_data(file_path=None, crs="EPSG:31983"):
    """ Carrega dados de rede viária enriquecidos. """
    if file_path is None:
        if 'ROADS_ENRICHED_PATH' not in globals() or not ROADS_ENRICHED_PATH:
             logger.error("Caminho do arquivo de estradas não fornecido e ROADS_ENRICHED_PATH global não está definido.")
             return None
        file_path = ROADS_ENRICHED_PATH

    logger.info(f"Tentando carregar dados de estradas de: {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"ERRO FATAL: Arquivo não encontrado: {file_path}")
        # Dicas adicionais
        if IN_COLAB:
            logger.error("No Colab, verifique se o Google Drive está montado corretamente e se o caminho/nome do arquivo está exato.")
        else:
            logger.error("No ambiente local, verifique o caminho completo do arquivo.")
        return None

    gdf = None
    try:
        layers = None
        if HAS_FIONA:
            try:
                layers = fiona.listlayers(file_path)
                logger.info(f"Camadas encontradas: {layers}")
            except Exception as e:
                logger.warning(f"Erro ao listar camadas com Fiona: {e}. Tentando ler sem especificar camada.")
        else:
             logger.warning("Fiona não disponível. Tentando ler a primeira camada ou sem especificar.")

        if layers:
            preferred_layer = next((l for l in layers if 'road' in l.lower() or 'via' in l.lower()), layers[0])
            logger.info(f"Tentando carregar camada: {preferred_layer}")
            gdf = gpd.read_file(file_path, layer=preferred_layer)
        else:
            logger.info("Tentando carregar arquivo sem especificar camada...")
            gdf = gpd.read_file(file_path)
        logger.info(f"Arquivo {os.path.basename(file_path)} carregado com sucesso.")

    except Exception as e:
        logger.exception(f"Erro fatal ao carregar o arquivo GPKG '{file_path}': {e}")
        return None

    if gdf.empty:
         logger.warning(f"GeoDataFrame carregado de '{file_path}' está vazio.")

    # Verificar e definir/reprojetar CRS
    try:
        target_epsg = int(crs.split(':')[-1])
        if gdf.crs is None:
            logger.warning(f"CRS não definido. Definindo como {crs}.")
            gdf.crs = crs
        elif gdf.crs.to_epsg() != target_epsg:
            logger.info(f"Reprojetando de {gdf.crs.name} ({gdf.crs.to_string()}) para {crs}")
            gdf = gdf.to_crs(crs)
        else:
             logger.info(f"CRS já está como {crs}.")
    except Exception as e:
        logger.error(f"Erro ao verificar/definir CRS para {crs}: {e}. Verifique o código EPSG.")
        logger.warning(f"Continuando com o CRS original: {gdf.crs}")

    # Verificar colunas essenciais
    essential_cols = ['geometry', 'highway']
    missing_cols = [col for col in essential_cols if col not in gdf.columns]
    if missing_cols:
        if 'geometry' in missing_cols:
             logger.error("ERRO FATAL: Coluna 'geometry' ausente. Impossível continuar.")
             return None
        logger.warning(f"Coluna(s) essencial(is) ausente(s): {missing_cols}. Funcionalidades podem ser limitadas.")

    # Criar índice espacial
    if not gdf.has_sindex and not gdf.empty:
        logger.info("Criando índice espacial (sindex)...")
        try:
            gdf.sindex
            logger.info("Índice espacial criado.")
        except Exception as e:
            logger.error(f"Erro ao criar índice espacial: {e}")

    logger.info(f"Carregamento finalizado: {len(gdf)} feições.")
    return gdf

def explode_multilines_improved(gdf):
    """ Processa MultiLineStrings usando GeoPandas explode. """
    if gdf.empty: return gdf
    logger.info("--- Processando MultiLineStrings ---")
    initial_len = len(gdf)
    geom_types = gdf.geometry.type.unique()
    logger.info(f"Tipos de geometria iniciais: {geom_types}")

    if 'MultiLineString' not in geom_types:
        logger.info("Nenhuma MultiLineString encontrada. Nenhuma explosão necessária.")
        return gdf

    multi_mask = gdf.geometry.type == "MultiLineString"
    multi_count = multi_mask.sum()
    logger.info(f"Encontradas {multi_count} MultiLineStrings para explodir.")

    try:
        gdf_exploded = gdf.explode(index_parts=False) # index_parts=False é geralmente mais simples
        exploded_count = len(gdf_exploded)
        logger.info(f"GeoDataFrame explodido. Total de feições agora: {exploded_count} (antes: {initial_len})")

        # Verificar tipos resultantes e filtrar se necessário
        final_types = gdf_exploded.geometry.type.unique()
        logger.info(f"Tipos de geometria após explosão: {final_types}")
        if not all(t == 'LineString' for t in final_types if t is not None):
             logger.warning("Tipos não-LineString encontrados após explosão. Filtrando para manter apenas LineStrings.")
             gdf_exploded = gdf_exploded[gdf_exploded.geometry.type == 'LineString'].copy()
             logger.info(f"Feições após filtrar por LineString: {len(gdf_exploded)}")

        # Resetar índice
        gdf_exploded = gdf_exploded.reset_index(drop=True)
        return gdf_exploded

    except Exception as e:
        logger.exception("Erro durante gdf.explode(). Retornando GeoDataFrame original.")
        return gdf # Retorna original em caso de erro

def calculate_sinuosity(gdf):
    """ Calcula sinuosidade (comprimento / distância reta entre endpoints). """
    logger.info("Calculando sinuosidade...")
    if gdf.empty or 'geometry' not in gdf.columns:
        return pd.Series(dtype=float)

    geometries = gdf.geometry
    sinuosities = np.full(len(gdf), np.nan) # Inicializa com NaN
    
    # Certifique-se de que valid_mask e valid_geoms estão corretos
    valid_mask = (geometries.notna()) & (geometries.geom_type == 'LineString')
    if valid_mask.sum() == 0:
        logger.warning("Nenhuma geometria LineString válida encontrada para calcular sinuosidade")
        return pd.Series(sinuosities, index=gdf.index)
        
    valid_geoms = geometries[valid_mask]

    if not valid_geoms.empty:
        line_lengths = valid_geoms.length
        try:
            # Extrai coordenadas de início e fim
            coords = valid_geoms.apply(lambda g: (g.coords[0], g.coords[-1]) if len(g.coords) >= 2 else (None, None))
            start_points = gpd.GeoSeries(coords.apply(lambda x: Point(x[0]) if x and x[0] else None), crs=gdf.crs)
            end_points = gpd.GeoSeries(coords.apply(lambda x: Point(x[1]) if x and x[1] else None), crs=gdf.crs)
            
            # Verifica se start_points e end_points não são None antes de calcular distância
            valid_dist_mask = start_points.notna() & end_points.notna()
            if valid_dist_mask.sum() == 0:
                logger.warning("Nenhum par de pontos válidos para calcular sinuosidade")
                return pd.Series(sinuosities, index=gdf.index)
                
            straight_distances = pd.Series(np.nan, index=valid_geoms.index) # Inicializa com NaN
            straight_distances[valid_dist_mask] = start_points[valid_dist_mask].distance(end_points[valid_dist_mask])

            # Calcular sinuosidade onde a distância reta é > 0
            mask_calc = (straight_distances > 1e-9) & (line_lengths > 1e-9) & valid_dist_mask
            
            if mask_calc.sum() > 0:
                # Verificar limites de índices para evitar IndexError
                valid_idx = valid_mask.loc[mask_calc].index
                # Garanta que todos os índices estão dentro dos limites
                valid_idx = [idx for idx in valid_idx if idx < len(sinuosities)]
                if valid_idx:
                    sinuosities[valid_idx] = (line_lengths[mask_calc] / straight_distances[mask_calc]).values

                # Sinuosidade é 1 para linhas retas ou muito curtas
                mask_straight = ((straight_distances <= 1e-9) | (line_lengths <= 1e-9)) & valid_dist_mask
                valid_idx_straight = valid_mask.loc[mask_straight].index
                valid_idx_straight = [idx for idx in valid_idx_straight if idx < len(sinuosities)]
                if valid_idx_straight:
                    sinuosities[valid_idx_straight] = 1.0

                # Lidar com casos onde straight_distance é quase zero, mas length não é (loops?) -> Infinito teoricamente
                mask_loops = (straight_distances <= 1e-9) & (line_lengths > 1e-9) & valid_dist_mask
                valid_idx_loops = valid_mask.loc[mask_loops].index
                valid_idx_loops = [idx for idx in valid_idx_loops if idx < len(sinuosities)]
                if valid_idx_loops:
                    sinuosities[valid_idx_loops] = np.inf # Ou um valor grande, e.g., 999
            else:
                logger.warning("Nenhum segmento válido para calcular sinuosidade (após filtros)")

        except Exception as e:
             logger.error(f"Erro durante cálculo vetorizado de sinuosidade: {e}. Retornando NaNs.")
             # sinuosities já está como NaN por padrão

    sinuosity_series = pd.Series(sinuosities, index=gdf.index)
    valid_count = sinuosity_series.notna().sum()
    if valid_count > 0:
        logger.info(f"Sinuosidade calculada para {valid_count} segmentos.")
    else:
        logger.warning("Não foi possível calcular sinuosidade para nenhum segmento.")
    return sinuosity_series

def clean_road_data(gdf):
    """ Limpa e valida dados de estradas: remove nulos/inválidos/duplicados, padroniza 'highway', calcula 'length_m', 'sinuosity', 'edge_id'. """
    if gdf.empty: return gdf
    logger.info("--- Limpando e Validando Dados Geoespaciais ---")
    initial_count = len(gdf)
    logger.info(f"Contagem inicial: {initial_count} feições")

    # 1. Remover geometrias nulas ou vazias
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if len(gdf) < initial_count:
        logger.warning(f"Removidas {initial_count - len(gdf)} geometrias nulas/vazias.")
        if gdf.empty: logger.warning("GDF vazio após remover nulos."); return gdf

    # 2. Corrigir geometrias inválidas
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        logger.warning(f"Encontradas {invalid_mask.sum()} geometrias inválidas. Tentando corrigir com buffer(0)...")
        # Usar .loc para evitar SettingWithCopyWarning
        gdf.loc[invalid_mask, 'geometry'] = gdf.loc[invalid_mask, 'geometry'].buffer(0)
        # Verificar novamente após correção
        still_invalid_mask = ~gdf.geometry.is_valid & invalid_mask # Verificar apenas as que eram inválidas
        if still_invalid_mask.any():
            logger.warning(f"Removendo {still_invalid_mask.sum()} geometrias ainda inválidas após correção.")
            gdf = gdf[~still_invalid_mask].copy()
            if gdf.empty: logger.warning("GDF vazio após remover inválidos."); return gdf

    # 3. Remover duplicatas geométricas (usando WKT para performance)
    logger.info("Removendo duplicatas geométricas...")
    before_dedup = len(gdf)
    try:
        # Usar representação WKB (Well-Known Binary) que pode ser mais robusta que WKT
        gdf['wkb_temp'] = gdf.geometry.apply(lambda g: g.wkb)
        gdf = gdf.drop_duplicates(subset=['wkb_temp']).drop(columns=['wkb_temp']).copy()
        logger.info(f"Removidas {before_dedup - len(gdf)} geometrias duplicadas (usando WKB).")
    except Exception as e:
        logger.warning(f"Erro ao usar WKB para drop_duplicates ({e}). Tentando com objeto geometry (mais lento)...")
        try:
            gdf = gdf.drop_duplicates(subset=['geometry']).copy()
            logger.info(f"Removidas {before_dedup - len(gdf)} geometrias duplicadas (método fallback).")
        except Exception as e2:
            logger.error(f"Erro fatal ao remover duplicatas geométricas: {e2}. Retornando GDF atual.")

    if gdf.empty: logger.warning("GDF vazio após remover duplicatas."); return gdf

    # 4. Padronizar 'highway' e criar 'road_category'
    if 'highway' in gdf.columns:
        logger.info("Padronizando 'highway' e criando 'road_category'...")
        # Garantir que 'highway' seja string antes de operar
        gdf['highway'] = gdf['highway'].astype(str).str.lower().str.strip().replace(['nan', 'none', '', '<na>', 'null'], 'unclassified')
        gdf.loc[gdf['highway'].isna(), 'highway'] = 'unclassified' # Tratar NaNs que podem surgir

        # Usar o mapeamento global HIGHWAY_TO_IDX para consistência
        # Criar um mapeamento reverso de índice para nome de categoria principal (se necessário)
        # Ou definir um mapeamento direto para 'road_category' baseado na importância/agrupamento
        category_mapping = {
             'motorway': 'motorway', 'trunk': 'trunk', 'primary': 'primary', 'secondary': 'secondary', 'tertiary': 'tertiary',
             'residential': 'residential', 'unclassified': 'residential', 'living_street': 'residential', 'service': 'residential', 'road': 'residential', # Agrupado
             'other': 'other', 'track': 'other', 'path': 'other', 'footway': 'other', 'cycleway': 'other', 'steps': 'other', 'pedestrian': 'other', 'construction': 'other',
             'unknown': 'unknown' # Manter desconhecido
        }
        # Aplicar mapeamento para 'road_category'
        gdf['road_category'] = gdf['highway'].map(lambda x: category_mapping.get(x, 'unknown'))
        logger.info(f"Distribuição de 'road_category':\n{gdf['road_category'].value_counts()}")
    else:
        logger.warning("Coluna 'highway' não encontrada. 'road_category' não criada.")
        gdf['road_category'] = 'unknown' # Criar coluna padrão

    # 5. Calcular 'length_m' e 'sinuosity'
    logger.info("Calculando 'length_m'...")
    gdf['length_m'] = gdf.geometry.length
    gdf['sinuosity'] = calculate_sinuosity(gdf) # Já loga internamente

    # 6. Garantir 'edge_id' único (importante para referenciar arestas depois)
    gdf = gdf.reset_index(drop=True)
    gdf['edge_id'] = gdf.index
    logger.info("Coluna 'edge_id' criada/atualizada.")

    final_count = len(gdf)
    retention = (final_count / initial_count * 100) if initial_count > 0 else 0
    logger.info(f"Limpeza geoespacial concluída. Feições finais: {final_count} ({retention:.1f}% das iniciais)")
    return gdf

def improve_topology(gdf, tolerance=1.0):
    """ Tenta corrigir topologia conectando endpoints próximos (snapping). """
    logger.info(f"--- Aplicando Correções Topológicas (Snapping, Tolerância: {tolerance}m) ---")
    if gdf.empty or 'geometry' not in gdf.columns: return gdf.copy()
    if tolerance <= 0:
        logger.info("Tolerância zero ou negativa, snapping ignorado.")
        return gdf.copy()

    gdf_corrected = gdf.copy()
    # Criar GeoDataFrame de endpoints
    endpoints_list = []
    for idx, row in gdf_corrected.iterrows():
        geom = row.geometry
        if geom and geom.geom_type == 'LineString' and len(geom.coords) >= 2:
            start_pt = Point(geom.coords[0])
            end_pt = Point(geom.coords[-1])
            # Usar edge_id se existir, senão o índice do GDF
            segment_ref_id = row.get('edge_id', idx)
            endpoints_list.append({'geometry': start_pt, 'segment_id': segment_ref_id, 'position': 'start', 'original_index': idx})
            endpoints_list.append({'geometry': end_pt, 'segment_id': segment_ref_id, 'position': 'end', 'original_index': idx})

    if not endpoints_list:
        logger.warning("Nenhum endpoint válido encontrado para snapping.")
        return gdf_corrected

    endpoints_gdf = gpd.GeoDataFrame(endpoints_list, crs=gdf_corrected.crs)
    if not endpoints_gdf.has_sindex:
        logger.info("Criando índice espacial para endpoints...")
        endpoints_gdf.sindex

    logger.info(f"Analisando {len(endpoints_gdf)} endpoints para snapping...")
    modifications = {} # Dicionário para armazenar modificações: {original_index: {'start': new_coord, 'end': new_coord}}
    processed_indices = set() # Índices do endpoints_gdf já processados
    snap_count = 0

    # Usar KDTree para busca eficiente de vizinhos próximos
    try:
        coords_array = np.array([(p.x, p.y) for p in endpoints_gdf.geometry])
        kdtree = KDTree(coords_array)
        # Encontrar pares dentro da tolerância
        pairs = kdtree.query_pairs(r=tolerance)
        logger.info(f"Encontrados {len(pairs)} pares de endpoints dentro da tolerância {tolerance}m.")

        # Agrupar pontos próximos usando Union-Find (Disjoint Set Union)
        parent = list(range(len(endpoints_gdf)))
        def find_set(v):
            if v == parent[v]: return v
            parent[v] = find_set(parent[v])
            return parent[v]
        def unite_sets(a, b):
            a = find_set(a)
            b = find_set(b)
            if a != b: parent[b] = a

        for i, j in pairs:
            unite_sets(i, j)

        # Determinar o ponto alvo (centroide) para cada grupo de pontos a serem unidos
        clusters = {}
        for i in range(len(endpoints_gdf)):
            root = find_set(i)
            if root not in clusters: clusters[root] = []
            clusters[root].append(i)

        target_points = {} # cluster_root_id -> target_coordinate
        for root, indices in clusters.items():
            if len(indices) > 1: # Apenas clusters com mais de um ponto precisam de snapping
                cluster_points = endpoints_gdf.iloc[indices]
                # Calcular o centroide geométrico do cluster
                centroid = cluster_points.unary_union.centroid
                target_coord = (centroid.x, centroid.y)
                target_points[root] = target_coord
                # Marcar modificações para todos os pontos no cluster
                for idx in indices:
                    endpoint_info = endpoints_gdf.iloc[idx]
                    original_gdf_idx = endpoint_info['original_index']
                    position = endpoint_info['position']
                    if original_gdf_idx not in modifications: modifications[original_gdf_idx] = {}
                    # Só marca se a coordenada for diferente (evita snaps desnecessários)
                    current_coord = tuple(endpoint_info.geometry.coords[0])
                    if current_coord != target_coord:
                        if position not in modifications[original_gdf_idx] or modifications[original_gdf_idx][position] != target_coord:
                             modifications[original_gdf_idx][position] = target_coord
                             snap_count += 1 # Contar modificações reais

    except ImportError:
        logger.warning("Scipy (KDTree) não encontrado. Snapping será mais lento usando busca espacial do GeoPandas.")
        # Implementação Fallback (mais lenta) sem KDTree - omitida para brevidade, mas seguiria lógica similar com sindex.query

    except Exception as e:
        logger.error(f"Erro durante snapping com KDTree: {e}. Tentando continuar sem snapping.")
        return gdf_corrected # Retornar sem modificar em caso de erro inesperado

    logger.info(f"Aplicando {snap_count} modificações de snapping em {len(modifications)} segmentos...")
    geom_col = gdf_corrected.geometry.name # Nome da coluna de geometria

    # Aplicar modificações ao GeoDataFrame original
    indices_to_update = list(modifications.keys())
    geoms_to_update = []

    for original_idx in indices_to_update:
        try:
            original_geom = gdf_corrected.loc[original_idx, geom_col]
            if original_geom is None or original_geom.geom_type != 'LineString': continue

            coords = list(original_geom.coords)
            mod_info = modifications[original_idx]
            modified = False

            if 'start' in mod_info and tuple(coords[0]) != mod_info['start']:
                coords[0] = mod_info['start']
                modified = True
            if 'end' in mod_info and tuple(coords[-1]) != mod_info['end']:
                coords[-1] = mod_info['end']
                modified = True

            if modified:
                if len(coords) >= 2:
                    # Verificar se início e fim não colapsaram para o mesmo ponto
                    if Point(coords[0]).distance(Point(coords[-1])) > 1e-9:
                         geoms_to_update.append(LineString(coords))
                    else:
                         logger.warning(f"Snapping para segmento com índice {original_idx} (edge_id={gdf_corrected.loc[original_idx].get('edge_id', 'N/A')}) resultou em geometria degenerada (start=end). Segmento será removido.")
                         geoms_to_update.append(None) # Marcar para remoção
                else:
                    logger.warning(f"Snapping para segmento {original_idx} resultou em < 2 coordenadas. Segmento será removido.")
                    geoms_to_update.append(None) # Marcar para remoção
            else:
                 geoms_to_update.append(original_geom) # Manter original se não modificado

        except Exception as e:
            logger.error(f"Erro ao processar snapping para segmento com índice {original_idx}: {e}")
            geoms_to_update.append(gdf_corrected.loc[original_idx, geom_col]) # Manter original em caso de erro

    # Atualizar geometrias em lote
    gdf_corrected.loc[indices_to_update, geom_col] = geoms_to_update

    # Remover segmentos que se tornaram None ou inválidos
    initial_len = len(gdf_corrected)
    gdf_corrected = gdf_corrected[gdf_corrected[geom_col].notna()].copy()
    if len(gdf_corrected) < initial_len:
         logger.warning(f"Removidos {initial_len - len(gdf_corrected)} segmentos devido a snapping degenerado.")

    logger.info("Correções topológicas (snapping) aplicadas.")
    return gdf_corrected

def enrich_edge_features(gdf):
    """ Enriquece arestas com features morfológicas (curvatura, densidade de pontos, orientação, categorias). """
    logger.info("--- Enriquecendo Features das Arestas (Morfológicas) ---")
    if gdf.empty: return gdf.copy()
    gdf_enriched = gdf.copy()

    # --- Curvature (ângulo total / comprimento) ---
    curvatures = []
    for geom in gdf_enriched.geometry:
        total_angle, length, curvature = 0.0, 0.0, 0.0
        if geom and geom.geom_type == 'LineString':
            coords = list(geom.coords)
            length = geom.length
            if len(coords) >= 3 and length > 1e-6:
                for i in range(len(coords) - 2):
                    p1, p2, p3 = coords[i:i+3]
                    v1 = (p2[0]-p1[0], p2[1]-p1[1])
                    v2 = (p3[0]-p2[0], p3[1]-p2[1])
                    mag_v1 = (v1[0]**2 + v1[1]**2)**0.5
                    mag_v2 = (v2[0]**2 + v2[1]**2)**0.5
                    if mag_v1 > 1e-9 and mag_v2 > 1e-9:
                        dot = v1[0]*v2[0] + v1[1]*v2[1]
                        # Clamp dot product para evitar erros de domínio no arccos
                        cos_angle = max(-1.0, min(1.0, dot / (mag_v1 * mag_v2)))
                        total_angle += np.arccos(cos_angle) # Ângulo em radianos
                curvature = total_angle / length if length > 1e-6 else 0.0
        curvatures.append(curvature)
    gdf_enriched['curvature'] = curvatures
    logger.info("Calculada feature 'curvature'.")

    # --- Point Density (pontos / comprimento) ---
    gdf_enriched['point_density'] = gdf_enriched.geometry.apply(
        lambda g: len(list(g.coords))/g.length if g and g.geom_type=='LineString' and g.length > 1e-6 else 0
    )
    logger.info("Calculada feature 'point_density'.")

    # --- Bearing (orientação geral da linha: 0-360 graus, Norte=0) ---
    bearings = []
    for geom in gdf_enriched.geometry:
        bearing = 0.0 # Padrão para geometrias inválidas ou pontos
        if geom and geom.geom_type == 'LineString' and len(geom.coords) >= 2:
            start, end = geom.coords[0], geom.coords[-1]
            dx, dy = end[0]-start[0], end[1]-start[1]
            # Evitar divisão por zero ou atan2(0,0)
            if abs(dx) > 1e-9 or abs(dy) > 1e-9:
                # atan2(dx, dy) dá o ângulo em radianos com o eixo Y (Norte)
                angle_rad = np.arctan2(dx, dy)
                # Converter para graus e ajustar para 0-360
                bearing = (np.degrees(angle_rad) + 360) % 360
        bearings.append(bearing)
    gdf_enriched['bearing'] = bearings
    logger.info("Calculada feature 'bearing'.")

    # --- Categorias (baseadas em features existentes) ---
    # Length Category
    if 'length_m' in gdf_enriched.columns:
        bins = [-np.inf, 50, 100, 250, 500, 1000, np.inf]
        labels = ['muito_curto', 'curto', 'medio_curto', 'medio', 'longo', 'muito_longo']
        gdf_enriched['length_category'] = pd.cut(gdf_enriched['length_m'], bins=bins, labels=labels, right=False).astype(str).fillna('desconhecido')
        logger.info("Criada feature 'length_category'.")

    # Sinuosity Category
    if 'sinuosity' in gdf_enriched.columns:
        # Ajustar bins para incluir 1.0 na categoria 'reta' e tratar np.inf
        bins = [0, 1.001, 1.05, 1.15, 1.3, 1.5, np.inf] # Bin inicial 0, ligeiramente acima de 1 para 'reta'
        labels = ['reta', 'quase_reta', 'pouco_sinuosa', 'sinuosa', 'muito_sinuosa', 'loop_ou_erro']
        # Usar fillna antes de cut para tratar NaNs explicitamente se houver
        gdf_enriched['sinuosity_category'] = pd.cut(gdf_enriched['sinuosity'].fillna(1.0), bins=bins, labels=labels, right=False).astype(str).fillna('desconhecido')
        logger.info("Criada feature 'sinuosity_category'.")

    # Slope Category (se slope_pct existir, exemplo)
    if 'slope_pct' in gdf_enriched.columns:
        bins = [-np.inf, 3, 8, 15, 30, np.inf]
        labels = ['plana', 'suave', 'moderada', 'acentuada', 'muito_acentuada']
        # Usar valor absoluto da inclinação
        gdf_enriched['slope_category'] = pd.cut(gdf_enriched['slope_pct'].abs().fillna(0), bins=bins, labels=labels, right=False).astype(str).fillna('desconhecido')
        logger.info("Criada feature 'slope_category'.")
    else:
        logger.info("Coluna 'slope_pct' não encontrada, 'slope_category' não criada.")

    logger.info("Enriquecimento morfológico concluído.")
    return gdf_enriched

def integrate_contextual_data(gdf_roads, context_data):
    """
    (PLACEHOLDER) Integra dados contextuais (uso do solo, edificações, etc.) com as estradas.
    Esta função precisa ser implementada com a lógica específica de integração
    (e.g., buffer, spatial join, agregação).
    """
    logger.warning("--- Função 'integrate_contextual_data' é um placeholder ---")
    logger.warning("--- Nenhuma integração contextual foi realizada ---")
    # Exemplo básico (não funcional sem implementação):
    # if 'landuse' in context_data:
    #     # Ex: Fazer um spatial join para obter o uso do solo predominante perto da via
    #     gdf_roads = gpd.sjoin_nearest(gdf_roads, context_data['landuse'], how='left', max_distance=50) # Exemplo
    #     # Agregar ou selecionar a informação relevante
    #     # gdf_roads['predominant_landuse'] = ...
    # if 'buildings' in context_data:
    #     # Ex: Contar edificações num buffer ao redor da via
    #     # buffer = gdf_roads.geometry.buffer(20) # Exemplo
    #     # joined = gpd.sjoin(context_data['buildings'], gpd.GeoDataFrame(geometry=buffer), how='inner', predicate='within')
    #     # counts = joined.groupby(joined.index_right).size()
    #     # gdf_roads['building_count'] = counts
    #     # gdf_roads['building_count'] = gdf_roads['building_count'].fillna(0)
    # ... etc para outros dados contextuais ...

    # Apenas retorna o GDF original por enquanto
    return gdf_roads

# --- Funções de Construção e Preparação do Grafo ---

def create_road_graph(gdf):
    """
    Cria um grafo NetworkX a partir de um GeoDataFrame de estradas processado.
    Nós representam endpoints/interseções, arestas representam segmentos.
    """
    logger.info("--- Construindo Grafo da Rede Viária (NetworkX) ---")
    if gdf.empty or 'geometry' not in gdf.columns:
        logger.error("GeoDataFrame vazio ou sem coluna 'geometry'. Impossível criar grafo.")
        return nx.Graph() # Retorna grafo vazio

    G = nx.Graph()
    node_coords_map = {} # Mapeia coordenada (tupla) para ID do nó no grafo
    node_counter = 0
    edge_count = 0

    logger.info(f"Processando {len(gdf)} segmentos de estrada para criar o grafo...")
    for idx, row in gdf.iterrows():
        geom = row.geometry
        # Garantir que é LineString e tem pelo menos 2 pontos
        if geom is None or geom.geom_type != 'LineString' or len(geom.coords) < 2:
            # logger.debug(f"Ignorando geometria inválida ou não-LineString no índice {idx}")
            continue
        
        # Obter coordenadas de início e fim como tuplas
        start_coord = tuple(geom.coords[0])
        end_coord = tuple(geom.coords[-1])

        # Adicionar/obter nós para start/end points
        node_ids = []
        for coord in [start_coord, end_coord]:
            if coord not in node_coords_map:
                node_id = node_counter
                node_coords_map[coord] = node_id
                # Adicionar nó com coordenadas e índice original do GDF (pode ser útil)
                G.add_node(node_id, x=coord[0], y=coord[1], original_indices=set([idx]))
                node_counter += 1
            else:
                node_id = node_coords_map[coord]
                # Adicionar índice do GDF ao conjunto de índices originais do nó existente
                G.nodes[node_id]['original_indices'].add(idx)
            node_ids.append(node_id)

        start_node_id, end_node_id = node_ids

        # Adicionar aresta apenas se os nós de início e fim forem diferentes
        # e se a aresta (ou sua reversa) ainda não existir
        if start_node_id != end_node_id and not G.has_edge(start_node_id, end_node_id):
            # Coletar atributos da linha do GDF para a aresta
            # Excluir geometria e outros que não fazem sentido como attr da aresta
            excluded_cols = ['geometry', 'wkb_temp'] # Adicionar outros se necessário
            attributes = {k: v for k, v in row.items() if k not in excluded_cols}
            # Garantir que 'length_m' (ou 'length') esteja presente
            if 'length_m' not in attributes and 'length' not in attributes:
                attributes['length'] = geom.length # Calcular se não existir
            elif 'length_m' in attributes:
                 attributes['length'] = attributes['length_m'] # Padronizar para 'length'
            # Adicionar referência ao índice original do GDF (edge_id)
            attributes['original_edge_id'] = row.get('edge_id', idx)

            G.add_edge(start_node_id, end_node_id, **attributes)
            edge_count += 1

    logger.info(f"Grafo inicial construído com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas.")

    # --- Pós-processamento do Grafo ---
    # 1. Remover nós isolados (sem conexões)
    isolated_nodes = list(nx.isolates(G))
    if isolated_nodes:
        logger.info(f"Removendo {len(isolated_nodes)} nós isolados...")
        G.remove_nodes_from(isolated_nodes)
        logger.info(f"Grafo após remover isolados: {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")

    # 2. Verificar conectividade e manter apenas o maior componente conectado
    if G.number_of_nodes() > 0 and not nx.is_connected(G):
        num_components = nx.number_connected_components(G)
        logger.warning(f"Grafo não é conectado. Possui {num_components} componentes.")
        largest_cc = max(nx.connected_components(G), key=len)
        nodes_in_lcc = len(largest_cc)
        logger.info(f"Mantendo apenas o maior componente conectado com {nodes_in_lcc} nós ({nodes_in_lcc / G.number_of_nodes():.1%}).")
        # Criar um subgrafo a partir do maior componente
        G = G.subgraph(largest_cc).copy() # .copy() é importante para obter um grafo modificável
        logger.info(f"Grafo final (maior componente): {G.number_of_nodes()} nós, {G.number_of_edges()} arestas.")
    elif G.number_of_nodes() == 0:
         logger.warning("Grafo final está vazio após processamento.")
    else:
         logger.info("Grafo é conectado.")

    return G

def calculate_centrality_metrics(G):
    """ Calcula métricas de centralidade (Betweenness, Closeness) e adiciona como atributos dos nós. """
    logger.info("--- Calculando Métricas de Centralidade ---")
    if G.number_of_nodes() == 0:
        logger.warning("Grafo vazio, impossível calcular centralidade.")
        return G

    # Calcular grau e adicionar como atributo
    logger.info("Calculando grau dos nós...")
    node_degrees = dict(G.degree())
    nx.set_node_attributes(G, node_degrees, 'degree')
    logger.info(f"Grau médio: {np.mean(list(node_degrees.values())):.2f}")

    # Calcular Betweenness Centrality (importância como ponte)
    # Usar 'length' como peso, se disponível, para centralidade baseada em distância
    weight = 'length' if nx.get_edge_attributes(G, 'length') else None
    logger.info(f"Calculando Betweenness Centrality (peso='{weight}')...")
    try:
        # k=None usa todos os nós, pode ser lento. Considerar amostrar (e.g., k=int(G.number_of_nodes()*0.1)) para grafos grandes
        bc = nx.betweenness_centrality(G, weight=weight, normalized=True) # Normalizado para [0,1]
        nx.set_node_attributes(G, bc, 'betweenness')
        logger.info("Betweenness Centrality calculada.")
        # logger.debug(f"Betweenness (amostra): {dict(list(bc.items())[:5])}")
    except Exception as e:
        logger.error(f"Erro ao calcular Betweenness Centrality: {e}")

    # Calcular Closeness Centrality (proximidade aos outros nós)
    logger.info(f"Calculando Closeness Centrality (distância='{weight}')...")
    try:
        # Usa distância geodésica (menor caminho)
        cc = nx.closeness_centrality(G, distance=weight)
        nx.set_node_attributes(G, cc, 'closeness')
        logger.info("Closeness Centrality calculada.")
        # logger.debug(f"Closeness (amostra): {dict(list(cc.items())[:5])}")
    except Exception as e:
        logger.error(f"Erro ao calcular Closeness Centrality: {e}")

    # (Opcional) Calcular Eigenvector Centrality (influência no grafo)
    # logger.info(f"Calculando Eigenvector Centrality (peso='{weight}')...")
    # try:
    #     ec = nx.eigenvector_centrality_numpy(G, weight=weight) # Requer scipy
    #     nx.set_node_attributes(G, ec, 'eigenvector')
    #     logger.info("Eigenvector Centrality calculada.")
    # except Exception as e:
    #     logger.error(f"Erro ao calcular Eigenvector Centrality: {e}")

    return G

def assign_node_classes(G, highway_to_idx_map, default_class):
    """
    Atribui classes aos nós com base nos tipos de estradas ('highway') conectadas.
    A classe é determinada pelo tipo de estrada mais importante (menor índice no mapa) conectado ao nó.
    """
    logger.info("--- Atribuindo Classes aos Nós ---")
    if G.number_of_nodes() == 0:
        logger.warning("Grafo vazio, impossível atribuir classes.")
        return G

    node_classes = {}
    for node in G.nodes():
        connected_highway_indices = []
        # Iterar sobre as arestas conectadas a este nó
        for neighbor in G.neighbors(node):
            edge_data = G.get_edge_data(node, neighbor)
            if edge_data and 'highway' in edge_data:
                highway_type = str(edge_data['highway']).lower().strip()
                # Obter o índice correspondente do mapa, ou um valor alto se não encontrado
                idx = highway_to_idx_map.get(highway_type, default_class)
                connected_highway_indices.append(idx)

        # Determinar a classe do nó: o menor índice (tipo mais importante)
        if connected_highway_indices:
            node_class = min(connected_highway_indices)
        else:
            # Se o nó não tem arestas com 'highway' (pode acontecer se for isolado antes da limpeza ou erro)
            node_class = default_class
        node_classes[node] = node_class

    # Atribuir as classes calculadas aos nós do grafo
    nx.set_node_attributes(G, node_classes, 'class')

    # Contar e logar distribuição de classes
    class_counts = pd.Series(node_classes).value_counts().sort_index()
    logger.info("Distribuição de classes de nós atribuídas:")
    # Criar mapa reverso de índice para nome para log
    idx_to_highway_name = {v: k for k, v in highway_to_idx_map.items()}
    for class_id, count in class_counts.items():
         class_name = idx_to_highway_name.get(class_id, f"Classe_{class_id}")
         logger.info(f"  {class_name} (ID: {class_id}): {count} nós")

    # Definir o número de classes de saída para o modelo GNN
    model_params['output_dim'] = len(class_counts) # Ou max(node_classes.values()) + 1
    logger.info(f"Parâmetro 'output_dim' do modelo definido como: {model_params['output_dim']}")

        return G
    
def optimize_graph(G):
    """
    Otimiza o grafo removendo nós de grau 2 (passagem) e mesclando suas arestas.
    Preserva atributos somando 'length' e mantendo outros da primeira aresta encontrada.
    """
    logger.info("--- Otimizando Grafo (Removendo Nós de Grau 2) ---")
    if G.number_of_nodes() == 0:
        logger.warning("Grafo vazio, nenhuma otimização necessária.")
        return G

    G_optimized = G.copy() # Trabalhar em uma cópia
    nodes_to_remove = []
    edges_to_add = [] # Lista de tuplas (u, v, attributes)

    # Iterar sobre nós com grau 2
    degree_2_nodes = [node for node, degree in dict(G_optimized.degree()).items() if degree == 2]
    num_degree_2 = len(degree_2_nodes)
    logger.info(f"Encontrados {num_degree_2} nós de grau 2 para possível remoção.")

    processed_nodes = set() # Evitar processar o mesmo nó múltiplas vezes em cadeias

    for node in degree_2_nodes:
        if node in processed_nodes:
            continue
            
        # Verificar se o grau ainda é 2 (pode ter mudado em iteração anterior)
        if G_optimized.degree(node) != 2:
            continue

        neighbors = list(G_optimized.neighbors(node))
        if len(neighbors) != 2: # Segurança extra
            continue
        
        n1, n2 = neighbors
        # Obter dados das duas arestas conectadas ao nó de grau 2
        edge1_data = G_optimized.get_edge_data(n1, node)
        edge2_data = G_optimized.get_edge_data(node, n2)
        
        if not edge1_data or not edge2_data:
             logger.warning(f"Dados de aresta ausentes para nó de grau 2: {node}. Ignorando.")
            continue
            
        # Criar atributos para a nova aresta mesclada
        new_attrs = {}
        # Usar atributos da primeira aresta como base e atualizar/somar
        new_attrs.update(edge1_data)
                # Somar comprimentos
        new_attrs['length'] = edge1_data.get('length', 0) + edge2_data.get('length', 0)
        # Opcional: Lidar com outros atributos (ex: manter o mais restritivo, concatenar, etc.)
        # Aqui, simplesmente mantemos os da edge1, exceto pelo length.
        # Poderia adicionar lógica mais complexa se necessário.
        # Ex: new_attrs['highway'] = min(edge1_data.get('highway'), edge2_data.get('highway')) # Se highway for numérico/ordinal

        # Marcar nó para remoção e aresta para adição
        nodes_to_remove.append(node)
        edges_to_add.append((n1, n2, new_attrs))
        processed_nodes.add(node)
        # Marcar vizinhos como processados se eles também forem de grau 2, para evitar processamento duplo na mesma cadeia
        if G_optimized.degree(n1) == 2: processed_nodes.add(n1)
        if G_optimized.degree(n2) == 2: processed_nodes.add(n2)


    # Aplicar modificações ao grafo
    if nodes_to_remove:
        logger.info(f"Removendo {len(nodes_to_remove)} nós de grau 2 e adicionando {len(edges_to_add)} arestas mescladas.")
        G_optimized.remove_nodes_from(nodes_to_remove)
        # Adicionar novas arestas com cuidado para não sobrescrever existentes se n1 e n2 já estavam conectados
        for u, v, attrs in edges_to_add:
            if not G_optimized.has_edge(u,v): # Só adiciona se não existir
                 G_optimized.add_edge(u, v, **attrs)
            else:
                 logger.warning(f"Aresta entre {u} e {v} já existia após remover nó de grau 2. Atributos não mesclados.")

        logger.info(f"Grafo otimizado: {G_optimized.number_of_nodes()} nós, {G_optimized.number_of_edges()} arestas.")
    else:
        logger.info("Nenhuma otimização (remoção de nós de grau 2) foi realizada.")
    
    return G_optimized

# --- Funções de Preparação para PyTorch Geometric ---

def load_pytorch_geometric_data(G, node_feature_keys=None, edge_feature_keys=None):
    """
    Converte um grafo NetworkX para um objeto Data do PyTorch Geometric.
    
    Args:
        G (nx.Graph): Grafo NetworkX processado.
        node_feature_keys (list, optional): Lista de chaves de atributos dos nós para usar como features (data.x).
                                            Padrão: ['x', 'y', 'degree', 'betweenness', 'closeness'].
        edge_feature_keys (list, optional): Lista de chaves de atributos das arestas para usar como features (data.edge_attr).
                                            Padrão: ['length'].
    
    Returns:
        torch_geometric.data.Data or None: Objeto Data para PyG, ou None se o grafo for vazio ou ocorrer erro.
    """
    logger.info("--- Convertendo Grafo NetworkX para PyTorch Geometric Data ---")
    if not HAS_TORCH:
        logger.error("PyTorch ou PyTorch Geometric não estão instalados. Impossível criar objeto Data.")
        return None
    if G.number_of_nodes() == 0:
        logger.warning("Grafo NetworkX está vazio. Retornando None.")
        return None

    # Definir features padrão se não foram especificados
    if node_feature_keys is None:
        # Verificar quais atributos de centralidade realmente existem no grafo
        default_node_features = ['x', 'y']
        if nx.get_node_attributes(G, 'degree'): default_node_features.append('degree')
        if nx.get_node_attributes(G, 'betweenness'): default_node_features.append('betweenness')
        if nx.get_node_attributes(G, 'closeness'): default_node_features.append('closeness')
        # Adicionar 'eigenvector' se calculado
        # if nx.get_node_attributes(G, 'eigenvector'): default_node_features.append('eigenvector')
        node_feature_keys = default_node_features
    logger.info(f"Usando features de nó: {node_feature_keys}")

    if edge_feature_keys is None:
        default_edge_features = []
        if nx.get_edge_attributes(G, 'length'): default_edge_features.append('length')
        # Adicionar outros atributos de aresta se existirem e forem desejados
        # if nx.get_edge_attributes(G, 'sinuosity'): default_edge_features.append('sinuosity')
        # if nx.get_edge_attributes(G, 'speed_kmh'): default_edge_features.append('speed_kmh')
        edge_feature_keys = default_edge_features
    logger.info(f"Usando features de aresta: {edge_feature_keys}")

    # --- Extração de Dados ---
    node_list = list(G.nodes())
    node_map = {node: i for i, node in enumerate(node_list)} # Mapeia ID do nó do NetworkX para índice 0..N-1

    # 1. Features dos Nós (x)
    x_list = []
    missing_node_features = set()
    for node_id in node_list:
        node_data = G.nodes[node_id]
        features = []
        for key in node_feature_keys:
            val = node_data.get(key)
            if val is not None:
                try:
                    features.append(float(val))
                except (ValueError, TypeError):
                    logger.warning(f"Não foi possível converter node feature '{key}' (valor: {val}) para float no nó {node_id}. Usando 0.0.")
                    features.append(0.0)
                    missing_node_features.add(key)
            else:
                # Usar valor padrão 0.0 para features ausentes e logar aviso
                features.append(0.0)
                missing_node_features.add(key)
        x_list.append(features)
    if missing_node_features:
         logger.warning(f"Features de nó ausentes encontradas e preenchidas com 0.0: {sorted(list(missing_node_features))}")
    x = torch.tensor(x_list, dtype=torch.float)
    logger.info(f"Tensor de features de nó (x) criado com shape: {x.shape}")
    # Atualizar input_dim do modelo
    model_params['input_dim'] = x.shape[1]
    logger.info(f"Parâmetro 'input_dim' do modelo definido como: {model_params['input_dim']}")


    # 2. Índices das Arestas (edge_index)
    edge_index_list = []
    for u, v in G.edges():
        # Adicionar aresta nos dois sentidos para grafo não direcionado
        edge_index_list.append([node_map[u], node_map[v]])
        edge_index_list.append([node_map[v], node_map[u]])
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    logger.info(f"Tensor de índices de aresta (edge_index) criado com shape: {edge_index.shape}")

    # 3. Features das Arestas (edge_attr) - Opcional
    edge_attr = None
    if edge_feature_keys:
        edge_attr_list = []
        missing_edge_features = set()
        # Iterar sobre as arestas originais (antes de duplicar para não direcionado)
        for u, v in G.edges():
            edge_data = G.get_edge_data(u, v)
            features = []
            for key in edge_feature_keys:
                val = edge_data.get(key)
                if val is not None:
                    try:
                        features.append(float(val))
                    except (ValueError, TypeError):
                        logger.warning(f"Não foi possível converter edge feature '{key}' (valor: {val}) para float na aresta ({u},{v}). Usando 0.0.")
                        features.append(0.0)
                        missing_edge_features.add(key)
        else:
                    features.append(0.0)
                    missing_edge_features.add(key)
            # Adicionar features para a aresta u->v e v->u
            edge_attr_list.append(features)
            edge_attr_list.append(features) # Duplicar para a aresta reversa

        if missing_edge_features:
             logger.warning(f"Features de aresta ausentes encontradas e preenchidas com 0.0: {sorted(list(missing_edge_features))}")

        if edge_attr_list:
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
            logger.info(f"Tensor de features de aresta (edge_attr) criado com shape: {edge_attr.shape}")
        else:
            logger.warning("Nenhuma feature de aresta válida encontrada ou extraída.")
            edge_attr = None # Garantir que é None se vazio
    else:
        logger.info("Nenhuma feature de aresta solicitada (edge_feature_keys está vazio).")


    # 4. Labels dos Nós (y) - Opcional (para tarefas supervisionadas)
    y = None
    if nx.get_node_attributes(G, 'class'):
        try:
            y_list = [G.nodes[node_id]['class'] for node_id in node_list]
            y = torch.tensor(y_list, dtype=torch.long)
            logger.info(f"Tensor de labels de nó (y) criado com shape: {y.shape}")
            # Verificar número de classes único
            unique_classes = torch.unique(y)
            logger.info(f"Classes únicas encontradas em y: {unique_classes.tolist()}")
            if model_params.get('output_dim') is not None and model_params['output_dim'] != len(unique_classes):
                 logger.warning(f"Número de classes únicas em y ({len(unique_classes)}) difere do 'output_dim' ({model_params['output_dim']}) definido anteriormente. Verifique a função assign_node_classes.")
                 # Poderia redefinir output_dim aqui se desejado:
                 # model_params['output_dim'] = len(unique_classes)
        except KeyError:
            logger.warning("Atributo 'class' não encontrado em todos os nós. Tensor 'y' não será criado.")
            y = None
    except Exception as e:
             logger.error(f"Erro ao extrair labels dos nós (y): {e}")
             y = None
    else:
        logger.info("Nenhum atributo 'class' encontrado nos nós. Tensor 'y' (labels) não criado.")

    # Criar objeto Data
    try:
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        logger.info("Objeto Data do PyTorch Geometric criado com sucesso.")
        # Adicionar atributos extras se necessário (ex: mapeamento de volta para ID original)
        data.node_map = node_map # Mapeamento idx -> node_id_original (invertido)
        data.original_node_ids = node_list # Lista de IDs originais na ordem 0..N-1
        return data
    except Exception as e:
        logger.exception(f"Erro ao criar objeto Data do PyTorch Geometric: {e}")
        return None

def normalize_node_features(data):
    """
    Normaliza as features dos nós (data.x) usando Z-score (média 0, desvio padrão 1).
    Modifica o objeto Data inplace.
    """
    if data is None or data.x is None:
        logger.warning("Objeto Data ou data.x é None. Impossível normalizar features.")
        return data

    logger.info("--- Normalizando Features dos Nós (Z-score) ---")
    try:
        mean = data.x.mean(dim=0, keepdim=True)
        std = data.x.std(dim=0, keepdim=True)

        # Substituir std zero por 1 para evitar divisão por zero (features constantes)
        std_safe = torch.where(std == 0, torch.ones_like(std), std)

        # Aplicar normalização Z-score: (x - mean) / std_safe
        data.x = (data.x - mean) / std_safe

        logger.info("Features dos nós normalizadas com sucesso.")
        # logger.debug(f"Média após normalização (deve ser próxima de 0): {data.x.mean(dim=0)}")
        # logger.debug(f"Desvio padrão após normalização (deve ser próximo de 1): {data.x.std(dim=0)}")
    except Exception as e:
        logger.error(f"Erro durante a normalização Z-score das features de nó: {e}")
    
    return data

def create_data_splits(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed_split=None):
    """
    Cria máscaras booleanas de treino/validação/teste para os nós no objeto Data.
    Garante que as proporções somem 1.0. Modifica o objeto Data inplace.
    
    Args:
        data (torch_geometric.data.Data): Objeto Data contendo os dados do grafo.
        train_ratio (float): Proporção de nós para treinamento.
        val_ratio (float): Proporção de nós para validação.
        test_ratio (float): Proporção de nós para teste.
        seed_split (int, optional): Semente para reprodutibilidade da divisão. Usa a semente global se None.
    
    Returns:
        torch_geometric.data.Data: O objeto Data modificado com as máscaras.
    """
    if data is None or data.x is None:
        logger.warning("Objeto Data ou data.x é None. Impossível criar splits.")
        return data
    if not HAS_TORCH:
         logger.error("PyTorch não disponível. Impossível criar splits.")
         return data

    logger.info("--- Criando Máscaras de Divisão Treino/Validação/Teste ---")

    num_nodes = data.num_nodes # Mais idiomático em PyG

    # Validar proporções
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        logger.error(f"As proporções de divisão não somam 1.0 ({train_ratio + val_ratio + test_ratio}). Ajustando teste para preencher.")
        test_ratio = 1.0 - train_ratio - val_ratio
        if test_ratio < 0: # Se ainda assim der negativo, ajustar val e test
             train_ratio = max(0.0, min(1.0, train_ratio))
             val_ratio = (1.0 - train_ratio) / 2.0
             test_ratio = (1.0 - train_ratio) / 2.0
             logger.warning(f"Proporções inválidas. Ajustadas para: Treino={train_ratio:.2f}, Val={val_ratio:.2f}, Teste={test_ratio:.2f}")

    # Usar semente global se nenhuma específica for fornecida
    current_seed = seed_split if seed_split is not None else seed
    torch.manual_seed(current_seed)
    logger.info(f"Usando semente {current_seed} para divisão dos dados.")

    # Criar permutação aleatória dos índices dos nós
    indices = torch.randperm(num_nodes)

    # Calcular tamanhos exatos dos conjuntos
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    # test_size é o restante
    test_size = num_nodes - train_size - val_size

    # Criar máscaras booleanas
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Atribuir índices às máscaras
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size : train_size + val_size]] = True
    test_mask[indices[train_size + val_size :]] = True # Pega o restante

    # Adicionar máscaras ao objeto Data
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    # Logar as contagens resultantes
    train_count = data.train_mask.sum().item()
    val_count = data.val_mask.sum().item()
    test_count = data.test_mask.sum().item()
    total_count = train_count + val_count + test_count

    logger.info(f"Divisão de dados criada ({total_count}/{num_nodes} nós):")
    logger.info(f"  Treinamento: {train_count} nós ({train_count/num_nodes:.2%})")
    logger.info(f"  Validação:   {val_count} nós ({val_count/num_nodes:.2%})")
    logger.info(f"  Teste:       {test_count} nós ({test_count/num_nodes:.2%})")

    # Opcional: Verificar se a divisão foi feita corretamente (todos os nós em exatamente uma máscara)
    if total_count != num_nodes:
         logger.error(f"Erro na divisão: contagem total ({total_count}) não bate com número de nós ({num_nodes}).")
    if torch.any(data.train_mask & data.val_mask) or \
       torch.any(data.train_mask & data.test_mask) or \
       torch.any(data.val_mask & data.test_mask):
        logger.error("Erro na divisão: máscaras têm sobreposição.")

    return data

# --- Relatório e Pipeline Principal ---

def generate_quality_report(input_path, output_gpkg_path, output_pyg_path, gdf_initial, gdf_final, G_final, pyg_data, pipeline_duration):
    """ Gera dicionário com relatório de qualidade do pré-processamento e construção do grafo. """
    logger.info("--- Gerando Relatório de Qualidade Final ---")

    def get_stats(series_or_list):
        # Lida com pd.Series, listas ou tensores PyTorch
        if isinstance(series_or_list, pd.Series):
            series = series_or_list.dropna()
            if series.empty: return {"min": None, "max": None, "mean": None, "median": None, "std": None, "count": 0}
            return {"min": float(series.min()), "max": float(series.max()), "mean": float(series.mean()), "median": float(series.median()), "std": float(series.std()), "count": int(series.count())}
        elif isinstance(series_or_list, torch.Tensor):
             if series_or_list.numel() == 0: return {"min": None, "max": None, "mean": None, "median": None, "std": None, "count": 0}
             # Remover NaNs e Infs se existirem no tensor
             valid_tensor = series_or_list[~torch.isnan(series_or_list) & ~torch.isinf(series_or_list)]
             if valid_tensor.numel() == 0: return {"min": None, "max": None, "mean": None, "median": None, "std": None, "count": 0}
             return {"min": float(valid_tensor.min()), "max": float(valid_tensor.max()), "mean": float(valid_tensor.mean()), "median": float(valid_tensor.median()), "std": float(valid_tensor.std()), "count": int(valid_tensor.numel())}
        elif isinstance(series_or_list, (list, np.ndarray)):
             arr = np.array(series_or_list)
             arr = arr[~np.isnan(arr) & ~np.isinf(arr)] # Remover NaNs/Infs
             if arr.size == 0: return {"min": None, "max": None, "mean": None, "median": None, "std": None, "count": 0}
             return {"min": float(np.min(arr)), "max": float(np.max(arr)), "mean": float(np.mean(arr)), "median": float(np.median(arr)), "std": float(np.std(arr)), "count": int(arr.size)}
        else:
             return {"min": None, "max": None, "mean": None, "median": None, "std": None, "count": 0}

    def get_value_counts(series_or_list):
        if isinstance(series_or_list, pd.Series):
            if series_or_list is None or series_or_list.empty: return {}
            return series_or_list.astype(str).value_counts().to_dict()
        elif isinstance(series_or_list, torch.Tensor):
             if series_or_list is None or series_or_list.numel() == 0: return {}
             counts = torch.unique(series_or_list, return_counts=True)
             return {str(k.item()): v.item() for k, v in zip(*counts)}
        elif isinstance(series_or_list, (list, np.ndarray)):
             if series_or_list is None or len(series_or_list) == 0: return {}
             unique, counts = np.unique(series_or_list, return_counts=True)
             return {str(k): int(v) for k, v in zip(unique, counts)}
            else:
             return {}

    initial_count = len(gdf_initial) if gdf_initial is not None else 0
    final_count_gdf = len(gdf_final) if gdf_final is not None else 0
    retention_ratio = (final_count_gdf / initial_count) if initial_count > 0 else 0

    report = {
        "report_type": "road_preprocessing_graph_quality",
        "report_date": datetime.now().isoformat(),
        "timestamp_run": timestamp,
        "input_file": input_path,
        "output_geopackage": output_gpkg_path if output_gpkg_path else "N/A",
        "output_pytorch_geometric_data": output_pyg_path if output_pyg_path else "N/A",
        "pipeline_duration_seconds": round(pipeline_duration, 2),
        "environment": {
            "in_colab": IN_COLAB,
            "python_version": sys.version,
            "geopandas_version": gpd.__version__,
            "networkx_version": nx.__version__,
            "torch_version": torch.__version__ if HAS_TORCH else "N/A",
            "torch_geometric_version": torch_geometric.__version__ if HAS_TORCH and 'torch_geometric' in sys.modules else "N/A",
            "fiona_available": HAS_FIONA,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        },
        "geospatial_processing": {
            "initial_feature_count": initial_count,
            "final_feature_count": final_count_gdf,
            "feature_retention_ratio": round(retention_ratio, 4),
            "initial_columns": list(gdf_initial.columns) if gdf_initial is not None else [],
            "final_columns": list(gdf_final.columns) if gdf_final is not None else [],
            "added_columns": sorted(list(set(gdf_final.columns) - set(gdf_initial.columns))) if gdf_initial is not None and gdf_final is not None else [],
            "removed_columns": sorted(list(set(gdf_initial.columns) - set(gdf_final.columns))) if gdf_initial is not None and gdf_final is not None else [],
            "initial_geom_types": get_value_counts(gdf_initial.geometry.type) if gdf_initial is not None else {},
            "final_geom_types": get_value_counts(gdf_final.geometry.type) if gdf_final is not None else {},
            "final_crs": str(gdf_final.crs) if gdf_final is not None else None,
            "final_total_bounds": list(gdf_final.total_bounds) if gdf_final is not None and not gdf_final.empty else None,
            "final_road_category_dist": get_value_counts(gdf_final.get('road_category')),
            "final_length_stats_m": get_stats(gdf_final.get('length_m')),
            "final_sinuosity_stats": get_stats(gdf_final.get('sinuosity')),
            "final_curvature_stats": get_stats(gdf_final.get('curvature')),
            # Adicionar outras métricas geo se desejado
        },
        "graph_construction": {
            "num_nodes": G_final.number_of_nodes() if G_final else 0,
            "num_edges": G_final.number_of_edges() if G_final else 0,
            "is_connected": nx.is_connected(G_final) if G_final and G_final.number_of_nodes() > 0 else None,
            "num_connected_components": nx.number_connected_components(G_final) if G_final and G_final.number_of_nodes() > 0 else None,
            "density": nx.density(G_final) if G_final else None,
            "node_degree_stats": get_stats(list(dict(G_final.degree()).values())) if G_final else {},
            "node_betweenness_stats": get_stats(list(nx.get_node_attributes(G_final, 'betweenness').values())) if G_final and nx.get_node_attributes(G_final, 'betweenness') else {},
            "node_closeness_stats": get_stats(list(nx.get_node_attributes(G_final, 'closeness').values())) if G_final and nx.get_node_attributes(G_final, 'closeness') else {},
            "node_class_distribution": get_value_counts(list(nx.get_node_attributes(G_final, 'class').values())) if G_final and nx.get_node_attributes(G_final, 'class') else {},
            "edge_length_stats": get_stats(list(nx.get_edge_attributes(G_final, 'length').values())) if G_final and nx.get_edge_attributes(G_final, 'length') else {},
        },
        "pytorch_geometric_data": {
            "data_object_created": pyg_data is not None,
            "num_nodes": pyg_data.num_nodes if pyg_data else 0,
            "num_edges": pyg_data.num_edges // 2 if pyg_data and pyg_data.is_undirected() else (pyg_data.num_edges if pyg_data else 0), # Contar arestas únicas
            "num_node_features": pyg_data.num_node_features if pyg_data else 0,
            "num_edge_features": pyg_data.num_edge_features if pyg_data and pyg_data.edge_attr is not None else 0,
            "has_isolated_nodes": pyg_data.has_isolated_nodes() if pyg_data else None,
            "is_undirected": pyg_data.is_undirected() if pyg_data else None,
            "node_feature_keys": list(pyg_data.x_keys) if pyg_data and hasattr(pyg_data, 'x_keys') else (model_params.get('node_feature_keys', [])), # Salvar as keys usadas
            "edge_feature_keys": list(pyg_data.edge_attr_keys) if pyg_data and hasattr(pyg_data, 'edge_attr_keys') else (model_params.get('edge_feature_keys', [])),
            "has_labels": pyg_data is not None and pyg_data.y is not None,
            "num_classes": int(torch.unique(pyg_data.y).numel()) if pyg_data and pyg_data.y is not None else None,
            "label_distribution": get_value_counts(pyg_data.y) if pyg_data and pyg_data.y is not None else {},
            "has_train_mask": pyg_data is not None and hasattr(pyg_data, 'train_mask'),
            "train_nodes": int(pyg_data.train_mask.sum()) if pyg_data and hasattr(pyg_data, 'train_mask') else 0,
            "val_nodes": int(pyg_data.val_mask.sum()) if pyg_data and hasattr(pyg_data, 'val_mask') else 0,
            "test_nodes": int(pyg_data.test_mask.sum()) if pyg_data and hasattr(pyg_data, 'test_mask') else 0,
        }
    }
    return report

def run_complete_pipeline(input_path=None, output_gpkg_path_template=None, output_pyg_path_template=None, save_intermediate_graph=False):
    """ Executa o pipeline completo: Pré-proc Geo -> Grafo NX -> Dados PyG. """
    pipeline_start_time = time.time()
    logger.info("="*60)
    logger.info("--- INICIANDO EXECUÇÃO COMPLETA DO PIPELINE (Geo + Grafo + PyG) ---")
    logger.info("="*60)

    # --- Configuração de Caminhos ---
    if input_path is None: input_path = ROADS_ENRICHED_PATH
    base_output_dir = PROCESSED_DATA_DIR if not IN_COLAB else DATA_DIR
    if output_gpkg_path_template is None:
        output_gpkg_path_template = os.path.join(base_output_dir, f"roads_processed_{timestamp}.gpkg")
    if output_pyg_path_template is None:
        output_pyg_path_template = os.path.join(base_output_dir, f"road_graph_pyg_{timestamp}.pt")

    output_gpkg_path = output_gpkg_path_template # Nome final pode ser ajustado
    output_pyg_path = output_pyg_path_template

    logger.info(f"Input GeoPackage: {input_path}")
    logger.info(f"Output GeoPackage Processado: {output_gpkg_path}")
    logger.info(f"Output PyTorch Geometric Data: {output_pyg_path}")
    if save_intermediate_graph:
        output_graphml_path = os.path.join(base_output_dir, f"road_graph_nx_{timestamp}.graphml")
        logger.info(f"Output Grafo NetworkX (GraphML): {output_graphml_path}")
        else:
        output_graphml_path = None

    gdf_processed = None
    G_final = None
    pyg_data = None
    quality_report = {}

    try:
        # === PARTE 1: PRÉ-PROCESSAMENTO GEOESPACIAL ===
        logger.info("--- [ETAPA 1/4] Pré-processamento Geoespacial ---")
        # 1. Carregar Dados
        gdf_initial = load_road_data(input_path) # Guardar original para relatório
        if gdf_initial is None or gdf_initial.empty: raise ValueError("Falha ao carregar dados iniciais ou GDF vazio.")
        gdf = gdf_initial.copy() # Trabalhar com cópia

        # 2. Explodir MultiLineStrings
        gdf = explode_multilines_improved(gdf)
        if gdf.empty: raise ValueError("GDF vazio após explodir MultiLineStrings.")

        # 3. Limpar Dados Geoespaciais
        gdf = clean_road_data(gdf) # Inclui cálculo de length, sinuosity, edge_id
        if gdf.empty: raise ValueError("GDF vazio após limpeza.")

        # 4. Correção Topológica (Snapping) - Opcional mas recomendado
        # Analisar conectividade antes para decidir se snapping é necessário
        # connectivity_before, _ = advanced_connectivity_analysis(gdf, tolerance=1.0) # Pode ser demorado
        # if connectivity_before.get('num_components', 1) > 1:
        gdf = improve_topology(gdf, tolerance=1.0) # Aplicar snapping
        # Recalcular features dependentes da geometria após snapping
        gdf['length_m'] = gdf.geometry.length
        gdf['sinuosity'] = calculate_sinuosity(gdf)
        logger.info("Recalculado length_m e sinuosity após correção topológica.")
        # else:
        #    logger.info("Correção topológica (snapping) ignorada (rede já conectada ou sem múltiplos componentes detectados).")

        # 5. Integrar Dados Contextuais (Placeholder)
        # context_data = load_contextual_data() # Carregar dados contextuais se existirem
        # if context_data: gdf = integrate_contextual_data(gdf, context_data)

        # 6. Enriquecer Features Morfológicas
        gdf = enrich_edge_features(gdf) # Curvatura, bearing, categorias, etc.

        gdf_processed = gdf # Resultado final do pré-processamento geoespacial
        logger.info("--- Pré-processamento Geoespacial Concluído ---")

        # === PARTE 2: CONSTRUÇÃO DO GRAFO NETWORKX ===
        logger.info("--- [ETAPA 2/4] Construção do Grafo NetworkX ---")
        # 7. Criar Grafo a partir do GDF Processado
        G_initial = create_road_graph(gdf_processed)
        if G_initial.number_of_nodes() == 0: raise ValueError("Grafo NetworkX vazio após construção inicial.")

        # 8. Calcular Métricas de Centralidade
        G_with_metrics = calculate_centrality_metrics(G_initial)

        # 9. Atribuir Classes aos Nós
        G_with_classes = assign_node_classes(G_with_metrics, HIGHWAY_TO_IDX, DEFAULT_NODE_CLASS)

        # 10. Otimizar Grafo (Remover Nós Grau 2)
        G_final = optimize_graph(G_with_classes)
        logger.info("--- Construção do Grafo NetworkX Concluída ---")

        # Salvar grafo intermediário se solicitado
        if save_intermediate_graph and output_graphml_path:
             try:
                 logger.info(f"Salvando grafo NetworkX intermediário em {output_graphml_path}...")
                 # Garantir que atributos sejam serializáveis
                 G_to_save = G_final.copy()
                 
                 # Converter atributos de nós problemáticos para tipos serializáveis
                 for node, data in G_to_save.nodes(data=True):
                     for key, value in list(data.items()):
                         if isinstance(value, set):
                             data[key] = str(list(value))  # Converter set para string de lista
                         elif isinstance(value, (type, np.ndarray)):
                             # Converter tipos não serializáveis para string
                             data[key] = str(value)
                         elif not isinstance(value, (str, int, float, bool, list, dict)) and value is not None:
                             # Converter outros tipos complexos para string
                             data[key] = str(value)
                 
                 # Converter atributos de arestas problemáticos
                 for u, v, data in G_to_save.edges(data=True):
                     for key, value in list(data.items()):
                         if isinstance(value, (type, np.ndarray)):
                             data[key] = str(value)
                         elif not isinstance(value, (str, int, float, bool, list, dict)) and value is not None:
                             data[key] = str(value)
                     
                 nx.write_graphml(G_to_save, output_graphml_path)
                 logger.info("Grafo NetworkX salvo com sucesso.")
             except Exception as e:
                 logger.error(f"Erro ao salvar grafo NetworkX em GraphML: {e}")


        # === PARTE 3: PREPARAÇÃO DOS DADOS PYTORCH GEOMETRIC ===
        logger.info("--- [ETAPA 3/4] Preparação dos Dados PyTorch Geometric ---")
        # 11. Converter Grafo NX para Objeto Data PyG
        # Definir quais features usar (pode vir de config ou ser padrão)
        node_features_to_use = ['x', 'y', 'degree', 'betweenness', 'closeness'] # Exemplo
        edge_features_to_use = ['length', 'sinuosity', 'curvature'] # Exemplo
        pyg_data_raw = load_pytorch_geometric_data(G_final, node_features_to_use, edge_features_to_use)
        if pyg_data_raw is None: raise ValueError("Falha ao converter grafo para formato PyTorch Geometric.")

        # 12. Normalizar Features dos Nós
        pyg_data_normalized = normalize_node_features(pyg_data_raw)

        # 13. Criar Máscaras de Treino/Validação/Teste
        pyg_data = create_data_splits(pyg_data_normalized, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15) # Usar seed global padrão
        logger.info("--- Preparação dos Dados PyTorch Geometric Concluída ---")

        # === PARTE 4: SALVAR RESULTADOS E RELATÓRIO ===
        logger.info("--- [ETAPA 4/4] Salvando Resultados e Gerando Relatório ---")
        # 14. Salvar GDF Processado
        try:
            logger.info(f"Salvando GeoDataFrame processado em {output_gpkg_path}...")
            output_dir = os.path.dirname(output_gpkg_path)
            os.makedirs(output_dir, exist_ok=True)
            # Preparar para salvar (converter tipos problemáticos se necessário)
            gdf_to_save = gdf_processed.copy()
            for col in gdf_to_save.select_dtypes(include=['object', 'category']).columns:
                 if col != gdf_to_save.geometry.name:
                     try:
                         # Tentar converter para string, lidar com possíveis erros
                         gdf_to_save[col] = gdf_to_save[col].apply(lambda x: str(x) if pd.notnull(x) else None)
                     except Exception as conv_err:
                         logger.warning(f"Não foi possível converter coluna '{col}' para string ao salvar GPKG: {conv_err}. Removendo coluna.")
                         gdf_to_save = gdf_to_save.drop(columns=[col])
            gdf_to_save.to_file(output_gpkg_path, driver="GPKG")
            logger.info("Arquivo GPKG salvo com sucesso.")
        except Exception as e:
            logger.exception(f"Erro ao salvar GPKG processado: {e}")
            output_gpkg_path = None # Indicar falha

        # 15. Salvar Objeto Data PyG
        if pyg_data is not None:
            try:
                logger.info(f"Salvando objeto Data PyTorch Geometric em {output_pyg_path}...")
                output_dir = os.path.dirname(output_pyg_path)
                os.makedirs(output_dir, exist_ok=True)
                torch.save(pyg_data, output_pyg_path)
                logger.info("Objeto Data PyG salvo com sucesso.")
            except Exception as e:
                logger.exception(f"Erro ao salvar objeto Data PyG: {e}")
                output_pyg_path = None # Indicar falha
    else:
            output_pyg_path = None


    except Exception as e:
        logger.exception(f"ERRO FATAL DURANTE A EXECUÇÃO DO PIPELINE: {e}")
        # Tentar gerar relatório parcial se possível
        pipeline_duration = time.time() - pipeline_start_time
        quality_report = generate_quality_report(
            input_path=input_path, output_gpkg_path=output_gpkg_path, output_pyg_path=output_pyg_path,
            gdf_initial=gdf_initial if 'gdf_initial' in locals() else None,
            gdf_final=gdf_processed, G_final=G_final, pyg_data=pyg_data,
            pipeline_duration=pipeline_duration
        )
        quality_report["pipeline_status"] = "FAILED"
        quality_report["error_message"] = str(e)

    else: # Bloco executado se não houver exceção no try
        pipeline_duration = time.time() - pipeline_start_time
        logger.info(f"--- Pipeline Completo Concluído em {pipeline_duration:.2f} segundos ---")
        # 16. Gerar Relatório Final
        quality_report = generate_quality_report(
            input_path=input_path, output_gpkg_path=output_gpkg_path, output_pyg_path=output_pyg_path,
            gdf_initial=gdf_initial, gdf_final=gdf_processed, G_final=G_final, pyg_data=pyg_data,
            pipeline_duration=pipeline_duration
        )
        quality_report["pipeline_status"] = "SUCCESS"

    # Salvar relatório final (JSON)
    quality_path = os.path.join(QUALITY_REPORT_DIR, f"road_pipeline_quality_report_{timestamp}.json")
    try:
        os.makedirs(QUALITY_REPORT_DIR, exist_ok=True)
        with open(quality_path, 'w', encoding='utf-8') as f:
            # Usar default=str para lidar com tipos não serializáveis (ex: numpy int64)
            json.dump(quality_report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Relatório de qualidade salvo em: {quality_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar relatório de qualidade JSON: {e}")

    logger.info("="*60)
    logger.info("--- FIM DA EXECUÇÃO DO PIPELINE ---")
    logger.info("="*60)

    # Retornar resultados principais
    return {
        'gdf_processed': gdf_processed,
        'graph_networkx': G_final,
        'data_pytorch_geometric': pyg_data,
        'quality_report': quality_report,
        'output_gpkg_path': output_gpkg_path,
        'output_pyg_path': output_pyg_path,
        'output_graphml_path': output_graphml_path,
        'processing_time': pipeline_duration,
        'status': quality_report.get("pipeline_status", "UNKNOWN")
    }


# --- Bloco Principal de Execução ---
if __name__ == "__main__":
    logger.info("="*50)
    logger.info("INICIANDO EXECUÇÃO DO SCRIPT PRINCIPAL (__main__)")
    logger.info("="*50)

    # --- Verificação de Ambiente e Dependências (Opcional, já feito no início) ---
    # Pode adicionar verificações extras aqui se necessário
    if not HAS_TORCH:
         logger.warning("PyTorch ou PyTorch Geometric não encontrados. O pipeline pode não gerar a saída PyG.")
         # sys.exit("Dependências PyTorch/PyG ausentes.") # Descomentar para exigir PyTorch/PyG

    # --- Executar Pipeline Completo ---
    pipeline_result = run_complete_pipeline(save_intermediate_graph=True) # Salvar grafo NX para inspeção

    # --- Resumo Final ---
    if pipeline_result and pipeline_result['status'] == "SUCCESS":
        logger.info("="*50)
        logger.info("PIPELINE CONCLUÍDO COM SUCESSO")
        logger.info("="*50)
        logger.info(f"Tempo total: {pipeline_result['processing_time']:.2f} segundos")
        logger.info(f"GPKG Processado: {pipeline_result['output_gpkg_path']}")
        if pipeline_result['output_graphml_path']:
             logger.info(f"Grafo NetworkX (GraphML): {pipeline_result['output_graphml_path']}")
        logger.info(f"Dados PyTorch Geometric: {pipeline_result['output_pyg_path']}")
        logger.info(f"Relatório salvo em: {os.path.join(QUALITY_REPORT_DIR, f'road_pipeline_quality_report_{timestamp}.json')}") # Caminho real do relatório

        # Acessar dados retornados se necessário
        # gdf = pipeline_result['gdf_processed']
        # G = pipeline_result['graph_networkx']
        # data = pipeline_result['data_pytorch_geometric']
        # print(f"\nAmostra do objeto Data PyG:\n{data}")

        logger.info("\nPróximos passos sugeridos:")
        logger.info(f"1. Carregar dados PyG: data = torch.load('{pipeline_result['output_pyg_path']}')")
        logger.info("2. Definir arquitetura do modelo GNN (ex: GCN, GAT).")
        logger.info("3. Implementar loop de treinamento e avaliação usando as máscaras (data.train_mask, etc.).")
        logger.info("4. Analisar resultados e métricas do modelo.")

                    else:
        logger.error("="*50)
        logger.error("EXECUÇÃO DO PIPELINE FALHOU")
        logger.error("="*50)
        if pipeline_result:
            logger.error(f"Status: {pipeline_result['status']}")
            logger.error(f"Erro: {pipeline_result.get('quality_report', {}).get('error_message', 'N/A')}")
        logger.error(f"Verifique o arquivo de log para detalhes: {os.path.join(OUTPUT_DIR, f'pipeline_gnn_road_{timestamp}.log') if os.path.exists(OUTPUT_DIR) else 'Console'}")
        sys.exit(1) # Sair com código de erro

    logger.info("="*50)
    logger.info("FIM DO SCRIPT")
    logger.info("="*50)

def analyze_spatial_patterns(G, output_dir=None, timestamp=None):
    """
    Executa análise espacial avançada e visualização da rede viária.
    
    Args:
        G (nx.Graph): Grafo NetworkX processado com atributos espaciais e topológicos.
        output_dir (str, optional): Diretório para salvar visualizações e relatórios. Se None, usa VISUALIZACOES_DIR.
        timestamp (str, optional): Timestamp para nomear arquivos. Se None, gera um novo.
    
    Returns:
        dict: Relatório com estatísticas e métricas da análise espacial.
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Configurar logger
    logger.info("="*50)
    logger.info("Iniciando análise espacial avançada do grafo")
    logger.info("="*50)
    
    # Verificar diretório de saída
    if output_dir is None:
        output_dir = VISUALIZACOES_DIR if 'VISUALIZACOES_DIR' in globals() else os.path.join(os.getcwd(), 'visualizacoes')
    
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Visualizações serão salvas em: {output_dir}")
    
    # Verificar se o grafo está vazio
    if G is None or G.number_of_nodes() == 0:
        logger.error("Grafo vazio ou None. Impossível realizar análise espacial.")
        return {"error": "Grafo vazio", "status": "failed"}
    
    # Verificar bibliotecas necessárias
    required_libs = {
        'matplotlib': False, 
        'seaborn': False, 
        'pandas': False,
        'geopandas': False,
        'shapely': False,
        'contextily': False
    }
    
    try:
        import matplotlib.pyplot as plt
        required_libs['matplotlib'] = True
        try:
            import seaborn as sns
            required_libs['seaborn'] = True
            sns.set_style("whitegrid")
        except ImportError:
            logger.warning("Seaborn não encontrado. Algumas visualizações serão limitadas.")
    except ImportError:
        logger.warning("Matplotlib não encontrado. Visualizações serão limitadas.")
    
    try:
        import pandas as pd
        required_libs['pandas'] = True
    except ImportError:
        logger.warning("Pandas não encontrado. Análise de dados será limitada.")
    
    try:
        import geopandas as gpd
        from shapely.geometry import Point, LineString
        required_libs['geopandas'] = True
        required_libs['shapely'] = True
        
        try:
            import contextily as cx
            required_libs['contextily'] = True
        except ImportError:
            logger.warning("Contextily não encontrado. Mapas base não estarão disponíveis.")
    except ImportError:
        logger.warning("GeoPandas ou Shapely não encontrados. Visualizações espaciais serão limitadas.")
    
    # Verificar bibliotecas opcionais para comunidades
    has_community_detection = False
    try:
        import community as community_louvain
        has_community_detection = True
        logger.info("Biblioteca python-louvain encontrada para detecção de comunidades.")
    except ImportError:
        logger.warning("Python-louvain não encontrado. Usando método de detecção de comunidades do NetworkX.")
    
    # Preparar DataFrames para análise
    logger.info("Preparando dados para análise...")
    
    # DataFrame de nós
    nodes_data = []
    for node_id, data in G.nodes(data=True):
        node_dict = {'node_id': node_id}
        node_dict.update(data)
        nodes_data.append(node_dict)
    
    # DataFrame de arestas
    edges_data = []
    for u, v, data in G.edges(data=True):
        edge_dict = {'source': u, 'target': v}
        edge_dict.update(data)
        edges_data.append(edge_dict)
    
    # Criar DataFrames se pandas disponível
    if required_libs['pandas']:
        nodes_df = pd.DataFrame(nodes_data)
        edges_df = pd.DataFrame(edges_data)
        logger.info(f"Criados DataFrames para {len(nodes_df)} nós e {len(edges_df)} arestas.")
        
        # Verificar métricas de centralidade disponíveis
        centrality_metrics = [col for col in ['degree', 'betweenness', 'closeness', 'eigenvector'] if col in nodes_df.columns]
        logger.info(f"Métricas de centralidade disponíveis: {centrality_metrics}")
        else:
        nodes_df = edges_df = None
        logger.warning("Pandas não disponível. Análise baseada em DataFrames será ignorada.")
    
    # Inicializar dicionário de resultados
    analysis_results = {
        'timestamp': timestamp,
        'graph_info': {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'is_directed': nx.is_directed(G),
            'is_connected': nx.is_connected(G),
            'num_components': nx.number_connected_components(G),
        },
        'centrality_analysis': {},
        'community_analysis': {},
        'spatial_metrics': {},
        'degree_distribution': {},
        'visualizations_saved': []
    }
    
    # === 1. Análise de Centralidade ===
    if required_libs['pandas'] and required_libs['matplotlib'] and len(centrality_metrics) > 0:
        logger.info("Analisando métricas de centralidade...")
        centrality_stats = {}
        
        # Estatísticas básicas
        for metric in centrality_metrics:
            if metric in nodes_df.columns:
                metric_stats = nodes_df[metric].describe().to_dict()
                centrality_stats[metric] = {
                    'min': metric_stats.get('min', 0),
                    'max': metric_stats.get('max', 0),
                    'mean': metric_stats.get('mean', 0),
                    'median': metric_stats.get('50%', 0),
                    'std': metric_stats.get('std', 0)
                }
        
        analysis_results['centrality_analysis'] = centrality_stats
        
        # Criar visualizações
        try:
            # Histogramas de centralidade
            plt.figure(figsize=(12, 8))
            for i, metric in enumerate(centrality_metrics, 1):
                plt.subplot(2, 2, i)
                if required_libs['seaborn']:
                    sns.histplot(nodes_df[metric], kde=True)
                else:
                    plt.hist(nodes_df[metric], bins=20, alpha=0.7)
                plt.title(f'Distribuição de {metric.capitalize()}')
                plt.xlabel(metric.capitalize())
                plt.ylabel('Frequência')
            
            plt.tight_layout()
            centrality_hist_path = os.path.join(output_dir, f'centrality_histograms_{timestamp}.png')
            plt.savefig(centrality_hist_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Histogramas de centralidade salvos em: {centrality_hist_path}")
            analysis_results['visualizations_saved'].append(centrality_hist_path)
            
            # Heatmap de correlação entre métricas de centralidade
            if len(centrality_metrics) > 1 and required_libs['seaborn']:
                plt.figure(figsize=(10, 8))
                correlation_matrix = nodes_df[centrality_metrics].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
                plt.title('Correlação entre Métricas de Centralidade')
                
                corr_heatmap_path = os.path.join(output_dir, f'centrality_correlation_{timestamp}.png')
                plt.savefig(corr_heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Heatmap de correlação de centralidade salvo em: {corr_heatmap_path}")
                analysis_results['visualizations_saved'].append(corr_heatmap_path)
        except Exception as e:
            logger.error(f"Erro ao criar visualizações de centralidade: {e}")
    
    # === 2. Análise de Estrutura de Comunidades ===
    logger.info("Analisando estrutura de comunidades...")
    
    # Detecção de comunidades
    communities = None
    try:
        if has_community_detection:
            # Usar algoritmo de Louvain
            logger.info("Aplicando algoritmo de Louvain para detecção de comunidades...")
            import community as community_louvain
            partition = community_louvain.best_partition(G)
            communities = {}
            for node, community_id in partition.items():
                if community_id not in communities:
                    communities[community_id] = []
                communities[community_id].append(node)
        else:
            # Fallback para métodos do NetworkX
            logger.info("Usando método de Girvan-Newman do NetworkX...")
            try:
                from networkx.algorithms import community
                communities_generator = community.girvan_newman(G)
                # Pegar apenas o primeiro nível de comunidades
                for communities_set in itertools.islice(communities_generator, 1):
                    communities = {i: list(c) for i, c in enumerate(communities_set)}
                    break
            except ImportError:
                logger.warning("Métodos de detecção de comunidades não disponíveis.")
                communities = None
        
        if communities:
            # Estatísticas de comunidades
            community_sizes = [len(members) for comm_id, members in communities.items()]
            analysis_results['community_analysis'] = {
                'num_communities': len(communities),
                'min_size': min(community_sizes),
                'max_size': max(community_sizes),
                'mean_size': sum(community_sizes) / len(community_sizes),
                'size_distribution': {str(size): count for size, count in pd.Series(community_sizes).value_counts().to_dict().items()}
            }
            
            # Adicionar comunidade como atributo de nó para visualização
            community_mapping = {}
            for comm_id, members in communities.items():
                for node in members:
                    community_mapping[node] = comm_id
            
            nx.set_node_attributes(G, community_mapping, 'community')
            logger.info(f"Detectadas {len(communities)} comunidades.")
            
            # Visualizar distribuição de tamanhos de comunidade
            if required_libs['matplotlib']:
                plt.figure(figsize=(10, 6))
                if required_libs['seaborn']:
                    sns.histplot(community_sizes, bins=20, kde=True)
            else:
                    plt.hist(community_sizes, bins=20, alpha=0.7)
                plt.title('Distribuição de Tamanhos de Comunidades')
                plt.xlabel('Tamanho da Comunidade (Número de Nós)')
                plt.ylabel('Frequência')
                
                community_hist_path = os.path.join(output_dir, f'community_size_distribution_{timestamp}.png')
                plt.savefig(community_hist_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Histograma de tamanhos de comunidades salvo em: {community_hist_path}")
                analysis_results['visualizations_saved'].append(community_hist_path)
    except Exception as e:
        logger.error(f"Erro durante análise de comunidades: {e}")
    
    # === 3. Visualizações Espaciais ===
    if required_libs['geopandas'] and required_libs['matplotlib']:
        logger.info("Criando visualizações espaciais...")
        
        # Criar GeoDataFrames
        try:
            # GeoDataFrame de nós
            geometry = [Point(data.get('x', 0), data.get('y', 0)) for _, data in G.nodes(data=True)]
            node_attrs = [dict(data) for _, data in G.nodes(data=True)]
            for i, attrs in enumerate(node_attrs):
                node_attrs[i]['node_id'] = list(G.nodes())[i]
                # Remover atributos problemáticos para GeoDataFrame
                for key in ['x', 'y', 'original_indices']:
                    if key in attrs:
                        del attrs[key]
            
            nodes_gdf = gpd.GeoDataFrame(node_attrs, geometry=geometry, crs="EPSG:31983")
            
            # GeoDataFrame de arestas
            edge_geoms = []
            edge_attrs = []
            
            for u, v, data in G.edges(data=True):
                u_data = G.nodes[u]
                v_data = G.nodes[v]
                try:
                    if 'x' in u_data and 'y' in u_data and 'x' in v_data and 'y' in v_data:
                        line = LineString([(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])])
                        edge_geoms.append(line)
                        
                        # Atributos da aresta
                        edge_attr = dict(data)
                        edge_attr['source'] = u
                        edge_attr['target'] = v
                        edge_attrs.append(edge_attr)
                except Exception as e:
                    logger.warning(f"Erro ao criar geometria para aresta ({u}, {v}): {e}")
            
            edges_gdf = gpd.GeoDataFrame(edge_attrs, geometry=edge_geoms, crs="EPSG:31983")
            
            logger.info(f"Criados GeoDataFrames para visualização: {len(nodes_gdf)} nós, {len(edges_gdf)} arestas")
            
            # Calcular métricas espaciais adicionais
            bounds = nodes_gdf.total_bounds
            x_range = bounds[2] - bounds[0]
            y_range = bounds[3] - bounds[1]
            area = x_range * y_range
            
            analysis_results['spatial_metrics'] = {
                'x_min': bounds[0],
                'y_min': bounds[1],
                'x_max': bounds[2],
                'y_max': bounds[3],
                'x_range': x_range,
                'y_range': y_range,
                'area_sq_m': area,
                'network_length_m': float(edges_gdf.geometry.length.sum()),
                'network_density_m_per_sqkm': float(edges_gdf.geometry.length.sum() / (area / 1_000_000)) if area > 0 else 0
            }
            
            # Criar visualizações temáticas
            for metric in centrality_metrics:
                if metric in nodes_gdf.columns:
                    try:
                        fig, ax = plt.subplots(figsize=(12, 10))
                        edges_gdf.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5)
                        nodes_gdf.plot(ax=ax, column=metric, cmap='viridis', 
                                       markersize=20, legend=True, alpha=0.7)
                        
                        if required_libs['contextily']:
                            try:
                                nodes_gdf_wm = nodes_gdf.to_crs(epsg=3857)  # Web Mercator para compatibilidade com contextily
                                edges_gdf_wm = edges_gdf.to_crs(epsg=3857)
                                
                                fig, ax = plt.subplots(figsize=(12, 10))
                                edges_gdf_wm.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5)
                                nodes_gdf_wm.plot(ax=ax, column=metric, cmap='viridis', 
                                               markersize=20, legend=True, alpha=0.7)
                                
                                cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
                                metric_map_path = os.path.join(output_dir, f'map_{metric}_with_basemap_{timestamp}.png')
                            except Exception as e:
                                logger.warning(f"Erro ao adicionar mapa base para {metric}: {e}")
                                metric_map_path = os.path.join(output_dir, f'map_{metric}_{timestamp}.png')
                        else:
                            metric_map_path = os.path.join(output_dir, f'map_{metric}_{timestamp}.png')
                        
                        ax.set_title(f'Distribuição Espacial de {metric.capitalize()}')
                        plt.tight_layout()
                        plt.savefig(metric_map_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        logger.info(f"Mapa de {metric} salvo em: {metric_map_path}")
                        analysis_results['visualizations_saved'].append(metric_map_path)
                    except Exception as e:
                        logger.error(f"Erro ao criar mapa para {metric}: {e}")
            
            # Mapa de comunidades se disponível
            if 'community' in nodes_gdf.columns:
                try:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    edges_gdf.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5)
                    nodes_gdf.plot(ax=ax, column='community', cmap='tab20', 
                                   categorical=True, markersize=20, legend=True, alpha=0.7)
                    
                    if required_libs['contextily']:
                        try:
                            nodes_gdf_wm = nodes_gdf.to_crs(epsg=3857)
                            edges_gdf_wm = edges_gdf.to_crs(epsg=3857)
                            
                            fig, ax = plt.subplots(figsize=(12, 10))
                            edges_gdf_wm.plot(ax=ax, color='gray', linewidth=0.5, alpha=0.5)
                            nodes_gdf_wm.plot(ax=ax, column='community', cmap='tab20', 
                                           categorical=True, markersize=20, legend=True, alpha=0.7)
                            
                            cx.add_basemap(ax, source=cx.providers.CartoDB.Positron)
                            community_map_path = os.path.join(output_dir, f'map_communities_with_basemap_{timestamp}.png')
                        except Exception as e:
                            logger.warning(f"Erro ao adicionar mapa base para comunidades: {e}")
                            community_map_path = os.path.join(output_dir, f'map_communities_{timestamp}.png')
                    else:
                        community_map_path = os.path.join(output_dir, f'map_communities_{timestamp}.png')
                    
                    ax.set_title('Estrutura de Comunidades')
                    plt.tight_layout()
                    plt.savefig(community_map_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Mapa de comunidades salvo em: {community_map_path}")
                    analysis_results['visualizations_saved'].append(community_map_path)
                except Exception as e:
                    logger.error(f"Erro ao criar mapa de comunidades: {e}")
        
        except Exception as e:
            logger.error(f"Erro ao criar GeoDataFrames para visualizações espaciais: {e}")
    
    # === 4. Análise Topológica Avançada ===
    logger.info("Realizando análise topológica avançada...")
    
    # Métricas avançadas para grafo conectado
    if nx.is_connected(G):
        try:
            diameter = nx.diameter(G)
            radius = nx.radius(G)
            center = list(nx.center(G))
            periphery = list(nx.periphery(G))
            
            analysis_results['advanced_topology'] = {
                'diameter': diameter,
                'radius': radius,
                'center_size': len(center),
                'periphery_size': len(periphery)
            }
            logger.info(f"Diâmetro do grafo: {diameter}, Raio: {radius}")
            
            # Calcular distância média do caminho mais curto
            # Usar amostragem se o grafo for grande
            if G.number_of_nodes() > 500:
                logger.info("Grafo grande, calculando distância média por amostragem...")
                import random
                sample_size = min(500, G.number_of_nodes())
                sample_nodes = random.sample(list(G.nodes()), sample_size)
                path_lengths = []
                
                for i, source in enumerate(sample_nodes):
                    lengths = nx.single_source_shortest_path_length(G, source)
                    path_lengths.extend(lengths.values())
                
                avg_path_length = sum(path_lengths) / len(path_lengths)
            else:
                avg_path_length = nx.average_shortest_path_length(G)
            
            analysis_results['advanced_topology']['avg_path_length'] = avg_path_length
            logger.info(f"Distância média do caminho mais curto: {avg_path_length:.4f}")
        except Exception as e:
            logger.error(f"Erro ao calcular métricas topológicas avançadas: {e}")
    else:
        logger.warning("Grafo não conectado. Algumas métricas avançadas não serão calculadas.")
    
    # Análise de distribuição de grau
    degrees = [d for _, d in G.degree()]
    degree_counts = {}
    for d in degrees:
        degree_counts[d] = degree_counts.get(d, 0) + 1
    
    analysis_results['degree_distribution'] = {
        'min_degree': min(degrees),
        'max_degree': max(degrees),
        'mean_degree': sum(degrees) / len(degrees),
        'distribution': {str(k): v for k, v in sorted(degree_counts.items())}
    }
    
    # Visualizar distribuição de grau
    if required_libs['matplotlib']:
        plt.figure(figsize=(10, 6))
        if required_libs['seaborn']:
            sns.histplot(degrees, bins=20, kde=True)
    else:
            plt.hist(degrees, bins=20, alpha=0.7)
        plt.title('Distribuição de Grau dos Nós')
        plt.xlabel('Grau')
        plt.ylabel('Frequência')
        
        # Adicionar linha para grau médio
        plt.axvline(x=analysis_results['degree_distribution']['mean_degree'], 
                   color='r', linestyle='--', label=f"Média: {analysis_results['degree_distribution']['mean_degree']:.2f}")
        plt.legend()
        
        degree_dist_path = os.path.join(output_dir, f'degree_distribution_{timestamp}.png')
        plt.savefig(degree_dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Distribuição de grau salva em: {degree_dist_path}")
        analysis_results['visualizations_saved'].append(degree_dist_path)
    
    # === 5. Gerar relatório final em JSON ===
    report_path = os.path.join(output_dir, f'spatial_analysis_report_{timestamp}.json')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Relatório de análise espacial salvo em: {report_path}")
    except Exception as e:
        logger.error(f"Erro ao salvar relatório de análise espacial: {e}")
    
    logger.info("="*50)
    logger.info("Análise espacial concluída")
    logger.info("="*50)
    
    return analysis_results

def generate_graph_statistics(G):
    """
    Gera estatísticas completas para o grafo.
    
    Args:
        G (nx.Graph): Grafo NetworkX processado.
    
    Returns:
        dict: Estatísticas completas do grafo.
    """
    if G is None or G.number_of_nodes() == 0:
        return {"error": "Grafo vazio ou None"}
    
    stats = {
        'basic': {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'is_directed': nx.is_directed(G),
            'is_connected': nx.is_connected(G),
            'density': nx.density(G),
            'num_components': nx.number_connected_components(G)
        },
        'degree': {
            'min': min(dict(G.degree()).values()),
            'max': max(dict(G.degree()).values()),
            'mean': sum(dict(G.degree()).values()) / G.number_of_nodes(),
            'median': sorted(dict(G.degree()).values())[G.number_of_nodes() // 2]
        }
    }
    
    # Centralidade de nós (se calculadas)
    for metric in ['betweenness', 'closeness', 'eigenvector']:
        attr_values = nx.get_node_attributes(G, metric)
        if attr_values:
            values = list(attr_values.values())
            stats[metric] = {
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'median': sorted(values)[len(values) // 2]
            }
    
    # Atributos de arestas (ex: comprimento)
    for attr in ['length', 'sinuosity', 'curvature']:
        attr_values = nx.get_edge_attributes(G, attr)
        if attr_values:
            values = list(attr_values.values())
            stats[f'edge_{attr}'] = {
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'sum': sum(values),
                'median': sorted(values)[len(values) // 2]
            }
    
    # Estatísticas avançadas para grafos conectados
    if nx.is_connected(G):
        try:
            stats['advanced'] = {
                'diameter': nx.diameter(G),
                'radius': nx.radius(G),
                'center_size': len(nx.center(G)),
                'avg_path_length': nx.average_shortest_path_length(G)
            }
        except:
            pass  # Ignorar se falhar em grafos grandes
    
    return stats

