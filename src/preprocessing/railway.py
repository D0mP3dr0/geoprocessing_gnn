"""
Funções para processamento de dados ferroviários com processamento paralelo e aceleração Numba.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import os
import json
from typing import Union, List, Dict, Optional, Tuple, Any
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, unary_union
import warnings
from concurrent.futures import ProcessPoolExecutor
import psutil
from tqdm import tqdm
import time
import numba
from numba import jit, prange, float64, int64, boolean

# Configurar processamento paralelo
N_WORKERS = min(psutil.cpu_count(logical=False), 8)  # Usar número físico de cores, máximo 8
PARTITION_SIZE = 1000  # Tamanho do chunk para processamento em paralelo

print(f"Configuração do sistema:")
print(f"- Número de workers: {N_WORKERS}")
print(f"- Memória disponível: {psutil.virtual_memory().available / (1024*1024*1024):.2f} GB")
print(f"- Tamanho dos chunks: {PARTITION_SIZE}")
print(f"- Usando aceleração Numba: {numba.__version__}")

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
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
REPORT_DIR = os.path.join(os.path.dirname(__file__), 'quality_reports', 'railway')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Arquivo de entrada das ferrovias
RAILWAY_FILE = os.path.join(INPUT_DIR, 'sorocaba_railway.gpkg')

# Arquivo do polígono de Sorocaba para recorte
SOROCABA_SHAPEFILE = os.path.join(INPUT_DIR, 'sorocaba.gpkg')

# Carrega o polígono de Sorocaba uma única vez no início do programa
def load_area_of_interest():
    """Carrega o polígono de área de interesse para filtrar os dados."""
    try:
        print(f"Carregando polígono de área de interesse de: {SOROCABA_SHAPEFILE}")
        aoi = gpd.read_file(SOROCABA_SHAPEFILE)
        
        # Reprojetar para CRS UTM para cálculos de área
        aoi_proj = aoi.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        area_km2 = aoi_proj.area.sum() / 1e6
        
        print(f"Polígono carregado. CRS: {aoi.crs}, Área: {area_km2:.2f} km²")
        return aoi
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o polígono de área de interesse: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

# Carrega o polígono de Sorocaba no início
try:
    AREA_OF_INTEREST = load_area_of_interest()
    print(f"Área de interesse carregada com sucesso. Formato: {AREA_OF_INTEREST.shape}")
except Exception as e:
    print(f"AVISO: Processamento continuará sem filtro espacial. Erro: {str(e)}")
    AREA_OF_INTEREST = None

# Define colunas essenciais e metadados para as ferrovias
RAILWAY_COLUMNS = {
    'osm_id': {'type': 'str', 'description': 'ID único da ferrovia no OpenStreetMap'},
    'name': {'type': 'str', 'description': 'Nome da ferrovia'},
    'railway': {'type': 'str', 'description': 'Tipo de ferrovia'},
    'z_order': {'type': 'int32', 'description': 'Ordem de sobreposição da ferrovia'},
    'length_km': {'type': 'float64', 'description': 'Comprimento da ferrovia em quilômetros', 'validation': {'min': 0}},
    'gauge_mm': {'type': 'float64', 'description': 'Bitola da ferrovia em milímetros', 'validation': {'min': 0}},
    'electrified': {'type': 'str', 'description': 'Indica se a ferrovia é eletrificada'},
    'operator': {'type': 'str', 'description': 'Operador da ferrovia'},
    'service': {'type': 'str', 'description': 'Tipo de serviço da ferrovia'},
    'sinuosity': {'type': 'float64', 'description': 'Índice de sinuosidade da ferrovia', 'validation': {'min': 1}},
}

@jit(nopython=True, parallel=True)
def validate_numeric_array(arr, min_val=None, max_val=None):
    """
    Valida um array numérico usando Numba para aceleração.
    
    Args:
        arr: Array NumPy para validar
        min_val: Valor mínimo permitido
        max_val: Valor máximo permitido
        
    Returns:
        Máscara booleana indicando quais valores são inválidos
    """
    n = len(arr)
    result = np.zeros(n, dtype=np.bool_)
    
    for i in prange(n):
        if np.isnan(arr[i]):
            continue
            
        if min_val is not None and arr[i] < min_val:
            result[i] = True
        elif max_val is not None and arr[i] > max_val:
            result[i] = True
    
    return result

@jit(nopython=True)
def calc_statistics(arr):
    """
    Calcula estatísticas básicas usando Numba para aceleração.
    
    Args:
        arr: Array NumPy para calcular estatísticas
        
    Returns:
        Tuple de (min, max, mean, std, median)
    """
    # Remover NaNs
    arr_clean = arr[~np.isnan(arr)]
    
    if len(arr_clean) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    
    # Calcular estatísticas
    min_val = np.min(arr_clean)
    max_val = np.max(arr_clean)
    mean_val = np.mean(arr_clean)
    std_val = np.std(arr_clean)
    
    # Calcular mediana
    sorted_arr = np.sort(arr_clean)
    n = len(sorted_arr)
    if n % 2 == 0:
        median_val = (sorted_arr[n//2 - 1] + sorted_arr[n//2]) / 2
    else:
        median_val = sorted_arr[n//2]
    
    return min_val, max_val, mean_val, std_val, median_val

def filter_by_area_of_interest(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filtra o GeoDataFrame para incluir apenas features que intersectam a área de interesse."""
    if AREA_OF_INTEREST is None:
        print("Aviso: Área de interesse não disponível. Processando todos os dados.")
        return df
        
    # Verificar se os dados têm geometria válida
    if df.geometry.isna().any() or (~df.geometry.is_valid).any():
        print("Aviso: Corrigindo geometrias inválidas antes do filtro espacial...")
        df = df.copy()
        invalid_mask = df.geometry.isna() | (~df.geometry.is_valid)
        if invalid_mask.any():
            df.loc[invalid_mask, 'geometry'] = df.loc[invalid_mask, 'geometry'].apply(
                lambda x: None if pd.isna(x) else x.buffer(0)
            )
            # Remover linhas onde não foi possível corrigir a geometria
            df = df.dropna(subset=['geometry'])
            print(f"  - {invalid_mask.sum()} geometrias corrigidas ou removidas")
            
    # Verificar projeções e reajustar se necessário
    if not df.crs:
        print("Erro: GeoDataFrame não possui CRS definido. Assumindo EPSG:4674 (SIRGAS 2000).")
        df.set_crs(epsg=4674, inplace=True)
        
    if not AREA_OF_INTEREST.crs:
        print("Erro: Área de interesse não possui CRS definido. Assumindo EPSG:4674 (SIRGAS 2000).")
        AREA_OF_INTEREST.set_crs(epsg=4674, inplace=True)
        
    if df.crs != AREA_OF_INTEREST.crs:
        print(f"Reprojetando dados de {df.crs} para {AREA_OF_INTEREST.crs}")
        df = df.to_crs(AREA_OF_INTEREST.crs)
        
    # Contar feições antes do filtro
    count_before = len(df)
    
    # Aplicar filtro espacial
    start_time = time.time()
    print("Aplicando filtro espacial...")
    
    try:
        # Dissolve o polígono de área de interesse para um único polígono
        # Usando o método union_all() em vez de unary_union (depreciado)
        aoi_polygon = AREA_OF_INTEREST.geometry.union_all()
        
        # Use intersects para melhor performance
        spatial_index = df.sindex
        possible_matches_index = list(spatial_index.intersection(aoi_polygon.bounds))
        possible_matches = df.iloc[possible_matches_index]
        
        # Refina os matches - realiza o teste de intersecção real
        precise_matches = possible_matches[possible_matches.intersects(aoi_polygon)]
        
        # Salva os resultados de volta para um GeoDataFrame
        filtered_df = gpd.GeoDataFrame(precise_matches, crs=df.crs)
        
        elapsed_time = time.time() - start_time
        count_after = len(filtered_df)
        reduction = (1 - (count_after / count_before)) * 100 if count_before > 0 else 0
        
        print(f"Filtro espacial concluído em {elapsed_time:.2f} segundos.")
        print(f"Feições antes: {count_before}, depois: {count_after} (redução de {reduction:.1f}%)")
        
        # Verificar se há resultados após o filtro
        if count_after == 0:
            print("AVISO: Nenhuma feição restante após o filtro espacial. Verifique a área de interesse.")
            print("Retornando conjunto de dados original para evitar perda de dados.")
            return df
            
        return filtered_df
        
    except Exception as e:
        print(f"ERRO no filtro espacial: {str(e)}")
        print("Retornando conjunto de dados original.")
        import traceback
        print(traceback.format_exc())
        return df

def clean_column_names(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Limpa e padroniza nomes de colunas."""
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    df.columns = df.columns.str.strip().str.lower()
    return df

def process_chunk(args):
    """Processa um único chunk de dados."""
    chunk, operation = args
    
    if operation == 'validate':
        return validate_chunk(chunk)
    elif operation == 'geometry':
        return process_geometry_chunk(chunk)
    elif operation == 'attributes':
        return process_attributes_chunk(chunk)
    return chunk

def validate_chunk(chunk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Valida um chunk de dados usando Numba para aceleração."""
    # Validar comprimento das vias
    if 'length_km' in chunk.columns:
        arr = chunk['length_km'].values
        invalid_mask = validate_numeric_array(arr, min_val=0)
        if invalid_mask.any():
            chunk.loc[invalid_mask, 'length_km'] = np.nan
    
    # Validar sinuosidade
    if 'sinuosity' in chunk.columns:
        arr = chunk['sinuosity'].values
        invalid_mask = validate_numeric_array(arr, min_val=1)
        if invalid_mask.any():
            chunk.loc[invalid_mask, 'sinuosity'] = np.nan
    
    # Validar bitola
    if 'gauge_mm' in chunk.columns:
        arr = chunk['gauge_mm'].values
        invalid_mask = validate_numeric_array(arr, min_val=0)
        if invalid_mask.any():
            chunk.loc[invalid_mask, 'gauge_mm'] = np.nan
    
    return chunk

def process_geometry_chunk(chunk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Processa um chunk de geometrias."""
    # Verificar se existem linhas
    if len(chunk) == 0:
        return chunk
        
    # Fazer uma cópia para evitar SettingWithCopyWarning
    chunk = chunk.copy()
    
    # Remove geometrias nulas
    null_geom_mask = chunk.geometry.isna()
    if null_geom_mask.any():
        print(f"Removendo {null_geom_mask.sum()} geometrias nulas")
        chunk = chunk.dropna(subset=['geometry'])
        
    # Remove geometrias inválidas
    invalid_mask = ~chunk.geometry.is_valid
    if invalid_mask.any():
        print(f"Corrigindo {invalid_mask.sum()} geometrias inválidas")
        chunk.loc[invalid_mask, 'geometry'] = chunk.loc[invalid_mask, 'geometry'].buffer(0)
        
        # Verificar novamente após a correção
        still_invalid = ~chunk.geometry.is_valid
        if still_invalid.any():
            print(f"Não foi possível corrigir {still_invalid.sum()} geometrias - estas serão removidas")
            chunk = chunk[~still_invalid]
    
    # Converter MultiLineString para LineString quando possível
    multiline_mask = chunk.geometry.type == 'MultiLineString'
    if multiline_mask.any():
        chunk.loc[multiline_mask, 'geometry'] = chunk.loc[multiline_mask, 'geometry'].apply(
            lambda x: linemerge(x) if x.geom_type == 'MultiLineString' else x
        )
    
    try:
        # Reprojetar para CRS projetado para cálculos métricos
        chunk_proj = chunk.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        
        # Calcular comprimento em quilômetros
        chunk['length_km'] = chunk_proj.geometry.length / 1000
        
        # Calcular sinuosidade
        chunk['sinuosity'] = chunk_proj.geometry.apply(calculate_sinuosity)
        
    except Exception as e:
        print(f"ERRO no processamento de geometria: {str(e)}")
        # Garantir que as colunas existam mesmo em caso de erro
        if 'length_km' not in chunk.columns:
            chunk['length_km'] = np.nan
        if 'sinuosity' not in chunk.columns:
            chunk['sinuosity'] = np.nan
    
    return chunk

def process_attributes_chunk(chunk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Processa atributos de um chunk."""
    # Extrair a bitola da ferrovia das tags OSM
    if 'other_tags' in chunk.columns:
        chunk['gauge_mm'] = chunk['other_tags'].apply(extract_gauge)
    else:
        chunk['gauge_mm'] = np.nan
        
    # Extrair se a ferrovia é eletrificada
    if 'other_tags' in chunk.columns:
        chunk['electrified'] = chunk['other_tags'].apply(extract_electrified)
    else:
        chunk['electrified'] = 'unknown'
        
    # Extrair o operador da ferrovia
    if 'other_tags' in chunk.columns:
        chunk['operator'] = chunk['other_tags'].apply(extract_operator)
    else:
        chunk['operator'] = None
        
    # Extrair o tipo de serviço
    if 'other_tags' in chunk.columns:
        chunk['service'] = chunk['other_tags'].apply(extract_service)
    else:
        chunk['service'] = None
    
    return chunk

def extract_gauge(other_tags):
    """Extrai a bitola da ferrovia das tags OSM."""
    if pd.isna(other_tags):
        return np.nan
    
    try:
        # Procura por bitola nas tags
        if 'gauge' in other_tags:
            # Extrai o valor de gauge usando regexp
            import re
            gauge_match = re.search(r'\"gauge\"=>\"(\d+)\"', other_tags)
            if gauge_match:
                return float(gauge_match.group(1))
        return np.nan
    except Exception:
        return np.nan

def extract_electrified(other_tags):
    """Extrai se a ferrovia é eletrificada das tags OSM."""
    if pd.isna(other_tags):
        return 'unknown'
    
    try:
        # Procura por electrified nas tags
        if 'electrified' in other_tags:
            # Extrai o valor de electrified usando regexp
            import re
            electrified_match = re.search(r'\"electrified\"=>\"([^\"]+)\"', other_tags)
            if electrified_match:
                return electrified_match.group(1)
        return 'unknown'
    except Exception:
        return 'unknown'

def extract_operator(other_tags):
    """Extrai o operador da ferrovia das tags OSM."""
    if pd.isna(other_tags):
        return None
    
    try:
        # Procura por operator nas tags
        if 'operator' in other_tags:
            # Extrai o valor de operator usando regexp
            import re
            operator_match = re.search(r'\"operator\"=>\"([^\"]+)\"', other_tags)
            if operator_match:
                return operator_match.group(1)
        return None
    except Exception:
        return None

def extract_service(other_tags):
    """Extrai o tipo de serviço da ferrovia das tags OSM."""
    if pd.isna(other_tags):
        return None
    
    try:
        # Procura por service nas tags
        if 'service' in other_tags:
            # Extrai o valor de service usando regexp
            import re
            service_match = re.search(r'\"service\"=>\"([^\"]+)\"', other_tags)
            if service_match:
                return service_match.group(1)
        return None
    except Exception:
        return None

def calculate_sinuosity(geometry) -> float:
    """Calcula o índice de sinuosidade de uma via."""
    if not isinstance(geometry, (LineString, MultiLineString)):
        return np.nan
        
    if isinstance(geometry, MultiLineString):
        # Tentar mesclar em uma única linha
        geometry = linemerge(geometry)
        if isinstance(geometry, MultiLineString):
            # Se ainda for MultiLineString, usar o segmento mais longo
            geometry = max(geometry.geoms, key=lambda x: x.length)
    
    if len(geometry.coords) < 2:
        return np.nan
        
    # Comprimento real
    real_length = geometry.length
    
    # Distância em linha reta entre pontos inicial e final
    start_point = Point(geometry.coords[0])
    end_point = Point(geometry.coords[-1])
    straight_length = start_point.distance(end_point)
    
    # Evitar divisão por zero
    if straight_length == 0:
        return np.nan
        
    return real_length / straight_length

def parallel_process(df: gpd.GeoDataFrame, operation: str) -> gpd.GeoDataFrame:
    """Processa dados em paralelo usando multiprocessing."""
    # Se o dataframe estiver vazio, retornar imediatamente
    if len(df) == 0:
        print(f"Aviso: DataFrame vazio. Pulando processamento de {operation}.")
        return df
        
    # Resolver o aviso de depreciação de swapaxes
    pd.options.future.infer_string = True
    
    # Dividir em chunks
    n_chunks = max(1, len(df) // PARTITION_SIZE)
    
    # Usar abordagem alternativa para dividir o DataFrame
    chunks = []
    chunk_size = len(df) // n_chunks
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else len(df)
        chunks.append(df.iloc[start_idx:end_idx].copy())
    
    # Preparar argumentos
    args = [(chunk, operation) for chunk in chunks]
    
    # Processar em paralelo com barra de progresso
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        processed_chunks = list(tqdm(
            executor.map(process_chunk, args),
            total=len(chunks),
            desc=f"Processando {operation}"
        ))
    
    # Combinar resultados
    return pd.concat(processed_chunks) if processed_chunks else df

def analyze_network_connectivity(df: gpd.GeoDataFrame) -> Dict:
    """
    Analisa a conectividade da rede ferroviária usando NetworkX.
    
    Args:
        df: GeoDataFrame com a rede ferroviária
        
    Returns:
        Dicionário com métricas de conectividade
    """
    try:
        # Verificar se NetworkX está disponível
        import networkx as nx
        
        # Projetar para um sistema métrico
        df_proj = df.to_crs(epsg=31983)
        
        # Criar grafo da rede
        print("Criando grafo da rede ferroviária...")
        G = nx.Graph()
        
        # Adicionar nós e arestas
        for idx, row in df_proj.iterrows():
            if isinstance(row.geometry, LineString) and len(row.geometry.coords) >= 2:
                # Usar como identificadores os pontos inicial e final da linha
                start = row.geometry.coords[0]
                end = row.geometry.coords[-1]
                G.add_edge(start, end, id=idx, length=row.geometry.length, railway_type=row.get('railway', 'unknown'))
                
        # Estatísticas básicas do grafo
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        print(f"Grafo criado com {num_nodes} nós e {num_edges} arestas.")
        
        # Análise de componentes conectados
        connected_components = list(nx.connected_components(G))
        num_components = len(connected_components)
        largest_component = max(connected_components, key=len) if connected_components else set()
        largest_component_size = len(largest_component)
        largest_component_pct = (largest_component_size / num_nodes) * 100 if num_nodes > 0 else 0
        
        # Análise de densidade da rede
        all_geoms = df_proj.geometry.union_all()
        area_km2 = all_geoms.convex_hull.area / 1e6  # Área aproximada em km²
        total_length_km = sum(nx.get_edge_attributes(G, 'length').values()) / 1000  # Comprimento total em km
        density_km_per_km2 = total_length_km / area_km2 if area_km2 > 0 else 0
        
        # Análise de interseções
        intersections = [node for node, degree in G.degree() if degree > 2]
        num_intersections = len(intersections)
        intersection_density = num_intersections / area_km2 if area_km2 > 0 else 0
        
        # Relações topológicas
        alpha_index = (num_edges - num_nodes + num_components) / (2 * num_nodes - 5) if num_nodes >= 3 else 0
        beta_index = num_edges / num_nodes if num_nodes > 0 else 0
        gamma_index = num_edges / (3 * (num_nodes - 2)) if num_nodes > 2 else 0
        
        # Organizar resultados
        report = {
            'network_size': {
                'nodes': num_nodes,
                'edges': num_edges,
                'total_length_km': float(total_length_km)
            },
            'connectivity': {
                'connected_components': num_components,
                'largest_component_nodes': largest_component_size,
                'largest_component_pct': float(largest_component_pct),
                'intersections': num_intersections
            },
            'density': {
                'area_km2': float(area_km2),
                'rail_density_km_per_km2': float(density_km_per_km2),
                'intersection_density_per_km2': float(intersection_density)
            },
            'topology_indices': {
                'alpha_index': float(alpha_index),
                'beta_index': float(beta_index),
                'gamma_index': float(gamma_index)
            }
        }
        
        return report
    
    except ImportError:
        print("AVISO: NetworkX não está instalado. A análise de conectividade não será realizada.")
        return {"error": "NetworkX não instalado"}
    
    except Exception as e:
        print(f"ERRO na análise de conectividade: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

def check_topology(df: gpd.GeoDataFrame) -> Dict:
    """Verifica a topologia da rede ferroviária."""
    # Verificar se há dados
    if len(df) == 0:
        return {
            'total_features': 0,
            'error': 'Nenhuma feição para análise topológica'
        }
        
    try:
        # Reprojetar para CRS projetado para cálculos métricos
        df_proj = df.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        
        topology_report = {
            'total_features': len(df),
            'invalid_geometries': (~df.geometry.is_valid).sum(),
            'total_length_km': float(df_proj['length_km'].sum()) if 'length_km' in df_proj.columns else 0,
            'disconnected_segments': 0,
            'intersections': 0
        }
        
        try:
            # Criar grafo da rede para análise de conectividade
            import networkx as nx
            G = nx.Graph()
            
            # Adicionar arestas ao grafo
            edges_added = 0
            for idx, row in df_proj.iterrows():
                if isinstance(row.geometry, LineString) and len(row.geometry.coords) >= 2:
                    start = row.geometry.coords[0]
                    end = row.geometry.coords[-1]
                    G.add_edge(start, end, id=idx)
                    edges_added += 1
            
            topology_report['edges_added_to_graph'] = edges_added
            
            if edges_added > 0:
                # Encontrar componentes conectados
                connected_components = list(nx.connected_components(G))
                topology_report['connected_components'] = len(connected_components)
                topology_report['disconnected_segments'] = len(connected_components) - 1 if len(connected_components) > 0 else 0
                
                # Contar interseções (nós com grau > 2)
                intersections = sum(1 for node, degree in G.degree() if degree > 2)
                topology_report['intersections'] = intersections
            else:
                topology_report['warning'] = 'Nenhuma aresta válida para análise de conectividade'
            
        except ImportError:
            topology_report['topology_error'] = "Módulo NetworkX não encontrado. Instale com 'pip install networkx'"
        except Exception as e:
            topology_report['topology_error'] = str(e)
        
        return topology_report
        
    except Exception as e:
        return {
            'total_features': len(df),
            'error': f'Erro na análise topológica: {str(e)}'
        }

def calculate_statistics_with_numba(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Calcula estatísticas de uma coluna usando Numba para aceleração."""
    if col not in df.columns or df[col].isna().all():
        return {
            'count': 0,
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'std': np.nan,
            'median': np.nan
        }
    
    # Usar Numba para calcular estatísticas
    try:
        arr = df[col].values
        min_val, max_val, mean_val, std_val, median_val = calc_statistics(arr)
        
        return {
            'count': len(df[col].dropna()),
            'min': float(min_val),
            'max': float(max_val),
            'mean': float(mean_val),
            'std': float(std_val),
            'median': float(median_val)
        }
    except Exception as e:
        print(f"Erro ao calcular estatísticas para {col}: {str(e)}")
        return {
            'count': len(df[col].dropna()),
            'error': str(e)
        }

def create_quality_report(df: gpd.GeoDataFrame, output_file: str):
    """Cria um relatório de qualidade para a rede ferroviária."""
    # Reprojetar para CRS projetado para cálculos métricos
    df_proj = df.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
    
    report = {
        'data_summary': {
            'total_features': len(df),
            'crs': str(df.crs),
            'columns': list(df.columns)
        },
        'topology': check_topology(df),
        'statistics': {}
    }
    
    # Calcular estatísticas para colunas numéricas
    for col in df.select_dtypes(include=['number']).columns:
        if col in ['length_km', 'gauge_mm', 'sinuosity']:
            report['statistics'][col] = calculate_statistics_with_numba(df, col)
    
    # Contar valores únicos para colunas categóricas
    for col in ['railway', 'electrified', 'operator', 'service']:
        if col in df.columns:
            value_counts = df[col].value_counts().to_dict()
            report['statistics'][col] = {
                'unique_values': len(value_counts),
                'value_counts': value_counts,
                'missing': int(df[col].isna().sum())
            }
    
    # Adicionar análise de conectividade da rede
    report['network_analysis'] = analyze_network_connectivity(df)
    
    # Salvar relatório como JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, cls=NpEncoder)
    
    print(f"Relatório de qualidade salvo em: {output_file}")
    return report

def process_railway_data(input_file: str, output_file: str) -> gpd.GeoDataFrame:
    """
    Processa dados ferroviários.
    
    Args:
        input_file: Caminho para o arquivo GPKG de entrada
        output_file: Caminho para salvar os dados processados
        
    Returns:
        GeoDataFrame processado
    """
    start_time = time.time()
    print(f"Processando dados ferroviários de: {input_file}")
    
    # Carregar dados
    try:
        df = gpd.read_file(input_file)
        print(f"Carregados {len(df)} registros com {len(df.columns)} colunas")
    except Exception as e:
        print(f"ERRO ao carregar dados: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise
    
    # Limpar nomes de colunas
    df = clean_column_names(df)
    
    # Filtrar por área de interesse (Sorocaba)
    df = filter_by_area_of_interest(df)
    
    # Processar geometria em paralelo
    print("Processando geometrias...")
    df = parallel_process(df, 'geometry')
    
    # Processar atributos em paralelo
    print("Processando atributos...")
    df = parallel_process(df, 'attributes')
    
    # Validar dados em paralelo
    print("Validando dados...")
    df = parallel_process(df, 'validate')
    
    # Criar relatório de qualidade
    quality_report_file = os.path.join(REPORT_DIR, 'railway_quality_report.json')
    create_quality_report(df, quality_report_file)
    
    # Salvar dados processados
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_file(output_file, driver="GPKG")
        print(f"Dados processados salvos em: {output_file}")
    except Exception as e:
        print(f"ERRO ao salvar dados processados: {str(e)}")
    
    elapsed_time = time.time() - start_time
    print(f"Tempo total de processamento: {elapsed_time:.2f} segundos")
    
    return df

def main():
    """Função principal com configuração de processamento paralelo e aceleração Numba."""
    start_time = time.time()
    input_file = RAILWAY_FILE
    output_file = os.path.join(OUTPUT_DIR, 'railway_processed.gpkg')
    
    try:
        df = process_railway_data(input_file, output_file)
        print("\nProcessamento dos dados ferroviários concluído com sucesso")
        print(f"Processados {len(df)} registros")
        
        # Calcular e mostrar algumas estatísticas básicas
        if 'length_km' in df.columns:
            total_length = df['length_km'].sum()
            print(f"Comprimento total da rede ferroviária: {total_length:.2f} km")
        
        if 'railway' in df.columns:
            railway_types = df['railway'].value_counts()
            print("\nTipos de ferrovias:")
            for railway_type, count in railway_types.items():
                print(f"- {railway_type}: {count} registros")
        
        total_time = time.time() - start_time
        print(f"Tempo total de execução: {total_time:.2f} segundos")
        
    except Exception as e:
        print(f"Erro ao processar dados ferroviários: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
