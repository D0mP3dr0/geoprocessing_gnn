"""
Funções para processamento de dados viários com processamento paralelo e aceleração Numba.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import os
import json
from typing import Union, List, Dict, Optional, Tuple, Any
from shapely.geometry import LineString, MultiLineString, Point, Polygon, MultiPolygon
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
REPORT_DIR = os.path.join(os.path.dirname(__file__), 'quality_reports', 'roads')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Arquivo de entrada das estradas
ROADS_FILE = os.path.join(INPUT_DIR, 'sorocaba_roads.gpkg')

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

# Define colunas essenciais e metadados para as estradas
ROAD_COLUMNS = {
    'osm_id': {'type': 'str', 'description': 'ID único da via no OpenStreetMap'},
    'name': {'type': 'str', 'description': 'Nome da via'},
    'highway': {'type': 'str', 'description': 'Tipo/classificação da via'},
    'railway': {'type': 'str', 'description': 'Tipo de ferrovia, se aplicável'},
    'z_order': {'type': 'int32', 'description': 'Ordem de sobreposição da via'},
    'length_km': {'type': 'float64', 'description': 'Comprimento da via em quilômetros', 'validation': {'min': 0}},
    'road_class': {'type': 'str', 'description': 'Classificação hierárquica da via'},
    'connectivity': {'type': 'int32', 'description': 'Número de conexões da via'},
    'sinuosity': {'type': 'float64', 'description': 'Índice de sinuosidade da via', 'validation': {'min': 1}},
}

# Mapeamento de classificação de vias
ROAD_CLASS_MAPPING = {
    'motorway': 'arterial',
    'trunk': 'arterial',
    'primary': 'arterial',
    'secondary': 'collector',
    'tertiary': 'collector',
    'residential': 'local',
    'service': 'local',
    'unclassified': 'local',
    'living_street': 'local',
    'pedestrian': 'pedestrian',
    'footway': 'pedestrian',
    'cycleway': 'cycleway',
    'path': 'pedestrian'
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
        if 'length_km' not in chunk.columns:
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
    # Classificar vias
    if 'highway' in chunk.columns:
        chunk['road_class'] = chunk['highway'].map(ROAD_CLASS_MAPPING).fillna('other')
    
    # Calcular conectividade (número de interseções)
    if 'connectivity' not in chunk.columns:
        chunk['connectivity'] = 0  # Será calculado posteriormente na análise de rede
    
    return chunk

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
    
    # Usar abordagem alternativa para dividir o DataFrame sem usar numpy.array_split
    # que causa o warning de swapaxes
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
    Analisa a conectividade da rede viária usando NetworkX.
    
    Args:
        df: GeoDataFrame com a rede viária
        
    Returns:
        Dicionário com métricas de conectividade
    """
    try:
        # Verificar se NetworkX está disponível
        import networkx as nx
        
        # Projetar para um sistema métrico
        df_proj = df.to_crs(epsg=31983)
        
        # Criar grafo da rede
        print("Criando grafo da rede viária...")
        G = nx.Graph()
        
        # Adicionar nós e arestas
        for idx, row in df_proj.iterrows():
            if isinstance(row.geometry, LineString) and len(row.geometry.coords) >= 2:
                # Usar como identificadores os pontos inicial e final da linha
                start = row.geometry.coords[0]
                end = row.geometry.coords[-1]
                G.add_edge(start, end, id=idx, length=row.geometry.length, road_class=row.get('road_class', 'unknown'))
                
        # Estatísticas básicas do grafo
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        print(f"Grafo criado com {num_nodes} nós e {num_edges} arestas.")
        
        # Análise de componentes conectados
        connected_components = list(nx.connected_components(G))
        num_components = len(connected_components)
        largest_component = max(connected_components, key=len)
        largest_component_size = len(largest_component)
        largest_component_pct = (largest_component_size / num_nodes) * 100 if num_nodes > 0 else 0
        
        # Análise de densidade da rede
        # Usando union_all() em vez de unary_union (depreciado)
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
                'road_density_km_per_km2': float(density_km_per_km2),
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
    """Verifica a topologia da rede viária."""
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
            'total_length_km': float(df_proj['length_km'].sum()) if 'length_km' in df.columns else 0,
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

def create_quality_report(df: gpd.GeoDataFrame, output_file: str):
    """Cria um relatório de qualidade para a rede viária."""
    # Reprojetar para CRS projetado para cálculos métricos
    df_proj = df.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
    
    report = {
        'data_summary': {
            'total_features': len(df),
            'crs': str(df.crs),
            'geometry_types': df.geometry.type.unique().tolist(),
            'total_columns': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        },
        'completeness': {
            'missing_values': df.isnull().sum().to_dict(),
            'total_missing': int(df.isnull().sum().sum())
        },
        'road_statistics': {
            'total_length_km': float(df_proj['length_km'].sum()),
            'road_classes': df['road_class'].value_counts().to_dict() if 'road_class' in df.columns else {},
            'highway_types': df['highway'].value_counts().to_dict() if 'highway' in df.columns else {}
        },
        'numeric_statistics': {},
        'topology': check_topology(df),
        'network_analysis': analyze_network_connectivity(df),
        'bounds': {
            'minx': float(df.total_bounds[0]),
            'miny': float(df.total_bounds[1]),
            'maxx': float(df.total_bounds[2]),
            'maxy': float(df.total_bounds[3])
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Calcular estatísticas para colunas numéricas
    numeric_cols = ['length_km', 'sinuosity', 'connectivity']
    for col in numeric_cols:
        if col in df.columns:
            report['numeric_statistics'][col] = calculate_statistics_with_numba(df, col)
    
    # Adicionar informação sobre área de interesse
    if AREA_OF_INTEREST is not None:
        aoi_proj = AREA_OF_INTEREST.to_crs(epsg=31983)
        report['area_of_interest'] = {
            'file': SOROCABA_SHAPEFILE,
            'crs': str(AREA_OF_INTEREST.crs),
            'area_km2': float(aoi_proj.area.sum() / 1e6)
        }
    
    # Salvar relatório
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, cls=NpEncoder)
        
    # Criar também uma versão resumida em texto para fácil leitura
    text_report_file = os.path.join(os.path.dirname(output_file), 'quality_report_roads.txt')
    with open(text_report_file, 'w', encoding='utf-8') as f:
        f.write("===== RELATÓRIO DE QUALIDADE DA REDE VIÁRIA =====\n\n")
        f.write(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("== RESUMO DOS DADOS ==\n")
        f.write(f"Total de feições: {report['data_summary']['total_features']}\n")
        f.write(f"CRS: {report['data_summary']['crs']}\n")
        f.write(f"Tipos de geometria: {', '.join(report['data_summary']['geometry_types'])}\n")
        f.write(f"Uso de memória: {report['data_summary']['memory_usage_mb']:.2f} MB\n\n")
        
        f.write("== ESTATÍSTICAS DA REDE VIÁRIA ==\n")
        f.write(f"Comprimento total: {report['road_statistics']['total_length_km']:.2f} km\n")
        
        if 'road_classes' in report['road_statistics']:
            f.write("\nClasses de via:\n")
            for cls, count in report['road_statistics']['road_classes'].items():
                f.write(f"- {cls}: {count} vias\n")
                
        if 'highway_types' in report['road_statistics']:
            f.write("\nTipos de highways (OpenStreetMap):\n")
            for hw_type, count in report['road_statistics']['highway_types'].items():
                f.write(f"- {hw_type}: {count} vias\n")
        
        f.write("\n== ANÁLISE TOPOLÓGICA ==\n")
        if 'topology' in report:
            f.write(f"Geometrias inválidas: {report['topology'].get('invalid_geometries', 'N/A')}\n")
            f.write(f"Componentes conectados: {report['topology'].get('connected_components', 'N/A')}\n")
            f.write(f"Interseções: {report['topology'].get('intersections', 'N/A')}\n")
        
        if 'network_analysis' in report and 'error' not in report['network_analysis']:
            network = report['network_analysis']
            f.write("\n== ANÁLISE DE REDE ==\n")
            f.write(f"Nós (junções): {network['network_size']['nodes']}\n")
            f.write(f"Arestas (vias): {network['network_size']['edges']}\n")
            f.write(f"Comprimento total: {network['network_size']['total_length_km']:.2f} km\n")
            f.write(f"Densidade viária: {network['density']['road_density_km_per_km2']:.2f} km/km²\n")
            f.write(f"Densidade de interseções: {network['density']['intersection_density_per_km2']:.2f} interseções/km²\n")
            f.write(f"Índice Beta (conectividade): {network['topology_indices']['beta_index']:.2f}\n")
            f.write(f"Índice Gamma (circuitos): {network['topology_indices']['gamma_index']:.2f}\n")
        
    print(f"Relatório de texto salvo em: {text_report_file}")

def calculate_statistics_with_numba(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """Calcula estatísticas para uma coluna numérica usando Numba."""
    arr = df[col].values
    min_val, max_val, mean_val, std_val, median_val = calc_statistics(arr)
    
    return {
        'min': float(min_val) if not np.isnan(min_val) else None,
        'max': float(max_val) if not np.isnan(max_val) else None,
        'mean': float(mean_val) if not np.isnan(mean_val) else None,
        'std': float(std_val) if not np.isnan(std_val) else None,
        'median': float(median_val) if not np.isnan(median_val) else None
    }

def process_road_data(input_file: str, output_dir: str) -> gpd.GeoDataFrame:
    """
    Processa os dados da rede viária.
    
    Args:
        input_file: Caminho para o arquivo GPKG de entrada
        output_dir: Diretório para salvar os dados processados
        
    Returns:
        GeoDataFrame com os dados processados
    """
    start_time = time.time()
    print(f"Processando dados viários de: {input_file}")
    
    # Verificar se o arquivo de entrada existe
    if not os.path.exists(input_file):
        print(f"ERRO: Arquivo de entrada não encontrado: {input_file}")
        return None
    
    try:
        # Ler dados
        df = gpd.read_file(input_file)
        print(f"Dados carregados: {len(df)} registros, {len(df.columns)} colunas")
        print(f"CRS: {df.crs}")
        
        # Verificar se o DataFrame está vazio
        if len(df) == 0:
            print("ERRO: Nenhum dado encontrado no arquivo de entrada.")
            return None
            
        # Verificar se há coluna de geometria
        if 'geometry' not in df.columns:
            print("ERRO: Coluna de geometria não encontrada.")
            return None
            
        # Verificar CRS
        if not df.crs:
            print("AVISO: CRS não definido. Assumindo SIRGAS 2000 (EPSG:4674).")
            df = df.set_crs(epsg=4674)
        
        # Filtrar por área de interesse
        print("Filtrando dados para a área de Sorocaba...")
        df = filter_by_area_of_interest(df)
        
        # Se não houver dados após o filtro, avisa e retorna
        if len(df) == 0:
            print("ERRO: Nenhum dado encontrado na área de interesse.")
            return None
        
        # Limpar e padronizar nomes de colunas
        df = clean_column_names(df)
        
        # Fazer backup antes do processamento
        try:
            if len(df) > 0:
                backup_file = os.path.join(output_dir, 'roads_backup.gpkg')
                df.to_file(backup_file, driver='GPKG')
                print(f"Backup salvo em: {backup_file}")
        except Exception as e:
            print(f"AVISO: Não foi possível criar backup: {str(e)}")
        
        # Processar em paralelo
        print("Validando dados...")
        df = parallel_process(df, 'validate')
        
        print("Processando geometrias...")
        df = parallel_process(df, 'geometry')
        
        print("Processando atributos...")
        df = parallel_process(df, 'attributes')
        
        # Verificar se há dados após processamento
        if len(df) == 0:
            print("ERRO: Todos os dados foram filtrados durante o processamento.")
            return None
            
        # Verificar e corrigir dados faltantes
        missing_lengths = df['length_km'].isna().sum()
        if missing_lengths > 0:
            print(f"AVISO: {missing_lengths} registros sem comprimento calculado.")
            
        # Gerar relatório
        report_file = os.path.join(REPORT_DIR, 'quality_report_roads.json')
        create_quality_report(df, report_file)
        print(f"Relatório de qualidade salvo em: {report_file}")
        
        # Salvar dados processados
        output_file = os.path.join(output_dir, 'roads_processed.gpkg')
        df.to_file(output_file, driver='GPKG')
        print(f"Dados processados salvos em: {output_file}")
        
        elapsed_time = time.time() - start_time
        print(f"Tempo total de processamento: {elapsed_time:.2f} segundos")
        
        # Verificar qualidade dos dados processados
        if 'length_km' in df.columns:
            total_length = df['length_km'].sum()
            if total_length < 1.0:  # Se comprimento total for muito pequeno
                print(f"AVISO: Comprimento total da rede muito pequeno: {total_length:.2f} km")
                print("      Verifique se a reprojeção está correta ou se os dados estão filtrados incorretamente")
        
        return df
        
    except Exception as e:
        print(f"ERRO ao processar dados viários: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def main():
    """Função principal."""
    print(f"\n{'='*80}")
    print(f"{'Processamento de Rede Viária':^80}")
    print(f"{'='*80}\n")
    
    print(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Diretório de trabalho: {WORKSPACE_DIR}")
    print(f"Arquivo de entrada: {ROADS_FILE}")
    print(f"Diretório de saída: {OUTPUT_DIR}")
    print(f"Relatórios: {REPORT_DIR}\n")
    
    # Verificar dependências
    try:
        import networkx
        print("NetworkX disponível para análise de conectividade: Versão", networkx.__version__)
    except ImportError:
        print("AVISO: NetworkX não instalado. A análise de conectividade será limitada.")
        print("Instale com: pip install networkx")
        
    try:
        processed_data = process_road_data(ROADS_FILE, OUTPUT_DIR)
        if processed_data is not None:
            print("\nProcessamento dos dados viários concluído com sucesso")
            print(f"Total de vias processadas: {len(processed_data)}")
            
            if 'road_class' in processed_data.columns:
                print("\nDistribuição por classe de via:")
                for class_name, count in processed_data['road_class'].value_counts().items():
                    print(f"  - {class_name}: {count} vias")
                    
            if 'length_km' in processed_data.columns:
                total_length = processed_data['length_km'].sum()
                print(f"\nComprimento total da rede: {total_length:.2f} km")
                
                # Comprimento por classe
                if 'road_class' in processed_data.columns:
                    print("Comprimento por classe de via:")
                    for class_name, group in processed_data.groupby('road_class'):
                        class_length = group['length_km'].sum()
                        print(f"  - {class_name}: {class_length:.2f} km")
                        
            # Salvar um sumário em texto para referência fácil
            try:
                summary_file = os.path.join(REPORT_DIR, 'roads_summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"Sumário do Processamento de Rede Viária\n")
                    f.write(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"Total de vias: {len(processed_data)}\n")
                    
                    if 'road_class' in processed_data.columns:
                        f.write("\nDistribuição por classe de via:\n")
                        for class_name, count in processed_data['road_class'].value_counts().items():
                            f.write(f"  - {class_name}: {count} vias\n")
                            
                    if 'length_km' in processed_data.columns:
                        f.write(f"\nComprimento total da rede: {processed_data['length_km'].sum():.2f} km\n")
                        
                print(f"\nSumário salvo em: {summary_file}")
            except Exception as e:
                print(f"AVISO: Não foi possível salvar o sumário: {str(e)}")
        else:
            print("ERRO: Não foi possível processar os dados viários")
            
    except Exception as e:
        print(f"ERRO ao executar processamento: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
    finally:
        print(f"\n{'='*80}")
        print(f"{'Fim do Processamento':^80}")
        print(f"{'='*80}\n")

if __name__ == "__main__":
    main() 