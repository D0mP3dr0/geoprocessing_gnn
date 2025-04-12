"""
Funções para processamento de dados hidrográficos com processamento paralelo e aceleração Numba.
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
from pyogrio import list_layers
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
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
REPORT_DIR = os.path.join(os.path.dirname(__file__), 'quality_reports', 'hidrografia')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Arquivo do polígono de Sorocaba para recorte
SOROCABA_SHAPEFILE = os.path.join(INPUT_DIR, 'sorocaba.gpkg')

# Carrega o polígono de Sorocaba uma única vez no início do programa
def load_area_of_interest():
    """Carrega o polígono de área de interesse para filtrar os dados."""
    try:
        print(f"Carregando polígono de área de interesse de: {SOROCABA_SHAPEFILE}")
        aoi = gpd.read_file(SOROCABA_SHAPEFILE)
        print(f"Polígono carregado. CRS: {aoi.crs}, Área: {aoi.area.sum() / 1e6:.2f} km²")
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

# Define colunas essenciais e metadados para cada tipo de camada
LAYER_COLUMNS = {
    'trecho_drenagem': {
        'drn_pk': {'type': 'int32', 'description': 'Chave primária do trecho de drenagem'},
        'cotrecho': {'type': 'int32', 'description': 'Código do trecho'},
        'noorigem': {'type': 'int32', 'description': 'Nó de origem'},
        'nodestino': {'type': 'int32', 'description': 'Nó de destino'},
        'cocursodag': {'type': 'str', 'description': 'Código do curso d\'água'},
        'cobacia': {'type': 'str', 'description': 'Código da bacia'},
        'nucomptrec': {'type': 'float64', 'description': 'Comprimento do trecho em quilômetros', 'validation': {'min': 0}},
        'nustrahler': {'type': 'float64', 'description': 'Ordem de Strahler', 'validation': {'min': 1, 'max': 10}}
    },
    'area_drenagem': {
        'are_pk': {'type': 'int32', 'description': 'Chave primária da área de drenagem'},
        'cobacia': {'type': 'str', 'description': 'Código da bacia'},
        'nuarea': {'type': 'float64', 'description': 'Área em quilômetros quadrados', 'validation': {'min': 0}},
        'nunivotto': {'type': 'int16', 'description': 'Nível Otto Pfafstetter'}
    },
    'ponto_drenagem': {
        'pto_pk': {'type': 'int32', 'description': 'Chave primária do ponto de drenagem'},
        'noponto': {'type': 'str', 'description': 'Nome do ponto'},
        'tpponto': {'type': 'str', 'description': 'Tipo do ponto'}
    },
    'curso_dagua': {
        'cda_pk': {'type': 'int32', 'description': 'Chave primária do curso d\'água'},
        'cocursodag': {'type': 'str', 'description': 'Código do curso d\'água'},
        'nucomprio': {'type': 'float64', 'description': 'Comprimento do rio em quilômetros', 'validation': {'min': 0}},
        'noriocomp': {'type': 'str', 'description': 'Nome completo do rio'}
    },
    'linha_costa': {
        'cos_pk': {'type': 'int32', 'description': 'Chave primária da linha de costa'},
        'tipocosta': {'type': 'str', 'description': 'Tipo de costa'}
    }
}

# Funções aceleradas com Numba para validação numérica
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
    
    # Calcular mediana (ordenar e pegar o valor do meio)
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
        
    # Verificar projeções e reajustar se necessário
    if df.crs != AREA_OF_INTEREST.crs:
        print(f"Reprojetando dados de {df.crs} para {AREA_OF_INTEREST.crs}")
        df = df.to_crs(AREA_OF_INTEREST.crs)
        
    # Contar feições antes do filtro
    count_before = len(df)
    
    # Aplicar filtro espacial
    start_time = time.time()
    print("Aplicando filtro espacial...")
    
    # Dissolve o polígono de área de interesse para um único polígono
    aoi_polygon = AREA_OF_INTEREST.unary_union
    
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
    
    return filtered_df

def get_layer_name(layer_path: str) -> str:
    """Extrai o nome principal da camada a partir do caminho completo."""
    return layer_path.split('.')[-1].replace('geoft_bho_', '')

def clean_column_names(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Limpa e padroniza nomes de colunas."""
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    df.columns = df.columns.str.strip().str.lower()
    return df

def process_chunk(args):
    """Processa um único chunk de dados."""
    chunk, layer_type, operation = args
    
    if operation == 'validate':
        return validate_chunk(chunk, layer_type)
    elif operation == 'geometry':
        return process_geometry_chunk(chunk, layer_type)
    return chunk

def validate_chunk(chunk: gpd.GeoDataFrame, layer_type: str) -> gpd.GeoDataFrame:
    """Valida um chunk de dados usando Numba para aceleração."""
    if layer_type not in LAYER_COLUMNS:
        return chunk
        
    for col, meta in LAYER_COLUMNS[layer_type].items():
        if col not in chunk.columns:
            continue
            
        if 'validation' in meta and col in chunk.select_dtypes(include=[np.number]).columns:
            min_val = meta['validation'].get('min')
            max_val = meta['validation'].get('max')
            
            # Extrair array NumPy para processamento com Numba
            arr = chunk[col].values
            
            # Usar função acelerada com Numba para validação
            invalid_mask = validate_numeric_array(arr, min_val, max_val)
            
            # Aplicar máscara de volta ao DataFrame
            if invalid_mask.any():
                chunk.loc[invalid_mask, col] = np.nan
    
    return chunk

def process_geometry_chunk(chunk: gpd.GeoDataFrame, layer_type: str) -> gpd.GeoDataFrame:
    """Processa um chunk de geometrias."""
    # Remove geometrias inválidas
    invalid_mask = ~chunk.geometry.is_valid
    if invalid_mask.any():
        chunk.loc[invalid_mask, 'geometry'] = chunk.loc[invalid_mask, 'geometry'].buffer(0)
    
    # Validação específica por tipo
    if layer_type in ['trecho_drenagem', 'curso_dagua', 'linha_costa']:
        multiline_mask = chunk.geometry.type == 'MultiLineString'
        if multiline_mask.any():
            chunk.loc[multiline_mask, 'geometry'] = chunk.loc[multiline_mask, 'geometry'].apply(
                lambda x: linemerge(x) if x.geom_type == 'MultiLineString' else x
            )
    
    elif layer_type == 'area_drenagem':
        multipoly_mask = chunk.geometry.type == 'MultiPolygon'
        if multipoly_mask.any():
            chunk.loc[multipoly_mask, 'geometry'] = chunk.loc[multipoly_mask, 'geometry'].apply(
                lambda x: max(x.geoms, key=lambda g: g.area) if x.geom_type == 'MultiPolygon' else x
            )
    
    return chunk

def parallel_process(df: gpd.GeoDataFrame, layer_type: str, operation: str) -> gpd.GeoDataFrame:
    """Processa dados em paralelo usando multiprocessing."""
    # Se o dataframe estiver vazio, retornar imediatamente
    if len(df) == 0:
        print(f"Aviso: DataFrame vazio. Pulando processamento de {operation}.")
        return df
        
    # Dividir em chunks
    n_chunks = max(1, len(df) // PARTITION_SIZE)
    chunks = np.array_split(df, n_chunks)
    
    # Preparar argumentos
    args = [(chunk, layer_type, operation) for chunk in chunks]
    
    # Processar em paralelo com barra de progresso
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        processed_chunks = list(tqdm(
            executor.map(process_chunk, args),
            total=len(chunks),
            desc=f"Processando {operation}"
        ))
    
    # Combinar resultados
    return pd.concat(processed_chunks) if processed_chunks else df

def check_topology(df: gpd.GeoDataFrame, layer_type: str) -> Dict:
    """
    Verifica a topologia com base no tipo de camada.
    
    Args:
        df: GeoDataFrame de entrada
        layer_type: Tipo da camada a verificar
        
    Returns:
        Dicionário com resultados da verificação topológica
    """
    topology_report = {
        'total_features': len(df),
        'invalid_geometries': (~df.geometry.is_valid).sum()
    }
    
    if layer_type in ['trecho_drenagem', 'curso_dagua']:
        # Verificar conectividade da rede
        if 'noorigem' in df.columns and 'nodestino' in df.columns:
            all_nodes = set(df['noorigem'].unique()) | set(df['nodestino'].unique())
            origin_nodes = set(df['noorigem'].unique())
            destination_nodes = set(df['nodestino'].unique())
            
            topology_report.update({
                'network_starts': len(origin_nodes - destination_nodes),
                'network_ends': len(destination_nodes - origin_nodes),
                'total_nodes': len(all_nodes)
            })
        
        # Verificar segmentos sobrepostos
        try:
            if df.geometry.type.isin(['LineString', 'MultiLineString']).all():
                overlaps = df.geometry.overlaps(df.geometry.union_all())
                topology_report['overlapping_segments'] = int(overlaps.sum())
        except Exception as e:
            topology_report['overlapping_segments_error'] = str(e)
    
    elif layer_type == 'area_drenagem':
        # Verificar lacunas e sobreposições em áreas de drenagem
        try:
            topology_report.update({
                'overlapping_areas': int(df.geometry.overlaps(df.geometry.union_all()).sum()),
                'total_area_km2': float(df.geometry.area.sum() / 1_000_000)  # Converter para km²
            })
        except Exception as e:
            topology_report['topology_error'] = str(e)
    
    return topology_report

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

def create_quality_report(df: gpd.GeoDataFrame, layer_type: str, output_file: str):
    """Cria um relatório de qualidade para a camada utilizando Numba para aceleração."""
    report = {
        'layer_type': layer_type,
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
        'numeric_statistics': {},
        'categorical_statistics': {},
        'topology': check_topology(df, layer_type),
        'bounds': {
            'minx': float(df.total_bounds[0]) if len(df) > 0 else None,
            'miny': float(df.total_bounds[1]) if len(df) > 0 else None,
            'maxx': float(df.total_bounds[2]) if len(df) > 0 else None,
            'maxy': float(df.total_bounds[3]) if len(df) > 0 else None
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Calcula estatísticas para colunas numéricas usando Numba
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in df.columns:
            report['numeric_statistics'][col] = calculate_statistics_with_numba(df, col)
    
    # Calcula estatísticas para colunas categóricas
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in df.columns and col != 'geometry':
            value_counts = df[col].value_counts()
            report['categorical_statistics'][col] = {
                'unique_values': len(value_counts),
                'top_values': value_counts.head(10).to_dict(),
                'null_count': int(df[col].isnull().sum())
            }
    
    # Adicionar informação sobre área de interesse
    if AREA_OF_INTEREST is not None:
        report['area_of_interest'] = {
            'file': SOROCABA_SHAPEFILE,
            'crs': str(AREA_OF_INTEREST.crs),
            'area_km2': float(AREA_OF_INTEREST.area.sum() / 1e6)
        }
    
    # Salva relatório
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, cls=NpEncoder)

def process_layer(input_file: str, layer: str, output_dir: str) -> Tuple[str, gpd.GeoDataFrame]:
    """Processa uma única camada com processamento paralelo e aceleração Numba."""
    start_time = time.time()
    print(f"\nProcessando camada: {layer}")
    
    try:
        # Pre-compilar funções Numba na primeira execução
        if not hasattr(process_layer, 'numba_initialized'):
            print("Pré-compilando funções Numba (primeira execução)...")
            # Pré-aquecer as funções compiladas JIT com dados dummy
            dummy_arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
            validate_numeric_array(dummy_arr, 0.0, 10.0)
            calc_statistics(dummy_arr)
            process_layer.numba_initialized = True
            print("Pré-compilação concluída.")
    
        # Ler dados
        df = gpd.read_file(input_file, layer=layer)
        print(f"Dados carregados: {len(df)} registros, {len(df.columns)} colunas")
        
        # Primeiro passo: filtrar por área de interesse
        print("Filtrando dados para a área de Sorocaba...")
        df = filter_by_area_of_interest(df)
        
        # Se não houver dados após o filtro, avisa e retorna
        if len(df) == 0:
            print(f"Aviso: Nenhum dado encontrado na área de interesse para a camada {layer}.")
            report_file = os.path.join(REPORT_DIR, f'quality_report_{layer}_empty.json')
            with open(report_file, 'w') as f:
                json.dump({
                    "layer": layer,
                    "status": "empty",
                    "message": "Nenhum dado encontrado na área de interesse"
                }, f)
            return None, None
        
        layer_type = get_layer_name(layer)
        df = clean_column_names(df)
        
        # Processar em paralelo
        print("Validando dados...")
        df = parallel_process(df, layer_type, 'validate')
        
        print("Validando geometrias...")
        df = parallel_process(df, layer_type, 'geometry')
        
        # Gerar relatório
        report_file = os.path.join(REPORT_DIR, f'quality_report_{layer_type}.json')
        create_quality_report(df, layer_type, report_file)
        print(f"Relatório de qualidade salvo em: {report_file}")
        
        # Salvar dados processados
        output_file = os.path.join(output_dir, f'hidrografia_{layer_type}_processed.gpkg')
        df.to_file(output_file, driver='GPKG')
        print(f"Dados processados salvos em: {output_file}")
        
        elapsed_time = time.time() - start_time
        print(f"Tempo de processamento: {elapsed_time:.2f} segundos")
        
        return layer_type, df
        
    except Exception as e:
        print(f"Erro ao processar camada {layer}: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None, None

def process_hydrographic_data(input_file: str, output_dir: str) -> Dict[str, gpd.GeoDataFrame]:
    """
    Processa todas as camadas do arquivo de dados hidrográficos.
    
    Args:
        input_file: Caminho para o arquivo GPKG de entrada
        output_dir: Diretório para salvar os dados processados
        
    Returns:
        Dicionário mapeando tipos de camada para seus GeoDataFrames processados
    """
    start_time = time.time()
    print(f"Processando dados hidrográficos de: {input_file}")
    
    # Listar todas as camadas no arquivo
    layers = list_layers(input_file)
    
    # Extrair apenas os nomes das camadas (o retorno do list_layers inclui o tipo da geometria)
    layer_names = []
    for layer in layers:
        # Apenas o nome da camada, sem o tipo de geometria
        if isinstance(layer, (list, tuple, np.ndarray)) and len(layer) > 0:
            # Certifique-se de que estamos pegando apenas o nome da camada, não o array inteiro
            layer_names.append(str(layer[0]))
        else:
            layer_names.append(str(layer))
    
    print(f"Encontradas {len(layer_names)} camadas: {', '.join(layer_names)}")
    
    # Processar cada camada
    processed_layers = {}
    for layer_name in layer_names:
        layer_type, df = process_layer(input_file, layer_name, output_dir)
        if layer_type and df is not None:
            processed_layers[layer_type] = df
    
    elapsed_time = time.time() - start_time
    print(f"Tempo total de processamento: {elapsed_time:.2f} segundos")
    
    return processed_layers

def main():
    """Função principal com configuração de processamento paralelo e aceleração Numba."""
    start_time = time.time()
    input_file = os.path.join(INPUT_DIR, 'hidrografia-001.gpkg')
    
    try:
        processed_layers = process_hydrographic_data(input_file, OUTPUT_DIR)
        print("\nProcessamento dos dados hidrográficos concluído com sucesso")
        print(f"Processadas {len(processed_layers)} camadas")
        
        total_time = time.time() - start_time
        print(f"Tempo total de execução: {total_time:.2f} segundos")
        
    except Exception as e:
        print(f"Erro ao processar dados hidrográficos: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 