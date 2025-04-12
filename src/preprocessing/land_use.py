"""
Módulo de processamento de dados de uso do solo com processamento paralelo e aceleração Numba.

Este script realiza o pré-processamento de dados brutos de uso do solo,
aplicando validações, limpezas e transformações necessárias para
produzir um conjunto de dados pronto para análise.

Autor: Usuário
Data: 2024
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import os
import json
import time
import datetime
from typing import Union, List, Dict, Optional, Tuple, Any
from shapely.geometry import LineString, MultiLineString, Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import warnings
from pyogrio import list_layers
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import psutil
from tqdm import tqdm
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
REPORT_DIR = os.path.join(os.path.dirname(__file__), 'quality_reports', 'land_use')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Arquivo de entrada para dados de uso do solo
LANDUSE_FILE = os.path.join(INPUT_DIR, 'sorocaba_landuse.gpkg')

# Arquivo do polígono de Sorocaba para recorte
SOROCABA_SHAPEFILE = os.path.join(INPUT_DIR, 'sorocaba.gpkg')

# Carrega o polígono de Sorocaba uma única vez no início do programa
def load_area_of_interest():
    """
    Carrega o polígono de área de interesse para filtrar os dados.
    
    Returns:
        GeoDataFrame: Contendo o polígono da área de interesse
    """
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

# Define colunas essenciais e metadados para dados de uso do solo
LANDUSE_COLUMNS = {
    'osm_id': {'type': 'str', 'description': 'ID único na base OSM'},
    'osm_way_id': {'type': 'str', 'description': 'ID da via no OSM'},
    'name': {'type': 'str', 'description': 'Nome da área'},
    'landuse': {'type': 'str', 'description': 'Tipo de uso do solo'},
    'area_km2': {'type': 'float64', 'description': 'Área em quilômetros quadrados', 'validation': {'min': 0}},
    'perimeter_km': {'type': 'float64', 'description': 'Perímetro em quilômetros', 'validation': {'min': 0}},
    'compactness': {'type': 'float64', 'description': 'Índice de compacidade (4πA/P²)', 'validation': {'min': 0, 'max': 1}},
    'land_category': {'type': 'str', 'description': 'Categoria de uso do solo simplificada'},
}

# Mapeamento para categorias simplificadas de uso do solo
LANDUSE_CATEGORY_MAPPING = {
    'residential': 'urban',
    'commercial': 'urban',
    'retail': 'urban',
    'industrial': 'urban',
    'construction': 'urban',
    'brownfield': 'urban',
    'garages': 'urban',
    'railway': 'urban',
    'military': 'urban',
    'education': 'institutional',
    'government': 'institutional',
    'governmental': 'institutional',
    'institutional': 'institutional',
    'religious': 'institutional',
    'hospital': 'institutional',
    'cemetery': 'institutional',
    'recreation_ground': 'green',
    'park': 'green',
    'village_green': 'green',
    'grass': 'green',
    'meadow': 'green',
    'orchard': 'agriculture',
    'vineyard': 'agriculture',
    'farmland': 'agriculture',
    'farmyard': 'agriculture',
    'forest': 'forest',
    'quarry': 'extraction',
    'basin': 'water',
    'reservoir': 'water',
    'allotments': 'agriculture'
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
    """
    Filtra o GeoDataFrame para incluir apenas features que intersectam a área de interesse.
    
    Args:
        df: GeoDataFrame a ser filtrado
        
    Returns:
        GeoDataFrame filtrado contendo apenas as feições que intersectam a área de interesse
    """
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
        print("Aviso: GeoDataFrame não possui CRS definido. Assumindo EPSG:4674 (SIRGAS 2000).")
        df.set_crs(epsg=4674, inplace=True)
        
    if not AREA_OF_INTEREST.crs:
        print("Aviso: Área de interesse não possui CRS definido. Assumindo EPSG:4674 (SIRGAS 2000).")
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
        aoi_polygon = AREA_OF_INTEREST.geometry.unary_union
        
        # Use spatial index para melhor performance
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
    """
    Limpa e padroniza nomes de colunas.
    
    Args:
        df: GeoDataFrame com colunas a serem padronizadas
        
    Returns:
        GeoDataFrame com nomes de colunas padronizados
    """
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    df.columns = df.columns.str.strip().str.lower()
    return df

def process_chunk(args):
    """
    Processa um único chunk de dados.
    
    Args:
        args: Tupla contendo (chunk, operation)
        
    Returns:
        Chunk processado
    """
    chunk, operation = args
    
    if operation == 'validate':
        return validate_chunk(chunk)
    elif operation == 'geometry':
        return process_geometry_chunk(chunk)
    elif operation == 'attributes':
        return process_attributes_chunk(chunk)
    return chunk

def validate_chunk(chunk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Valida um chunk de dados usando Numba para aceleração.
    
    Args:
        chunk: GeoDataFrame com dados a serem validados
        
    Returns:
        GeoDataFrame com valores inválidos tratados
    """
    # Fazer uma cópia para evitar SettingWithCopyWarning
    chunk = chunk.copy()
    
    # Validar área
    if 'area_km2' in chunk.columns:
        arr = chunk['area_km2'].values
        invalid_mask = validate_numeric_array(arr, min_val=0)
        if invalid_mask.any():
            chunk.loc[invalid_mask, 'area_km2'] = np.nan
    
    # Validar perímetro
    if 'perimeter_km' in chunk.columns:
        arr = chunk['perimeter_km'].values
        invalid_mask = validate_numeric_array(arr, min_val=0)
        if invalid_mask.any():
            chunk.loc[invalid_mask, 'perimeter_km'] = np.nan
    
    # Validar compacidade
    if 'compactness' in chunk.columns:
        arr = chunk['compactness'].values
        invalid_mask = validate_numeric_array(arr, min_val=0, max_val=1)
        if invalid_mask.any():
            chunk.loc[invalid_mask, 'compactness'] = np.nan
    
    return chunk

def process_geometry_chunk(chunk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Processa um chunk de geometrias.
    
    Args:
        chunk: GeoDataFrame com geometrias a serem processadas
        
    Returns:
        GeoDataFrame com geometrias processadas e métricas calculadas
    """
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
    
    # Converter MultiPolygon para o polígono principal quando possível
    multipoly_mask = chunk.geometry.type == 'MultiPolygon'
    if multipoly_mask.any():
        chunk.loc[multipoly_mask, 'geometry'] = chunk.loc[multipoly_mask, 'geometry'].apply(
            lambda x: max(x.geoms, key=lambda g: g.area) if isinstance(x, MultiPolygon) else x
        )
    
    try:
        # Reprojetar para CRS projetado para cálculos métricos
        chunk_proj = chunk.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        
        # Calcular área em quilômetros quadrados
        chunk['area_km2'] = chunk_proj.geometry.area / 1e6
        
        # Calcular perímetro em quilômetros
        chunk['perimeter_km'] = chunk_proj.geometry.length / 1000
        
        # Calcular índice de compacidade (4πA/P²) - valor entre 0 e 1, sendo 1 um círculo perfeito
        areas = chunk_proj.geometry.area
        perimeters = chunk_proj.geometry.length
        chunk['compactness'] = (4 * np.pi * areas) / (perimeters * perimeters)
        # Limitar valores a intervalo [0, 1]
        chunk.loc[chunk['compactness'] > 1, 'compactness'] = 1.0
            
    except Exception as e:
        print(f"ERRO no processamento de geometria: {str(e)}")
        import traceback
        print(traceback.format_exc())
        # Garantir que as colunas existam mesmo em caso de erro
        if 'area_km2' not in chunk.columns:
            chunk['area_km2'] = np.nan
        if 'perimeter_km' not in chunk.columns:
            chunk['perimeter_km'] = np.nan
        if 'compactness' not in chunk.columns:
            chunk['compactness'] = np.nan
    
    return chunk

def process_attributes_chunk(chunk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Processa atributos de um chunk.
    
    Args:
        chunk: GeoDataFrame com atributos a serem processados
        
    Returns:
        GeoDataFrame com atributos processados
    """
    # Fazer uma cópia para evitar SettingWithCopyWarning
    chunk = chunk.copy()
    
    # Classificar uso do solo em categorias simplificadas
    if 'landuse' in chunk.columns:
        chunk['land_category'] = chunk['landuse'].map(LANDUSE_CATEGORY_MAPPING).fillna('other')
    
    return chunk

def parallel_process(df: gpd.GeoDataFrame, operation: str) -> gpd.GeoDataFrame:
    """
    Processa dados em paralelo usando multiprocessing.
    
    Args:
        df: GeoDataFrame a ser processado
        operation: Tipo de operação ('validate', 'geometry', 'attributes')
        
    Returns:
        GeoDataFrame processado
    """
    # Se o dataframe estiver vazio, retornar imediatamente
    if len(df) == 0:
        print(f"Aviso: DataFrame vazio. Pulando processamento de {operation}.")
        return df
        
    # Resolver o aviso de depreciação de swapaxes
    pd.options.future.infer_string = True
    
    print(f"Iniciando processamento paralelo: {operation} com {N_WORKERS} workers")
    
    # Dividir em chunks
    n_chunks = max(1, min(N_WORKERS*2, len(df) // PARTITION_SIZE))
    
    # Usar abordagem alternativa para dividir o DataFrame
    chunks = []
    chunk_size = len(df) // n_chunks
    for i in range(n_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_chunks - 1 else len(df)
        chunks.append(df.iloc[start_idx:end_idx].copy())
    
    print(f"Dados divididos em {len(chunks)} chunks de ~{chunk_size} registros cada")
    
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
    result = pd.concat(processed_chunks) if processed_chunks else df
    print(f"Processamento de {operation} concluído: {len(result)} registros")
    
    return result

def check_topology(df: gpd.GeoDataFrame) -> Dict:
    """
    Verifica a topologia do uso do solo.
    
    Args:
        df: GeoDataFrame a ser analisado
        
    Returns:
        Dicionário com relatório de topologia
    """
    # Verificar se há dados
    if len(df) == 0:
        return {
            'total_features': 0,
            'error': 'Nenhuma feição para análise topológica'
        }
        
    try:
        topology_report = {
            'total_features': len(df),
            'invalid_geometries': (~df.geometry.is_valid).sum(),
            'multipolygon_count': (df.geometry.type == 'MultiPolygon').sum(),
            'polygon_count': (df.geometry.type == 'Polygon').sum(),
            'self_intersections': 0  # Será calculado abaixo
        }
        
        # Detectar auto-interseções
        self_intersect_count = 0
        for idx, geom in df.geometry.items():
            if not pd.isna(geom) and not geom.is_simple:
                self_intersect_count += 1
        topology_report['self_intersections'] = self_intersect_count
        
        # Verificar sobreposições
        try:
            # Criar buffer mínimo para facilitar operações de interseção
            buffered = df.copy()
            buffered.geometry = buffered.geometry.buffer(0.000001)  # Buffer mínimo
            
            # Usar spatial index para encontrar geometrias que potencialmente se sobrepõem
            sindex = buffered.sindex
            potential_overlaps = 0
            
            # Limitar o número de feições verificadas para evitar processamento excessivo
            sample_size = min(len(buffered), 500)
            sample_indices = np.random.choice(buffered.index, sample_size, replace=False)
            
            for idx in tqdm(sample_indices, desc="Verificando sobreposições"):
                geom = buffered.loc[idx, 'geometry']
                if pd.isna(geom):
                    continue
                    
                # Encontrar possíveis interseções
                possible_matches_idx = list(sindex.intersection(geom.bounds))
                if idx in possible_matches_idx:
                    possible_matches_idx.remove(idx)  # Remover a própria geometria
                
                if possible_matches_idx:
                    possible_matches = buffered.loc[possible_matches_idx, 'geometry']
                    for match_idx, match_geom in possible_matches.items():
                        if pd.isna(match_geom):
                            continue
                        if geom.intersects(match_geom):
                            potential_overlaps += 1
                            break
            
            # Estimar o total baseado na amostra
            if sample_size < len(buffered):
                potential_overlaps = int(potential_overlaps * (len(buffered) / sample_size))
                
            topology_report['potential_overlaps'] = potential_overlaps
            
        except Exception as e:
            topology_report['overlap_error'] = str(e)
            import traceback
            print(f"Erro ao verificar sobreposições: {e}")
            print(traceback.format_exc())
        
        return topology_report
        
    except Exception as e:
        import traceback
        print(f"Erro na análise topológica: {e}")
        print(traceback.format_exc())
        return {
            'total_features': len(df),
            'error': f'Erro na análise topológica: {str(e)}'
        }

def create_quality_report(df: gpd.GeoDataFrame, output_file: str):
    """
    Cria um relatório de qualidade para dados de uso do solo.
    
    Args:
        df: GeoDataFrame com dados processados
        output_file: Caminho para salvar o relatório JSON
    """
    try:
        start_time = time.time()
        print(f"Gerando relatório de qualidade...")
        
        # Reprojetar para CRS projetado para cálculos métricos
        df_proj = df.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        
        report = {
            'timestamp': datetime.datetime.now().isoformat(),
            'data_summary': {
                'total_features': len(df),
                'crs': str(df.crs),
                'geometry_types': df.geometry.type.unique().tolist(),
                'total_columns': len(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            },
            'completeness': {
                'missing_values': {col: int(df[col].isnull().sum()) for col in df.columns},
                'total_missing': int(df.isnull().sum().sum())
            },
            'landuse_statistics': {
                'total_area_km2': float(df_proj['area_km2'].sum()),
                'landuse_types': {k: int(v) for k, v in df['landuse'].value_counts().to_dict().items()} if 'landuse' in df.columns else {},
                'land_categories': {k: int(v) for k, v in df['land_category'].value_counts().to_dict().items()} if 'land_category' in df.columns else {}
            },
            'numeric_statistics': {},
            'topology': check_topology(df),
            'bounds': {
                'minx': float(df.total_bounds[0]),
                'miny': float(df.total_bounds[1]),
                'maxx': float(df.total_bounds[2]),
                'maxy': float(df.total_bounds[3])
            }
        }
        
        # Calcular estatísticas para colunas numéricas
        numeric_cols = ['area_km2', 'perimeter_km', 'compactness']
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
        
        # Adicionar análise de distribuição por categoria
        if 'area_km2' in df.columns and 'land_category' in df.columns:
            area_by_category = df.groupby('land_category')['area_km2'].sum().to_dict()
            total_area = df['area_km2'].sum()
            area_percentage = {cat: (area/total_area)*100 for cat, area in area_by_category.items()}
            
            report['landuse_statistics']['area_by_category_km2'] = area_by_category
            report['landuse_statistics']['area_percentage'] = area_percentage
        
        # Salvar relatório
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, cls=NpEncoder)
            
        # Criar também uma versão resumida em texto para fácil leitura
        text_report_file = os.path.join(os.path.dirname(output_file), 'quality_report_landuse.txt')
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write("===== RELATÓRIO DE QUALIDADE DE USO DO SOLO =====\n\n")
            f.write(f"Data/Hora: {datetime.datetime.now().isoformat()}\n\n")
            
            f.write("== RESUMO DOS DADOS ==\n")
            f.write(f"Total de feições: {report['data_summary']['total_features']}\n")
            f.write(f"CRS: {report['data_summary']['crs']}\n")
            f.write(f"Tipos de geometria: {', '.join(report['data_summary']['geometry_types'])}\n")
            f.write(f"Uso de memória: {report['data_summary']['memory_usage_mb']:.2f} MB\n\n")
            
            f.write("== ESTATÍSTICAS DE USO DO SOLO ==\n")
            f.write(f"Área total mapeada: {report['landuse_statistics']['total_area_km2']:.2f} km²\n")
            
            if 'area_by_category_km2' in report['landuse_statistics']:
                f.write("\nÁrea por categoria (km²):\n")
                for cat, area in report['landuse_statistics']['area_by_category_km2'].items():
                    percentage = report['landuse_statistics']['area_percentage'][cat]
                    f.write(f"- {cat}: {area:.2f} km² ({percentage:.1f}%)\n")
            
            if 'landuse_types' in report['landuse_statistics']:
                f.write("\nTipos de uso do solo:\n")
                for landuse, count in report['landuse_statistics']['landuse_types'].items():
                    f.write(f"- {landuse}: {count} polígonos\n")
            
            if 'land_categories' in report['landuse_statistics'] and report['landuse_statistics']['land_categories']:
                f.write("\nCategorias simplificadas:\n")
                for category, count in report['landuse_statistics']['land_categories'].items():
                    f.write(f"- {category}: {count} polígonos\n")
            
            f.write("\n== ANÁLISE TOPOLÓGICA ==\n")
            if 'topology' in report:
                f.write(f"Geometrias inválidas: {report['topology'].get('invalid_geometries', 'N/A')}\n")
                f.write(f"Polígonos: {report['topology'].get('polygon_count', 'N/A')}\n")
                f.write(f"MultiPolígonos: {report['topology'].get('multipolygon_count', 'N/A')}\n")
                f.write(f"Auto-interseções: {report['topology'].get('self_intersections', 'N/A')}\n")
                f.write(f"Sobreposições potenciais: {report['topology'].get('potential_overlaps', 'N/A')}\n")
            
            # Adicionar resumo das estatísticas numéricas
            f.write("\n== ESTATÍSTICAS NUMÉRICAS ==\n")
            for col, stats in report['numeric_statistics'].items():
                f.write(f"\n{col}:\n")
                f.write(f"- Mínimo: {stats['min']:.4f}\n")
                f.write(f"- Máximo: {stats['max']:.4f}\n")
                f.write(f"- Média: {stats['mean']:.4f}\n")
                f.write(f"- Mediana: {stats['median']:.4f}\n")
                f.write(f"- Desvio padrão: {stats['std']:.4f}\n")
                
        elapsed_time = time.time() - start_time
        print(f"Relatório de qualidade gerado em {elapsed_time:.2f} segundos")
        print(f"Relatório JSON salvo em: {output_file}")
        print(f"Relatório de texto salvo em: {text_report_file}")
            
    except Exception as e:
        print(f"ERRO ao criar relatório de qualidade: {str(e)}")
        import traceback
        print(traceback.format_exc())

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

def process_landuse_data(input_file: str = None, output_dir: str = None) -> gpd.GeoDataFrame:
    """
    Processa os dados de uso do solo.
    
    Args:
        input_file: Caminho para o arquivo GPKG de entrada. Se None, usa o padrão.
        output_dir: Diretório para salvar os dados processados. Se None, usa o padrão.
        
    Returns:
        GeoDataFrame com os dados processados
    """
    # Definir caminhos padrão se não fornecidos
    if input_file is None:
        input_file = LANDUSE_FILE
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    start_time = time.time()
    print(f"Processando dados de uso do solo de: {input_file}")
    
    # Verificar se o arquivo de entrada existe
    if not os.path.exists(input_file):
        print(f"ERRO: Arquivo de entrada não encontrado: {input_file}")
        return None
    
    try:
        # Pre-compilar funções Numba na primeira execução
        if not hasattr(process_landuse_data, 'numba_initialized'):
            print("Pré-compilando funções Numba (primeira execução)...")
            # Pré-aquecer as funções compiladas JIT com dados dummy
            dummy_arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
            validate_numeric_array(dummy_arr, 0.0, 10.0)
            calc_statistics(dummy_arr)
            process_landuse_data.numba_initialized = True
            print("Pré-compilação concluída.")
    
        # Ler dados
        print(f"Lendo arquivo {input_file}...")
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
                backup_file = os.path.join(output_dir, 'landuse_backup.gpkg')
                df.to_file(backup_file, driver='GPKG')
                print(f"Backup salvo em: {backup_file}")
        except Exception as e:
            print(f"AVISO: Não foi possível criar backup: {str(e)}")
        
        # Processar em paralelo - validação
        print("\nValidando dados...")
        df = parallel_process(df, 'validate')
        
        # Processar em paralelo - geometrias
        print("\nProcessando geometrias...")
        df = parallel_process(df, 'geometry')
        
        # Processar em paralelo - atributos
        print("\nProcessando atributos...")
        df = parallel_process(df, 'attributes')
        
        # Verificar se há dados após processamento
        if len(df) == 0:
            print("ERRO: Todos os dados foram filtrados durante o processamento.")
            return None
            
        # Verificar e corrigir dados faltantes
        missing_areas = df['area_km2'].isna().sum()
        if missing_areas > 0:
            print(f"AVISO: {missing_areas} registros sem área calculada.")
        
        # Gerar relatório
        report_file = os.path.join(REPORT_DIR, 'quality_report_landuse.json')
        create_quality_report(df, report_file)
        
        # Salvar dados processados
        output_file = os.path.join(output_dir, 'landuse_processed.gpkg')
        df.to_file(output_file, driver='GPKG')
        print(f"Dados processados salvos em: {output_file}")
        
        elapsed_time = time.time() - start_time
        print(f"\nProcessamento concluído em {elapsed_time:.2f} segundos")
        print(f"Total de registros processados: {len(df)}")
        
        # Imprimir estatísticas básicas
        if 'land_category' in df.columns:
            category_counts = df['land_category'].value_counts()
            print("\nDistribuição por categoria de uso do solo:")
            for category, count in category_counts.items():
                print(f"  - {category}: {count}")
                
        if 'area_km2' in df.columns:
            total_area = df['area_km2'].sum()
            print(f"\nÁrea total mapeada: {total_area:.2f} km²")
        
        return df
        
    except Exception as e:
        print(f"ERRO ao processar dados de uso do solo: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
        # Tentar salvar o que foi possível processar até o erro
        try:
            if 'df' in locals() and len(df) > 0:
                backup_file = os.path.join(output_dir, 'landuse_error_backup.gpkg')
                df.to_file(backup_file, driver='GPKG')
                print(f"Backup parcial salvo em: {backup_file}")
        except Exception as be:
            print(f"Não foi possível salvar backup dos dados parciais: {str(be)}")
            
        return None

def main():
    """
    Função principal para processamento dos dados de uso do solo.
    Configura o ambiente, processa os dados e gera relatórios.
    """
    start_time = time.time()
    
    print("="*80)
    print(" PROCESSAMENTO DE DADOS DE USO DO SOLO ".center(80, '='))
    print("="*80)
    print(f"Data/Hora: {datetime.datetime.now().isoformat()}")
    print(f"Arquivo de entrada: {LANDUSE_FILE}")
    print(f"Diretório de saída: {OUTPUT_DIR}")
    print(f"Diretório de relatórios: {REPORT_DIR}")
    print("="*80)
    
    try:
        # Processar dados
        processed_data = process_landuse_data()
        
        if processed_data is not None:
            print("\nProcessamento concluído com sucesso!")
            total_time = time.time() - start_time
            print(f"Tempo total de execução: {total_time:.2f} segundos")
            print("="*80)
        else:
            print("\nProcessamento não pôde ser concluído. Verifique os erros acima.")
            print("="*80)
            
    except Exception as e:
        print(f"\nERRO FATAL: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print("="*80)

if __name__ == "__main__":
    main()