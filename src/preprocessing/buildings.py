"""
Funções para processamento de dados de edifícios com processamento paralelo e aceleração Numba.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import os
import json
from typing import Union, List, Dict, Optional, Tuple, Any
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
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
REPORT_DIR = os.path.join(os.path.dirname(__file__), 'quality_reports', 'buildings')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Arquivo de entrada dos edifícios
BUILDINGS_FILE = os.path.join(INPUT_DIR, 'sorocaba_buildings.gpkg')

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

# Define colunas essenciais e metadados para os edifícios
BUILDING_COLUMNS = {
    'osm_id': {'type': 'str', 'description': 'ID único do edifício no OpenStreetMap'},
    'name': {'type': 'str', 'description': 'Nome do edifício'},
    'building': {'type': 'str', 'description': 'Tipo de edifício'},
    'amenity': {'type': 'str', 'description': 'Serviço/Facilidade do edifício'},
    'shop': {'type': 'str', 'description': 'Tipo de comércio, se aplicável'},
    'landuse': {'type': 'str', 'description': 'Uso do solo, se aplicável'},
    'height': {'type': 'float64', 'description': 'Altura do edifício em metros', 'validation': {'min': 0}},
    'levels': {'type': 'int32', 'description': 'Número de andares', 'validation': {'min': 1}},
    'area_m2': {'type': 'float64', 'description': 'Área do edifício em metros quadrados', 'validation': {'min': 0}},
    'perimeter_m': {'type': 'float64', 'description': 'Perímetro do edifício em metros', 'validation': {'min': 0}},
    'building_class': {'type': 'str', 'description': 'Classificação do edifício'},
    'compactness_index': {'type': 'float64', 'description': 'Índice de compacidade (circularidade) do edifício', 'validation': {'min': 0, 'max': 1}}
}

# Mapeamento para uso do solo/edifícios
BUILDING_CLASS_MAPPING = {
    'house': 'residential',
    'residential': 'residential',
    'apartments': 'residential',
    'detached': 'residential',
    'semidetached_house': 'residential',
    'terrace': 'residential',
    'dormitory': 'residential',
    
    'commercial': 'commercial',
    'retail': 'commercial',
    'shop': 'commercial',
    'supermarket': 'commercial',
    'department_store': 'commercial',
    'mall': 'commercial',
    'kiosk': 'commercial',
    
    'industrial': 'industrial',
    'warehouse': 'industrial',
    'factory': 'industrial',
    'manufacture': 'industrial',
    
    'office': 'office',
    'government': 'office',
    'civic': 'office',
    
    'school': 'educational',
    'university': 'educational',
    'college': 'educational',
    'kindergarten': 'educational',
    'education': 'educational',
    
    'hospital': 'healthcare',
    'clinic': 'healthcare',
    'doctors': 'healthcare',
    'dentist': 'healthcare',
    'pharmacy': 'healthcare',
    
    'church': 'religious',
    'mosque': 'religious',
    'temple': 'religious',
    'synagogue': 'religious',
    'shrine': 'religious',
    'place_of_worship': 'religious',
    
    'garage': 'utility',
    'garages': 'utility',
    'parking': 'utility',
    'service': 'utility',
    'shed': 'utility',
    'storage_tank': 'utility',
    
    'hotel': 'accommodation',
    'hostel': 'accommodation',
    'guest_house': 'accommodation',
    'motel': 'accommodation',
    
    'sports_centre': 'leisure',
    'stadium': 'leisure',
    'grandstand': 'leisure',
    'pavilion': 'leisure',
    'restaurant': 'leisure',
    'fast_food': 'leisure',
    'cafe': 'leisure',
    'pub': 'leisure',
    'bar': 'leisure',
    'cinema': 'leisure',
    'theatre': 'leisure',
    'library': 'leisure',
    'museum': 'leisure',
    
    'construction': 'construction',
    'ruins': 'abandoned',
    'collapsed': 'abandoned',
    'damaged': 'abandoned',
    'demolished': 'abandoned',
    
    'yes': 'unclassified',
    'roof': 'unclassified',
    'hut': 'unclassified',
    'cabin': 'unclassified'
}

# Funções aceleradas com Numba para cálculos geométricos
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
    # Validar altura e número de andares
    if 'height' in chunk.columns:
        arr = chunk['height'].values
        invalid_mask = validate_numeric_array(arr, min_val=0)
        if invalid_mask.any():
            chunk.loc[invalid_mask, 'height'] = np.nan
    
    if 'levels' in chunk.columns:
        arr = chunk['levels'].values
        invalid_mask = validate_numeric_array(arr, min_val=1)
        if invalid_mask.any():
            chunk.loc[invalid_mask, 'levels'] = np.nan
    
    # Validar área e perímetro
    if 'area_m2' in chunk.columns:
        arr = chunk['area_m2'].values
        invalid_mask = validate_numeric_array(arr, min_val=0)
        if invalid_mask.any():
            chunk.loc[invalid_mask, 'area_m2'] = np.nan
    
    if 'perimeter_m' in chunk.columns:
        arr = chunk['perimeter_m'].values
        invalid_mask = validate_numeric_array(arr, min_val=0)
        if invalid_mask.any():
            chunk.loc[invalid_mask, 'perimeter_m'] = np.nan
    
    # Validar índice de compacidade
    if 'compactness_index' in chunk.columns:
        arr = chunk['compactness_index'].values
        invalid_mask = validate_numeric_array(arr, min_val=0, max_val=1)
        if invalid_mask.any():
            chunk.loc[invalid_mask, 'compactness_index'] = np.nan
    
    return chunk

def process_geometry_chunk(chunk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Processa um chunk de geometrias."""
    # Verificar se existem geometrias
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
    
    # Converter MultiPolygon para Polygon quando possível (pegar o maior polígono)
    multipoly_mask = chunk.geometry.type == 'MultiPolygon'
    if multipoly_mask.any():
        chunk.loc[multipoly_mask, 'geometry'] = chunk.loc[multipoly_mask, 'geometry'].apply(
            lambda x: max(x.geoms, key=lambda g: g.area) if x.geom_type == 'MultiPolygon' else x
        )
    
    try:
        # Reprojetar para CRS projetado para cálculos métricos
        chunk_proj = chunk.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        
        # Calcular área em m²
        if 'area_m2' not in chunk.columns:
            chunk['area_m2'] = chunk_proj.geometry.area
        
        # Calcular perímetro em m
        if 'perimeter_m' not in chunk.columns:
            chunk['perimeter_m'] = chunk_proj.geometry.length
        
        # Calcular índice de compacidade (circularidade)
        # Um círculo tem índice = 1, formas irregulares têm valores menores
        if 'compactness_index' not in chunk.columns:
            # Fórmula: 4π*área / perímetro²
            chunk['compactness_index'] = (4 * np.pi * chunk['area_m2']) / (chunk['perimeter_m'] ** 2)
            # Limitar valores entre 0 e 1 (evitar problemas com geometrias inválidas)
            chunk['compactness_index'] = chunk['compactness_index'].clip(0, 1)
        
    except Exception as e:
        print(f"ERRO no processamento de geometria: {str(e)}")
        # Garantir que as colunas existam mesmo em caso de erro
        for col in ['area_m2', 'perimeter_m', 'compactness_index']:
            if col not in chunk.columns:
                chunk[col] = np.nan
    
    return chunk

def process_attributes_chunk(chunk: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Processa atributos de um chunk."""
    # Extrair informações de altura e níveis de other_tags se necessário
    if 'other_tags' in chunk.columns:
        # Extrair building:levels
        if 'levels' not in chunk.columns or chunk['levels'].isna().all():
            chunk['levels'] = chunk['other_tags'].apply(extract_levels)
        
        # Extrair height
        if 'height' not in chunk.columns or chunk['height'].isna().all():
            chunk['height'] = chunk['other_tags'].apply(extract_height)
    
    # Classificar edifícios
    if 'building' in chunk.columns:
        chunk['building_class'] = chunk['building'].map(BUILDING_CLASS_MAPPING).fillna('unclassified')
    elif 'amenity' in chunk.columns:
        # Se não há building mas há amenity, usar amenity para classificação
        chunk['building_class'] = chunk['amenity'].map(
            lambda x: BUILDING_CLASS_MAPPING.get(x, 'unclassified')
        ).fillna('unclassified')
    elif 'shop' in chunk.columns:
        # Se não há building/amenity mas há shop, classificar como comercial
        chunk.loc[~chunk['shop'].isna(), 'building_class'] = 'commercial'
    else:
        chunk['building_class'] = 'unclassified'
    
    return chunk

def extract_levels(other_tags):
    """Extrai o número de andares de uma string other_tags."""
    if pd.isna(other_tags):
        return np.nan
    
    try:
        if '"building:levels"=>' in other_tags:
            # Exemplo: '"building:levels"=>"2"'
            parts = other_tags.split('"building:levels"=>')
            if len(parts) > 1:
                level_part = parts[1].split(',')[0].strip('"')
                # Remover aspas extras
                level_part = level_part.strip('"\'')
                return int(float(level_part))
    except:
        pass
    
    return np.nan

def extract_height(other_tags):
    """Extrai a altura de uma string other_tags."""
    if pd.isna(other_tags):
        return np.nan
    
    try:
        if '"height"=>' in other_tags:
            # Exemplo: '"height"=>"10"' ou '"height"=>"10m"'
            parts = other_tags.split('"height"=>')
            if len(parts) > 1:
                height_part = parts[1].split(',')[0].strip('"')
                # Remover aspas extras
                height_part = height_part.strip('"\'')
                # Remover 'm' se presente
                height_part = height_part.replace('m', '')
                return float(height_part)
    except:
        pass
    
    return np.nan

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

def check_topology(df: gpd.GeoDataFrame) -> Dict:
    """Verifica a topologia dos edifícios."""
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
            'multipoly_count': (df.geometry.type == 'MultiPolygon').sum(),
            'overlapping_buildings': None,  # Será calculado abaixo
            'total_area_m2': float(df['area_m2'].sum()) if 'area_m2' in df.columns else None
        }
        
        # Verificar edifícios sobrepostos
        # Esta operação pode ser intensiva para conjuntos grandes de dados
        if len(df) < 10000:  # Limitar para conjuntos menores de dados
            try:
                # Reprojetar para sistema métrico para cálculos de área
                df_proj = df.to_crs(epsg=31983)
                
                # Contar quantos edifícios se sobrepõem
                overlaps = 0
                for i, building in df_proj.iterrows():
                    # Criar índice espacial para otimizar
                    spatial_index = df_proj.sindex
                    possible_matches_index = list(spatial_index.intersection(building.geometry.bounds))
                    possible_matches = df_proj.iloc[possible_matches_index]
                    
                    # Remover o próprio edifício
                    possible_matches = possible_matches[possible_matches.index != i]
                    
                    # Verificar interseções
                    for j, other_building in possible_matches.iterrows():
                        if building.geometry.intersects(other_building.geometry):
                            overlaps += 1
                            break  # Parar após encontrar o primeiro edifício sobreposto
                
                topology_report['overlapping_buildings'] = overlaps
                
            except Exception as e:
                print(f"Erro ao verificar sobreposições: {str(e)}")
                topology_report['overlapping_buildings_error'] = str(e)
        else:
            topology_report['overlapping_buildings'] = "Não calculado (conjunto de dados muito grande)"
        
        return topology_report
        
    except Exception as e:
        return {
            'total_features': len(df),
            'error': f'Erro na análise topológica: {str(e)}'
        }

def create_quality_report(df: gpd.GeoDataFrame, output_file: str):
    """Cria um relatório de qualidade para os edifícios."""
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
        'building_statistics': {
            'total_area_m2': float(df_proj['area_m2'].sum()),
            'mean_area_m2': float(df_proj['area_m2'].mean()),
            'building_classes': df['building_class'].value_counts().to_dict() if 'building_class' in df.columns else {},
            'building_types': df['building'].value_counts().to_dict() if 'building' in df.columns else {}
        },
        'numeric_statistics': {},
        'topology': check_topology(df),
        'bounds': {
            'minx': float(df.total_bounds[0]),
            'miny': float(df.total_bounds[1]),
            'maxx': float(df.total_bounds[2]),
            'maxy': float(df.total_bounds[3])
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Adicionar informação sobre área de interesse
    if AREA_OF_INTEREST is not None:
        aoi_proj = AREA_OF_INTEREST.to_crs(epsg=31983)
        report['area_of_interest'] = {
            'file': SOROCABA_SHAPEFILE,
            'crs': str(AREA_OF_INTEREST.crs),
            'area_km2': float(aoi_proj.area.sum() / 1e6)
        }
    
    # Calcular estatísticas para colunas numéricas
    numeric_cols = ['area_m2', 'perimeter_m', 'height', 'levels', 'compactness_index']
    for col in numeric_cols:
        if col in df.columns:
            report['numeric_statistics'][col] = calculate_statistics_with_numba(df, col)
    
    # Salvar relatório
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, cls=NpEncoder)
        
    # Criar também uma versão resumida em texto para fácil leitura
    text_report_file = os.path.join(os.path.dirname(output_file), 'quality_report_buildings.txt')
    with open(text_report_file, 'w', encoding='utf-8') as f:
        f.write("===== RELATÓRIO DE QUALIDADE DOS EDIFÍCIOS =====\n\n")
        f.write(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("== RESUMO DOS DADOS ==\n")
        f.write(f"Total de feições: {report['data_summary']['total_features']}\n")
        f.write(f"CRS: {report['data_summary']['crs']}\n")
        f.write(f"Tipos de geometria: {', '.join(report['data_summary']['geometry_types'])}\n")
        f.write(f"Uso de memória: {report['data_summary']['memory_usage_mb']:.2f} MB\n\n")
        
        f.write("== ESTATÍSTICAS DOS EDIFÍCIOS ==\n")
        f.write(f"Área total: {report['building_statistics']['total_area_m2']:.2f} m²\n")
        f.write(f"Área média: {report['building_statistics']['mean_area_m2']:.2f} m²\n")
        
        if 'building_classes' in report['building_statistics']:
            f.write("\nClasses de edifícios:\n")
            for cls, count in report['building_statistics']['building_classes'].items():
                f.write(f"- {cls}: {count} edifícios\n")
                
        if 'building_types' in report['building_statistics']:
            f.write("\nTipos de edifícios (OpenStreetMap):\n")
            total_count = 0
            for btype, count in sorted(report['building_statistics']['building_types'].items(), key=lambda x: x[1], reverse=True):
                total_count += count
                # Limitar a 15 tipos mais comuns para evitar relatórios muito grandes
                if total_count <= len(df) * 0.9:  # Até 90% dos edifícios
                    f.write(f"- {btype}: {count} edifícios\n")
        
        f.write("\n== ANÁLISE TOPOLÓGICA ==\n")
        if 'topology' in report:
            f.write(f"Geometrias inválidas: {report['topology'].get('invalid_geometries', 'N/A')}\n")
            f.write(f"Multipolígonos: {report['topology'].get('multipoly_count', 'N/A')}\n")
            f.write(f"Edifícios sobrepostos: {report['topology'].get('overlapping_buildings', 'N/A')}\n")
        
        f.write("\n== ESTATÍSTICAS DE ALTURA E TAMANHO ==\n")
        if 'height' in report['numeric_statistics']:
            height_stats = report['numeric_statistics']['height']
            f.write(f"Altura: média {height_stats['mean']:.2f}m, min {height_stats['min']:.2f}m, max {height_stats['max']:.2f}m\n")
        
        if 'levels' in report['numeric_statistics']:
            levels_stats = report['numeric_statistics']['levels']
            f.write(f"Número de andares: média {levels_stats['mean']:.2f}, min {levels_stats['min']:.0f}, max {levels_stats['max']:.0f}\n")
        
        if 'compactness_index' in report['numeric_statistics']:
            compactness_stats = report['numeric_statistics']['compactness_index']
            f.write(f"Índice de compacidade: média {compactness_stats['mean']:.2f}, min {compactness_stats['min']:.2f}, max {compactness_stats['max']:.2f}\n")
    
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

def analyze_building_density(df: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame = None) -> Dict:
    """
    Analisa a densidade de edifícios por área.
    
    Args:
        df: GeoDataFrame com os edifícios
        aoi: GeoDataFrame com a área de interesse (opcional)
        
    Returns:
        Dicionário com métricas de densidade
    """
    # Reprojetar para sistema métrico para cálculos de área
    df_proj = df.to_crs(epsg=31983)
    
    # Definir área de estudo
    if aoi is not None:
        study_area = aoi.to_crs(epsg=31983)
        area_km2 = study_area.area.sum() / 1e6
    else:
        # Usar o envelope convexo dos edifícios como área de estudo
        total_geom = df_proj.geometry.union_all()
        study_area = gpd.GeoDataFrame(geometry=[total_geom.convex_hull], crs=df_proj.crs)
        area_km2 = study_area.area.sum() / 1e6
    
    # Calcular métricas de densidade
    building_count = len(df_proj)
    buildings_per_km2 = building_count / area_km2
    
    # Calcular área construída total e proporção em relação à área de estudo
    total_building_area_km2 = df_proj['area_m2'].sum() / 1e6
    built_up_ratio = total_building_area_km2 / area_km2
    
    return {
        'study_area_km2': float(area_km2),
        'building_count': building_count,
        'buildings_per_km2': float(buildings_per_km2),
        'total_building_area_km2': float(total_building_area_km2),
        'built_up_ratio': float(built_up_ratio),
        'built_up_percentage': float(built_up_ratio * 100)
    }

def process_building_data(input_file: str, output_dir: str) -> gpd.GeoDataFrame:
    """
    Processa os dados de edifícios.
    
    Args:
        input_file: Caminho para o arquivo GPKG de entrada
        output_dir: Diretório para salvar os dados processados
        
    Returns:
        GeoDataFrame com os dados processados
    """
    start_time = time.time()
    print(f"Processando dados de edifícios de: {input_file}")
    
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
                backup_file = os.path.join(output_dir, 'buildings_backup.gpkg')
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
        
        # Gerar relatório
        report_file = os.path.join(REPORT_DIR, 'quality_report_buildings.json')
        create_quality_report(df, report_file)
        print(f"Relatório de qualidade salvo em: {report_file}")
        
        # Calcular densidade de edifícios
        density_report = analyze_building_density(df, AREA_OF_INTEREST)
        density_file = os.path.join(REPORT_DIR, 'density_report_buildings.json')
        with open(density_file, 'w', encoding='utf-8') as f:
            json.dump(density_report, f, indent=4, cls=NpEncoder)
        print(f"Relatório de densidade salvo em: {density_file}")
        
        # Salvar dados processados
        output_file = os.path.join(output_dir, 'buildings_processed.gpkg')
        df.to_file(output_file, driver='GPKG')
        print(f"Dados processados salvos em: {output_file}")
        
        elapsed_time = time.time() - start_time
        print(f"Tempo total de processamento: {elapsed_time:.2f} segundos")
        
        # Verificar qualidade dos dados processados
        if 'area_m2' in df.columns:
            total_area = df['area_m2'].sum()
            if total_area < 1.0:  # Se área total for muito pequena
                print(f"AVISO: Área total dos edifícios muito pequena: {total_area:.2f} m²")
                print("      Verifique se a reprojeção está correta ou se os dados estão filtrados incorretamente")
        
        return df
        
    except Exception as e:
        print(f"ERRO ao processar dados de edifícios: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def main():
    """Função principal."""
    print(f"\n{'='*80}")
    print(f"{'Processamento de Edifícios':^80}")
    print(f"{'='*80}\n")
    
    print(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Diretório de trabalho: {WORKSPACE_DIR}")
    print(f"Arquivo de entrada: {BUILDINGS_FILE}")
    print(f"Diretório de saída: {OUTPUT_DIR}")
    print(f"Relatórios: {REPORT_DIR}\n")
    
    try:
        processed_data = process_building_data(BUILDINGS_FILE, OUTPUT_DIR)
        if processed_data is not None:
            print("\nProcessamento dos dados de edifícios concluído com sucesso")
            print(f"Total de edifícios processados: {len(processed_data)}")
            
            if 'building_class' in processed_data.columns:
                print("\nDistribuição por classe de edifícios:")
                for class_name, count in processed_data['building_class'].value_counts().items():
                    percent = (count / len(processed_data)) * 100
                    print(f"  - {class_name}: {count} edifícios ({percent:.1f}%)")
                    
            if 'area_m2' in processed_data.columns:
                total_area = processed_data['area_m2'].sum()
                print(f"\nÁrea total construída: {total_area:.2f} m²")
                
                # Área por classe
                if 'building_class' in processed_data.columns:
                    print("Área por classe de edifício:")
                    for class_name, group in processed_data.groupby('building_class'):
                        class_area = group['area_m2'].sum()
                        percent = (class_area / total_area) * 100
                        print(f"  - {class_name}: {class_area:.2f} m² ({percent:.1f}%)")
                
            # Verificar altura média
            if 'height' in processed_data.columns and not processed_data['height'].isna().all():
                height_mean = processed_data['height'].mean()
                height_median = processed_data['height'].median()
                print(f"\nAltura média dos edifícios: {height_mean:.2f} m (mediana: {height_median:.2f} m)")
            
            # Verificar andares médios
            if 'levels' in processed_data.columns and not processed_data['levels'].isna().all():
                levels_mean = processed_data['levels'].mean()
                levels_median = processed_data['levels'].median()
                print(f"Número médio de andares: {levels_mean:.2f} (mediana: {levels_median:.2f})")
                
            # Salvar um sumário em texto para referência fácil
            try:
                summary_file = os.path.join(REPORT_DIR, 'buildings_summary.txt')
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(f"Sumário do Processamento de Edifícios\n")
                    f.write(f"Data/Hora: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"Total de edifícios: {len(processed_data)}\n")
                    
                    if 'building_class' in processed_data.columns:
                        f.write("\nDistribuição por classe de edifícios:\n")
                        for class_name, count in processed_data['building_class'].value_counts().items():
                            percent = (count / len(processed_data)) * 100
                            f.write(f"  - {class_name}: {count} edifícios ({percent:.1f}%)\n")
                            
                    if 'area_m2' in processed_data.columns:
                        f.write(f"\nÁrea total construída: {processed_data['area_m2'].sum():.2f} m²\n")
                        
                print(f"\nSumário salvo em: {summary_file}")
            except Exception as e:
                print(f"AVISO: Não foi possível salvar o sumário: {str(e)}")
        else:
            print("ERRO: Não foi possível processar os dados de edifícios")
            
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