#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Módulo para processamento de dados de setores censitários.

Este script realiza o pré-processamento dos dados de setores censitários, incluindo:
- Carregamento de dados
- Validação e correção de geometrias
- Cálculo de métricas socioeconômicas e de densidade populacional
- Geração de relatórios de qualidade
- Exportação de dados processados

O script utiliza processamento paralelo e aceleração via numba para otimizar 
o desempenho em grandes conjuntos de dados.
"""

import os
import sys
import time
import json
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from typing import Dict, List, Tuple, Any, Optional
import warnings
import logging
import multiprocessing as mp
from numba import jit, prange
from functools import partial
from datetime import datetime

# Configurar número de workers baseado em CPU físicas (deixar pelo menos 1 núcleo livre)
NUM_WORKERS = max(1, mp.cpu_count() - 1)
print(f"Configurando processamento paralelo com {NUM_WORKERS} workers")

# Configurar diretórios
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'raw')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
REPORT_DIR = os.path.join(WORKSPACE_DIR, 'src', 'preprocessing', 'quality_reports', 'setores_censitarios')

# Criar diretórios se não existirem
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(REPORT_DIR, 'processing.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('setor_censitario')

# Encoder personalizado para numpy
class NpEncoder(json.JSONEncoder):
    """Classe para serializar tipos do NumPy para JSON."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

# Definir colunas essenciais para setores censitários
ESSENTIAL_COLUMNS = [
    'CD_SETOR', 'NM_MUNIC', 'CD_GEOCODM', 'AREA_KM2', 
    'geometry'
]

# Colunas para calcular métricas socioeconômicas se existirem
SOCIO_COLUMNS = [
    'POP', 'DOM', 'RENDA', 'DENS_POP', 'DENS_DOM'
]

def load_area_of_interest():
    """
    Carrega o polígono da área de interesse (município de Sorocaba).
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame contendo o polígono da área de interesse
    """
    try:
        sorocaba_file = os.path.join(INPUT_DIR, 'sorocaba.gpkg')
        if not os.path.exists(sorocaba_file):
            logger.warning(f"Arquivo de área de interesse não encontrado: {sorocaba_file}")
            return None
            
        sorocaba = gpd.read_file(sorocaba_file)
        
        if sorocaba.empty:
            logger.warning("Arquivo de área de interesse está vazio")
            return None
            
        # Verificar CRS
        if sorocaba.crs is None:
            logger.warning("CRS não definido para área de interesse, assumindo SIRGAS 2000 (EPSG:4674)")
            sorocaba.set_crs(epsg=4674, inplace=True)
        elif sorocaba.crs != "EPSG:4674":
            logger.info(f"Convertendo CRS da área de interesse de {sorocaba.crs} para SIRGAS 2000 (EPSG:4674)")
            sorocaba = sorocaba.to_crs(epsg=4674)
        
        # Verificar geometrias
        invalid_geoms = ~sorocaba.is_valid
        if invalid_geoms.any():
            logger.warning(f"Corrigindo {invalid_geoms.sum()} geometrias inválidas na área de interesse")
            sorocaba.geometry = sorocaba.geometry.buffer(0)
        
        logger.info(f"Área de interesse carregada: {len(sorocaba)} polígonos")
        return sorocaba
    except Exception as e:
        logger.error(f"Erro ao carregar área de interesse: {str(e)}")
        return None

@jit(nopython=True, parallel=True)
def validate_numeric_array(arr, min_val=None, max_val=None):
    """
    Valida e corrige valores numéricos em um array.
    
    Args:
        arr: Array numpy com valores numéricos
        min_val: Valor mínimo permitido (opcional)
        max_val: Valor máximo permitido (opcional)
        
    Returns:
        tuple: (array corrigido, contagem de valores corrigidos)
    """
    # Criar cópia do array para não modificar o original
    result = arr.copy()
    corrections = 0
    
    # Processar em paralelo
    for i in prange(len(arr)):
        # Tratar valores NaN
        if np.isnan(arr[i]):
            result[i] = 0
            corrections += 1
            continue
            
        # Aplicar limites mínimos se especificado
        if min_val is not None and arr[i] < min_val:
            result[i] = min_val
            corrections += 1
            
        # Aplicar limites máximos se especificado
        if max_val is not None and arr[i] > max_val:
            result[i] = max_val
            corrections += 1
            
    return result, corrections

@jit(nopython=True)
def calc_statistics(arr):
    """
    Calcula estatísticas básicas para um array numérico usando Numba para aceleração.
    
    Args:
        arr: Array numpy com valores numéricos
        
    Returns:
        dict: Dicionário com estatísticas calculadas
    """
    # Remover NaN para cálculos
    valid_arr = arr[~np.isnan(arr)]
    
    if len(valid_arr) == 0:
        # Retornar valores nulos se não houver dados válidos
        return {
            'count': 0,
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'sum': np.nan
        }
    
    # Calcular estatísticas básicas
    return {
        'count': len(valid_arr),
        'min': float(np.min(valid_arr)),
        'max': float(np.max(valid_arr)),
        'mean': float(np.mean(valid_arr)),
        'median': float(np.median(valid_arr)),
        'std': float(np.std(valid_arr)),
        'sum': float(np.sum(valid_arr))
    }

def filter_by_area_of_interest(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Filtra setores censitários pela área de interesse.
    
    Args:
        df: GeoDataFrame com os setores censitários
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame filtrado
    """
    try:
        # Carregar área de interesse
        aoi = load_area_of_interest()
        
        if aoi is None:
            logger.warning("Área de interesse não disponível, não será aplicado filtro espacial")
            return df
            
        # Garantir mesmo CRS
        if df.crs != aoi.crs:
            logger.info(f"Convertendo CRS dos setores de {df.crs} para {aoi.crs}")
            df = df.to_crs(aoi.crs)
        
        # Contar setores antes da filtragem
        count_before = len(df)
        
        # Usar spatial join para verificar interseção com a área de interesse
        logger.info("Aplicando filtro espacial pela área de interesse")
        filtered_df = gpd.sjoin(df, aoi, predicate='intersects', how='inner')
        
        # Remover colunas duplicadas do join
        for col in filtered_df.columns:
            if col.endswith('_right') and col.replace('_right', '') in filtered_df.columns:
                filtered_df = filtered_df.drop(col, axis=1)
        
        # Contar setores após filtragem
        count_after = len(filtered_df)
        logger.info(f"Filtro espacial: {count_before} setores antes, {count_after} setores após filtragem")
        
        return filtered_df
    except Exception as e:
        logger.error(f"Erro ao filtrar por área de interesse: {str(e)}")
        return df

def clean_column_names(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Limpa e padroniza nomes de colunas.
    
    Args:
        df: GeoDataFrame com os dados
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame com nomes de colunas padronizados
    """
    # Criar uma cópia para não modificar o original
    result = df.copy()
    
    # Mapear nomes de colunas comuns para o padrão
    column_mapping = {
        'CD_GEOCODI': 'CD_SETOR',
        'CD_GEOCODIG': 'CD_SETOR',
        'GEOCODIGO': 'CD_SETOR',
        'CD_GEOCODB': 'CD_GEOCODM',
        'NM_MUNICIP': 'NM_MUNIC',
        'NOME_MUNIC': 'NM_MUNIC',
        'Shape_Area': 'AREA_KM2',
        'AREA': 'AREA_KM2',
        'NOME_DISTRITO': 'NM_DISTRI',
        'NOME_SUBDISTRITO': 'NM_SUBDIST'
    }
    
    # Aplicar mapeamento apenas para colunas que existem
    for old_col, new_col in column_mapping.items():
        if old_col in result.columns and new_col not in result.columns:
            result = result.rename(columns={old_col: new_col})
    
    return result

def process_chunk(args):
    """
    Processa um chunk de dados em paralelo.
    
    Args:
        args: Tupla contendo (chunk, operação, tipo)
        
    Returns:
        gpd.GeoDataFrame: Chunk processado
    """
    chunk, operation, layer_type = args
    
    if operation == 'validate':
        return validate_chunk(chunk, layer_type)
    elif operation == 'process':
        return process_geometry_chunk(chunk, layer_type)
    else:
        return chunk

def validate_chunk(chunk: gpd.GeoDataFrame, layer_type: str) -> gpd.GeoDataFrame:
    """
    Valida e corrige dados em um chunk.
    
    Args:
        chunk: GeoDataFrame com os dados do chunk
        layer_type: Tipo de camada (para logging)
        
    Returns:
        gpd.GeoDataFrame: Chunk com dados validados
    """
    # Criar uma cópia para não modificar o original
    result = chunk.copy()
    
    # Verificar e tratar valores ausentes
    for col in ESSENTIAL_COLUMNS:
        if col in result.columns and col != 'geometry':
            missing = result[col].isna().sum()
            if missing > 0:
                logger.warning(f"Coluna {col}: {missing} valores ausentes")
                
                # Para códigos de setor e município, não podemos ter valores nulos
                if col in ['CD_SETOR', 'CD_GEOCODM']:
                    # Não é possível recuperar esses valores críticos, remover linhas
                    result = result.dropna(subset=[col])
                elif col == 'AREA_KM2':
                    # Calcular área para valores ausentes
                    mask = result[col].isna()
                    if mask.any():
                        # Converter para projeção métrica para cálculo de área
                        temp = result[mask].to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
                        result.loc[mask, 'AREA_KM2'] = temp.geometry.area / 1_000_000  # metros² para km²
    
    # Verificar e corrigir geometrias
    if 'geometry' in result.columns:
        # Verificar geometrias vazias
        empty_geoms = result.geometry.is_empty
        if empty_geoms.any():
            logger.warning(f"Removendo {empty_geoms.sum()} geometrias vazias")
            result = result[~empty_geoms]
        
        # Verificar geometrias inválidas
        invalid_geoms = ~result.geometry.is_valid
        if invalid_geoms.any():
            logger.warning(f"Corrigindo {invalid_geoms.sum()} geometrias inválidas")
            result.loc[invalid_geoms, 'geometry'] = result.loc[invalid_geoms, 'geometry'].buffer(0)
    
    return result

def process_geometry_chunk(chunk: gpd.GeoDataFrame, layer_type: str) -> gpd.GeoDataFrame:
    """
    Processa geometrias em um chunk.
    
    Args:
        chunk: GeoDataFrame com os dados do chunk
        layer_type: Tipo de camada (para logging)
        
    Returns:
        gpd.GeoDataFrame: Chunk com geometrias processadas
    """
    # Criar uma cópia para não modificar o original
    result = chunk.copy()
    
    # Calcular ou recalcular área se não existir
    if 'AREA_KM2' not in result.columns:
        # Converter para projeção métrica para cálculo de área
        temp = result.to_crs(epsg=31983)  # SIRGAS 2000 / UTM zone 23S
        result['AREA_KM2'] = temp.geometry.area / 1_000_000  # metros² para km²
    
    # Calcular perímetro
    temp = result.to_crs(epsg=31983)
    result['PERIMETER_KM'] = temp.geometry.length / 1_000  # metros para km
    
    # Calcular compacidade (razão de circularidade)
    # Compacidade = 4π × Área / Perímetro²
    # Valor de 1 = círculo perfeito, valores menores = mais irregular
    result['COMPACTNESS'] = (4 * np.pi * result['AREA_KM2']) / (result['PERIMETER_KM'] ** 2)
    
    # Calcular densidade populacional se houver dados de população
    if 'POP' in result.columns and 'AREA_KM2' in result.columns:
        result['DENS_POP'] = result['POP'] / result['AREA_KM2']
    
    # Calcular densidade de domicílios se houver dados
    if 'DOM' in result.columns and 'AREA_KM2' in result.columns:
        result['DENS_DOM'] = result['DOM'] / result['AREA_KM2']
    
    return result

def parallel_process(df: gpd.GeoDataFrame, layer_type: str, operation: str) -> gpd.GeoDataFrame:
    """
    Realiza processamento paralelo em um GeoDataFrame.
    
    Args:
        df: GeoDataFrame com os dados
        layer_type: Tipo de camada (para logging)
        operation: Tipo de operação ('validate' ou 'process')
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame processado
    """
    if len(df) == 0:
        return df
    
    # Determinar número de chunks baseado em CPU e tamanho dos dados
    n_chunks = min(NUM_WORKERS * 2, max(NUM_WORKERS, int(len(df) / 1000)))
    chunk_size = max(1, len(df) // n_chunks)
    
    # Dividir em chunks
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    logger.info(f"Processando {len(df)} linhas em {len(chunks)} chunks")
    
    # Preparar argumentos para processamento paralelo
    args = [(chunk, operation, layer_type) for chunk in chunks]
    
    # Processar em paralelo
    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_chunk, args)
    
    # Combinar resultados
    result = pd.concat(results)
    
    return result

def check_topology(df: gpd.GeoDataFrame, layer_type: str) -> Dict:
    """
    Verifica problemas de topologia nos dados.
    
    Args:
        df: GeoDataFrame com os dados
        layer_type: Tipo de camada (para logging)
        
    Returns:
        dict: Relatório com problemas de topologia
    """
    topology_report = {
        'overlaps': 0,
        'gaps': 0,
        'invalid_geometries': 0,
        'self_intersections': 0
    }
    
    try:
        # Verificar geometrias inválidas
        topology_report['invalid_geometries'] = (~df.is_valid).sum()
        
        # Para setores censitários (polígonos), verificar sobreposições
        # Isso pode ser computacionalmente intensivo, então limitamos aos primeiros N setores
        if len(df) > 100:
            sample_size = 100
            logger.info(f"Verificando sobreposições em uma amostra de {sample_size} setores")
            sample = df.sample(sample_size) if len(df) > sample_size else df
        else:
            sample = df
        
        overlaps = 0
        for i, row1 in sample.iterrows():
            for j, row2 in sample.iterrows():
                if i < j:  # Evitar comparações duplicadas
                    if row1.geometry.intersects(row2.geometry):
                        intersection = row1.geometry.intersection(row2.geometry)
                        if not intersection.is_empty and intersection.area > 0:
                            overlaps += 1
        
        # Extrapolar para o conjunto completo
        if len(df) > sample_size:
            overlap_ratio = overlaps / (sample_size * (sample_size - 1) / 2)
            estimated_overlaps = int(overlap_ratio * (len(df) * (len(df) - 1) / 2))
            topology_report['overlaps'] = estimated_overlaps
        else:
            topology_report['overlaps'] = overlaps
        
        # Verificar auto-interseções (mais comum em linhas, mas pode ocorrer em polígonos)
        self_intersections = 0
        for geom in sample.geometry:
            if not geom.is_simple:
                self_intersections += 1
        
        # Extrapolar para o conjunto completo
        if len(df) > sample_size:
            self_intersection_ratio = self_intersections / sample_size
            estimated_self_intersections = int(self_intersection_ratio * len(df))
            topology_report['self_intersections'] = estimated_self_intersections
        else:
            topology_report['self_intersections'] = self_intersections
        
        return topology_report
    except Exception as e:
        logger.error(f"Erro ao verificar topologia: {str(e)}")
        return topology_report

def calculate_statistics_with_numba(df: pd.DataFrame, col: str) -> Dict[str, Any]:
    """
    Calcula estatísticas para uma coluna usando Numba.
    
    Args:
        df: DataFrame com os dados
        col: Nome da coluna
        
    Returns:
        dict: Estatísticas calculadas
    """
    try:
        if col not in df.columns:
            return {}
            
        # Converter para array numpy para processamento com Numba
        arr = np.array(df[col], dtype=np.float64)
        return calc_statistics(arr)
    except Exception as e:
        logger.error(f"Erro ao calcular estatísticas para {col}: {str(e)}")
        return {}

def create_quality_report(df: gpd.GeoDataFrame, layer_type: str, output_file: str):
    """
    Cria relatório de qualidade para os dados.
    
    Args:
        df: GeoDataFrame com os dados
        layer_type: Tipo de camada (para logging)
        output_file: Caminho para salvar o relatório
        
    Returns:
        dict: Relatório de qualidade
    """
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'layer_type': layer_type,
            'total_features': len(df),
            'columns': list(df.columns),
            'geometry_type': df.geometry.geom_type.value_counts().to_dict(),
            'validation': {
                'missing_values': {col: int(df[col].isna().sum()) for col in df.columns if col != 'geometry'},
                'invalid_geometries': int((~df.geometry.is_valid).sum()),
                'empty_geometries': int(df.geometry.is_empty.sum())
            },
            'statistics': {},
            'topology': check_topology(df, layer_type)
        }
        
        # Calcular estatísticas para colunas numéricas
        numeric_columns = df.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            report['statistics'][col] = calculate_statistics_with_numba(df, col)
        
        # Calcular estatísticas para setores específicas
        report['setor_stats'] = {
            'area_km2': calculate_statistics_with_numba(df, 'AREA_KM2'),
            'perimeter_km': calculate_statistics_with_numba(df, 'PERIMETER_KM'),
            'compactness': calculate_statistics_with_numba(df, 'COMPACTNESS')
        }
        
        # Adicionar estatísticas socioeconômicas se existirem
        if 'POP' in df.columns:
            report['setor_stats']['population'] = calculate_statistics_with_numba(df, 'POP')
        
        if 'DENS_POP' in df.columns:
            report['setor_stats']['population_density'] = calculate_statistics_with_numba(df, 'DENS_POP')
        
        if 'DOM' in df.columns:
            report['setor_stats']['households'] = calculate_statistics_with_numba(df, 'DOM')
        
        if 'DENS_DOM' in df.columns:
            report['setor_stats']['household_density'] = calculate_statistics_with_numba(df, 'DENS_DOM')
        
        if 'RENDA' in df.columns:
            report['setor_stats']['income'] = calculate_statistics_with_numba(df, 'RENDA')
        
        # Salvar relatório como JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, cls=NpEncoder, ensure_ascii=False)
        
        logger.info(f"Relatório de qualidade salvo em {output_file}")
        
        # Também exportar para formato de texto para fácil visualização
        txt_file = output_file.replace('.json', '.txt')
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"=== RELATÓRIO DE QUALIDADE: {layer_type} ===\n\n")
            f.write(f"Timestamp: {report['timestamp']}\n")
            f.write(f"Total de feições: {report['total_features']}\n")
            f.write(f"Tipo de geometria: {report['geometry_type']}\n\n")
            
            f.write("=== VALIDAÇÃO ===\n")
            f.write(f"Geometrias inválidas: {report['validation']['invalid_geometries']}\n")
            f.write(f"Geometrias vazias: {report['validation']['empty_geometries']}\n")
            f.write("Valores ausentes por coluna:\n")
            for col, count in report['validation']['missing_values'].items():
                if count > 0:
                    f.write(f"  - {col}: {count}\n")
            
            f.write("\n=== TOPOLOGIA ===\n")
            for issue, count in report['topology'].items():
                f.write(f"{issue}: {count}\n")
            
            f.write("\n=== ESTATÍSTICAS DOS SETORES ===\n")
            for stat_name, values in report['setor_stats'].items():
                f.write(f"{stat_name}:\n")
                for metric, value in values.items():
                    f.write(f"  - {metric}: {value}\n")
                f.write("\n")
        
        logger.info(f"Relatório de texto salvo em {txt_file}")
        
        return report
    except Exception as e:
        logger.error(f"Erro ao criar relatório de qualidade: {str(e)}")
        return {'error': str(e)}

def process_setores_censitarios(input_file: str, output_file: str) -> gpd.GeoDataFrame:
    """
    Processa dados de setores censitários.
    
    Args:
        input_file: Caminho para o arquivo de entrada
        output_file: Caminho para salvar o arquivo processado
        
    Returns:
        gpd.GeoDataFrame: GeoDataFrame processado
    """
    start_time = time.time()
    logger.info(f"Iniciando processamento de setores censitários: {input_file}")
    
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(input_file):
            logger.error(f"Arquivo não encontrado: {input_file}")
            return None
        
        # Carregar dados
        logger.info("Carregando dados...")
        df = gpd.read_file(input_file)
        
        if df.empty:
            logger.error("Arquivo está vazio")
            return None
        
        # Verificar e definir CRS
        if df.crs is None:
            logger.warning("CRS não definido, assumindo SIRGAS 2000 (EPSG:4674)")
            df.set_crs(epsg=4674, inplace=True)
        elif df.crs != "EPSG:4674":
            logger.info(f"Convertendo CRS de {df.crs} para SIRGAS 2000 (EPSG:4674)")
            df = df.to_crs(epsg=4674)
        
        # Limpar e padronizar nomes de colunas
        logger.info("Padronizando nomes de colunas...")
        df = clean_column_names(df)
        
        # Validar dados em paralelo
        logger.info("Validando dados...")
        df = parallel_process(df, 'setores_censitarios', 'validate')
        
        # Filtrar pela área de interesse (Sorocaba)
        logger.info("Filtrando pela área de interesse...")
        df = filter_by_area_of_interest(df)
        
        # Processar geometrias em paralelo
        logger.info("Processando geometrias...")
        df = parallel_process(df, 'setores_censitarios', 'process')
        
        # Criar relatório de qualidade
        logger.info("Gerando relatório de qualidade...")
        quality_report_file = os.path.join(REPORT_DIR, 'quality_report_setores_censitarios.json')
        create_quality_report(df, 'setores_censitarios', quality_report_file)
        
        # Salvar dados processados
        logger.info(f"Salvando dados processados em {output_file}")
        df.to_file(output_file, driver='GPKG')
        
        elapsed_time = time.time() - start_time
        logger.info(f"Processamento concluído em {elapsed_time:.2f} segundos")
        
        # Informações sobre os dados processados
        logger.info(f"Total de setores processados: {len(df)}")
        if 'POP' in df.columns:
            total_pop = df['POP'].sum()
            logger.info(f"População total: {total_pop}")
        
        return df
    except Exception as e:
        logger.error(f"Erro ao processar setores censitários: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    """Função principal para processamento de setores censitários."""
    start_time = time.time()
    
    input_file = os.path.join(INPUT_DIR, 'sorocaba_setores_censitarios.gpkg')
    output_file = os.path.join(OUTPUT_DIR, 'setores_censitarios_processed.gpkg')
    
    try:
        # Processar setores censitários
        df = process_setores_censitarios(input_file, output_file)
        
        if df is not None:
            print("\nProcessamento dos setores censitários concluído com sucesso")
            print(f"Processados {len(df)} setores")
            
            # Mostrar estatísticas básicas
            if 'AREA_KM2' in df.columns:
                total_area = df['AREA_KM2'].sum()
                print(f"Área total: {total_area:.2f} km²")
            
            if 'POP' in df.columns:
                total_pop = df['POP'].sum()
                print(f"População total: {total_pop}")
            
            total_time = time.time() - start_time
            print(f"Tempo total de execução: {total_time:.2f} segundos")
            
        else:
            print("\nErro no processamento dos setores censitários")
            
    except Exception as e:
        print(f"Erro ao processar dados de setores censitários: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
