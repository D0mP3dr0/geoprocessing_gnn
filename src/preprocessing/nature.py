#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para pré-processamento de dados de áreas naturais.

Este script realiza o processamento dos dados brutos de áreas naturais,
aplicando validações, limpezas e transformações necessárias para
produzir um conjunto de dados pronto para análise.

Autor: Usuário
Data: 2024
"""

import os
import time
import json
import datetime
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely import make_valid
from numba import jit, prange
import numba as nb
import warnings
warnings.filterwarnings('ignore', 'GeoSeries.isna', FutureWarning)
import traceback
import multiprocessing
import matplotlib.pyplot as plt

# Configurações
N_WORKERS = os.cpu_count() - 1 or 1  # Deixar pelo menos 1 CPU livre
PARTITION_SIZE = 1000  # Tamanho dos chunks para processamento

# Definições de caminhos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_PATH = os.path.join(BASE_DIR, "data/raw/sorocaba_natural.gpkg")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/processed/natural_areas_processed.gpkg")
REPORT_PATH = os.path.join(BASE_DIR, "src/preprocessing/quality_reports/nature")

# Definições das colunas essenciais e metadados
ESSENTIAL_COLUMNS = [
    'geometry',    # Geometria do polígono (obrigatório)
    'name',        # Nome da área natural (opcional)
    'type',        # Tipo de área natural (floresta, wetland, etc.)
    'protected'    # Status de proteção (booleano)
]

# Funções de validação e processamento
def validate_data(gdf):
    """
    Valida um GeoDataFrame de áreas naturais
    
    Args:
        gdf (GeoDataFrame): DataFrame a ser validado
        
    Returns:
        tuple: (relatório de validação, GeoDataFrame validado)
    """
    validation_report = {
        "missing_columns": [],
        "invalid_values": {},
        "null_counts": {},
        "total_records": len(gdf),
        "valid_records": 0
    }
    
    # Verificar colunas existentes - apenas reporta, não adiciona
    for col in ESSENTIAL_COLUMNS:
        if col not in gdf.columns:
            if col != 'geometry':  # Geometria é obrigatória
                validation_report["missing_columns"].append(col)
            else:
                raise ValueError(f"Coluna obrigatória 'geometry' não encontrada no arquivo")
    
    # Verificar valores nulos
    for col in gdf.columns:
        null_count = gdf[col].isna().sum()
        if null_count > 0:
            validation_report["null_counts"][col] = int(null_count)
    
    # Verificar geometrias nulas ou inválidas
    invalid_geom_mask = gdf.geometry.isna() | ~gdf.geometry.is_valid
    validation_report["invalid_geometries"] = int(invalid_geom_mask.sum())
    
    # Verificar tipo da área natural - verifica apenas se a coluna existe
    if 'type' in gdf.columns:
        invalid_types = gdf[~gdf['type'].isin(['forest', 'wetland', 'grassland', 'park', 'reserve', 'other'])].index.tolist()
        if invalid_types:
            validation_report["invalid_values"]["type"] = len(invalid_types)
            # Definir valores inválidos como 'other'
            gdf.loc[invalid_types, 'type'] = 'other'
    
    # Verificar status de proteção - verifica apenas se a coluna existe
    if 'protected' in gdf.columns:
        # Certifique-se de que protected é booleano
        if gdf['protected'].dtype != bool:
            try:
                # Tenta converter para booleano
                gdf['protected'] = gdf['protected'].astype(bool)
            except:
                # Se falhar, reporta mas não modifica
                validation_report["invalid_values"]["protected"] = len(gdf)
    
    # Registros válidos são aqueles sem problemas de geometria
    validation_report["valid_records"] = int((~invalid_geom_mask).sum())
    
    return validation_report, gdf

def clean_and_standardize_columns(gdf):
    """
    Padroniza nomes de colunas e mantém apenas as colunas necessárias
    
    Args:
        gdf (GeoDataFrame): DataFrame a ser processado
        
    Returns:
        GeoDataFrame: DataFrame com colunas padronizadas
    """
    # Criar uma cópia para não modificar o original
    gdf = gdf.copy()
    
    # Padronizar nomes de colunas (minúsculos, sem espaços)
    gdf.columns = [col.lower().replace(' ', '_') for col in gdf.columns]
    
    # Verificar e adicionar apenas colunas realmente essenciais (mínimas para processamento)
    # Mantenha a coluna natural original para compatibilidade com dados OSM
    if 'natural' in gdf.columns and 'type' not in gdf.columns:
        gdf['type'] = gdf['natural']
    
    # Adicione a coluna de área em hectares e perímetro para cálculos
    # Estas serão preenchidas pelo calculate_spatial_metrics
    
    # Mantenha todas as colunas originais, sem adicionar colunas vazias
    
    return gdf

def validate_and_fix_geometries(gdf):
    """
    Valida e tenta corrigir geometrias inválidas
    
    Args:
        gdf (GeoDataFrame): DataFrame com geometrias a serem validadas
        
    Returns:
        tuple: (GeoDataFrame com geometrias válidas, relatório de geometria)
    """
    geometry_report = {
        "total_geometries": len(gdf),
        "initially_invalid": 0,
        "fixed_geometries": 0,
        "unfixable_geometries": 0
    }
    
    # Verificar geometrias inválidas
    invalid_mask = ~gdf.geometry.is_valid
    geometry_report["initially_invalid"] = int(invalid_mask.sum())
    
    if geometry_report["initially_invalid"] > 0:
        # Identificar linhas com geometrias inválidas
        invalid_indices = gdf[invalid_mask].index
        
        # Tentar corrigir geometrias inválidas
        for idx in invalid_indices:
            try:
                # Tentar fazer a geometria válida
                fixed_geom = make_valid(gdf.loc[idx, 'geometry'])
                
                # Verificar se a correção funcionou
                if fixed_geom.is_valid:
                    gdf.loc[idx, 'geometry'] = fixed_geom
                    geometry_report["fixed_geometries"] += 1
                else:
                    geometry_report["unfixable_geometries"] += 1
            except Exception as e:
                print(f"Erro ao corrigir geometria no índice {idx}: {e}")
                geometry_report["unfixable_geometries"] += 1
    
    # Remover registros com geometrias ainda inválidas
    still_invalid = ~gdf.geometry.is_valid
    if still_invalid.sum() > 0:
        print(f"Removendo {still_invalid.sum()} registros com geometrias inválidas não corrigíveis")
        gdf = gdf[~still_invalid]
    
    return gdf, geometry_report

def calculate_spatial_metrics(gdf):
    """
    Calcula métricas espaciais para áreas naturais
    
    Args:
        gdf (GeoDataFrame): DataFrame com geometrias
        
    Returns:
        GeoDataFrame: DataFrame com métricas espaciais calculadas
    """
    # Criar uma cópia para não modificar o original
    gdf = gdf.copy()
    
    # Calcular área em hectares (convertendo de m² para ha)
    if gdf.crs and gdf.crs.is_projected:
        # Se já estiver em sistema projetado, calcular área diretamente
        gdf['area_ha'] = gdf.geometry.area / 10000  # m² para hectares
    else:
        # Se estiver em sistema geográfico, reprojetar para calcular área
        gdf_projected = gdf.to_crs(epsg=3857)  # Web Mercator para cálculos
        gdf['area_ha'] = gdf_projected.geometry.area / 10000  # m² para hectares
    
    # Calcular perímetro
    if gdf.crs and gdf.crs.is_projected:
        gdf['perimeter_m'] = gdf.geometry.length
    else:
        gdf_projected = gdf.to_crs(epsg=3857)
        gdf['perimeter_m'] = gdf_projected.geometry.length
    
    # Calcular índice de forma (Shape Index)
    # SI = P / (2 * sqrt(π * A)), onde P é o perímetro e A é a área
    # Um círculo perfeito tem SI = 1, valores maiores indicam formas mais complexas
    gdf['shape_index'] = gdf['perimeter_m'] / (2 * np.sqrt(np.pi * gdf['area_ha'] * 10000))
    
    # Classificar importância ecológica com base na área e índice de forma
    conditions = [
        (gdf['area_ha'] > 100) & (gdf['shape_index'] < 1.5),  # Grandes áreas com forma compacta
        (gdf['area_ha'] > 100) | (gdf['shape_index'] < 1.5),  # Grandes áreas ou forma compacta
        (gdf['area_ha'] > 10) & (gdf['shape_index'] < 2.0),   # Áreas médias com forma razoável
        (gdf['area_ha'] > 1)                                  # Áreas pequenas
    ]
    choices = ['very_high', 'high', 'medium', 'low']
    gdf['ecological_importance'] = np.select(conditions, choices, default='very_low')
    
    return gdf

def filter_by_area_of_interest(gdf, aoi_path=None):
    """
    Filtra o GeoDataFrame por uma área de interesse
    
    Args:
        gdf (GeoDataFrame): DataFrame a ser filtrado
        aoi_path (str, opcional): Caminho para o arquivo de área de interesse
        
    Returns:
        GeoDataFrame: DataFrame filtrado
    """
    if not aoi_path or not os.path.exists(aoi_path):
        print("Caminho para área de interesse não fornecido ou inválido. Usando todos os dados.")
        return gdf
    
    try:
        # Carregar área de interesse
        aoi = gpd.read_file(aoi_path)
        
        # Garantir que ambos tenham o mesmo CRS
        if gdf.crs != aoi.crs:
            aoi = aoi.to_crs(gdf.crs)
        
        # Filtrar por interseção com a área de interesse
        print(f"Filtrando {len(gdf)} registros pela área de interesse...")
        filtered_gdf = gpd.sjoin(gdf, aoi, predicate='intersects', how='inner')
        
        # Remover colunas da junção
        drop_cols = [col for col in filtered_gdf.columns if col.endswith('_right')]
        filtered_gdf = filtered_gdf.drop(columns=drop_cols)
        
        print(f"Restaram {len(filtered_gdf)} registros após filtro pela área de interesse")
        return filtered_gdf
    
    except Exception as e:
        print(f"Erro ao filtrar por área de interesse: {e}")
        return gdf

def main():
    """
    Função principal para processamento dos dados de áreas naturais.
    """
    start_time = time.time()
    print(f"Iniciando processamento de áreas naturais: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Carregar dados
    try:
        print(f"Carregando dados de {INPUT_PATH}...")
        
        # Verificar se o arquivo existe
        if not os.path.exists(INPUT_PATH):
            print(f"ERRO: Arquivo de entrada não encontrado em {INPUT_PATH}")
            return False
            
        # Listar camadas disponíveis
        import fiona
        layers = fiona.listlayers(str(INPUT_PATH))
        print(f"Camadas disponíveis: {layers}")
        
        # Carregar a camada correta
        if len(layers) > 1:
            # Se houver múltiplas camadas, tentar carregar a camada de natureza
            nature_layers = [l for l in layers if 'natural' in l.lower()]
            if nature_layers:
                layer_name = nature_layers[0]
                print(f"Usando camada: {layer_name}")
                gdf = gpd.read_file(INPUT_PATH, layer=layer_name)
            else:
                # Usar a primeira camada se não encontrar uma específica
                layer_name = layers[0]
                print(f"Usando primeira camada: {layer_name}")
                gdf = gpd.read_file(INPUT_PATH, layer=layer_name)
        else:
            # Se houver apenas uma camada, carregá-la diretamente
            gdf = gpd.read_file(INPUT_PATH)
            
        print(f"Dados carregados: {len(gdf)} features")
        print(f"Colunas disponíveis: {gdf.columns.tolist()}")
        print(f"Sistema de coordenadas: {gdf.crs}")
        
        # Filtrar por área de interesse
        gdf = filter_by_area_of_interest(gdf)
        
        # Processar dados
        gdf = process_natural_areas(gdf)
        
        # Gerar relatório de qualidade
        quality_report = generate_quality_report(gdf)
        
        # Salvar dados processados
        print(f"Salvando dados processados em {OUTPUT_PATH}...")
        gdf.to_file(OUTPUT_PATH, driver="GPKG")
        
        # Salvar relatório de qualidade
        report_file = REPORT_PATH / "quality_report_nature.json"
        with open(report_file, 'w') as f:
            json.dump(quality_report, f, indent=4)
            
        elapsed_time = time.time() - start_time
        print(f"Processamento concluído em {elapsed_time:.2f} segundos.")
        print(f"Total de áreas naturais processadas: {len(gdf)}")
        
        # Imprimir estatísticas básicas
        if 'type' in gdf.columns:
            type_counts = gdf['type'].value_counts()
            print("\nDistribuição por tipo de área natural:")
            for type_name, count in type_counts.items():
                print(f"  - {type_name}: {count}")
                
        if 'area_ha' in gdf.columns:
            total_area = gdf['area_ha'].sum()
            print(f"\nÁrea total: {total_area:.2f} hectares")
            
        return True
        
    except Exception as e:
        print(f"ERRO durante o processamento: {str(e)}")
        traceback.print_exc()
        return False

def process_natural_areas(input_file=None, output_file=None, aoi_path=None):
    """
    Processa dados de áreas naturais.
    
    Args:
        input_file (str): Caminho para o arquivo de entrada. Se None, usa o padrão.
        output_file (str): Caminho para o arquivo de saída. Se None, usa o padrão.
        aoi_path (str): Caminho para o arquivo de área de interesse. Se None, não filtra.
        
    Returns:
        GeoDataFrame processado ou None em caso de erro
    """
    try:
        # Definir caminhos padrão se não fornecidos
        if input_file is None:
            input_file = INPUT_PATH
        
        if output_file is None:
            output_file = OUTPUT_PATH
        
        # Criar diretório de saída se não existir
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Caminho para relatório de qualidade
        report_dir = Path(REPORT_PATH)
        os.makedirs(report_dir, exist_ok=True)
        report_path = report_dir / "quality_report_natural_areas.json"
        
        print(f"Processando áreas naturais...")
        print(f"Arquivo de entrada: {input_file}")
        print(f"Arquivo de saída: {output_file}")
        
        # Carregar dados
        print("Carregando dados...")
        gdf = gpd.read_file(input_file)
        print(f"Carregados {len(gdf)} registros")
        print(f"Colunas disponíveis: {gdf.columns.tolist()}")
        
        # Verificar CRS e reprojetar se necessário
        target_crs = "EPSG:4674"  # SIRGAS 2000
        if gdf.crs is None:
            print("AVISO: CRS não definido. Atribuindo SIRGAS 2000 (EPSG:4674)")
            gdf.crs = target_crs
        elif gdf.crs != target_crs:
            print(f"Reprojetando de {gdf.crs} para {target_crs}")
            gdf = gdf.to_crs(target_crs)
        
        # Filtrar por área de interesse
        if aoi_path and os.path.exists(aoi_path):
            print(f"Filtrando por área de interesse: {aoi_path}")
            gdf = filter_by_area_of_interest(gdf, aoi_path)
        
        # Padronizar nomes das colunas sem adicionar colunas novas
        gdf = clean_and_standardize_columns(gdf)
        
        # Verificar o tipo de dado, apenas para o relatório
        print("Validando dados...")
        validation_report, gdf = validate_data(gdf)
        
        # Validar geometrias
        print("Validando geometrias...")
        gdf, geometry_report = validate_and_fix_geometries(gdf)
        
        # Calcular métricas espaciais
        print("Calculando métricas espaciais...")
        gdf = calculate_spatial_metrics(gdf)
        
        # Colunas a manter no resultado final - apenas as relevantes
        # Se houver outras colunas de interesse específicas dos dados de entrada, adicionar aqui
        columns_to_keep = [
            'geometry', 'area_ha', 'perimeter_m', 'shape_index', 
            'ecological_importance'
        ]
        
        # Adicionar colunas opcionais se existirem nos dados originais
        optional_columns = ['name', 'type', 'natural', 'landuse', 'leisure', 'protected']
        for col in optional_columns:
            if col in gdf.columns:
                columns_to_keep.append(col)
        
        # Manter apenas as colunas necessárias
        columns_to_keep = [col for col in columns_to_keep if col in gdf.columns]
        gdf = gdf[columns_to_keep]
        
        # Gerar relatório de qualidade
        print("Gerando relatório de qualidade...")
        quality_report = generate_quality_report(gdf, validation_report, geometry_report)
        
        # Salvar relatório
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=4)
        print(f"Relatório de qualidade salvo em: {report_path}")
        
        # Salvar dados processados
        print(f"Salvando {len(gdf)} feições processadas...")
        gdf.to_file(output_file, driver="GPKG")
        print(f"Dados processados salvos em: {output_file}")
        
        # Imprimir resumo
        print("\nResumo do processamento:")
        print(f"Total de áreas naturais: {len(gdf)}")
        
        if 'type' in gdf.columns:
            type_counts = gdf['type'].value_counts()
            print("\nDistribuição por tipo de área natural:")
            for type_name, count in type_counts.items():
                print(f"  - {type_name}: {count}")
                
        if 'protected' in gdf.columns:
            protected_count = gdf['protected'].sum()
            print(f"\nÁreas protegidas: {protected_count} ({100 * protected_count / len(gdf):.1f}%)")
                
        if 'area_ha' in gdf.columns:
            total_area = gdf['area_ha'].sum()
            print(f"Área total: {total_area:.2f} hectares")
            
        if 'ecological_importance' in gdf.columns:
            eco_counts = gdf['ecological_importance'].value_counts()
            print("\nImportância ecológica:")
            for importance, count in eco_counts.items():
                print(f"  - {importance}: {count}")
        
        return gdf
    
    except Exception as e:
        print(f"ERRO ao processar áreas naturais: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Tentar salvar o que foi possível processar até o erro
        try:
            if 'gdf' in locals() and len(gdf) > 0:
                backup_file = str(output_file).replace('.gpkg', '_error_backup.gpkg')
                gdf.to_file(backup_file, driver="GPKG")
                print(f"Backup parcial salvo em: {backup_file}")
        except Exception as be:
            print(f"Não foi possível salvar backup dos dados parciais: {str(be)}")
        
        return None

def standardize_column_names(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Padroniza os nomes das colunas para facilitar o processamento.
    
    Args:
        gdf: GeoDataFrame com os dados brutos
        
    Returns:
        GeoDataFrame com nomes de colunas padronizados
    """
    # Mapeamento de possíveis nomes para nomes padronizados
    name_mapping = {
        'nome': 'name',
        'tipo': 'type',
        'class': 'type',
        'classification': 'type',
        'protegido': 'protected',
        'protection': 'protected',
        'area': 'area_m2',
        'perimetro': 'perimeter_m',
        'perimeter': 'perimeter_m'
    }
    
    # Criar cópia para não modificar o original
    gdf = gdf.copy()
    
    # Converter nomes para minúsculas e remover espaços
    gdf.columns = [col.lower().strip().replace(' ', '_') for col in gdf.columns]
    
    # Aplicar mapeamento
    for old_name, new_name in name_mapping.items():
        if old_name in gdf.columns and new_name not in gdf.columns:
            gdf = gdf.rename(columns={old_name: new_name})
            
    return gdf

def generate_quality_report(gdf, validation_report=None, geometry_report=None):
    """
    Gera um relatório de qualidade para os dados processados
    
    Args:
        gdf (GeoDataFrame): DataFrame processado
        validation_report (dict): Relatório de validação
        geometry_report (dict): Relatório de geometria
        
    Returns:
        dict: Relatório completo de qualidade
    """
    # Criar um relatório básico se não fornecido
    if validation_report is None:
        validation_report = {
            "total_records": len(gdf),
            "valid_records": len(gdf[gdf.geometry.is_valid]),
            "invalid_geometries": len(gdf[~gdf.geometry.is_valid]),
            "null_counts": {col: int(gdf[col].isna().sum()) for col in gdf.columns}
        }
    
    if geometry_report is None:
        geometry_report = {
            "total_geometries": len(gdf),
            "initially_invalid": (~gdf.geometry.is_valid).sum(),
            "fixed_geometries": 0,
            "unfixable_geometries": 0
        }
    
    # Estatísticas gerais
    quality_report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "total_features": len(gdf),
        "validation_results": validation_report,
        "geometry_report": geometry_report,
        "crs": str(gdf.crs),
        "area_statistics": {},
        "type_distribution": {},
        "protection_status": {},
        "ecological_importance": {}
    }
    
    # Estatísticas de área
    if 'area_ha' in gdf.columns:
        quality_report["area_statistics"] = {
            "min_area_ha": float(gdf['area_ha'].min()),
            "max_area_ha": float(gdf['area_ha'].max()),
            "mean_area_ha": float(gdf['area_ha'].mean()),
            "median_area_ha": float(gdf['area_ha'].median()),
            "total_area_ha": float(gdf['area_ha'].sum()),
            "std_dev_area_ha": float(gdf['area_ha'].std())
        }
    
    # Distribuição por tipo
    if 'type' in gdf.columns:
        type_counts = gdf['type'].value_counts().to_dict()
        quality_report["type_distribution"] = {str(k): int(v) for k, v in type_counts.items()}
    
    # Status de proteção
    if 'protected' in gdf.columns:
        protected_count = int(gdf['protected'].sum())
        quality_report["protection_status"] = {
            "protected": protected_count,
            "unprotected": len(gdf) - protected_count,
            "percent_protected": round(100 * protected_count / len(gdf), 2) if len(gdf) > 0 else 0
        }
    
    # Importância ecológica
    if 'ecological_importance' in gdf.columns:
        eco_counts = gdf['ecological_importance'].value_counts().to_dict()
        quality_report["ecological_importance"] = {str(k): int(v) for k, v in eco_counts.items()}
    
    # Salvar relatório em forma de string
    report_json = json.dumps(quality_report, indent=4)
    
    return quality_report

def process_chunk(chunk, operation):
    """
    Processa um chunk do GeoDataFrame.
    
    Args:
        chunk: Pedaço do GeoDataFrame
        operation: Operação a ser realizada ('validate', 'geometry', 'metrics')
        
    Returns:
        Chunk processado
    """
    result = chunk.copy()
    
    if operation == 'validate':
        # Validar dados
        for idx, row in result.iterrows():
            if not row.geometry.is_valid:
                result.at[idx, 'geometry'] = make_valid(row.geometry)
    
    elif operation == 'geometry':
        # Processar geometrias (simplificar, corrigir, etc.)
        for idx, row in result.iterrows():
            # Simplificar geometria para reduzir complexidade
            if hasattr(row.geometry, 'simplify'):
                tolerance = 0.00001  # Ajuste conforme necessário
                result.at[idx, 'geometry'] = row.geometry.simplify(tolerance, preserve_topology=True)
    
    elif operation == 'metrics':
        # Calcular métricas espaciais
        result['area_ha'] = result.geometry.area / 10000  # Convertendo m² para hectares
        result['perimeter_m'] = result.geometry.length
        result['compactness'] = 4 * np.pi * result.geometry.area / (result.geometry.length ** 2)
    
    return result

def parallel_process(gdf, operation, n_cores=None):
    """
    Processa o GeoDataFrame em paralelo.
    
    Args:
        gdf: GeoDataFrame a ser processado
        operation: Operação a ser realizada ('validate', 'geometry', 'metrics')
        n_cores: Número de núcleos a usar. Se None, usa todos menos 1.
        
    Returns:
        GeoDataFrame processado
    """
    if n_cores is None:
        n_cores = max(1, os.cpu_count() - 1)
    
    # Determinar o tamanho dos chunks
    n_rows = len(gdf)
    if n_rows < 1000:
        # Para conjuntos pequenos, não usar paralelização
        return process_chunk(gdf, operation)
    
    # Calcular tamanho dos chunks
    chunk_size = max(1, n_rows // (n_cores * 4))
    chunks = [gdf.iloc[i:i+chunk_size] for i in range(0, n_rows, chunk_size)]
    
    print(f"Processando {n_rows} feições usando {n_cores} núcleos em {len(chunks)} chunks...")
    
    # Processar em paralelo
    processed_chunks = []
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        # Mapear a função process_chunk para cada chunk
        futures = [executor.submit(process_chunk, chunk, operation) for chunk in chunks]
        
        # Coletar resultados
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processando {operation}"):
            try:
                result = future.result()
                processed_chunks.append(result)
            except Exception as e:
                print(f"Erro ao processar chunk: {e}")
    
    # Concatenar chunks processados
    result = pd.concat(processed_chunks, ignore_index=True)
    
    return result

def create_quality_report(original_gdf, processed_gdf):
    """
    Cria um relatório de qualidade comparando os dados originais e processados.
    
    Args:
        original_gdf: GeoDataFrame original
        processed_gdf: GeoDataFrame processado
        
    Returns:
        Dicionário com o relatório
    """
    report = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_count": len(original_gdf),
        "processed_count": len(processed_gdf),
        "reduction_percentage": (1 - len(processed_gdf) / len(original_gdf)) * 100 if len(original_gdf) > 0 else 0,
        "geometry_validity": {
            "original_invalid": (~original_gdf.geometry.is_valid).sum(),
            "processed_invalid": (~processed_gdf.geometry.is_valid).sum(),
        },
        "crs": str(processed_gdf.crs),
        "metrics": {}
    }
    
    # Estatísticas para colunas numéricas
    numeric_columns = processed_gdf.select_dtypes(include=['number']).columns
    
    for col in numeric_columns:
        if col in processed_gdf.columns:
            stats = processed_gdf[col].describe().to_dict()
            report["metrics"][col] = {
                "min": stats.get('min'),
                "max": stats.get('max'),
                "mean": stats.get('mean'),
                "std": stats.get('std'),
                "null_count": processed_gdf[col].isna().sum(),
                "null_percentage": (processed_gdf[col].isna().sum() / len(processed_gdf)) * 100
            }
    
    # Estatísticas para colunas categóricas
    categorical_columns = processed_gdf.select_dtypes(include=['object']).columns
    
    for col in categorical_columns:
        if col in processed_gdf.columns and col != 'geometry':
            value_counts = processed_gdf[col].value_counts().to_dict()
            report["metrics"][col] = {
                "value_counts": value_counts,
                "unique_values": processed_gdf[col].nunique(),
                "null_count": processed_gdf[col].isna().sum(),
                "null_percentage": (processed_gdf[col].isna().sum() / len(processed_gdf)) * 100
            }
    
    # Métricas espaciais
    if 'area_ha' in processed_gdf.columns:
        report["spatial_metrics"] = {
            "total_area_ha": processed_gdf['area_ha'].sum(),
            "mean_area_ha": processed_gdf['area_ha'].mean(),
            "min_area_ha": processed_gdf['area_ha'].min(),
            "max_area_ha": processed_gdf['area_ha'].max()
        }
    
    if 'perimeter_m' in processed_gdf.columns:
        report["spatial_metrics"]["total_perimeter_m"] = processed_gdf['perimeter_m'].sum()
        report["spatial_metrics"]["mean_perimeter_m"] = processed_gdf['perimeter_m'].mean()
    
    # Métricas de biodiversidade (se existirem)
    if 'biodiversity_index' in processed_gdf.columns:
        report["biodiversity_metrics"] = {
            "mean_biodiversity_index": processed_gdf['biodiversity_index'].mean(),
            "max_biodiversity_index": processed_gdf['biodiversity_index'].max(),
            "min_biodiversity_index": processed_gdf['biodiversity_index'].min()
        }
    
    # Salvar relatório em arquivo JSON
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORT_PATH / f"quality_report_nature_{timestamp}.json"
    
    try:
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"Relatório de qualidade salvo em: {report_path}")
    except Exception as e:
        print(f"Erro ao salvar relatório: {e}")
    
    return report

def make_valid(geom):
    """
    Corrige geometrias inválidas usando buffer(0) ou outras técnicas.
    
    Args:
        geom: Geometria Shapely a ser corrigida
        
    Returns:
        Geometria corrigida
    """
    if geom is None:
        return None
    
    if geom.is_valid:
        return geom
    
    # Tentar diferentes técnicas para corrigir geometria
    try:
        # Técnica 1: Buffer zero
        valid_geom = geom.buffer(0)
        if valid_geom.is_valid:
            return valid_geom
        
        # Técnica 2: Simplificação
        valid_geom = geom.simplify(0.01)
        if valid_geom.is_valid:
            return valid_geom
        
        # Técnica 3: Convex hull
        valid_geom = geom.convex_hull
        if valid_geom.is_valid:
            return valid_geom
        
        # Se nenhuma técnica funcionar, retornar None
        return None
    except Exception:
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Processa áreas naturais para análise espacial.")
    parser.add_argument("--input", help="Caminho para o arquivo de entrada GPKG", default=INPUT_PATH)
    parser.add_argument("--output", help="Caminho para o arquivo de saída GPKG", default=OUTPUT_PATH)
    parser.add_argument("--aoi", help="Caminho para arquivo da área de interesse", default=None)
    args = parser.parse_args()
    
    process_natural_areas(input_file=args.input, 
                          output_file=args.output, 
                          aoi_path=args.aoi)