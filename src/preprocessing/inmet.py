"""
Functions for processing INMET weather station data.
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Union, List, Dict, Optional
from shapely.geometry import Point

# Classe personalizada para serialização JSON de tipos NumPy
class NpEncoder(json.JSONEncoder):
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
REPORT_DIR = os.path.join(os.path.dirname(__file__), 'quality_reports', 'inmet')
VISUALIZATION_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualizations', 'inmet')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# Define essential columns and their metadata
ESSENTIAL_COLUMNS = {
    'date': {
        'type': 'datetime64[ns]',
        'description': 'Date of measurement'
    },
    'time_utc': {
        'type': 'str',
        'description': 'Time in UTC'
    },
    'latitude': {
        'type': 'float64',
        'description': 'Station latitude',
        'validation': {'min': -90, 'max': 90}
    },
    'longitude': {
        'type': 'float64',
        'description': 'Station longitude',
        'validation': {'min': -180, 'max': 180}
    },
    'altitude_m': {
        'type': 'float64',
        'description': 'Station altitude in meters'
    },
    'temperature_c': {
        'type': 'float64',
        'description': 'Air temperature in Celsius',
        'validation': {'min': -40, 'max': 50}
    },
    'humidity_pct': {
        'type': 'float64',
        'description': 'Relative humidity percentage',
        'validation': {'min': 0, 'max': 100}
    },
    'pressure_mb': {
        'type': 'float64',
        'description': 'Atmospheric pressure in millibars',
        'validation': {'min': 800, 'max': 1100}
    },
    'wind_speed_ms': {
        'type': 'float64',
        'description': 'Wind speed in meters per second',
        'validation': {'min': 0, 'max': 100}
    },
    'wind_direction_deg': {
        'type': 'float64',
        'description': 'Wind direction in degrees',
        'validation': {'min': 0, 'max': 360}
    },
    'precipitation_mm': {
        'type': 'float64',
        'description': 'Precipitation in millimeters',
        'validation': {'min': 0, 'max': 500}
    },
    'radiation_kjm2': {
        'type': 'float64',
        'description': 'Global radiation in kilojoules per square meter',
        'validation': {'min': 0}
    }
}

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate data ranges and flag suspicious values.
    
    Args:
        df: Input DataFrame with INMET data
        
    Returns:
        DataFrame with validated data (invalid values set to NaN)
    """
    for col, meta in ESSENTIAL_COLUMNS.items():
        if col not in df.columns:
            continue
            
        if 'validation' in meta:
            min_val = meta['validation'].get('min')
            max_val = meta['validation'].get('max')
            
            if min_val is not None:
                invalid_mask = df[col] < min_val
                if invalid_mask.any():
                    print(f"Warning: Found {invalid_mask.sum()} values below minimum ({min_val}) in {col}")
                    df.loc[invalid_mask, col] = np.nan
                    
            if max_val is not None:
                invalid_mask = df[col] > max_val
                if invalid_mask.any():
                    print(f"Warning: Found {invalid_mask.sum()} values above maximum ({max_val}) in {col}")
                    df.loc[invalid_mask, col] = np.nan
    
    return df

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize INMET column names.
    
    Args:
        df: Input DataFrame with original INMET columns
        
    Returns:
        DataFrame with cleaned column names
    """
    # First, clean any unnamed columns
    unnamed_cols = [col for col in df.columns if 'Unnamed' in col]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    
    # Standard column mapping
    column_mapping = {
        'DATA (YYYY-MM-DD)': 'date',
        'HORA (UTC)': 'time_utc',
        'PRECIPITAÇÃO TOTAL, HORÁRIO (mm)': 'precipitation_mm',
        'PRECIPITACAO TOTAL, HORARIO (mm)': 'precipitation_mm',
        'PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'pressure_mb',
        'PRESSÃO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)': 'pressure_mb',
        'RADIACAO GLOBAL (KJ/m²)': 'radiation_kjm2',
        'RADIAÇÃO GLOBAL (KJ/m²)': 'radiation_kjm2',
        'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)': 'temperature_c',
        'TEMPERATURA DO AR - BULBO SECO, HORÁRIA (°C)': 'temperature_c',
        'UMIDADE RELATIVA DO AR, HORARIA (%)': 'humidity_pct',
        'UMIDADE RELATIVA DO AR, HORÁRIA (%)': 'humidity_pct',
        'VENTO, DIREÇÃO HORARIA (gr) (° (gr))': 'wind_direction_deg',
        'VENTO, DIREÇÃO HORÁRIA (gr) (° (gr))': 'wind_direction_deg',
        'VENTO, VELOCIDADE HORARIA (m/s)': 'wind_speed_ms',
        'VENTO, VELOCIDADE HORÁRIA (m/s)': 'wind_speed_ms',
        'LATITUDE': 'latitude',
        'LONGITUDE': 'longitude',
        'ALTITUDE': 'altitude_m'
    }
    
    # Clean column names
    df.columns = df.columns.str.strip().str.upper()
    
    # Apply mapping for known columns
    for old_name, new_name in column_mapping.items():
        if old_name.upper() in df.columns:
            df = df.rename(columns={old_name.upper(): new_name})
    
    return df

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types.
    
    Args:
        df: Input DataFrame with string/object columns
        
    Returns:
        DataFrame with proper data types
    """
    for col, meta in ESSENTIAL_COLUMNS.items():
        if col not in df.columns:
            continue
            
        if meta['type'] == 'float64':
            # Handle both string and numeric inputs
            if df[col].dtype == object:  # If string
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            else:  # If already numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
        elif meta['type'] == 'datetime64[ns]':
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    return df

def filter_relevant_columns(df: pd.DataFrame, include_radiation: bool = False) -> pd.DataFrame:
    """
    Filter only relevant columns for the analysis.
    
    Args:
        df: Input DataFrame
        include_radiation: Whether to include radiation data
        
    Returns:
        DataFrame with only relevant columns
    """
    essential_columns = list(ESSENTIAL_COLUMNS.keys())
    if not include_radiation and 'radiation_kjm2' in essential_columns:
        essential_columns.remove('radiation_kjm2')
    
    # Add region, state and station metadata if available
    for meta_col in ['region', 'state', 'station', 'wmo_code']:
        if meta_col in df.columns:
            if meta_col not in essential_columns:
                essential_columns.append(meta_col)
    
    # Always include latitude and longitude if available
    for geo_col in ['latitude', 'longitude']:
        if geo_col not in essential_columns and geo_col in df.columns:
            essential_columns.append(geo_col)
    
    # Keep only columns that exist in the DataFrame
    available_columns = [col for col in essential_columns if col in df.columns]
    
    if not available_columns:
        raise ValueError("No essential columns found in the DataFrame")
        
    # Check if latitude and longitude are present
    if 'latitude' not in available_columns or 'longitude' not in available_columns:
        # Try to find them in the metadata
        if 'LATITUDE:' in df.columns and 'latitude' not in available_columns:
            lat_value = df['LATITUDE:'].iloc[0]
            if isinstance(lat_value, str) and ';' in lat_value:
                lat_value = lat_value.split(';')[0]
            df['latitude'] = float(str(lat_value).replace(',', '.'))
            available_columns.append('latitude')
            
        if 'LONGITUDE:' in df.columns and 'longitude' not in available_columns:
            lon_value = df['LONGITUDE:'].iloc[0]
            if isinstance(lon_value, str) and ';' in lon_value:
                lon_value = lon_value.split(';')[0]
            df['longitude'] = float(str(lon_value).replace(',', '.'))
            available_columns.append('longitude')
    
    return df[available_columns]

def read_inmet_data(file_path: Union[str, Path], include_radiation: bool = False) -> pd.DataFrame:
    """
    Read and preprocess INMET weather station data from CSV files.
    
    Args:
        file_path: Path to the INMET CSV file
        include_radiation: Whether to include radiation data
        
    Returns:
        DataFrame containing processed INMET data
    """
    try:
        # Read the first few lines to get metadata
        with open(file_path, 'r', encoding='utf-8') as f:
            metadata = {}
            lines = []
            for i in range(8):  # Read first 8 lines for metadata
                line = f.readline().strip()
                lines.append(line)
                if ':' in line:
                    parts = line.split(';')
                    if len(parts) > 1:
                        key, value = parts[0:2]
                        metadata[key] = value

        # Extract coordinates from metadata
        coords = {}
        for line in lines:
            if 'LATITUDE:' in line:
                parts = line.split(';')
                if len(parts) > 1:
                    lat_str = parts[1].strip()
                    try:
                        coords['latitude'] = float(lat_str.replace(',', '.'))
                    except:
                        print(f"Warning: Could not parse latitude from {lat_str}")
            
            if 'LONGITUDE:' in line:
                parts = line.split(';')
                if len(parts) > 1:
                    lon_str = parts[1].strip()
                    try:
                        coords['longitude'] = float(lon_str.replace(',', '.'))
                    except:
                        print(f"Warning: Could not parse longitude from {lon_str}")
            
            if 'ALTITUDE:' in line:
                parts = line.split(';')
                if len(parts) > 1:
                    alt_str = parts[1].strip()
                    try:
                        coords['altitude_m'] = float(alt_str.replace(',', '.'))
                    except:
                        print(f"Warning: Could not parse altitude from {alt_str}")

        # Read the actual data, skipping metadata rows
        df = pd.read_csv(
            file_path, 
            sep=';', 
            decimal=',', 
            thousands='.',
            encoding='utf-8',
            skiprows=8,  # Skip metadata rows
            low_memory=False  # Prevent mixed type inference
        )
        
        # Clean column names first
        df = clean_column_names(df)
        
        # Add coordinates from metadata if available
        for coord_name, coord_value in coords.items():
            if coord_name not in df.columns:
                df[coord_name] = coord_value
        
        # Add metadata columns
        if metadata:
            df['region'] = metadata.get('REGIAO', '')
            df['state'] = metadata.get('UF', '')
            df['station'] = metadata.get('ESTACAO', '')
            df['wmo_code'] = metadata.get('CODIGO (WMO)', '')
        
        # Process the data
        df = convert_data_types(df)
        df = validate_data(df)
        df = filter_relevant_columns(df, include_radiation)
        
        return df
        
    except Exception as e:
        print(f"Error processing file {file_path}:")
        print(str(e))
        raise

def analyze_column_distribution(df: pd.DataFrame, column_name: str, output_dir: str):
    """Analisa e visualiza a distribuição dos dados em uma coluna específica."""
    if column_name not in df.columns:
        return
    
    try:
        if df[column_name].dtype.kind in 'ifc':  # Numérico
            # Remover valores nulos e infinitos
            data = df[column_name].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(data) == 0:
                print(f"Aviso: Coluna {column_name} não possui dados válidos para análise")
                return
            
            plt.figure(figsize=(15, 6))
            
            # Histograma com KDE
            plt.subplot(1, 3, 1)
            sns.histplot(data=data, kde=True)
            plt.title(f'Distribuição: {column_name}')
            plt.xticks(rotation=45)
            
            # Boxplot
            plt.subplot(1, 3, 2)
            sns.boxplot(y=data)
            plt.title(f'Boxplot: {column_name}')
            
            # QQ plot para verificar normalidade
            plt.subplot(1, 3, 3)
            stats.probplot(data, dist="norm", plot=plt)
            plt.title('Q-Q Plot')
            
            plt.tight_layout()
            # Salvar visualização no diretório de visualizações
            output_file = os.path.join(VISUALIZATION_DIR, f'distribuicao_{column_name}.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calcular estatísticas descritivas
            stats_dict = {
                'media': float(data.mean()),
                'mediana': float(data.median()),
                'desvio_padrao': float(data.std()),
                'minimo': float(data.min()),
                'maximo': float(data.max()),
                'q1': float(data.quantile(0.25)),
                'q3': float(data.quantile(0.75)),
                'contagem': int(len(data)),
                'valores_nulos': int(df[column_name].isnull().sum()),
                'percentual_nulos': float(df[column_name].isnull().mean() * 100)
            }
            
            # Salvar estatísticas em arquivo JSON (mantém no diretório de relatórios)
            stats_file = os.path.join(output_dir, f'estatisticas_{column_name}.json')
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_dict, f, indent=4, cls=NpEncoder)
                
    except Exception as e:
        print(f"Erro ao analisar distribuição da coluna {column_name}: {str(e)}")
        import traceback
        print(traceback.format_exc())

def process_inmet_files(data_dir: Union[str, Path], pattern: str = "*.CSV", include_radiation: bool = False) -> pd.DataFrame:
    """Process all INMET files in the directory."""
    print(f"Processando arquivos INMET em: {data_dir}")
    
    all_data = []
    files_processed = 0
    
    # Relatórios de qualidade
    missing_values_report = {}
    outlier_report = {}
    
    # Verificar se o diretório de dados existe
    if not os.path.exists(data_dir):
        print(f"Diretório de dados não encontrado: {data_dir}")
        return pd.DataFrame()
    
    # Listar todos os arquivos CSV no diretório
    csv_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.upper().endswith('.CSV') and 'INMET' in file.upper():
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"Nenhum arquivo CSV do INMET encontrado em: {data_dir}")
        return pd.DataFrame()
    
    print(f"Encontrados {len(csv_files)} arquivos CSV do INMET para processar")
    
    for file_path in csv_files:
        try:
            print(f"Processando arquivo: {os.path.basename(file_path)}")
            df = read_inmet_data(file_path, include_radiation)
            
            if df is not None and not df.empty:
                all_data.append(df)
                files_processed += 1
                
                # Análise de valores ausentes
                for col in df.columns:
                    null_count = df[col].isnull().sum()
                    if col not in missing_values_report:
                        missing_values_report[col] = {'contagem_nulos': 0, 'percentual': 0}
                    missing_values_report[col]['contagem_nulos'] += null_count
                    missing_values_report[col]['percentual'] = (null_count / len(df)) * 100
                
                # Análise de outliers para colunas numéricas
                for col in df.columns:
                    if df[col].dtype.kind in 'ifc':
                        # Identificar outliers usando IQR
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 3 * IQR
                        upper_bound = Q3 + 3 * IQR
                        
                        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                        outlier_count = len(outliers)
                        
                        if col not in outlier_report:
                            outlier_report[col] = {
                                'contagem_outliers': 0,
                                'percentual': 0,
                                'valor_min': float('inf'),
                                'valor_max': float('-inf'),
                                'limite_inferior': lower_bound,
                                'limite_superior': upper_bound
                            }
                        
                        outlier_report[col]['contagem_outliers'] += outlier_count
                        outlier_report[col]['percentual'] = (outlier_count / len(df)) * 100
                        outlier_report[col]['valor_min'] = min(outlier_report[col]['valor_min'], df[col].min())
                        outlier_report[col]['valor_max'] = max(outlier_report[col]['valor_max'], df[col].max())
                
                # Análise de distribuição
                for col in df.columns:
                    analyze_column_distribution(df, col, REPORT_DIR)
                    
        except Exception as e:
            print(f"Erro ao processar arquivo {file_path}: {str(e)}")
            continue
    
    if not all_data:
        print("Nenhum dado válido encontrado!")
        return pd.DataFrame()
    
    # Combinar todos os dados
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Processados {files_processed} arquivos, total de {len(combined_df)} registros")
    
    # Salvar relatórios de qualidade
    with open(os.path.join(REPORT_DIR, 'missing_values_report.json'), 'w', encoding='utf-8') as f:
        json.dump(missing_values_report, f, indent=4, cls=NpEncoder)
    
    with open(os.path.join(REPORT_DIR, 'outlier_report.json'), 'w', encoding='utf-8') as f:
        json.dump(outlier_report, f, indent=4, cls=NpEncoder)
    
    # Gerar relatório de qualidade geral
    quality_metrics = {
        'arquivos_processados': files_processed,
        'registros_totais': len(combined_df),
        'colunas_totais': len(combined_df.columns),
        'valores_ausentes': missing_values_report,
        'outliers': outlier_report,
        'colunas_numericas': [col for col in combined_df.columns if combined_df[col].dtype.kind in 'ifc'],
        'colunas_categoricas': [col for col in combined_df.columns if combined_df[col].dtype == 'object'],
        'colunas_temporais': [col for col in combined_df.columns if combined_df[col].dtype == 'datetime64[ns]']
    }
    
    with open(os.path.join(REPORT_DIR, 'quality_report.json'), 'w', encoding='utf-8') as f:
        json.dump(quality_metrics, f, indent=4, cls=NpEncoder)
    
    return combined_df

def create_inmet_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame from the processed INMET data.
    Uses the centroid of Sorocaba study area as location instead of original coordinates.
    
    Args:
        df: DataFrame with processed INMET data
        
    Returns:
        GeoDataFrame with point geometries
    """
    # Create a copy of the dataframe to avoid modifying the original
    df_clean = df.copy()
    
    # Replace latitude and longitude with Sorocaba centroid coordinates
    sorocaba_centroid = (-23.464555685955755, -47.44676542591232)  # lat, lon from sorocaba.gpkg centroid
    
    print(f"Substituindo coordenadas originais pelo centroide de Sorocaba: {sorocaba_centroid}")
    
    # Assign centroid coordinates to all rows
    df_clean['latitude'] = sorocaba_centroid[0]
    df_clean['longitude'] = sorocaba_centroid[1]
    
    # Create point geometries
    geometry = [Point(xy) for xy in zip(df_clean['longitude'], df_clean['latitude'])]
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df_clean, geometry=geometry, crs="EPSG:4674")  # SIRGAS 2000
    
    print(f"GeoDataFrame criado com {len(gdf)} registros usando o centroide de Sorocaba")
    return gdf

def main():
    """Main function to process INMET data."""
    print("Processando dados do INMET...")
    
    try:
        # Verificar se há arquivos CSV no diretório
        csv_files = []
        for root, _, files in os.walk(INPUT_DIR):
            for file in files:
                if file.upper().endswith('.CSV') and 'INMET' in file.upper():
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            print(f"Nenhum arquivo CSV do INMET encontrado em: {INPUT_DIR}")
            print("Por favor, verifique se os arquivos CSV do INMET estão no diretório.")
            return
            
        print(f"Encontrados {len(csv_files)} arquivos CSV do INMET para processar")
        for file in csv_files:
            print(f"- {os.path.basename(file)}")
        
        # Processar arquivos CSV
        df = process_inmet_files(INPUT_DIR, pattern="*INMET*.CSV")
        
        if df.empty:
            print("Nenhum dado válido foi processado.")
            return
            
        # Criar GeoDataFrame
        try:
            gdf = create_inmet_geodataframe(df)
            
            # Salvar resultado processado
            output_file = os.path.join(OUTPUT_DIR, 'inmet_processed.gpkg')
            gdf.to_file(output_file, driver='GPKG')
            print(f"Dados processados salvos em: {output_file}")
            
        except ValueError as e:
            print(f"Erro ao criar GeoDataFrame: {str(e)}")
            print("Salvando dados em formato CSV para análise...")
            output_csv = os.path.join(OUTPUT_DIR, 'inmet_processed.csv')
            df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"Dados salvos em CSV: {output_csv}")
            
    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 