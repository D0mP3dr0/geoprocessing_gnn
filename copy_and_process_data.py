import os
import shutil
import geopandas as gpd
import pandas as pd
import logging
from datetime import datetime
import sys

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_preparation')

# Definição de caminhos
LOCAL_RAW_DIR = r"F:\TESE_MESTRADO\geoprocessing\data\raw"
LOCAL_ENRICHED_DIR = r"F:\TESE_MESTRADO\geoprocessing\data\enriched"

# Caminhos no Google Drive
DRIVE_BASE_DIR = "/content/drive/MyDrive/geoprocessamento_gnn"
DRIVE_DATA_DIR = os.path.join(DRIVE_BASE_DIR, "data")
DRIVE_OUTPUT_DIR = os.path.join(DRIVE_BASE_DIR, "OUTPUT")
DRIVE_QUALITY_DIR = os.path.join(DRIVE_BASE_DIR, "QUALITY_REPORT")
DRIVE_VIS_DIR = os.path.join(DRIVE_BASE_DIR, "VISUALIZACOES")

def is_colab():
    """Verifica se o código está sendo executado no Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def create_directories():
    """Cria os diretórios necessários no Google Drive"""
    if is_colab():
        for dir_path in [DRIVE_BASE_DIR, DRIVE_DATA_DIR, DRIVE_OUTPUT_DIR, DRIVE_QUALITY_DIR, DRIVE_VIS_DIR]:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Diretório verificado/criado: {dir_path}")
    else:
        logger.warning("Não estamos no Google Colab. Operando em modo local.")

def copy_raw_files():
    """Copia os arquivos brutos necessários para o Google Drive"""
    if not is_colab():
        logger.warning("Função disponível apenas no Google Colab")
        return

    # Lista de arquivos a serem copiados
    files_to_copy = [
        "sorocaba_buildings.gpkg",
        "sorocaba_landuse.gpkg",
        "sorocaba_roads.gpkg",
        "sorocaba_setores_censitarios.gpkg"
    ]

    for filename in files_to_copy:
        source_path = os.path.join(LOCAL_RAW_DIR, filename)
        destination_path = os.path.join(DRIVE_DATA_DIR, filename)
        
        # Verifica se o arquivo existe no ambiente local
        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, destination_path)
                logger.info(f"Arquivo copiado: {source_path} -> {destination_path}")
            except Exception as e:
                logger.error(f"Erro ao copiar {filename}: {str(e)}")
        else:
            logger.warning(f"Arquivo não encontrado: {source_path}")

def basic_enrich_buildings(input_path, output_path):
    """Realiza um enriquecimento básico de dados de edificações"""
    try:
        # Carregar o arquivo
        logger.info(f"Carregando dados de edificações de {input_path}")
        gdf = gpd.read_file(input_path)
        
        # Quantidade inicial de registros
        initial_count = len(gdf)
        logger.info(f"Carregados {initial_count} registros de edificações")
        
        # Adicionar coluna de área se não existir
        if 'area_m2' not in gdf.columns:
            gdf['area_m2'] = gdf.geometry.area
            logger.info("Calculada área das edificações")
        
        # Adicionar coluna de perímetro se não existir
        if 'perimeter_m' not in gdf.columns:
            gdf['perimeter_m'] = gdf.geometry.length
            logger.info("Calculado perímetro das edificações")
        
        # Índice de compacidade
        if 'compactness_index' not in gdf.columns:
            gdf['compactness_index'] = (4 * 3.14159 * gdf['area_m2']) / (gdf['perimeter_m'] ** 2)
            logger.info("Calculado índice de compacidade")
        
        # Altura padrão para edificações sem altura
        if 'height' not in gdf.columns:
            gdf['height'] = 3.0  # Valor padrão de 3 metros
            logger.info("Adicionada coluna de altura com valor padrão")
        
        # Níveis da edificação
        if 'levels' not in gdf.columns:
            gdf['levels'] = 1.0  # Valor padrão de 1 nível
            logger.info("Adicionada coluna de níveis com valor padrão")
        
        # Classificação simples de edificações
        if 'building_class' not in gdf.columns:
            def classify_building(row):
                if pd.isna(row['building']):
                    return 'unknown'
                elif row['building'] in ['house', 'residential', 'apartments', 'detached']:
                    return 'residential'
                elif row['building'] in ['commercial', 'retail', 'office', 'supermarket']:
                    return 'commercial'
                elif row['building'] in ['industrial', 'warehouse', 'factory']:
                    return 'industrial'
                elif row['building'] in ['school', 'university', 'college', 'kindergarten']:
                    return 'education'
                elif row['building'] in ['hospital', 'clinic', 'healthcare']:
                    return 'healthcare'
                else:
                    return 'other'
            
            gdf['building_class'] = gdf.apply(classify_building, axis=1)
            logger.info("Adicionada coluna de classificação de edificações")
        
        # Salvar arquivo enriquecido
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path_with_timestamp = output_path.replace('.gpkg', f'_{timestamp}.gpkg')
        gdf.to_file(output_path_with_timestamp, driver='GPKG')
        logger.info(f"Arquivo enriquecido salvo em {output_path_with_timestamp}")
        
        return output_path_with_timestamp
    except Exception as e:
        logger.error(f"Erro ao enriquecer edificações: {str(e)}")
        return None

def basic_enrich_landuse(input_path, output_path):
    """Realiza um enriquecimento básico de dados de uso do solo"""
    try:
        # Carregar o arquivo
        logger.info(f"Carregando dados de uso do solo de {input_path}")
        gdf = gpd.read_file(input_path)
        
        # Quantidade inicial de registros
        initial_count = len(gdf)
        logger.info(f"Carregados {initial_count} registros de uso do solo")
        
        # Adicionar coluna de área se não existir
        if 'area_m2' not in gdf.columns:
            gdf['area_m2'] = gdf.geometry.area
            logger.info("Calculada área das parcelas de uso do solo")
        
        # Classificação do uso do solo
        if 'landuse_class' not in gdf.columns:
            def classify_landuse(row):
                if pd.isna(row['landuse']):
                    return 'unknown'
                elif row['landuse'] in ['residential', 'apartment']:
                    return 'residential'
                elif row['landuse'] in ['retail', 'commercial']:
                    return 'commercial'
                elif row['landuse'] in ['industrial', 'warehouse']:
                    return 'industrial'
                elif row['landuse'] in ['education', 'university', 'school']:
                    return 'education'
                elif row['landuse'] in ['forest', 'grass', 'meadow', 'park']:
                    return 'green_area'
                elif row['landuse'] in ['farmland', 'farm', 'farmyard']:
                    return 'agricultural'
                else:
                    return 'other'
            
            gdf['landuse_class'] = gdf.apply(classify_landuse, axis=1)
            logger.info("Adicionada coluna de classificação de uso do solo")
        
        # Densidade (se a coluna 'building' existir)
        if 'density_class' not in gdf.columns:
            gdf['density_class'] = 'unknown'  # Valor padrão
            logger.info("Adicionada coluna de classe de densidade")
        
        # Porcentagem impermeável (estimativa)
        if 'impervious_pct' not in gdf.columns:
            def estimate_impervious(row):
                if row['landuse_class'] == 'residential':
                    return 70.0
                elif row['landuse_class'] == 'commercial':
                    return 90.0
                elif row['landuse_class'] == 'industrial':
                    return 80.0
                elif row['landuse_class'] == 'green_area':
                    return 10.0
                elif row['landuse_class'] == 'agricultural':
                    return 20.0
                else:
                    return 50.0  # Valor padrão
            
            gdf['impervious_pct'] = gdf.apply(estimate_impervious, axis=1)
            logger.info("Adicionada coluna de porcentagem impermeável estimada")
        
        # Salvar arquivo enriquecido
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path_with_timestamp = output_path.replace('.gpkg', f'_{timestamp}.gpkg')
        gdf.to_file(output_path_with_timestamp, driver='GPKG')
        logger.info(f"Arquivo enriquecido salvo em {output_path_with_timestamp}")
        
        return output_path_with_timestamp
    except Exception as e:
        logger.error(f"Erro ao enriquecer uso do solo: {str(e)}")
        return None

def basic_enrich_setores(input_path, output_path):
    """Realiza um enriquecimento básico de dados de setores censitários"""
    try:
        # Carregar o arquivo
        logger.info(f"Carregando dados de setores censitários de {input_path}")
        gdf = gpd.read_file(input_path)
        
        # Quantidade inicial de registros
        initial_count = len(gdf)
        logger.info(f"Carregados {initial_count} registros de setores censitários")
        
        # Adicionar coluna de área se não existir
        if 'area_m2' not in gdf.columns:
            gdf['area_m2'] = gdf.geometry.area
            logger.info("Calculada área dos setores censitários")
        
        # Densidade populacional (exemplo com valores fictícios)
        if 'pop_density' not in gdf.columns:
            # Aqui seria ideal usar dados reais do IBGE
            gdf['pop_density'] = 50.0  # Valor padrão em pessoas/hectare
            logger.info("Adicionada coluna de densidade populacional estimada")
        
        # Classificação de urbanização
        if 'urban_class' not in gdf.columns:
            gdf['urban_class'] = 'urban'  # Valor padrão
            logger.info("Adicionada coluna de classificação urbana")
        
        # Salvar arquivo enriquecido
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path_with_timestamp = output_path.replace('.gpkg', f'_{timestamp}.gpkg')
        gdf.to_file(output_path_with_timestamp, driver='GPKG')
        logger.info(f"Arquivo enriquecido salvo em {output_path_with_timestamp}")
        
        return output_path_with_timestamp
    except Exception as e:
        logger.error(f"Erro ao enriquecer setores censitários: {str(e)}")
        return None

def basic_enrich_roads(input_path, output_path):
    """Realiza um enriquecimento básico de dados de vias"""
    try:
        # Carregar o arquivo
        logger.info(f"Carregando dados de vias de {input_path}")
        gdf = gpd.read_file(input_path)
        
        # Quantidade inicial de registros
        initial_count = len(gdf)
        logger.info(f"Carregados {initial_count} registros de vias")
        
        # Adicionar coluna de comprimento se não existir
        if 'length_km' not in gdf.columns:
            gdf['length_km'] = gdf.geometry.length / 1000
            logger.info("Calculado comprimento das vias em km")
        
        # Adicionar coluna de sinuosidade
        if 'sinuosity' not in gdf.columns:
            def calculate_sinuosity(geom):
                if geom.geom_type == 'LineString':
                    start_point = geom.coords[0]
                    end_point = geom.coords[-1]
                    from shapely.geometry import Point
                    straight_line = Point(start_point).distance(Point(end_point))
                    if straight_line > 0:
                        return geom.length / straight_line
                    else:
                        return 1.0
                elif geom.geom_type == 'MultiLineString':
                    # Para MultiLineString, calculamos a média ponderada da sinuosidade
                    total_length = 0
                    total_sinuosity_weighted = 0
                    for line in geom.geoms:
                        start_point = line.coords[0]
                        end_point = line.coords[-1]
                        from shapely.geometry import Point
                        straight_line = Point(start_point).distance(Point(end_point))
                        if straight_line > 0:
                            sinuosity = line.length / straight_line
                        else:
                            sinuosity = 1.0
                        total_sinuosity_weighted += sinuosity * line.length
                        total_length += line.length
                    
                    if total_length > 0:
                        return total_sinuosity_weighted / total_length
                    else:
                        return 1.0
                else:
                    return 1.0
            
            gdf['sinuosity'] = gdf.geometry.apply(calculate_sinuosity)
            logger.info("Calculada sinuosidade das vias")
        
        # Classificação de vias
        if 'road_class' not in gdf.columns:
            def classify_road(row):
                if pd.isna(row['highway']):
                    return 'other'
                elif row['highway'] in ['motorway', 'trunk']:
                    return 'primary'
                elif row['highway'] in ['primary']:
                    return 'primary'
                elif row['highway'] in ['secondary']:
                    return 'secondary'
                elif row['highway'] in ['tertiary']:
                    return 'tertiary'
                elif row['highway'] in ['residential', 'living_street']:
                    return 'residential'
                elif row['highway'] in ['service', 'track']:
                    return 'service'
                else:
                    return 'other'
            
            gdf['road_class'] = gdf.apply(classify_road, axis=1)
            logger.info("Adicionada coluna de classificação de vias")
        
        # Conectividade estimada
        if 'connectivity' not in gdf.columns:
            gdf['connectivity'] = 2  # Valor padrão: cada via conecta 2 nós
            logger.info("Adicionada coluna de conectividade estimada")
        
        # Valores fictícios para elevação
        if 'elevation_min' not in gdf.columns:
            gdf['elevation_min'] = 0.0
            logger.info("Adicionada coluna de elevação mínima")
        
        if 'elevation_max' not in gdf.columns:
            gdf['elevation_max'] = 0.0
            logger.info("Adicionada coluna de elevação máxima")
        
        if 'elevation_mean' not in gdf.columns:
            gdf['elevation_mean'] = 0.0
            logger.info("Adicionada coluna de elevação média")
        
        if 'elevation_range' not in gdf.columns:
            gdf['elevation_range'] = 0.0
            logger.info("Adicionada coluna de variação de elevação")
        
        # Adicionar inclinação (valores fictícios)
        if 'slope_pct' not in gdf.columns:
            gdf['slope_pct'] = 0.0
            logger.info("Adicionada coluna de inclinação em porcentagem")
        
        if 'slope_deg' not in gdf.columns:
            gdf['slope_deg'] = 0.0
            logger.info("Adicionada coluna de inclinação em graus")
        
        if 'slope_class' not in gdf.columns:
            gdf['slope_class'] = 'flat'
            logger.info("Adicionada coluna de classe de inclinação")
        
        # Salvar arquivo enriquecido
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path_with_timestamp = output_path.replace('.gpkg', f'_{timestamp}.gpkg')
        gdf.to_file(output_path_with_timestamp, driver='GPKG')
        logger.info(f"Arquivo enriquecido salvo em {output_path_with_timestamp}")
        
        return output_path_with_timestamp
    except Exception as e:
        logger.error(f"Erro ao enriquecer vias: {str(e)}")
        return None

def process_all_data():
    """Processa todos os dados necessários"""
    logger.info("Iniciando processamento de todos os dados...")
    
    # Verificar se estamos no Colab
    if not is_colab():
        logger.warning("Esta função é destinada a ser executada no Google Colab")
        return
    
    # Criar diretórios
    create_directories()
    
    # Copiar arquivos brutos
    copy_raw_files()
    
    # Processar edificações
    buildings_input = os.path.join(DRIVE_DATA_DIR, "sorocaba_buildings.gpkg")
    buildings_output = os.path.join(DRIVE_DATA_DIR, "buildings_enriched.gpkg")
    if os.path.exists(buildings_input):
        buildings_processed = basic_enrich_buildings(buildings_input, buildings_output)
        logger.info(f"Edificações processadas: {buildings_processed}")
    else:
        logger.warning(f"Arquivo de edificações não encontrado: {buildings_input}")
    
    # Processar uso do solo
    landuse_input = os.path.join(DRIVE_DATA_DIR, "sorocaba_landuse.gpkg")
    landuse_output = os.path.join(DRIVE_DATA_DIR, "landuse_enriched.gpkg")
    if os.path.exists(landuse_input):
        landuse_processed = basic_enrich_landuse(landuse_input, landuse_output)
        logger.info(f"Uso do solo processado: {landuse_processed}")
    else:
        logger.warning(f"Arquivo de uso do solo não encontrado: {landuse_input}")
    
    # Processar setores censitários
    setores_input = os.path.join(DRIVE_DATA_DIR, "sorocaba_setores_censitarios.gpkg")
    setores_output = os.path.join(DRIVE_DATA_DIR, "setores_censitarios_enriched.gpkg")
    if os.path.exists(setores_input):
        setores_processed = basic_enrich_setores(setores_input, setores_output)
        logger.info(f"Setores censitários processados: {setores_processed}")
    else:
        logger.warning(f"Arquivo de setores censitários não encontrado: {setores_input}")
    
    # Processar vias
    roads_input = os.path.join(DRIVE_DATA_DIR, "sorocaba_roads.gpkg")
    roads_output = os.path.join(DRIVE_DATA_DIR, "roads_enriched.gpkg")
    if os.path.exists(roads_input):
        roads_processed = basic_enrich_roads(roads_input, roads_output)
        logger.info(f"Vias processadas: {roads_processed}")
    else:
        logger.warning(f"Arquivo de vias não encontrado: {roads_input}")
    
    logger.info("Processamento de todos os dados concluído!")

if __name__ == "__main__":
    if is_colab():
        logger.info("Ambiente Google Colab detectado")
        # Montar Google Drive
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("Google Drive montado com sucesso em /content/drive")
        except:
            logger.error("Erro ao montar o Google Drive")
            sys.exit(1)
        
        process_all_data()
    else:
        logger.info("Este script é destinado a ser executado no Google Colab") 