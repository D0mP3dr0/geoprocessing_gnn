# Análise e Enriquecimento de Dados de ERBs para Machine Learning Geoespacial
# Instala as dependências necessárias para processamento geoespacial

# Instalação das dependências (remova o # no Colab)
# pip install geopandas folium matplotlib contextily scikit-learn numpy pandas pyproj shapely networkx h3 rtree scipy tqdm seaborn

# Configuração dos caminhos para Google Colab
import os
import time
import logging
from datetime import datetime

# Novos caminhos para o Google Colab
BASE_PATH = "/content/drive/MyDrive/geoprocessing_gnn"
DATA_PATH = os.path.join(BASE_PATH, "data")
PROCESSED_DATA_PATH = os.path.join(DATA_PATH, "processed") 
ENRICHED_DATA_PATH = os.path.join(DATA_PATH, "enriched_data")
VISUALIZATION_DIR = os.path.join(BASE_PATH, "outputs/visualize_enriched_data/rbs")
REPORT_DIR = os.path.join(BASE_PATH, "outputs/reports")

# Criar diretórios necessários
def ensure_directories():
    """Cria todos os diretórios necessários para o processamento."""
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    os.makedirs(ENRICHED_DATA_PATH, exist_ok=True)
    os.makedirs(VISUALIZATION_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, "json"), exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, "summary"), exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, "csv"), exist_ok=True)
    
    print(f"Diretórios criados/verificados:")
    print(f"- Dados processados: {PROCESSED_DATA_PATH}")
    print(f"- Dados enriquecidos: {ENRICHED_DATA_PATH}")
    print(f"- Visualizações: {VISUALIZATION_DIR}")
    print(f"- Relatórios: {REPORT_DIR}")

# Configuração do logging
def setup_logging():
    """Configura o sistema de logging."""
    log_dir = os.path.join(BASE_PATH, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"erb_enrichment_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("erb_enrichment")

# Verificar e criar diretórios
ensure_directories()

# Importando bibliotecas e definindo funções para carregar dados
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
import json
import matplotlib.pyplot as plt
import contextily as ctx
from scipy.spatial import Voronoi
import networkx as nx
from sklearn.cluster import DBSCAN
import h3
import multiprocessing as mp
import shapely
from tqdm import tqdm

# Variável global para o logger
logger = None

# Função para carregar os dados
def load_data():
    """
    Carrega dados de ERBs.
    
    Returns:
        geopandas.GeoDataFrame: DataFrame com dados de ERBs ou None em caso de erro
    """
    # Iniciar o logger
    global logger
    if logger is None:
        logger = setup_logging()
    
    # Caminho para o arquivo de dados
    # Modifique este caminho para o local do seu arquivo de entrada
    input_file = os.path.join(PROCESSED_DATA_PATH, "erbs_data.gpkg")
    
    try:
        # Carregar dados
        logger.info(f"Carregando dados de {input_file}")
        data = gpd.read_file(input_file)
        
        # Verificar se há dados
        if data.empty:
            logger.error("Arquivo carregado, mas não contém dados")
            return None
        
        logger.info(f"Dados carregados com sucesso: {len(data)} registros, {len(data.columns)} colunas")
        
        # Verificar o sistema de coordenadas
        if data.crs is None:
            logger.warning("Dados sem sistema de coordenadas definido, assumindo WGS84")
            data.set_crs(epsg=4326, inplace=True)
        else:
            logger.info(f"Sistema de coordenadas: {data.crs}")
        
        # Verificar se há geometrias válidas
        invalid_geoms = ~data.geometry.is_valid
        if invalid_geoms.any():
            n_invalid = invalid_geoms.sum()
            logger.warning(f"Encontradas {n_invalid} geometrias inválidas ({n_invalid/len(data)*100:.2f}%)")
            
            # Tentar corrigir geometrias inválidas
            data.geometry = data.geometry.buffer(0)
            still_invalid = ~data.geometry.is_valid
            if still_invalid.any():
                logger.warning(f"Ainda existem {still_invalid.sum()} geometrias inválidas após correção")
        
        # Verificar tipo das geometrias (esperado: Point)
        geom_types = data.geometry.type.unique()
        logger.info(f"Tipos de geometria encontrados: {geom_types}")
        if 'Point' not in geom_types:
            logger.warning("Não foram encontradas geometrias do tipo Point!")
        
        return data
    
    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado: {input_file}")
        return None
    except Exception as e:
        logger.error(f"Erro ao carregar dados: {str(e)}")
        return None

# Funções para cálculo de potência e cobertura
def calculate_eirp(gdf):
    """
    Calcula o EIRP (Potência Isotrópica Efetivamente Radiada) em dBm para cada ERB.
    
    EIRP = Potência do Transmissor (dBm) + Ganho da Antena (dBi)
    
    Args:
        gdf (geopandas.GeoDataFrame): DataFrame com dados de ERB
        
    Returns:
        geopandas.GeoDataFrame: DataFrame atualizado com EIRP calculado
    """
    # Iniciar o logger
    global logger
    if logger is None:
        logger = setup_logging()
    
    logger.info("Calculando EIRP para cada ERB")
    
    # Fazer uma cópia para não modificar o original
    result = gdf.copy()
    
    try:
        # Verificar se as colunas necessárias existem
        if 'PotenciaTransmissorWatts' not in result.columns:
            logger.error("Coluna 'PotenciaTransmissorWatts' não encontrada")
            return gdf
        
        if 'GanhoAntena' not in result.columns:
            logger.error("Coluna 'GanhoAntena' não encontrada")
            return gdf
        
        # Converter potência de Watts para dBm se necessário
        # dBm = 10 * log10(1000 * Potência em Watts)
        if 'PotenciaTxdBm' not in result.columns:
            # Converter para valores numéricos e substituir valores nulos
            result['PotenciaTransmissorWatts'] = pd.to_numeric(result['PotenciaTransmissorWatts'], errors='coerce')
            mask_nulos = result['PotenciaTransmissorWatts'].isna()
            
            if mask_nulos.any():
                n_nulos = mask_nulos.sum()
                logger.warning(f"Encontrados {n_nulos} valores nulos em 'PotenciaTransmissorWatts'")
                # Usar valor padrão para registros com potência nula (1 Watt = 30 dBm)
                result.loc[mask_nulos, 'PotenciaTransmissorWatts'] = 1.0
            
            # Converter potência para dBm
            result['PotenciaTxdBm'] = 10 * np.log10(1000 * result['PotenciaTransmissorWatts'])
            logger.info("Potência convertida de Watts para dBm")
        
        # Converter ganho para valores numéricos
        result['GanhoAntena'] = pd.to_numeric(result['GanhoAntena'], errors='coerce')
        
        # Substituir valores nulos no ganho
        mask_ganho_nulo = result['GanhoAntena'].isna()
        if mask_ganho_nulo.any():
            n_ganho_nulo = mask_ganho_nulo.sum()
            logger.warning(f"Encontrados {n_ganho_nulo} valores nulos em 'GanhoAntena'")
            # Usar valor padrão para ganho nulo (0 dBi)
            result.loc[mask_ganho_nulo, 'GanhoAntena'] = 0.0
        
        # Calcular EIRP (dBm)
        result['EIRP_dBm'] = result['PotenciaTxdBm'] + result['GanhoAntena']
        
        # Registrar estatísticas
        eirp_mean = result['EIRP_dBm'].mean()
        eirp_min = result['EIRP_dBm'].min()
        eirp_max = result['EIRP_dBm'].max()
        
        logger.info(f"EIRP calculado: média={eirp_mean:.2f} dBm, min={eirp_min:.2f} dBm, max={eirp_max:.2f} dBm")
        
        return result
    
    except Exception as e:
        logger.error(f"Erro ao calcular EIRP: {str(e)}")
        return gdf

def calculate_coverage_radius(gdf):
    """
    Calcula o raio de cobertura para cada ERB baseado no EIRP e frequência.
    
    Usa um modelo de propagação simplificado:
    - Raio (km) = 10^((EIRP - Limiar - 32.44 - 20*log10(f)) / 20)
    onde f é a frequência em MHz e o Limiar é a sensibilidade do receptor (-100 dBm por padrão)
    
    Args:
        gdf (geopandas.GeoDataFrame): DataFrame com dados de ERB
        
    Returns:
        geopandas.GeoDataFrame: DataFrame atualizado com raio de cobertura calculado
    """
    # Iniciar o logger
    global logger
    if logger is None:
        logger = setup_logging()
    
    logger.info("Calculando raio de cobertura para cada ERB")
    
    # Fazer uma cópia para não modificar o original
    result = gdf.copy()
    
    try:
        # Verificar se o EIRP foi calculado
        if 'EIRP_dBm' not in result.columns:
            logger.warning("EIRP não encontrado, calculando primeiro")
            result = calculate_eirp(result)
        
        # Verificar se temos a frequência
        if 'Frequencia' not in result.columns:
            logger.warning("Coluna 'Frequencia' não encontrada, usando valor padrão de 900 MHz")
            result['Frequencia'] = 900  # Valor padrão em MHz
        else:
            # Converter para valores numéricos
            result['Frequencia'] = pd.to_numeric(result['Frequencia'], errors='coerce')
            # Substituir valores nulos
            mask_freq_nula = result['Frequencia'].isna()
            if mask_freq_nula.any():
                logger.warning(f"Encontrados {mask_freq_nula.sum()} valores nulos em 'Frequencia', usando 900 MHz")
                result.loc[mask_freq_nula, 'Frequencia'] = 900
        
        # Limiar de sensibilidade do receptor (padrão: -100 dBm)
        threshold_dbm = -100
        
        # Calcular raio de cobertura usando modelo simplificado
        # Raio (km) = 10^((EIRP - Limiar - 32.44 - 20*log10(f)) / 20)
        result['Raio_Cobertura_km'] = 10 ** ((result['EIRP_dBm'] - threshold_dbm - 32.44 - 20 * np.log10(result['Frequencia'])) / 20)
        
        # Calcular área de cobertura aproximada (assumindo cobertura circular)
        result['Area_Cobertura_km2'] = np.pi * result['Raio_Cobertura_km'] ** 2
        
        # Limitar o raio máximo para valores razoáveis (ex: 50 km)
        max_raio = 50.0
        mask_raio_grande = result['Raio_Cobertura_km'] > max_raio
        if mask_raio_grande.any():
            logger.warning(f"Encontrados {mask_raio_grande.sum()} raios maiores que {max_raio} km, limitando")
            result.loc[mask_raio_grande, 'Raio_Cobertura_km'] = max_raio
            result.loc[mask_raio_grande, 'Area_Cobertura_km2'] = np.pi * max_raio ** 2
        
        # Registrar estatísticas
        raio_mean = result['Raio_Cobertura_km'].mean()
        area_mean = result['Area_Cobertura_km2'].mean()
        
        logger.info(f"Raio médio de cobertura: {raio_mean:.2f} km")
        logger.info(f"Área média de cobertura: {area_mean:.2f} km²")
        
        return result
    
    except Exception as e:
        logger.error(f"Erro ao calcular raio de cobertura: {str(e)}")
        return gdf

def create_coverage_sectors(gdf):
    """
    Cria setores de cobertura para cada ERB com base no azimute e raio de cobertura.
    
    Cria um polígono representando a cobertura setorial da antena:
    - Se o ângulo de abertura da antena estiver disponível, usa-o
    - Caso contrário, assume uma abertura padrão de 120 graus
    
    Args:
        gdf (geopandas.GeoDataFrame): DataFrame com dados de ERB
        
    Returns:
        geopandas.GeoDataFrame: DataFrame atualizado com setores de cobertura
    """
    # Iniciar o logger
    global logger
    if logger is None:
        logger = setup_logging()
    
    logger.info("Criando setores de cobertura para cada ERB")
    
    # Fazer uma cópia para não modificar o original
    result = gdf.copy()
    
    try:
        # Verificar se temos Azimute
        if 'Azimute' not in result.columns:
            logger.error("Coluna 'Azimute' não encontrada, necessária para criar setores")
            return gdf
        
        # Verificar se o raio de cobertura foi calculado
        if 'Raio_Cobertura_km' not in result.columns:
            logger.warning("Raio de cobertura não encontrado, calculando primeiro")
            result = calculate_coverage_radius(result)
        
        # Converter azimute para numérico
        result['Azimute'] = pd.to_numeric(result['Azimute'], errors='coerce')
        
        # Verificar se temos ângulo de abertura
        abertura_padrao = 120  # Ângulo de abertura padrão em graus
        
        if 'AnguloAbertura' not in result.columns:
            logger.warning(f"Coluna 'AnguloAbertura' não encontrada, usando valor padrão de {abertura_padrao} graus")
            result['AnguloAbertura'] = abertura_padrao
        else:
            # Converter para valores numéricos
            result['AnguloAbertura'] = pd.to_numeric(result['AnguloAbertura'], errors='coerce')
            # Substituir valores nulos
            mask_abertura_nula = result['AnguloAbertura'].isna()
            if mask_abertura_nula.any():
                logger.warning(f"Encontrados {mask_abertura_nula.sum()} valores nulos em 'AnguloAbertura', usando {abertura_padrao} graus")
                result.loc[mask_abertura_nula, 'AnguloAbertura'] = abertura_padrao
        
        # Criar coluna para os setores
        result['setor_geometria'] = None
        
        # Função para criar um polígono setorial
        def criar_setor(row):
            try:
                # Obter coordenadas, azimute e raio
                lon, lat = row.geometry.x, row.geometry.y
                azimute = row['Azimute']
                raio_km = row['Raio_Cobertura_km']
                abertura = row['AnguloAbertura']
                
                # Converter para coordenadas projetadas para cálculos
                ponto = Point(lon, lat)
                
                # Evitar cálculos para raios muito pequenos
                if raio_km < 0.01:
                    return None
                
                # Calcular ângulos do setor
                # Azimute: 0 = Norte, 90 = Leste, 180 = Sul, 270 = Oeste
                angulo_inicio = (azimute - abertura / 2) % 360
                angulo_fim = (azimute + abertura / 2) % 360
                
                # Converter para radianos
                angulo_inicio_rad = np.radians(angulo_inicio)
                angulo_fim_rad = np.radians(angulo_fim)
                
                # Número de pontos para criar o arco
                num_pontos = 20
                
                # Criar pontos do setor
                pontos = [ponto]  # Primeiro ponto é a localização da ERB
                
                # Determinar se precisamos atravessar 0/360 graus
                if angulo_fim < angulo_inicio:
                    # Precisamos dividir em dois arcos: inicio-360 e 0-fim
                    angulos = np.linspace(angulo_inicio_rad, 2 * np.pi, num_pontos // 2)
                    angulos = np.append(angulos, np.linspace(0, angulo_fim_rad, num_pontos // 2))
                else:
                    # Arco simples de inicio a fim
                    angulos = np.linspace(angulo_inicio_rad, angulo_fim_rad, num_pontos)
                
                # Converter para quilômetros (aproximação para coordenadas geográficas)
                # Fator de conversão aproximado: 1 grau = 111 km
                km_por_grau_lat = 111.0
                km_por_grau_lon = 111.0 * np.cos(np.radians(lat))
                
                # Adicionar pontos do arco
                for angulo in angulos:
                    # Calcular deslocamento em quilômetros
                    dx_km = raio_km * np.sin(angulo)  # Deslocamento leste-oeste
                    dy_km = raio_km * np.cos(angulo)  # Deslocamento norte-sul
                    
                    # Converter para graus
                    dx_graus = dx_km / km_por_grau_lon
                    dy_graus = dy_km / km_por_grau_lat
                    
                    # Adicionar ponto
                    pontos.append(Point(lon + dx_graus, lat + dy_graus))
                
                # Adicionar o primeiro ponto novamente para fechar o polígono
                pontos.append(ponto)
                
                # Criar polígono a partir dos pontos
                coordenadas = [(p.x, p.y) for p in pontos]
                poligono = Polygon(coordenadas)
                
                return poligono
            
            except Exception as e:
                logger.warning(f"Erro ao criar setor para ERB: {e}")
                return None
        
        # Aplicar função para cada ERB
        for idx, row in tqdm(result.iterrows(), total=len(result), desc="Criando setores"):
            result.at[idx, 'setor_geometria'] = criar_setor(row)
        
        # Contar setores criados com sucesso
        setores_validos = result['setor_geometria'].notna().sum()
        logger.info(f"Criados {setores_validos} setores de cobertura de um total de {len(result)} ERBs")
        
        return result
    
    except Exception as e:
        logger.error(f"Erro ao criar setores de cobertura: {str(e)}")
        return gdf

# Execução de teste com dados de exemplo
# Para usar em seu notebook, descomente este código

# # Inicialização do logger
# logger = setup_logging()

# # Exemplo: Criar dados de teste quando o arquivo não estiver disponível
# def create_sample_data():
#     """Cria dados de teste para exemplificar o processamento."""
#     # Criar 5 pontos em uma região
#     lats = np.random.uniform(-23.65, -23.55, 5)  # Latitudes na região de São Paulo
#     lons = np.random.uniform(-46.75, -46.65, 5)  # Longitudes na região de São Paulo
#     
#     # Criar pontos
#     geometria = [Point(lon, lat) for lon, lat in zip(lons, lats)]
#     
#     # Criar dados fictícios
#     dados = {
#         'NomeEntidade': ['Operadora A', 'Operadora B', 'Operadora A', 'Operadora C', 'Operadora B'],
#         'PotenciaTransmissorWatts': [10.0, 15.0, 12.0, 8.0, 20.0],
#         'GanhoAntena': [17.0, 15.0, 16.0, 14.0, 18.0],
#         'Frequencia': [900, 1800, 900, 2100, 1800],
#         'Azimute': [0, 120, 240, 60, 180],
#         'AnguloAbertura': [120, 120, 120, 90, 90],
#         'geometry': geometria
#     }
#     
#     # Criar GeoDataFrame
#     gdf = gpd.GeoDataFrame(dados, geometry='geometry', crs="EPSG:4326")
#     
#     # Salvar arquivo de exemplo
#     os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
#     output_file = os.path.join(PROCESSED_DATA_PATH, "erbs_data.gpkg")
#     gdf.to_file(output_file, driver="GPKG")
#     
#     print(f"Dados de exemplo criados e salvos em {output_file}")
#     return gdf

# # Tentar carregar dados ou criar dados de exemplo se necessário
# def load_or_create_sample_data():
#     """Tenta carregar dados reais ou cria dados de exemplo se necessário."""
#     data = load_data()
#     if data is None:
#         print("Criando dados de exemplo para demonstração...")
#         data = create_sample_data()
#     return data

# # Processar dados
# print("Iniciando processamento de enriquecimento...")
# data = load_or_create_sample_data()

# # Calcular EIRP e raio de cobertura
# data = calculate_eirp(data)
# data = calculate_coverage_radius(data)

# # Criar setores de cobertura
# data = create_coverage_sectors(data)

# # Exibir estatísticas dos dados processados
# print("\nEstatísticas dos dados processados:")
# print(f"Total de ERBs: {len(data)}")
# print(f"EIRP médio: {data['EIRP_dBm'].mean():.2f} dBm")
# print(f"Raio médio: {data['Raio_Cobertura_km'].mean():.2f} km")
# print(f"Setores válidos: {data['setor_geometria'].notna().sum()}")

# # Visualizar dados em um mapa
# print("\nGerando visualização básica...")
# plt.figure(figsize=(12, 10))
# ax = plt.subplot(111)

# # Plotar locais das ERBs
# data.plot(ax=ax, markersize=50, color='red', alpha=0.7)

# # Plotar setores de cobertura se disponíveis
# if 'setor_geometria' in data.columns and data['setor_geometria'].notna().any():
#     # Criar um GeoDataFrame temporário apenas com os setores válidos
#     setores_gdf = gpd.GeoDataFrame(
#         {'NomeEntidade': data.loc[data['setor_geometria'].notna(), 'NomeEntidade']},
#         geometry=data.loc[data['setor_geometria'].notna(), 'setor_geometria'],
#         crs=data.crs
#     )
#     
#     # Plotar setores com cores por operadora
#     setores_gdf.plot(ax=ax, column='NomeEntidade', alpha=0.5, legend=True)

# # Adicionar mapa base
# try:
#     ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
# except Exception as e:
#     print(f"Erro ao adicionar mapa base: {e}")

# # Adicionar título
# plt.title('Localização e Cobertura das ERBs', fontsize=16)

# # Mostrar mapa
# plt.tight_layout()
# plt.savefig(os.path.join(VISUALIZATION_DIR, "mapa_cobertura_erbs.png"), dpi=300, bbox_inches='tight')
# print(f"Mapa salvo em {os.path.join(VISUALIZATION_DIR, 'mapa_cobertura_erbs.png')}")

# # Exibir o mapa no notebook
# plt.show()

print("Módulo de enriquecimento de dados de ERBs carregado com sucesso!")
print("Pronto para processamento. Consulte a função 'main()' para execução completa.")
print("Veja a próxima célula para continuar com a análise e criação de hexágonos e diagramas de Voronoi.") 