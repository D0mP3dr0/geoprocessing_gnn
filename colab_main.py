import os
import sys
import logging
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import glob
import json
import traceback

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('colab_main')

def setup_environment():
    """Configuração do ambiente do Google Colab"""
    logger.info("Configurando ambiente do Google Colab...")
    
    # Verificar se estamos no Colab
    in_colab = 'google.colab' in sys.modules
    if not in_colab:
        logger.warning("Este script é destinado a ser executado no Google Colab")
        return False
    
    # Montar o Google Drive se ainda não estiver montado
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        logger.info("Google Drive montado com sucesso")
    except:
        logger.error("Erro ao montar o Google Drive")
        return False
    
    # Definir diretórios essenciais
    base_dir = "/content/drive/MyDrive/geoprocessamento_gnn"
    data_dir = os.path.join(base_dir, "data")
    output_dir = os.path.join(base_dir, "OUTPUT")
    quality_dir = os.path.join(base_dir, "QUALITY_REPORT")
    vis_dir = os.path.join(base_dir, "VISUALIZACOES")
    
    # Criar diretórios se não existirem
    for directory in [base_dir, data_dir, output_dir, quality_dir, vis_dir]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Diretório verificado/criado: {directory}")
    
    # Configurar variáveis de ambiente para os módulos
    os.environ['GEOPROCESSAMENTO_BASE_DIR'] = base_dir
    os.environ['GEOPROCESSAMENTO_DATA_DIR'] = data_dir
    os.environ['GEOPROCESSAMENTO_OUTPUT_DIR'] = output_dir
    os.environ['GEOPROCESSAMENTO_QUALITY_DIR'] = quality_dir
    os.environ['GEOPROCESSAMENTO_VIS_DIR'] = vis_dir
    
    # Verificar se os dados estão disponíveis
    brutos_existem = verify_raw_files(data_dir)
    if not brutos_existem:
        logger.error("Não foram encontrados arquivos brutos necessários no Google Drive")
        logger.info("Execute o script copy_and_process_data.py primeiro para preparar os dados")
        return False
    
    return True

def verify_raw_files(data_dir):
    """Verifica se os arquivos brutos necessários estão disponíveis"""
    arquivos_necessarios = [
        "sorocaba_roads.gpkg",
        "sorocaba_buildings.gpkg",
        "sorocaba_landuse.gpkg",
        "sorocaba_setores_censitarios.gpkg"
    ]
    
    arquivos_encontrados = []
    for arquivo in arquivos_necessarios:
        caminho = os.path.join(data_dir, arquivo)
        if os.path.exists(caminho):
            arquivos_encontrados.append(arquivo)
            logger.info(f"Arquivo encontrado: {arquivo}")
        else:
            logger.warning(f"Arquivo não encontrado: {arquivo}")
    
    return len(arquivos_encontrados) > 0

def process_data():
    """Processa os dados para análise"""
    logger.info("Iniciando processamento de dados...")
    
    # Obter caminhos dos diretórios das variáveis de ambiente
    base_dir = os.environ.get('GEOPROCESSAMENTO_BASE_DIR')
    data_dir = os.environ.get('GEOPROCESSAMENTO_DATA_DIR')
    output_dir = os.environ.get('GEOPROCESSAMENTO_OUTPUT_DIR')
    quality_dir = os.environ.get('GEOPROCESSAMENTO_QUALITY_DIR')
    
    # 1. Carregar e processar os dados de vias
    try:
        # Importar módulo de processamento de vias
        # Verificamos se os módulos do src/graph/road estão no path
        sys.path.append('/content/drive/MyDrive/geoprocessamento_gnn')
        
        from src.graph.road.pipeline.03_Inicializacao_Carregamento_Pre_processamento import load_road_data, preprocess_road_data, run_preprocessing_pipeline
        
        # Carregar ou processar os dados de vias
        roads_raw_path = os.path.join(data_dir, "sorocaba_roads.gpkg")
        roads_enriched_path = find_latest_file(data_dir, "roads_enriched_*.gpkg")
        
        if roads_enriched_path:
            logger.info(f"Usando arquivo de vias enriquecido existente: {roads_enriched_path}")
            roads_gdf = load_road_data(roads_enriched_path)
        else:
            logger.info("Arquivo de vias enriquecido não encontrado. Processando dados brutos...")
            if os.path.exists(roads_raw_path):
                roads_gdf = load_road_data(roads_raw_path)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                roads_processed_path = os.path.join(data_dir, f"roads_enriched_{timestamp}.gpkg")
                preprocess_road_data(roads_raw_path, roads_processed_path)
                roads_gdf = load_road_data(roads_processed_path)
            else:
                logger.error(f"Arquivo de vias não encontrado: {roads_raw_path}")
                return False
        
        if roads_gdf is None or len(roads_gdf) == 0:
            logger.error("Falha no carregamento de dados de vias")
            return False
        
        logger.info(f"Dados de vias carregados: {len(roads_gdf)} registros")
        
        # 2. Executar o pipeline completo de processamento
        result = run_preprocessing_pipeline()
        if not result:
            logger.error("Falha na execução do pipeline de pré-processamento")
            return False
        
        logger.info("Pré-processamento de vias concluído com sucesso")
        
        # 3. Construir e analisar o grafo da rede viária
        from src.graph.road.pipeline.04_CONSTRUÇÃO_DO_GRAFO import RoadNetworkBuilder, pipeline_completo
        
        # Carregamos os dados processados mais recentes
        roads_processed_path = find_latest_file(data_dir, "roads_processed_*.gpkg")
        if not roads_processed_path:
            logger.error("Não foi possível encontrar o arquivo de vias processadas")
            return False
        
        logger.info(f"Carregando vias processadas de: {roads_processed_path}")
        roads_processed = gpd.read_file(roads_processed_path)
        
        # Executar o pipeline completo
        builder, files = pipeline_completo(
            roads_processed, 
            output_dir=output_dir,
            calculate_centrality=True,
            detect_communities=True,
            spatial_analysis=True
        )
        
        if builder is None:
            logger.error("Falha na construção do grafo")
            return False
        
        logger.info(f"Grafo construído com sucesso: {builder.G.number_of_nodes()} nós, {builder.G.number_of_edges()} arestas")
        logger.info(f"Arquivos gerados: {list(files.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro durante o processamento dos dados: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def find_latest_file(directory, pattern):
    """Encontra o arquivo mais recente que corresponde ao padrão"""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    
    # Ordenar por data de modificação (mais recente primeiro)
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def show_processing_results():
    """Exibe o resumo dos resultados do processamento"""
    quality_dir = os.environ.get('GEOPROCESSAMENTO_QUALITY_DIR')
    if not quality_dir:
        logger.error("Diretório de relatórios de qualidade não definido")
        return
    
    # Encontrar o relatório de qualidade mais recente
    quality_files = glob.glob(os.path.join(quality_dir, "road_preprocessing_quality_*.json"))
    if not quality_files:
        logger.warning("Nenhum relatório de qualidade encontrado")
        return
    
    latest_quality_report = max(quality_files, key=os.path.getmtime)
    logger.info(f"Exibindo resultados do relatório de qualidade: {latest_quality_report}")
    
    try:
        with open(latest_quality_report, 'r') as f:
            report = json.load(f)
        
        print("\n" + "="*80)
        print(f"RELATÓRIO DE QUALIDADE DO PROCESSAMENTO DE VIAS")
        print("="*80)
        print(f"Data do relatório: {report.get('report_date', 'N/A')}")
        print(f"Arquivo original: {report.get('original_file', 'N/A')}")
        print(f"Arquivo processado: {report.get('processed_file', 'N/A')}")
        print("\nESTATÍSTICAS GERAIS:")
        print(f"- Features originais: {report.get('counts', {}).get('initial_features', 'N/A')}")
        print(f"- Features processadas: {report.get('counts', {}).get('final_features', 'N/A')}")
        
        print("\nDISTRIBUIÇÃO DE TIPOS DE VIAS:")
        for highway, count in report.get('road_types', {}).get('final_highway_dist', {}).items():
            print(f"- {highway}: {count}")
        
        print("\nESTATÍSTICAS DE COMPRIMENTO (metros):")
        length_stats = report.get('geometry', {}).get('final_length_stats_m', {})
        print(f"- Mínimo: {length_stats.get('min', 'N/A'):.2f}")
        print(f"- Máximo: {length_stats.get('max', 'N/A'):.2f}")
        print(f"- Média: {length_stats.get('mean', 'N/A'):.2f}")
        print(f"- Mediana: {length_stats.get('median', 'N/A'):.2f}")
        print(f"- Comprimento total (km): {report.get('geometry', {}).get('final_total_length_km', 'N/A'):.2f}")
        
        print("\nESTATÍSTICAS DE SINUOSIDADE:")
        sinuosity_stats = report.get('derived_metrics', {}).get('sinuosity_stats', {})
        print(f"- Mínimo: {sinuosity_stats.get('min', 'N/A'):.2f}")
        print(f"- Máximo: {sinuosity_stats.get('max', 'N/A'):.2f}")
        print(f"- Média: {sinuosity_stats.get('mean', 'N/A'):.2f}")
        
        print("\nCONECTIVIDADE DA REDE:")
        connectivity = report.get('connectivity', {}).get('final', {})
        print(f"- Rede conectada: {connectivity.get('is_connected', 'N/A')}")
        print(f"- Número de componentes: {connectivity.get('num_components', 'N/A')}")
        print(f"- Número de nós: {connectivity.get('num_nodes', 'N/A')}")
        print(f"- Número de arestas: {connectivity.get('num_edges', 'N/A')}")
        print(f"- Tamanho do maior componente: {connectivity.get('largest_component_size', 'N/A')}")
        print(f"- Percentual do maior componente: {connectivity.get('largest_component_percentage', 'N/A')*100:.2f}%")
        
        print("\nPREPARAÇÃO PARA GNN:")
        gnn_info = report.get('gnn_preparation', {})
        print(f"- Contagem de nós: {gnn_info.get('nodes_count', 'N/A')}")
        print(f"- Contagem de arestas: {gnn_info.get('edges_count', 'N/A')}")
        print(f"- Características dos nós: {gnn_info.get('nodes_features_count', 'N/A')}")
        print(f"- Características das arestas: {gnn_info.get('edges_features_count', 'N/A')}")
        print(f"- Arquivo CSV de nós: {gnn_info.get('nodes_output_csv', 'N/A')}")
        print(f"- Arquivo CSV de arestas: {gnn_info.get('edges_output_csv', 'N/A')}")
        
        print("\nTEMPO DE PROCESSAMENTO:")
        print(f"- Duração total (segundos): {report.get('processing_pipeline_duration_seconds', 'N/A'):.2f}")
        print("="*80 + "\n")
        
    except Exception as e:
        logger.error(f"Erro ao exibir relatório de qualidade: {str(e)}")

if __name__ == "__main__":
    logger.info("Iniciando script principal para o Google Colab")
    
    # Configurar o ambiente
    if not setup_environment():
        logger.error("Falha na configuração do ambiente. Abortando.")
        sys.exit(1)
    
    # Processar os dados
    if not process_data():
        logger.error("Falha no processamento dos dados. Abortando.")
        sys.exit(1)
    
    # Mostrar resultados
    show_processing_results()
    
    logger.info("Processamento concluído com sucesso!") 