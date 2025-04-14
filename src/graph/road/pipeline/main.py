# -*- coding: utf-8 -*-
"""
Main Road Network Analysis Pipeline

This script provides a complete pipeline for road network analysis,
including data loading, preprocessing, graph construction, GNN training,
and evaluation.
"""

import os
import argparse
import time
import json
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Caminho absoluto para pasta do Google Drive
DRIVE_PATH = "/content/drive/MyDrive/geoprocessamento_gnn"
DATA_DIR = os.path.join(DRIVE_PATH, "DATA")
OUTPUT_DIR = os.path.join(DRIVE_PATH, "OUTPUT")
REPORT_DIR = os.path.join(DRIVE_PATH, "QUALITY_REPORT")

# Mantenha todos os caminhos originais
ROADS_PROCESSED_PATH = os.path.join(DATA_DIR, "processed/roads_processed.gpkg")
ROADS_ENRICHED_PATH = os.path.join(DATA_DIR, "enriched/roads_enriched.gpkg")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")
ROAD_GRAPHS_DIR = os.path.join(OUTPUT_DIR, "road_graphs")

from pipeline.data_loading import load_road_data, load_from_osm, load_pytorch_geometric_data, mount_google_drive
from pipeline.preprocessing import explode_multilines, clean_road_data
from pipeline.graph_construction import create_road_graph, assign_node_classes
from pipeline.gnn_models import GNN, ImprovedGNN, AttentionGNN
from pipeline.training import train, evaluate, save_results, plot_training_history
from pipeline.visualization import plot_road_network, plot_graph, plot_node_classes, create_interactive_map
from pipeline.reporting import generate_quality_report

def ensure_directories():
    """Garantir que todos os diretórios originais existam."""
    for directory in [DATA_DIR, OUTPUT_DIR, REPORT_DIR, RESULTS_DIR, 
                    VISUALIZATIONS_DIR, MODELS_DIR, ROAD_GRAPHS_DIR]:
        os.makedirs(directory, exist_ok=True)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Road Network Analysis Pipeline')
    
    # Data loading options
    group = parser.add_argument_group('Data Loading')
    group.add_argument('--data_path', type=str, help='Path to road network data file')
    group.add_argument('--place_name', type=str, help='Name of place to download from OSM')
    group.add_argument('--bbox', type=float, nargs=4, help='Bounding box for OSM data (min_lat, min_lon, max_lat, max_lon)')
    
    # Processing options
    group = parser.add_argument_group('Processing')
    group.add_argument('--explode_multilines', action='store_true', help='Explode multilinestrings into individual linestrings')
    group.add_argument('--crs', type=str, default='EPSG:4326', help='Coordinate reference system')
    
    # Model options
    group = parser.add_argument_group('Model')
    group.add_argument('--model_type', type=str, default='GNN', choices=['GNN', 'ImprovedGNN', 'AttentionGNN'],
                      help='Type of GNN model to use')
    group.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension size')
    group.add_argument('--num_classes', type=int, default=5, help='Number of node classes')
    group.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    
    # Training options
    group = parser.add_argument_group('Training')
    group.add_argument('--epochs', type=int, default=200, help='Maximum number of epochs')
    group.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    group.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of nodes for training')
    group.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of nodes for validation')
    group.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Output options
    group = parser.add_argument_group('Output')
    group.add_argument('--output_dir', type=str, default='results', help='Output directory')
    group.add_argument('--save_model', action='store_true', help='Save the trained model')
    group.add_argument('--save_visualizations', action='store_true', help='Generate and save visualizations')
    group.add_argument('--interactive_map', action='store_true', help='Generate interactive map')
    
    return parser.parse_args()

def main():
    """Função principal do pipeline."""
    # Montar o Google Drive (se estiver no Colab)
    mounted = mount_google_drive()
    if not mounted:
        print("AVISO: Google Drive não montado. Certifique-se que os caminhos estão corretos.")
    
    # Garantir que os diretórios existam
    ensure_directories()
    
    # Registrar o tempo de início
    start_time = time.time()
    
    # Timestamp para nomes de arquivos
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Carregar dados de ruas já processados (mantendo caminho original)
    print("Carregando dados de ruas...")
    gdf = load_road_data(ROADS_PROCESSED_PATH)
    
    # Pré-processar os dados
    print("Pré-processando dados...")
    gdf = explode_multilines(gdf)
    gdf = clean_road_data(gdf)
    
    # Construir o grafo
    print("Construindo grafo da rede viária...")
    G = create_road_graph(gdf)
    
    # Criar mapeamento de tipos de rua para índices
    highway_types = set()
    for _, data in G.edges(data=True):
        if 'highway' in data:
            highway_types.add(data['highway'])
    highway_to_idx = {hw: i for i, hw in enumerate(sorted(highway_types))}
    
    # Atribuir classes aos nós
    G = assign_node_classes(G, highway_to_idx)
    
    # Preparar dados para PyTorch Geometric
    print("Preparando dados para PyTorch Geometric...")
    data = load_pytorch_geometric_data(G)
    
    # Obter features dos nós e labels
    x = data.x
    
    # Criar labels dos nós
    y = torch.tensor([G.nodes[i].get('class', 0) for i in range(len(G.nodes))], dtype=torch.long)
    data.y = y
    
    # Criar máscaras train/val/test
    num_nodes = len(G.nodes)
    train_ratio = 0.7
    val_ratio = 0.1
    indices = torch.randperm(num_nodes)
    
    train_size = int(train_ratio * num_nodes)
    val_size = int(val_ratio * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    
    # Criar e treinar o modelo
    print("Criando e treinando o modelo...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    # Criar modelo
    input_dim = data.x.size(1)
    hidden_dim = 64
    num_classes = len(highway_to_idx)
    dropout = 0.5
    
    model = GNN(input_dim=input_dim, hidden_dim=hidden_dim, 
               output_dim=num_classes, dropout=dropout)
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # Treinar o modelo
    model, history = train(model, optimizer, data, epochs=200)
    
    # Avaliar o modelo
    print("Avaliando o modelo...")
    test_results = evaluate(model, data, data.test_mask)
    print(f"Acurácia de Teste: {test_results['accuracy']:.4f}")
    
    # Gerar relatórios e visualizações
    print("Gerando relatórios e visualizações...")
    
    # Criar mapeamento de índices para nomes de classes
    idx_to_class = {i: hw for hw, i in highway_to_idx.items()}
    
    # Gerar previsões
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1).cpu().numpy()
    
    # Gerar relatório de qualidade
    reports = generate_quality_report(G, pred, data.y.cpu().numpy(), 
                                     idx_to_class, output_dir=REPORT_DIR)
    
    # Salvar resultados
    saved_paths = save_results(test_results, model, history, 
                              output_dir=MODELS_DIR, 
                              prefix="gnn_road")
    print(f"Modelo salvo em {saved_paths['model']}")
    
    # Plotar histórico de treinamento
    history_plot = plot_training_history(history)
    history_path = os.path.join(VISUALIZATIONS_DIR, f"training_history_{timestamp}.png")
    plt.savefig(history_path)
    
    # Plotar rede de ruas
    road_plot = plot_road_network(gdf)
    road_path = os.path.join(VISUALIZATIONS_DIR, f"road_network_{timestamp}.png")
    plt.savefig(road_path)
    
    # Plotar classes dos nós
    class_plot = plot_node_classes(G, node_class_attr='class')
    class_path = os.path.join(VISUALIZATIONS_DIR, f"node_classes_{timestamp}.png")
    plt.savefig(class_path)
    
    # Gerar mapa interativo
    map_path = os.path.join(VISUALIZATIONS_DIR, f"interactive_map_{timestamp}.html")
    create_interactive_map(gdf, output_path=map_path)
    print(f"Mapa interativo salvo em {map_path}")
    
    # Calcular e imprimir tempo de execução
    execution_time = time.time() - start_time
    print(f"Pipeline concluído em {execution_time:.2f} segundos")
    
    # Retornar resultados para possível processamento adicional
    return {
        'model': model,
        'history': history,
        'test_results': test_results,
        'reports': reports
    }

if __name__ == '__main__':
    main() 