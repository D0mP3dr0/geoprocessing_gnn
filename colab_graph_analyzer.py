"""
Analisador de Grafo PyTorch Geometric para Redes Viárias
-------------------------------------------------------
Este script carrega e analisa um grafo de rede viária salvo em formato PyTorch Geometric (.pt).
Código autocontido para execução em uma única célula do Google Colab.
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from google.colab import drive

# Montar o Google Drive (se não estiver montado)
try:
    drive.mount('/content/drive')
    print("Google Drive montado com sucesso")
except:
    print("Drive já está montado ou ocorreu um erro")

# Configurar caminho do arquivo PyTorch Geometric
BASE_DIR = '/content/drive/MyDrive/geoprocessamento_gnn'
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Listar arquivos .pt disponíveis
try:
    pt_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pt')]
    pt_files.sort(reverse=True)  # Mais recentes primeiro
    if pt_files:
        print(f"Arquivos .pt encontrados ({len(pt_files)}):")
        for i, file in enumerate(pt_files):
            print(f"  {i+1}. {file}")
        
        # Usar o arquivo mais recente por padrão
        latest_pt_file = pt_files[0]
        PT_FILE_PATH = os.path.join(DATA_DIR, latest_pt_file)
        print(f"\nUsando arquivo mais recente: {latest_pt_file}")
    else:
        print("Nenhum arquivo .pt encontrado em:", DATA_DIR)
        PT_FILE_PATH = None
except Exception as e:
    print(f"Erro ao listar arquivos: {e}")
    PT_FILE_PATH = None

# Função principal para carregar e analisar o grafo
def analyze_graph(file_path):
    if not os.path.exists(file_path):
        print(f"Arquivo não encontrado: {file_path}")
        return
    
    print(f"Carregando dados de: {file_path}")
    try:
        # Carregar o objeto Data do PyTorch Geometric
        data = torch.load(file_path)
        
        # 1. Informações básicas do grafo
        print("\n==== INFORMAÇÕES BÁSICAS DO GRAFO ====")
        print(f"Número de nós: {data.num_nodes}")
        print(f"Número de arestas: {data.num_edges//2 if data.is_undirected() else data.num_edges}")
        print(f"O grafo é não direcionado: {data.is_undirected()}")
        print(f"Possui nós isolados: {data.has_isolated_nodes()}")
        
        # 2. Informações sobre features
        print("\n==== FEATURES DO GRAFO ====")
        if hasattr(data, 'x') and data.x is not None:
            print(f"Dimensões das features dos nós (data.x): {data.x.shape}")
            print(f"Média das features dos nós: {data.x.mean(dim=0).numpy().round(4)}")
            print(f"Desvio padrão das features dos nós: {data.x.std(dim=0).numpy().round(4)}")
        else:
            print("Não há features de nós disponíveis")
        
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            print(f"Dimensões das features das arestas (data.edge_attr): {data.edge_attr.shape}")
            print(f"Média das features das arestas: {data.edge_attr.mean(dim=0).numpy().round(4)}")
            print(f"Desvio padrão das features das arestas: {data.edge_attr.std(dim=0).numpy().round(4)}")
        else:
            print("Não há features de arestas disponíveis")
        
        # 3. Informações sobre classes (labels)
        if hasattr(data, 'y') and data.y is not None:
            num_classes = len(torch.unique(data.y))
            print(f"\n==== CLASSES DOS NÓS ====")
            print(f"Número de classes: {num_classes}")
            class_counts = torch.bincount(data.y)
            for i, count in enumerate(class_counts):
                print(f"  Classe {i}: {count} nós ({count/data.num_nodes:.2%})")
        
        # 4. Informações sobre divisão de dados
        if hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask'):
            print(f"\n==== DIVISÃO DOS DADOS ====")
            train_count = data.train_mask.sum().item()
            val_count = data.val_mask.sum().item()
            test_count = data.test_mask.sum().item()
            
            print(f"Nós para treinamento: {train_count} ({train_count/data.num_nodes:.2%})")
            print(f"Nós para validação: {val_count} ({val_count/data.num_nodes:.2%})")
            print(f"Nós para teste: {test_count} ({test_count/data.num_nodes:.2%})")
            
            # Verificar sobreposição
            train_val_overlap = (data.train_mask & data.val_mask).sum().item()
            train_test_overlap = (data.train_mask & data.test_mask).sum().item()
            val_test_overlap = (data.val_mask & data.test_mask).sum().item()
            
            if train_val_overlap + train_test_overlap + val_test_overlap > 0:
                print(f"ATENÇÃO: Há sobreposição nas máscaras!")
                print(f"  Train-Val: {train_val_overlap}")
                print(f"  Train-Test: {train_test_overlap}")
                print(f"  Val-Test: {val_test_overlap}")
        
        # 5. Análise de distribuição de grau
        print("\n==== ANÁLISE DE DISTRIBUIÇÃO DE GRAU ====")
        edge_index = data.edge_index.numpy()
        
        # Contar graus dos nós
        node_degrees = {}
        for i in range(edge_index.shape[1]):
            node = edge_index[0, i]
            node_degrees[node] = node_degrees.get(node, 0) + 1
        
        # Separamos nós por grau
        degree_counts = {}
        for node, degree in node_degrees.items():
            degree_counts[degree] = degree_counts.get(degree, 0) + 1
        
        # Estatísticas de grau
        degrees = list(node_degrees.values())
        avg_degree = np.mean(degrees)
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0
        
        print(f"Grau médio: {avg_degree:.2f}")
        print(f"Grau mínimo: {min_degree}")
        print(f"Grau máximo: {max_degree}")
        print(f"Distribuição de grau (top 10):")
        
        # Mostrar os 10 graus mais comuns
        sorted_degrees = sorted(degree_counts.items(), key=lambda x: x[1], reverse=True)
        for degree, count in sorted_degrees[:10]:
            print(f"  Grau {degree}: {count} nós ({count/data.num_nodes:.2%})")
        
        # 6. Visualizações
        plt.figure(figsize=(15, 10))
        
        # 6.1 Visualizar distribuição de grau
        plt.subplot(2, 2, 1)
        plt.hist(degrees, bins=min(30, max_degree), alpha=0.7)
        plt.axvline(x=avg_degree, color='r', linestyle='--', label=f'Média: {avg_degree:.2f}')
        plt.title('Distribuição de Grau dos Nós')
        plt.xlabel('Grau')
        plt.ylabel('Frequência')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6.2 Visualizar amostra do grafo (até 200 nós)
        if data.num_nodes > 200:
            # Para grafos grandes, amostrar um subconjunto
            sample_size = 200
            sampled_nodes = np.random.choice(data.num_nodes, sample_size, replace=False)
            
            # Filtrar arestas conectando apenas os nós amostrados
            edge_mask = np.isin(edge_index[0], sampled_nodes) & np.isin(edge_index[1], sampled_nodes)
            sampled_edges = edge_index[:, edge_mask]
            
            # Criar mapeamento de índices originais para índices 0...sample_size-1
            node_map = {int(node): i for i, node in enumerate(sampled_nodes)}
            
            # Criar grafo NetworkX
            G = nx.Graph()
            for i in range(len(sampled_nodes)):
                G.add_node(i)
            
            for i in range(sampled_edges.shape[1]):
                src, dst = sampled_edges[0, i], sampled_edges[1, i]
                if src in node_map and dst in node_map:
                    G.add_edge(node_map[src], node_map[dst])
            
            graph_title = f'Amostra do Grafo ({sample_size} nós de {data.num_nodes})'
        else:
            # Para grafos menores, usar todos os nós
            G = nx.Graph()
            for i in range(data.num_nodes):
                G.add_node(i)
            
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                G.add_edge(src, dst)
            
            graph_title = f'Grafo Completo ({data.num_nodes} nós)'
        
        plt.subplot(2, 2, 2)
        pos = nx.spring_layout(G, seed=42)  # Posição dos nós
        nx.draw(G, pos, node_size=30, node_color='skyblue', edge_color='gray', 
                width=0.5, alpha=0.7, with_labels=False)
        plt.title(graph_title)
        
        # 6.3 Visualizar histograma de features (primeiras 2 features)
        if hasattr(data, 'x') and data.x is not None and data.x.shape[1] >= 2:
            plt.subplot(2, 2, 3)
            x_np = data.x.numpy()
            plt.hist(x_np[:, 0], bins=30, alpha=0.7, label='Feature 1')
            plt.title('Distribuição da Feature 1')
            plt.xlabel('Valor')
            plt.ylabel('Frequência')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(2, 2, 4)
            plt.hist(x_np[:, 1], bins=30, alpha=0.7, label='Feature 2')
            plt.title('Distribuição da Feature 2')
            plt.xlabel('Valor')
            plt.ylabel('Frequência')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 7. Se tiver classes, mostrar distribuição por classe
        if hasattr(data, 'y') and data.y is not None:
            plt.figure(figsize=(12, 5))
            y_np = data.y.numpy()
            class_names = [f'Classe {i}' for i in range(num_classes)]
            class_counts = torch.bincount(data.y).numpy()
            
            plt.bar(class_names, class_counts)
            plt.title('Distribuição de Classes dos Nós')
            plt.xlabel('Classe')
            plt.ylabel('Número de Nós')
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            
            # Adicionar valores sobre as barras
            for i, count in enumerate(class_counts):
                plt.text(i, count + 0.1, str(count), ha='center')
            
            plt.tight_layout()
            plt.show()
        
        # 8. Estatísticas das features como DataFrame (se tiver features)
        if hasattr(data, 'x') and data.x is not None:
            print("\n==== ESTATÍSTICAS DAS FEATURES DOS NÓS ====")
            x_np = data.x.numpy()
            feature_names = [f'Feature_{i+1}' for i in range(data.x.shape[1])]
            
            stats = {
                'Média': np.mean(x_np, axis=0),
                'Desvio Padrão': np.std(x_np, axis=0),
                'Mínimo': np.min(x_np, axis=0),
                '25%': np.percentile(x_np, 25, axis=0),
                'Mediana': np.median(x_np, axis=0),
                '75%': np.percentile(x_np, 75, axis=0),
                'Máximo': np.max(x_np, axis=0)
            }
            
            stats_df = pd.DataFrame(stats, index=feature_names)
            display(stats_df.round(4))
        
        return data
    except Exception as e:
        print(f"Erro ao analisar grafo: {e}")
        return None

# Executar análise se encontrou um arquivo
if PT_FILE_PATH:
    data = analyze_graph(PT_FILE_PATH)
else:
    print("\nNenhum arquivo .pt encontrado para análise. Verifique o caminho do diretório.")
    print(f"Caminho atual: {DATA_DIR}")
    print("\nSe o caminho estiver incorreto, ajuste a variável BASE_DIR no início do script.") 