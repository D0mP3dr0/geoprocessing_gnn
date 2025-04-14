#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Advanced GCN for Morphological Analysis of Road Networks (GPU L4 Optimized + Neighbor Sampling)

This script implements an enhanced GNN model optimized for GPU capabilities (e.g., L4)
and large graphs using neighbor sampling. It includes:
- Optimized GNN architecture (GCNConv, GATConv, LayerNorm, SiLU)
- Optimized training with Neighbor Sampling for large graphs.
- Specific optimizations for L4 GPUs.
- Memory-efficient data preprocessing and visualization.
- Google Drive integration for Colab persistence.
"""

import os
import time
import psutil  # For system monitoring
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from datetime import datetime
import json # For saving report

# Scikit-learn imports
from sklearn.manifold import TSNE
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             precision_score, recall_score, classification_report)
from sklearn.model_selection import train_test_split

# PyTorch Geometric imports
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, subgraph
from torch_geometric.loader import NeighborSampler # For mini-batch training

# PyTorch utilities
from torch.optim.lr_scheduler import ReduceLROnPlateau # Changed from OneCycleLR for sampling
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn # For SiLU, LayerNorm

# Profiling (optional)
# import torch.cuda.profiler as profiler
# import torch.cuda.nvtx as nvtx

# --- Configuration ---
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# --- L4 GPU Optimizations ---
def optimize_for_l4_gpu():
    """Configurações otimizadas para GPU L4 no Colab"""
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping GPU optimizations.")
        return

    torch.backends.cudnn.benchmark = True  # Acelera operações convolucionais se input size não varia muito
    torch.backends.cuda.matmul.allow_tf32 = True  # Permite TF32 na L4 para aceleração matmul
    torch.backends.cudnn.allow_tf32 = True  # Permite TF32 para operações cudnn (conv)

    prop = torch.cuda.get_device_properties(0)
    print(f"GPU: {prop.name}, Memória Total: {prop.total_memory / 1e9:.2f} GB")
    if 'L4' in prop.name:
        print("Detectada GPU L4. Aplicando otimizações específicas...")
        # Libera memória cache não utilizada ao iniciar
        torch.cuda.empty_cache()
        # Configuração de alocação de memória (experimental, use com cautela)
        # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        # print("Configurado PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128")
    else:
        print("GPU detectada não é L4. Otimizações padrão aplicadas.")

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Google Drive Integration ---
def setup_colab_integration(base_folder='GNN_Road_Networks'):
    """Configura integração com Google Drive para persistência no Colab"""
    try:
        from google.colab import drive
        drive_mounted = False
        drive_path = '/content/drive'

        # Montar Google Drive se não estiver montado
        if not os.path.exists(os.path.join(drive_path, 'MyDrive')):
            print("Montando Google Drive...")
            drive.mount(drive_path)
            drive_mounted = True
            print("Google Drive montado.")
        else:
            print("Google Drive já está montado.")
            drive_mounted = True # Assume montado se o caminho existe

        # Criar diretórios para salvar resultados e modelos
        save_path_base = os.path.join(drive_path, 'MyDrive', base_folder)
        models_path = os.path.join(save_path_base, 'models')
        results_path = os.path.join(save_path_base, 'results')

        os.makedirs(models_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)

        print(f"Caminho base para salvar no Drive: {save_path_base}")
        return save_path_base, models_path, results_path, drive_mounted

    except ImportError:
        print("Não estamos em ambiente Colab ou erro ao importar 'google.colab'. Usando diretório local.")
        save_path_base = '.' # Salvar no diretório atual
        models_path = os.path.join(save_path_base, 'models')
        results_path = os.path.join(save_path_base, 'results')
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        print(f"Resultados serão salvos em: {os.path.abspath(results_path)}")
        return save_path_base, models_path, results_path, False
    except Exception as e:
        print(f"Erro ao configurar integração com Google Drive: {e}")
        print("Usando diretório local como fallback.")
        save_path_base = '.'
        models_path = os.path.join(save_path_base, 'models')
        results_path = os.path.join(save_path_base, 'results')
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(results_path, exist_ok=True)
        print(f"Resultados serão salvos em: {os.path.abspath(results_path)}")
        return save_path_base, models_path, results_path, False


# --- Data Loading and Preprocessing ---

def load_data(data_path):
    """
    Loads graph data from a PyTorch Geometric file.
    (Same as previous version, kept for consistency)
    """
    try:
        # Load data onto CPU first to avoid potential GPU OOM during loading
        data = torch.load(data_path, map_location='cpu')
        print(f"Data loaded successfully from {data_path} onto CPU.")
        print(f"Number of nodes: {data.num_nodes}")
        if hasattr(data, 'edge_index'):
            print(f"Number of edges: {data.edge_index.shape[1]}")
        else:
            print("Warning: Edge index (data.edge_index) not found.")
            # return None # Decide if edges are essential

        # Check features and labels
        if hasattr(data, 'x'):
            print(f"Number of node features: {data.x.shape[1]}")
            if data.x.shape[0] != data.num_nodes:
                 print(f"Warning: Mismatch between data.x rows ({data.x.shape[0]}) and data.num_nodes ({data.num_nodes}).")
        else:
            print("Warning: Node features (data.x) not found.")
            return None # Cannot proceed without features

        if hasattr(data, 'y'):
            # Ensure labels are integer type
            if not torch.is_tensor(data.y):
                 data.y = torch.tensor(data.y) # Convert if not tensor
            if data.y.dtype != torch.long:
                 print(f"Warning: Converting data.y from {data.y.dtype} to torch.long.")
                 data.y = data.y.long()

            print(f"Number of classes: {len(torch.unique(data.y))}")
            if data.y.shape[0] != data.num_nodes:
                 print(f"Warning: Mismatch between data.y length ({data.y.shape[0]}) and data.num_nodes ({data.num_nodes}).")
        else:
            print("Warning: Node labels (data.y) not found.")
            return None # Cannot proceed without labels

        return data
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        return None

def create_stratified_masks(data, train_ratio=0.7, val_ratio=0.15, random_state=RANDOM_SEED):
    """
    Creates stratified train/validation/test masks preserving class proportions.
    (Same as previous version, kept for consistency)
    """
    if not hasattr(data, 'y'):
        print("Error: Cannot create stratified masks without 'data.y'.")
        return data

    num_nodes = data.num_nodes
    y = data.y.cpu().numpy()
    indices = np.arange(num_nodes)

    try:
        # Split into train and temp (val + test)
        train_idx, temp_idx, y_train, y_temp = train_test_split(
            indices, y, test_size=(1.0 - train_ratio), stratify=y, random_state=random_state
        )

        # Calculate test ratio relative to the temp set size
        # Ensure temp_idx is not empty before calculating test_ratio_adj
        if len(temp_idx) == 0:
             print("Warning: No samples left for validation/test split after initial train split.")
             val_idx, test_idx = np.array([], dtype=int), np.array([], dtype=int)
        else:
            test_ratio_adj = (1.0 - train_ratio - val_ratio) / (1.0 - train_ratio)
            if test_ratio_adj < 0 or test_ratio_adj > 1:
                 print(f"Warning: Calculated adjusted test ratio ({test_ratio_adj:.2f}) is invalid. Adjusting ratios.")
                 # Example adjustment: split remaining equally between val and test
                 test_ratio_adj = 0.5

            # Split temp into val and test
            val_idx, test_idx, _, _ = train_test_split(
                temp_idx, y_temp, test_size=test_ratio_adj, stratify=y_temp, random_state=random_state
            )

        # Create boolean masks
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True

        print("Stratified masks created:")
        print(f"  Train nodes: {data.train_mask.sum().item()}")
        print(f"  Validation nodes: {data.val_mask.sum().item()}")
        print(f"  Test nodes: {data.test_mask.sum().item()}")

    except ValueError as e:
        print(f"Error during stratified split (often due to few samples per class): {e}. Falling back to random split.")
        indices_perm = torch.randperm(num_nodes)
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[indices_perm[:train_size]] = True
        data.val_mask[indices_perm[train_size:train_size+val_size]] = True
        data.test_mask[indices_perm[train_size+val_size:]] = True
        print("Fallback random masks created.")

    return data

def preprocess_fragmented_graph(data, strategy='largest_component'):
    """
    Preprocesses a potentially fragmented graph, focusing on memory efficiency.

    Args:
        data: PyTorch Geometric Data object (on CPU).
        strategy: 'largest_component' or 'keep_all'.

    Returns:
        Processed Data object (on CPU) and indices of nodes kept.
    """
    if strategy == 'keep_all':
        print("Keeping all graph components.")
        return data, torch.arange(data.num_nodes)

    if strategy == 'largest_component':
        print("Analyzing graph connectivity (memory-efficient approach)...")
        original_num_nodes = data.num_nodes
        original_num_edges = data.num_edges if hasattr(data, 'num_edges') else data.edge_index.shape[1]

        # Use NetworkX on edge_index directly if possible, avoid full attribute transfer if large
        try:
            print("Converting edge index to NetworkX graph...")
            if torch.cuda.is_available(): torch.cuda.empty_cache() # Clear cache before potentially large operation
            # Convert only connectivity for component analysis
            g_nx = nx.Graph()
            g_nx.add_nodes_from(range(original_num_nodes))
            # Ensure edge_index is on CPU and is valid
            if not hasattr(data, 'edge_index'):
                 print("Error: data object has no 'edge_index'. Cannot analyze connectivity.")
                 return data, torch.arange(original_num_nodes)
            if data.edge_index.shape[0] != 2:
                 print(f"Error: data.edge_index has incorrect shape {data.edge_index.shape}. Expected shape (2, num_edges).")
                 return data, torch.arange(original_num_nodes)

            edges = data.edge_index.t().cpu().numpy()
            g_nx.add_edges_from(edges)
            del edges # Free memory
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            print("Finding connected components...")
            components = list(nx.connected_components(g_nx))
            num_components = len(components)
            print(f"Graph has {num_components} connected components.")

            if num_components <= 1:
                print("Graph is already connected or empty. No changes made.")
                del g_nx # Free memory
                return data, torch.arange(original_num_nodes)

            # Find the largest component
            largest_component_nodes = max(components, key=len)
            # Ensure indices are long type
            largest_component_indices = torch.tensor(sorted(list(largest_component_nodes)), dtype=torch.long)
            print(f"Largest component has {len(largest_component_indices)} nodes.")
            del g_nx, components, largest_component_nodes # Free memory

            # --- Memory-Efficient Subgraphing using PyG utils ---
            print("Creating subgraph using torch_geometric.utils.subgraph...")
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            # Ensure edge attributes are handled if they exist
            edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None

            # Perform subgraph operation
            edge_index_sub, edge_attr_sub = subgraph(
                largest_component_indices,
                data.edge_index,
                edge_attr=edge_attr,
                relabel_nodes=True, # Essential for the new node indexing
                num_nodes=original_num_nodes
            )
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            # Create the new Data object efficiently
            data_subgraph = Data(num_nodes=len(largest_component_indices))
            data_subgraph.edge_index = edge_index_sub
            if edge_attr_sub is not None:
                data_subgraph.edge_attr = edge_attr_sub

            # Transfer node features and labels
            if hasattr(data, 'x'):
                data_subgraph.x = data.x[largest_component_indices]
            if hasattr(data, 'y'):
                data_subgraph.y = data.y[largest_component_indices]

            # Transfer masks - Corrected Logic
            node_map = {orig_idx.item(): new_idx for new_idx, orig_idx in enumerate(largest_component_indices)}
            for mask_name in ['train_mask', 'val_mask', 'test_mask']:
                if hasattr(data, mask_name):
                    original_mask = getattr(data, mask_name)
                    # Get original indices where the mask is True AND are in the largest component
                    original_true_indices_in_component = largest_component_indices[original_mask[largest_component_indices]]
                    # Map these original indices to the new subgraph indices
                    subgraph_mask_indices = [node_map[idx.item()] for idx in original_true_indices_in_component if idx.item() in node_map]

                    new_mask = torch.zeros(data_subgraph.num_nodes, dtype=torch.bool)
                    if subgraph_mask_indices:
                        new_mask[subgraph_mask_indices] = True
                    setattr(data_subgraph, mask_name, new_mask)


            # Copy other relevant attributes (be selective to save memory)
            if hasattr(data, 'pos'):
                 if isinstance(data.pos, torch.Tensor) and data.pos.size(0) == original_num_nodes:
                     data_subgraph.pos = data.pos[largest_component_indices]

            print(f"Created subgraph with {data_subgraph.num_nodes} nodes and {data_subgraph.num_edges} edges.")
            # Verify mask sums
            if hasattr(data_subgraph, 'train_mask'): print(f"  Train nodes in subgraph: {data_subgraph.train_mask.sum().item()}")
            if hasattr(data_subgraph, 'val_mask'): print(f"  Validation nodes in subgraph: {data_subgraph.val_mask.sum().item()}")
            if hasattr(data_subgraph, 'test_mask'): print(f"  Test nodes in subgraph: {data_subgraph.test_mask.sum().item()}")

            # Return the subgraph and the original indices of the nodes kept
            return data_subgraph, largest_component_indices

        except Exception as e:
            print(f"Error processing fragmented graph: {e}")
            print("Returning original data.")
            # Ensure to return indices matching the original data if returning original
            return data, torch.arange(original_num_nodes)
    else:
        print(f"Unknown strategy: {strategy}. Returning original data.")
        return data, torch.arange(data.num_nodes)


# --- Model Definition (ImprovedGNN) ---

class ImprovedGNN(torch.nn.Module):
    """Arquitetura avançada otimizada para GPU L4 com multi-resolução e alta paralelização"""
    def __init__(self, input_dim, hidden_dim=384, output_dim=6, num_layers=4, heads=6, dropout=0.3):
        """
        Initializes the ImprovedGNN model.

        Args:
            input_dim: Dimensionality of input node features.
            hidden_dim: Dimensionality of hidden layers (aumentado para L4).
            output_dim: Number of output classes.
            num_layers: Total number of GNN layers (input + GCNs + GAT). Should be >= 1.
            heads: Number of attention heads for the GAT layer.
            dropout: Dropout probability.
        """
        super(ImprovedGNN, self).__init__()
        if num_layers < 1: # Need at least input layer
            raise ValueError("ImprovedGNN requires at least 1 layer.")

        self.hidden_dim = hidden_dim
        self.dropout = dropout
        # Number of message passing layers (GCNs + GAT)
        self.num_gnn_layers = num_layers
        
        # Ativação moderna (SiLU = Swish) aproveitando melhor a GPU
        self.activation = torch.nn.SiLU()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        
        # --- Camada de Normalização de Entrada ---
        self.input_norm = torch.nn.LayerNorm(input_dim)
        
        # --- Camadas de Mensagem ---
        self.layers = torch.nn.ModuleList()
        self.layer_norms = torch.nn.ModuleList()
        
        # --- Multi-Resolução: Caminhos Paralelos ---
        # A. Caminho Rápido/Global
        self.global_conv = GCNConv(input_dim, hidden_dim//2) 
        self.global_norm = torch.nn.LayerNorm(hidden_dim//2)
        
        # B. Caminho Detalhado/Local (sequencial)
        
        # 1. Input Layer (projeta para hidden_dim)
        self.layers.append(GCNConv(input_dim, hidden_dim))
        self.layer_norms.append(torch.nn.LayerNorm(hidden_dim))
        
        # 2. Camadas GCN Intermediárias (melhoram os detalhes locais)
        # num_layers - 2 porque uma é entrada, uma é a última camada GAT
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
            self.layer_norms.append(torch.nn.LayerNorm(hidden_dim))
            
        # 3. GAT como última camada de mensagem (melhor para capturar padrões complexos)
        if num_layers >= 2:
            # Aumentamos o número de cabeças para mais paralelismo na L4
            # Cada cabeça fica com hidden_dim//heads dimensões
            self.layers.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads,
                                      dropout=dropout, concat=True))
            self.layer_norms.append(torch.nn.LayerNorm(hidden_dim))
            
        # --- Camada de Fusão Multi-Resolução ---
        # Concatena e projeta os caminhos global e local
        self.fusion_layer = torch.nn.Linear(hidden_dim + hidden_dim//2, hidden_dim)
        self.fusion_norm = torch.nn.LayerNorm(hidden_dim)
        
        # --- Camada de Saída com MLP ---
        # Um pequeno MLP ao invés de projeção linear direta
        self.output_mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.LayerNorm(hidden_dim//2),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim//2, output_dim)
        )
        
        # --- Inicialização de Pesos ---
        self.reset_parameters()
        
    def reset_parameters(self):
        """Inicialização melhorada de pesos para evitar problemas de convergência"""
        # Xavier/Glorot para camadas lineares
        if hasattr(self, 'fusion_layer'):
            torch.nn.init.xavier_uniform_(self.fusion_layer.weight)
            if self.fusion_layer.bias is not None:
                torch.nn.init.zeros_(self.fusion_layer.bias)
                
        # Para camadas do output_mlp
        if hasattr(self, 'output_mlp'):
            for module in self.output_mlp:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)

    def forward(self, x, edge_index):
        """
        Standard forward pass for full graph inference.
        """
        # Normalização das features de entrada
        x = self.input_norm(x)
        
        # --- CAMINHO GLOBAL: Captura contexto amplo em um passo ---
        x_global = self.global_conv(x, edge_index)
        x_global = self.global_norm(x_global)
        x_global = self.activation(x_global)
        x_global = F.dropout(x_global, p=self.dropout, training=self.training)
        
        # --- CAMINHO LOCAL: Captura padrões detalhados com múltiplas camadas ---
        # Processamento através das camadas GNN sequenciais
        x_local = x
        for i in range(self.num_gnn_layers):
            x_res = x_local  # Armazena input para conexão residual
            x_local = self.layers[i](x_local, edge_index)
            x_local = self.layer_norms[i](x_local)
            x_local = self.activation(x_local)
            x_local = F.dropout(x_local, p=self.dropout, training=self.training)
            
            # Adiciona conexão residual (pula a primeira camada)
            if i > 0:
                if x_local.shape == x_res.shape:
                    x_local = x_local + x_res
                else:
                    print(f"Warning: Skipping residual connection at layer {i} due to shape mismatch")
        
        # --- FUSÃO MULTI-RESOLUÇÃO ---
        # Concatenar features globais e locais
        x_combined = torch.cat([x_local, x_global], dim=1)
        
        # Camada de fusão para integrar informações
        x_fused = self.fusion_layer(x_combined)
        x_fused = self.fusion_norm(x_fused)
        x_fused = self.activation(x_fused)
        x_fused = F.dropout(x_fused, p=self.dropout, training=self.training)
        
        # Camada de saída final (MLP)
        out = self.output_mlp(x_fused)
        
        return out

    def forward_sampled(self, x_input, adjs, batch_size_actual):
        """
        Corrigido para resolver o problema de size mismatch nas conexões residuais.
        Agora rastreia corretamente o tensor residual para cada camada.
        
        Args:
            x_input: Input features para os nós incluídos no grafo de computação amostrado.
            adjs: Lista de tuplas (edge_index, e_id, size) do NeighborSampler.
            batch_size_actual: Número real de nós alvo no batch atual.
            
        Returns:
            Tensor de saída para os nós do batch (shape: [batch_size_actual, output_dim]).
        """
        # Aplicar normalização de entrada
        x_input = self.input_norm(x_input)
        
        # --- CAMINHO GLOBAL ---
        # O primeiro edge_index e tamanho define o escopo do primeiro salto
        edge_index_first, _, size_first = adjs[0]
        
        # Computar features globais
        x_global = self.global_conv(x_input, edge_index_first)
        x_global = self.global_norm(x_global)
        x_global = self.activation(x_global)
        x_global = F.dropout(x_global, p=self.dropout, training=self.training)
        
        # Obter apenas as features globais para os nós alvo do primeiro salto
        x_global_target = x_global[:size_first[1]]
        
        # --- CAMINHO LOCAL ---
        x = x_input # Features iniciais
        
        # Verificar número de camadas vs. adjs
        if len(adjs) != self.num_gnn_layers:
            print(f"AVISO: Comprimento de adjs ({len(adjs)}) != número de camadas GNN ({self.num_gnn_layers}).")
            num_iterations = min(len(adjs), self.num_gnn_layers)
        else:
            num_iterations = self.num_gnn_layers
            
        # Para cada camada, precisamos manter o input correto para a conexão residual
        for i in range(num_iterations):
            edge_index, _, size = adjs[i]
            
            # Obtém os nós alvo para esta camada (nodes que receberão mensagens)
            target_nodes = x[:size[1]]
            
            # Validar edge_index antes de prosseguir
            if not isinstance(edge_index, torch.Tensor):
                print(f"Erro: edge_index não é um tensor (tipo: {type(edge_index)}) na camada {i}")
                raise TypeError("edge_index deve ser um Tensor")
                
            if edge_index.dtype != torch.long:
                edge_index = edge_index.long()
                
            if edge_index.dim() != 2 or edge_index.shape[0] != 2:
                print(f"Erro: shape de edge_index é {edge_index.shape}, esperado [2, num_edges] na camada {i}")
                raise ValueError("edge_index deve ter shape [2, num_edges]")
            
            # Salvar o tensor alvo para conexão residual
            x_target = target_nodes.clone()
            
            try:
                # Aplicar a camada i (GCN ou GAT)
                x_out = self.layers[i](x, edge_index)
                
                # Verificação de segurança
                if x_out.shape[0] != size[1]:
                    print(f"Erro crítico: Output da camada {i} tem shape[0]={x_out.shape[0]} mas deveria ser {size[1]}")
                
                x_out = self.layer_norms[i](x_out)
                x_out = self.activation(x_out)
                x_out = F.dropout(x_out, p=self.dropout, training=self.training)
                
                # Conexão residual (pular primeira camada)
                if i > 0:
                    if x_out.shape == x_target.shape:
                        x_out = x_out + x_target
                    else:
                        print(f"AVISO: Ignorando conexão residual na camada {i}:")
                        print(f"  x_out: {x_out.shape}, x_target: {x_target.shape}")
                
                # Atualizar x para a próxima iteração
                x = x_out
                
            except Exception as e:
                print(f"Erro na camada {i}: {str(e)}")
                print(f"Shape de x: {x.shape}, edge_index: {edge_index.shape}, size: {size}")
                raise e
        
        # O x final contém as features locais dos nós alvo do último salto
        # x_global_target contém as features globais
        
        # Verificar se o tamanho final é o esperado
        if x.shape[0] != x_global_target.shape[0]:
            print(f"AVISO: Tamanho final de features locais ({x.shape[0]}) não corresponde às globais ({x_global_target.shape[0]})")
            # Corrigir tamanho se necessário - pegar apenas os primeiros nós em comum
            min_size = min(x.shape[0], x_global_target.shape[0])
            x = x[:min_size]
            x_global_target = x_global_target[:min_size]
        
        # --- FUSÃO MULTI-RESOLUÇÃO ---
        x_combined = torch.cat([x, x_global_target], dim=1)
        x_fused = self.fusion_layer(x_combined)
        x_fused = self.fusion_norm(x_fused)
        x_fused = self.activation(x_fused)
        x_fused = F.dropout(x_fused, p=self.dropout, training=self.training)
        
        # Camada de saída final (MLP)
        out = self.output_mlp(x_fused)
        
        # Verificar tamanho do batch
        if batch_size_actual > out.shape[0]:
            print(f"AVISO: batch_size_actual ({batch_size_actual}) > tamanho de saída ({out.shape[0]})")
            batch_size_actual = out.shape[0]
            
        # Retornar apenas as predições para os nós do batch atual
        return out[:batch_size_actual]


# --- Loss Function ---

def weighted_cross_entropy_loss(predictions, targets, class_weights=None, num_classes=None, label_smoothing=0.1):
    """
    Weighted cross-entropy loss with optional label smoothing.
    (Minor modification to handle potential empty targets)
    """
    if targets.numel() == 0: # Handle empty target tensor
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)

    if class_weights is None:
        if num_classes is None:
             # Infer num_classes from predictions if not provided
             num_classes = predictions.size(1)
             if num_classes == 0: # Handle edge case where predictions might be empty
                  print("Warning: Cannot determine num_classes from predictions.")
                  return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        # Ensure targets are within the valid range [0, num_classes-1] before bincount
        if targets.max() >= num_classes or targets.min() < 0:
             print(f"Error: Target labels {targets.unique().tolist()} outside range [0, {num_classes-1}]. Cannot calculate weights.")
             # Return unweighted loss or zero loss? Let's return zero loss for safety.
             return torch.tensor(0.0, device=predictions.device, requires_grad=True)

        class_counts = torch.bincount(targets, minlength=num_classes)
        total_counts = class_counts.sum().float()
        # Add epsilon to prevent division by zero for classes not present
        class_weights = total_counts / (class_counts.float() + 1e-8)
        # Apply sqrt scaling to weights to moderate extreme values
        class_weights = torch.sqrt(class_weights)
        # Normalize weights
        class_weights = class_weights / (class_weights.sum() + 1e-8) * num_classes
        class_weights = class_weights.to(predictions.device)

    # Label smoothing
    if label_smoothing > 0.0 and num_classes is not None:
        # Ensure targets are still valid after potential filtering/sampling
        if targets.max() >= num_classes or targets.min() < 0:
             print(f"Error: Target labels {targets.unique().tolist()} outside range [0, {num_classes-1}] after sampling. Cannot apply label smoothing.")
             # Fallback to standard cross-entropy without smoothing
             return F.cross_entropy(predictions, targets, weight=class_weights)

        try:
            targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
            targets_smooth = (1.0 - label_smoothing) * targets_one_hot + label_smoothing / num_classes
            log_probs = F.log_softmax(predictions, dim=1)
            # Calculate loss per sample
            loss_per_sample = -(targets_smooth * log_probs).sum(dim=1)
            # Apply class weights per sample
            weights_per_sample = class_weights[targets]
            loss = (loss_per_sample * weights_per_sample).mean()
        except IndexError as e:
            print(f"IndexError during label smoothing or weighting: {e}")
            print(f"  Predictions shape: {predictions.shape}")
            print(f"  Targets shape: {targets.shape}, Max target: {targets.max()}, Min target: {targets.min()}")
            print(f"  Class weights shape: {class_weights.shape}")
            # Fallback to standard cross-entropy
            loss = F.cross_entropy(predictions, targets, weight=class_weights)
        except Exception as e:
            print(f"Unexpected error during label smoothing loss: {e}")
            # Fallback to standard cross-entropy
            loss = F.cross_entropy(predictions, targets, weight=class_weights)

    else:
        # Standard weighted cross-entropy
        loss = F.cross_entropy(predictions, targets, weight=class_weights)

    return loss


# --- Training Function (Neighbor Sampling) ---

def train_with_sampling(model, data, num_epochs=200, batch_size=2048, num_neighbors=[25, 10], 
                         lr=0.001, weight_decay=1e-4, use_amp=True,
                         patience=30, min_delta=0.001, label_smoothing=0.1,
                         checkpoint_dir='models', run_id='latest', 
                         gradient_accumulation_steps=1, lr_warmup_epochs=5,
                         batch_size_validation=8192):
    """
    Treinamento otimizado para GPU L4 utilizando amostragem de vizinhança.

    Args:
        model: O modelo GNN a ser treinado.
        data: Objeto PyTorch Geometric Data (na CPU).
        num_epochs: Número máximo de épocas.
        batch_size: Tamanho do batch para treinamento (aumentado para L4).
        num_neighbors: Lista de vizinhos a amostrar por camada.
        lr: Taxa de aprendizado inicial.
        weight_decay: Regularização L2.
        use_amp: Usar Automatic Mixed Precision.
        patience: Épocas a esperar antes de parar (early stopping).
        min_delta: Melhoria mínima para considerar progresso.
        label_smoothing: Fator de suavização de rótulos.
        checkpoint_dir: Diretório para salvar checkpoints.
        run_id: Identificador para esta execução.
        gradient_accumulation_steps: Número de batches a acumular antes de atualizar pesos.
        lr_warmup_epochs: Número de épocas de aquecimento da taxa de aprendizado.
        batch_size_validation: Tamanho do batch para validação (pode ser maior).

    Returns:
        Modelo treinado e dicionário com histórico de métricas.
    """
    print("\n--- Iniciando Treinamento Otimizado para L4 com Amostragem de Vizinhança ---")

    # Mover modelo para o dispositivo correto
    model.to(device)
    
    # Otimizador - AdamW com gradiente acumulado
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    # Configurações avançadas de scheduler
    # 1. Warm-up LR para estabilidade inicial
    # 2. Redução ajustiva baseada no F1 para épocas posteriores
    total_steps = (len(data.train_mask.nonzero()) // batch_size + 1) * num_epochs // gradient_accumulation_steps
    warmup_steps = (len(data.train_mask.nonzero()) // batch_size + 1) * lr_warmup_epochs // gradient_accumulation_steps
    
    # Definição de funções de scheduler personalizado
    def lr_lambda(current_step):
        # Fase de warmup: LR cresce linearmente até atingir o valor máximo
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Após warmup, mantem o LR até que ReduceLROnPlateau atue 
        return 1.0
    
    # Combinação de schedulers
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, verbose=True, 
        threshold=0.0001, threshold_mode='abs'
    )

    # Scaler para Mixed Precision
    scaler = GradScaler(enabled=(device.type == 'cuda' and use_amp))
    if device.type == 'cuda' and use_amp:
        print("Usando Automatic Mixed Precision (AMP)")

    # Configuração do DataLoader com NeighborSampler para treinamento
    if hasattr(data, 'train_mask') and data.train_mask.sum() > 0:
        train_idx = data.train_mask.nonzero(as_tuple=False).view(-1)
        
        # Ajustar num_workers para ambiente (recomendado: 0-4 para Colab)
        num_workers = min(4, max(1, os.cpu_count() // 2)) if os.cpu_count() else 0
        print(f"Configurando NeighborSampler com batch_size={batch_size}, num_workers={num_workers}")
        
        try:
            # Ajustar num_neighbors para corresponder ao número de camadas do modelo
            if len(num_neighbors) != model.num_gnn_layers:
                print(f"Ajustando comprimento de num_neighbors ({len(num_neighbors)}) para corresponder a model.num_gnn_layers ({model.num_gnn_layers}).")
                if len(num_neighbors) > model.num_gnn_layers:
                    num_neighbors = num_neighbors[:model.num_gnn_layers]
                else:
                    num_neighbors += [num_neighbors[-1]] * (model.num_gnn_layers - len(num_neighbors))
                print(f"  num_neighbors ajustado para: {num_neighbors}")

            # DataLoader com persistência para workers
            train_loader = NeighborSampler(
                data.edge_index, 
                node_idx=train_idx,
                sizes=num_neighbors,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                persistent_workers=(num_workers > 0)
            )
            
            # Validação com batch maior para aproveitar a memória maior da L4
            if hasattr(data, 'val_mask') and data.val_mask.sum() > 0:
                val_idx = data.val_mask.nonzero(as_tuple=False).view(-1)
                val_loader = NeighborSampler(
                    data.edge_index,
                    node_idx=val_idx,
                    sizes=num_neighbors,
                    batch_size=batch_size_validation,  # Batch maior para validação (eficiência)
                    shuffle=False,  # Não precisamos embaralhar para validação
                    num_workers=num_workers,
                    persistent_workers=(num_workers > 0)
                )
            else:
                val_loader = None
                print("Validação em batch não disponível (sem máscaras de validação).")
                
        except Exception as e:
            print(f"Erro ao criar NeighborSampler: {e}")
            print("Usando fallback com num_workers=0")
            train_loader = NeighborSampler(
                data.edge_index, 
                node_idx=train_idx, 
                sizes=num_neighbors,
                batch_size=batch_size, 
                shuffle=True, 
                num_workers=0
            )
            val_loader = None
    else:
        print("Erro: Nenhuma máscara de treinamento encontrada. Impossível configurar o amostrador.")
        return model, {}

    # Cálculo de pesos de classe para balanceamento
    num_classes = int(data.y.max().item() + 1) if data.y.numel() > 0 else 0
    class_weights = None
    if num_classes > 0 and hasattr(data, 'train_mask') and data.train_mask.sum() > 0:
        print("Calculando pesos de classe para função de perda...")
        class_counts = torch.bincount(data.y[data.train_mask], minlength=num_classes)
        total_counts = class_counts.sum().float()
        
        # Aplicar transformação mais suave com raiz quadrada
        class_weights = torch.sqrt(total_counts / (class_counts.float() + 1e-8))
        
        # Normalizar pesos
        class_weights = class_weights / (class_weights.sum() + 1e-8) * num_classes
        class_weights = class_weights.to(device)
        print(f"Pesos de classe calculados: {class_weights.cpu().numpy()}")
    else:
        print("Aviso: Não foi possível calcular pesos de classe.")

    # Early Stopping
    early_stopping = {
        'best_val_f1': 0.0,
        'best_model_state': None,
        'patience_counter': 0,
    }

    # Histórico de métricas
    history = {
        'train_loss': [], 'val_loss': [], 'val_acc': [], 
        'val_f1': [], 'val_f1_per_class': [], 'lr': [],
        'train_steps': 0  # Contador global de passos
    }

    # --- Loop de Treinamento ---
    global_step = 0  # Contador de passos para scheduler de warmup
    start_time = time.time()
    
    # Monitoramento de recursos
    process = psutil.Process()
    
    # Limpeza inicial de memória GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_gpu_memory = torch.cuda.memory_allocated(device) / (1024**2)
    else:
        initial_gpu_memory = 0
        
    initial_system_memory = process.memory_info().rss / (1024**2)
    print(f"Memória inicial - Sistema: {initial_system_memory:.1f}MB, GPU: {initial_gpu_memory:.1f}MB")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # --- Fase de Treinamento ---
        model.train()
        total_loss = 0.0
        total_nodes = 0
        num_batches = 0
        
        # Definir acumulação de gradiente
        optimizer.zero_grad(set_to_none=True)  # Inicialização mais eficiente
        
        # Loop pelos batches de treinamento
        for batch_idx, (batch_size_actual, n_id, adjs) in enumerate(train_loader):
            # Mover adjacências para o dispositivo
            adjs = [adj.to(device) for adj in adjs]
            
            # Gradient Accumulation
            is_accumulation_step = ((batch_idx + 1) % gradient_accumulation_steps != 0)
            
            # Forward pass com Autocast para precisão mista
            with autocast(enabled=(device.type == 'cuda' and use_amp)):
                try:
                    # Obter features para os nós necessários
                    x = data.x[n_id].to(device)
                    
                    # Obter rótulos APENAS para os nós alvo (primeiros batch_size_actual nós)
                    y = data.y[n_id[:batch_size_actual]].to(device)
                    
                    # Forward pass específico para dados amostrados
                    out = model.forward_sampled(x, adjs, batch_size_actual)
                    
                    # Calcular perda com os pesos de classe pré-calculados
                    loss = weighted_cross_entropy_loss(
                        out, y,
                        class_weights=class_weights,
                        num_classes=num_classes,
                        label_smoothing=label_smoothing
                    )
                    
                    # Escalar a perda para gradient accumulation
                    if gradient_accumulation_steps > 1:
                        loss = loss / gradient_accumulation_steps
                        
                except Exception as e:
                    print(f"Erro durante forward/loss no batch {batch_idx}: {e}")
                    continue
            
            # Backward pass com escala para precisão mista
            scaler.scale(loss).backward()
            
            # Só atualiza os pesos após acumular gradientes de múltiplos batches
            if not is_accumulation_step or batch_idx == len(train_loader) - 1:
                # Unscale para aplicar clipping corretamente
                scaler.unscale_(optimizer)
                
                # Gradient clipping para estabilidade
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Aplicar gradientes e atualizar modelo
                scaler.step(optimizer)
                scaler.update()
                
                # Atualizar scheduler de warmup baseado em steps
                if global_step < warmup_steps:
                    warmup_scheduler.step()
                
                # Zerar gradientes
                optimizer.zero_grad(set_to_none=True)
                
                # Incrementar contador global
                global_step += 1
                history['train_steps'] = global_step
            
            # Acumular estatísticas
            total_loss += loss.item() * (batch_size_actual * (gradient_accumulation_steps if is_accumulation_step else 1))
            total_nodes += batch_size_actual
            num_batches += 1
            
            # Limpeza de memória após cada batch
            del x, y, adjs, out, loss
            
            # Limpeza periódica de cache CUDA
            if (batch_idx + 1) % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calcular perda média por época
        avg_train_loss = total_loss / total_nodes if total_nodes > 0 else 0
        history['train_loss'].append(avg_train_loss)

        # --- Fase de Validação ---
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_f1 = 0.0
        val_f1_per_class = [0.0] * num_classes
        val_true_all = []
        val_pred_all = []
        val_total_nodes = 0
        
        # Validação com amostragem ou full-graph
        if val_loader is not None:
            # Validação em batches (para grafos grandes)
            try:
                with torch.no_grad():
                    for batch_size_val, n_id_val, adjs_val in val_loader:
                        # Mover adjacências para o dispositivo
                        adjs_val = [adj.to(device) for adj in adjs_val]
                        
                        # Usar autocast para validação também
                        with autocast(enabled=(device.type == 'cuda' and use_amp)):
                            # Obter features e labels
                            x_val = data.x[n_id_val].to(device)
                            y_val = data.y[n_id_val[:batch_size_val]].to(device)
                            
                            # Forward pass com amostragem
                            out_val = model.forward_sampled(x_val, adjs_val, batch_size_val)
                            
                            # Calcular métricas
                            val_loss_batch = F.cross_entropy(out_val, y_val).item()
                            _, pred_val = out_val.max(dim=1)
                            
                            # Acumular stats para cálculo posterior
                            val_true_all.append(y_val.cpu().numpy())
                            val_pred_all.append(pred_val.cpu().numpy())
                            
                            # Acumular perda
                            val_loss += val_loss_batch * batch_size_val
                            val_total_nodes += batch_size_val
                            
                            # Limpar memória
                            del x_val, y_val, adjs_val, out_val, pred_val
                            
                    # Calcular métricas finais
                    if val_total_nodes > 0:
                        val_loss /= val_total_nodes
                        
                        # Calcular métricas finais combinando resultados de todos os batches
                        y_true_val = np.concatenate(val_true_all) if val_true_all else np.array([])
                        y_pred_val = np.concatenate(val_pred_all) if val_pred_all else np.array([])
                        
                        if len(y_true_val) > 0:
                            val_acc = accuracy_score(y_true_val, y_pred_val)
                            val_f1 = f1_score(y_true_val, y_pred_val, average='macro', zero_division=0)
                            
                            # Calcular F1 por classe
                            val_f1_per_class = f1_score(
                                y_true_val, y_pred_val, average=None, 
                                labels=range(num_classes), zero_division=0
                            ).tolist()
            
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"Erro de memória durante validação em batch. Usando fallback.")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    val_loader = None  # Forçar fallback para validação full-graph
                else:
                    print(f"Erro durante validação: {e}")
        
        # Fallback: Validação full-graph se não houver loader ou se ele falhar
        if val_loader is None and hasattr(data, 'val_mask') and data.val_mask.sum() > 0:
            try:
                with torch.no_grad():
                    # Fallback para validação em grafo completo
                    print("Realizando validação em grafo completo...")
                    
                    # Mover dados para GPU
                    data.to(device)
                    
                    with autocast(enabled=(device.type == 'cuda' and use_amp)):
                        # Forward pass completo
                        out_val_full = model(data.x, data.edge_index)
                        
                        # Aplicar máscara de validação
                        val_mask = data.val_mask.to(device)
                        y_val_full = data.y[val_mask].to(device)
                        out_val_masked = out_val_full[val_mask]
                        
                        # Calcular perda
                        val_loss = F.cross_entropy(out_val_masked, y_val_full).item()
                        
                        # Obter predições
                        _, pred_val_full = out_val_masked.max(dim=1)
                        
                        # Calcular métricas
                        y_true_val = y_val_full.cpu().numpy()
                        y_pred_val = pred_val_full.cpu().numpy()
                        
                        val_acc = accuracy_score(y_true_val, y_pred_val)
                        val_f1 = f1_score(y_true_val, y_pred_val, average='macro', zero_division=0)
                        
                        # F1 por classe
                        val_f1_per_class = f1_score(
                            y_true_val, y_pred_val, average=None, 
                            labels=range(num_classes), zero_division=0
                        ).tolist()
                    
                    # Mover dados de volta para CPU
                    data.to('cpu')
                    
            except RuntimeError as e:
                print(f"Erro durante validação full-graph: {e}")
                # Assumir valores default em caso de falha
                val_loss, val_acc, val_f1 = 0.0, 0.0, 0.0
                val_f1_per_class = [0.0] * num_classes
            finally:
                # Garantir limpeza de memória
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Adicionar métricas ao histórico
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_f1_per_class'].append(val_f1_per_class)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Atualizar scheduler baseado em validação
        main_scheduler.step(val_f1)
        
        # Logging
        epoch_time = time.time() - epoch_start_time
        
        # Logging a cada 5 épocas ou na última
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            # Monitoramento de memória
            mem_info = psutil.virtual_memory()
            gpu_mem_alloc = torch.cuda.memory_allocated(device) / (1024**2) if device.type == 'cuda' else 0
            gpu_mem_res = torch.cuda.memory_reserved(device) / (1024**2) if device.type == 'cuda' else 0
            
            print(f"Época {epoch+1}/{num_epochs} | Tempo: {epoch_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Perda Treino: {avg_train_loss:.4f}")
            
            # Imprimir métricas de validação, se disponíveis
            if val_f1 > 0:
                print(f"  Val Perda: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
                
                # Top-3 classes com melhor/pior F1
                if len(val_f1_per_class) > 1:
                    f1_indices = np.argsort(val_f1_per_class)
                    worst_classes = f1_indices[:min(3, len(f1_indices))]
                    best_classes = f1_indices[-min(3, len(f1_indices)):][::-1]
                    
                    print(f"  Melhores F1 por classe: " + ", ".join([f"Classe {c}: {val_f1_per_class[c]:.4f}" for c in best_classes]))
                    print(f"  Piores F1 por classe: " + ", ".join([f"Classe {c}: {val_f1_per_class[c]:.4f}" for c in worst_classes]))
            
            print(f"  Mem Sist: {mem_info.used/(1024**3):.2f}/{mem_info.total/(1024**3):.2f} GB | GPU: {gpu_mem_alloc:.1f}/{gpu_mem_res:.1f} MB")
            
            # Salvar checkpoint periódico
            if (epoch + 1) % 20 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'model_checkpoint_{run_id}_epoch{epoch+1}.pt')
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': main_scheduler.state_dict(),
                        'val_f1': val_f1,
                        'val_f1_per_class': val_f1_per_class,
                    }, checkpoint_path)
                    print(f"  Checkpoint periódico salvo em {checkpoint_path}")
                except Exception as e:
                    print(f"  Erro ao salvar checkpoint: {e}")
        
        # Early Stopping
        is_improved = val_f1 > early_stopping['best_val_f1'] + min_delta
        
        if is_improved:
            early_stopping['best_val_f1'] = val_f1
            early_stopping['best_model_state'] = {k: v.cpu() for k, v in model.state_dict().items()}
            early_stopping['patience_counter'] = 0
            
            # Salvar melhor modelo
            best_model_path = os.path.join(checkpoint_dir, f'model_checkpoint_{run_id}_best.pt')
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': early_stopping['best_model_state'],
                    'val_f1': val_f1,
                    'val_f1_per_class': val_f1_per_class,
                    'class_weights': class_weights.cpu() if class_weights is not None else None,
                    'history': history,
                }, best_model_path)
                print(f"  Melhor modelo salvo com F1: {val_f1:.4f}")
            except Exception as e:
                print(f"  Erro ao salvar melhor modelo: {e}")
        else:
            early_stopping['patience_counter'] += 1
            if early_stopping['patience_counter'] >= patience:
                print(f"\nEarly stopping após {epoch+1} épocas sem melhoria.")
                break
        
        # Limpar memória no final de cada época
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # --- Fim do Treinamento ---
    total_time = time.time() - start_time
    print(f"\n--- Treinamento Concluído ---")
    print(f"Tempo total: {total_time:.2f} segundos")
    print(f"Melhor F1 de validação: {early_stopping['best_val_f1']:.4f}")
    
    # Restaurar melhor modelo
    if early_stopping['best_model_state'] is not None:
        print("Restaurando pesos do melhor modelo.")
        model.load_state_dict({k: v.to(device) for k, v in early_stopping['best_model_state'].items()})
    else:
        print("Aviso: Nenhum estado de melhor modelo salvo. Usando o estado final.")
    
    return model, history


# --- Evaluation and Visualization Functions ---
# (Keep visualize_training_history, analyze_node_embeddings,
#  memory_efficient_visualization, evaluate_model as they were in the previous version)
# ... (rest of the evaluation and visualization functions remain the same) ...
def visualize_training_history(history, output_path=None):
    """
    Visualizes training and validation loss, accuracy, F1, and LR curves.
    (Adapted from previous version to include F1)
    """
    epochs = range(1, len(history['train_loss']) + 1)
    # Determine number of plots needed
    num_plots = 3 # Loss, F1, LR by default
    has_val_acc = 'val_acc' in history and any(v is not None and v != 0.0 for v in history['val_acc'])
    if has_val_acc: # Add accuracy plot if available and non-zero
        num_plots = 4

    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 5 * num_plots), sharex=True)
    # If only 3 plots, axes might not be indexable as axes[3]
    if not isinstance(axes, np.ndarray): # Handle case where subplots returns a single Axes object
        axes = [axes] * num_plots # Make it iterable

    ax_idx = 0

    # Plot Loss
    axes[ax_idx].plot(epochs, history['train_loss'], 'bo-', label='Training Loss', markersize=3, alpha=0.8)
    if 'val_loss' in history and any(v is not None and v != 0.0 for v in history['val_loss']):
        axes[ax_idx].plot(epochs, history['val_loss'], 'ro-', label='Validation Loss', markersize=3, alpha=0.8)
    # axes[ax_idx].set_xlabel('Epoch') # Shared x-axis
    axes[ax_idx].set_ylabel('Loss')
    axes[ax_idx].set_title('Training and Validation Loss')
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)
    ax_idx += 1

    # Plot F1 Score (Primary metric for validation)
    if 'val_f1' in history and any(v is not None and v != 0.0 for v in history['val_f1']):
        # Only plot train F1 if calculated (might not be in sampling)
        if 'train_f1' in history and any(v is not None and v != 0.0 for v in history['train_f1']):
             axes[ax_idx].plot(epochs, history['train_f1'], 'bo-', label='Training F1 Score', markersize=3, alpha=0.8)
        axes[ax_idx].plot(epochs, history['val_f1'], 'ro-', label='Validation F1 Score', markersize=3, alpha=0.8)
        # axes[ax_idx].set_xlabel('Epoch') # Shared x-axis
        axes[ax_idx].set_ylabel('F1 Score (Macro)')
        axes[ax_idx].set_title('Training and Validation F1 Score')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    # Plot Accuracy (Secondary metric)
    if has_val_acc:
        if 'train_acc' in history and any(v is not None and v != 0.0 for v in history['train_acc']):
             axes[ax_idx].plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy', markersize=3, alpha=0.8)
        axes[ax_idx].plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy', markersize=3, alpha=0.8)
        # axes[ax_idx].set_xlabel('Epoch') # Shared x-axis
        axes[ax_idx].set_ylabel('Accuracy')
        axes[ax_idx].set_title('Training and Validation Accuracy')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
        ax_idx += 1

    # Plot Learning Rate
    axes[ax_idx].plot(epochs, history['lr'], 'go-', label='Learning Rate', markersize=3, alpha=0.8)
    axes[ax_idx].set_xlabel('Epoch') # Add x-label only to the bottom plot
    axes[ax_idx].set_ylabel('Learning Rate')
    axes[ax_idx].set_title('Learning Rate Schedule')
    axes[ax_idx].legend()
    axes[ax_idx].grid(True, alpha=0.3)
    if any(lr > 0 for lr in history['lr']): # Avoid log scale error if LR is always 0
        axes[ax_idx].set_yscale('log')

    plt.tight_layout()

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {output_path}")
        except Exception as e:
            print(f"Error saving training history plot: {e}")
    else:
        plt.show()
    plt.close(fig) # Close figure after saving/showing

    return fig

def analyze_node_embeddings(model, data, output_path=None, use_intermediate=True, n_nodes_tsne=5000):
    """
    Analyzes and visualizes node embeddings using t-SNE, sampling nodes if necessary.
    NOTE: Uses full graph inference, which might be memory-intensive.
    """
    print("\n--- Analyzing Node Embeddings (t-SNE) ---")
    model.eval()
    embeddings_np = None
    labels_np = None
    selected_indices = None

    try:
        # Perform full graph inference on CPU to get embeddings
        model.to('cpu')
        data.to('cpu')
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print("Performing inference on CPU for embedding extraction...")

        with torch.no_grad():
            # Get embeddings (intermediate or final)
            if use_intermediate:
                try:
                    # --- Get output of the last message passing layer (GAT) ---
                    x = data.x
                    # Apply layers up to the one before the output_layer
                    for i in range(model.num_gnn_layers): # Iterate through GNN layers
                        x_res = x
                        x = model.layers[i](x, data.edge_index)
                        x = model.layer_norms[i](x)
                        x = model.activation(x)
                        # No dropout during inference needed here
                        if i > 0 and x.shape == x_res.shape: # Apply residual if shapes match
                             x = x + x_res
                    embeddings = x # Output of the last GNN layer (GAT)
                    print("Using embeddings from the output of the last GNN layer (GAT).")
                except Exception as e:
                    print(f"Could not access intermediate layer ({e}). Falling back to final output.")
                    # Need standard forward pass for final output
                    x = data.x
                    for i in range(model.num_gnn_layers):
                         x_res = x
                         x = model.layers[i](x, data.edge_index)
                         x = model.layer_norms[i](x)
                         x = model.activation(x)
                         if i > 0 and x.shape == x_res.shape: x = x + x_res
                    embeddings = model.output_layer(x) # Apply final linear layer
            else:
                 # Standard forward pass for final output logits
                 x = data.x
                 for i in range(model.num_gnn_layers):
                      x_res = x
                      x = model.layers[i](x, data.edge_index)
                      x = model.layer_norms[i](x)
                      x = model.activation(x)
                      if i > 0 and x.shape == x_res.shape: x = x + x_res
                 embeddings = model.output_layer(x) # Apply final linear layer
                 print("Using final output logits as embeddings.")

            # Sample nodes for t-SNE if the graph is too large
            num_all_nodes = embeddings.shape[0]
            if num_all_nodes > n_nodes_tsne:
                print(f"Sampling {n_nodes_tsne} nodes out of {num_all_nodes} for t-SNE visualization.")
                selected_indices = np.random.choice(num_all_nodes, n_nodes_tsne, replace=False)
                embeddings_np = embeddings[selected_indices].detach().cpu().numpy()
                labels_np = data.y[selected_indices].cpu().numpy()
            else:
                embeddings_np = embeddings.detach().cpu().numpy()
                labels_np = data.y.cpu().numpy()
                selected_indices = np.arange(num_all_nodes) # All nodes selected

            print("Inference complete.")
            del embeddings, x # Free memory

    except RuntimeError as e:
         if "out of memory" in str(e).lower():
              print("Error: Ran out of memory during full graph inference for t-SNE.")
              model.to(device)
              return None
         else:
              print(f"An error occurred during embedding extraction: {e}")
              model.to(device)
              return None
    finally:
         model.to(device) # Ensure model is back on the original device
         if torch.cuda.is_available(): torch.cuda.empty_cache()

    if embeddings_np is None or labels_np is None:
        return None

    # Reduce dimensionality using t-SNE
    print(f"Running t-SNE on {embeddings_np.shape[0]} nodes (this may take a while)...")
    # Adjust perplexity if number of samples is small
    perplexity_value = min(30, max(5, embeddings_np.shape[0] - 1))
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=perplexity_value,
                n_iter=300, verbose=1, init='pca', learning_rate='auto')
    try:
        tsne_result = tsne.fit_transform(embeddings_np)
        print("t-SNE finished.")
    except Exception as e:
        print(f"Error during t-SNE: {e}")
        return None


    # Visualization
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels_np)
    num_unique_labels = len(unique_labels)
    colors = plt.cm.get_cmap('tab10', max(10, num_unique_labels))

    # Define class names (adjust based on your dataset)
    num_classes_total = int(data.y.max().item() + 1) if data.y.numel() > 0 else 0
    class_names = { i: f'Classe {i}' for i in range(num_classes_total) }
    # Example:
    # class_names = { 0: 'Motorway', 1: 'Trunk', ... }


    for i, class_idx in enumerate(unique_labels):
        mask = labels_np == class_idx
        class_label = class_names.get(class_idx, f'Classe {class_idx}')
        plt.scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                    color=colors(i % 10), label=class_label, alpha=0.6, s=10)

    plt.title(f'Visualização t-SNE de Embeddings ({embeddings_np.shape[0]} Nós Amostrados)', fontsize=14)
    plt.xlabel("Dimensão t-SNE 1")
    plt.ylabel("Dimensão t-SNE 2")
    plt.legend(fontsize=10, markerscale=2, loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Visualização t-SNE salva em {output_path}")
        except Exception as e:
            print(f"Error saving t-SNE plot: {e}")
    else:
        plt.show()
    plt.close()

    return {
        'embeddings': embeddings_np,
        'tsne_result': tsne_result,
        'selected_indices': selected_indices
    }

def memory_efficient_visualization(G, node_indices, predictions, class_names, output_path, max_nodes_vis=50000):
    """
    Versão de visualização que gerencia melhor a memória, limitando nós e usando lotes.

    Args:
        G: NetworkX graph object (full graph).
        node_indices: Numpy array of original indices corresponding to predictions.
        predictions: Numpy array of predicted labels.
        class_names: Dictionary mapping class indices to names.
        output_path: Path to save the visualization.
        max_nodes_vis: Maximum number of nodes to draw directly to prevent OOM.
    """
    if not G:
        print("Error: NetworkX graph G is required for visualization.")
        return
    if node_indices is None or predictions is None:
        print("Error: node_indices and predictions are required.")
        return

    print("\n--- Visualizing Morphological Classification (Memory-Efficient) ---")

    # Limpar memória antes da visualização
    plt.close('all') # Close any existing figures
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Limitar visualização a um máximo de nós se necessário
    num_predictions = len(node_indices)
    if num_predictions > max_nodes_vis:
        print(f"Warning: Limiting visualization to {max_nodes_vis} randomly sampled nodes out of {num_predictions} to conserve memory.")
        vis_indices = np.random.choice(num_predictions, max_nodes_vis, replace=False) # Sample random nodes
        node_indices_vis = node_indices[vis_indices]
        predictions_vis = predictions[vis_indices]
    else:
        node_indices_vis = node_indices
        predictions_vis = predictions

    # Obter posições eficientemente apenas para os nós a serem visualizados
    print(f"Extracting positions for {len(node_indices_vis)} nodes...")
    positions = {}
    nodes_to_get_pos = set(node_indices_vis) # Use set for faster lookup
    count = 0
    try:
        for node_id, node_data in G.nodes(data=True):
             if node_id in nodes_to_get_pos:
                  # Prioritize 'x', 'y' if they exist from original data mapping
                  pos_x = node_data.get('x', None)
                  pos_y = node_data.get('y', None)
                  if pos_x is not None and pos_y is not None:
                      positions[node_id] = (pos_x, pos_y)
                  else: # Fallback if 'x', 'y' are missing
                      positions[node_id] = (0,0) # Placeholder
                  count += 1
                  if count == len(nodes_to_get_pos): break
    except Exception as e:
         print(f"Error extracting node positions: {e}")
         # Fallback to random positions if extraction fails
         positions = {nid: (np.random.rand(), np.random.rand()) for nid in node_indices_vis}

    # Check if positions are valid or need fallback layout
    if not positions or all(p == (0, 0) for p in positions.values()):
        print("Warning: Node positions ('x', 'y') not found or zero. Using spring layout (might be slow/memory intensive).")
        if len(node_indices_vis) < 5000:
             try:
                  # Create subgraph only for layout calculation
                  G_layout = G.subgraph(node_indices_vis)
                  positions = nx.spring_layout(G_layout, seed=RANDOM_SEED)
                  del G_layout
             except Exception as e:
                  print(f"Spring layout failed: {e}. Using random positions.")
                  positions = {nid: (np.random.rand(), np.random.rand()) for nid in node_indices_vis}
        else:
             print("Skipping spring layout due to large number of nodes. Using random positions.")
             positions = {nid: (np.random.rand(), np.random.rand()) for nid in node_indices_vis}


    # Reduzir resolução/qualidade para economizar memória
    plt.figure(figsize=(12, 10), dpi=100) # Reduzir tamanho e DPI

    # Mapear predições para nós
    node_to_pred = {node_id: pred for node_id, pred in zip(node_indices_vis, predictions_vis)}
    unique_preds = sorted(list(set(predictions_vis)))

    # Preparar cores e legenda
    num_classes = len(class_names)
    cmap = plt.cm.get_cmap('tab10', max(10, num_classes))
    legend_handles = []

    # Desenhar arestas (apenas entre nós visualizados) - Optional and potentially slow
    draw_edges = True
    if len(node_indices_vis) > 20000: # Heuristic: don't draw edges if too many nodes
        print("Skipping edge drawing due to large number of visualized nodes.")
        draw_edges = False

    if draw_edges:
        try:
            print("Drawing graph edges (between visualized nodes)...")
            # Filter edges where both source and target are in the visualized set
            edges_to_draw = [(u, v) for u, v in G.edges(node_indices_vis) if u in positions and v in positions]
            nx.draw_networkx_edges(G, positions, edgelist=edges_to_draw, alpha=0.05, width=0.5, edge_color='gray') # More transparent edges
            del edges_to_draw
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            print(f"Could not draw edges efficiently: {e}. Skipping edges.")


    # Desenhar nós em lotes (batches)
    print("Drawing graph nodes in batches...")
    node_size = 8 # Smaller node size
    batch_draw_size = 5000 # Adjust batch size based on memory

    for class_idx in unique_preds:
        nodes_in_class = [node_id for node_id, pred in node_to_pred.items() if pred == class_idx]
        if not nodes_in_class: continue

        color = cmap(class_idx % 10) # Use modulo for color index
        label = class_names.get(class_idx, f'Classe {class_idx}')
        added_label = False

        for i in range(0, len(nodes_in_class), batch_draw_size):
            batch_nodes = nodes_in_class[i:i+batch_draw_size]
            # Ensure positions exist for all nodes in the batch
            batch_pos = {n: positions[n] for n in batch_nodes if n in positions}
            if not batch_pos: continue

            # Only add label once per class for the legend
            current_label = label if i == 0 and not added_label else ""
            nx.draw_networkx_nodes(G, batch_pos, nodelist=list(batch_pos.keys()), node_color=[color],
                                   node_size=node_size, alpha=0.7, label=current_label)
            if i == 0 and not added_label:
                 added_label = True

        # Criar proxy artist para a legenda (only if label was added)
        if added_label:
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                             markerfacecolor=color, markersize=8, label=label))

    plt.title(f'Classificação Morfológica (Visualizando {len(node_indices_vis)} nós)', fontsize=14)
    # Only show legend if handles were created
    if legend_handles:
        plt.legend(handles=legend_handles, fontsize=9, loc='upper left', bbox_to_anchor=(1.02, 1))
    plt.axis('off')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout for legend

    try:
        plt.savefig(output_path, dpi=150, bbox_inches='tight') # Lower DPI for saving
        print(f"Visualização morfológica salva em {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {e}")

    plt.close() # Fechar figura após salvar
    if torch.cuda.is_available(): torch.cuda.empty_cache()


def evaluate_model(model, data, output_dir, class_names):
    """
    Evaluates the trained model on the test set and generates reports.
    (Adapted from previous version to include more metrics and plots)
    NOTE: Uses full graph inference. May need adaptation for very large graphs.
    """
    print("\n--- Evaluating Model on Test Set ---")
    model.eval() # Set model to evaluation mode

    # Ensure data has a test mask and it's not empty
    if not hasattr(data, 'test_mask') or data.test_mask.sum() == 0:
        print("Error: No test mask found or test mask is empty. Cannot evaluate.")
        return None, None

    y_true_test_cpu = None
    y_pred_test_cpu = None
    all_preds_np = None
    num_classes = len(class_names) # Get number of classes from class_names dict

    try:
        with torch.no_grad():
            # Perform inference on the full graph
            print("Performing full graph inference for evaluation...")
            data.to(device) # Move data to device for inference
            model.to(device)
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            # Use autocast for inference as well if memory is a concern
            with autocast(enabled=(device.type == 'cuda')):
                out = model(data.x, data.edge_index) # Use standard forward pass
            _, pred = out.max(dim=1)

            # Ensure mask and labels are on the correct device for filtering
            test_mask_tensor = data.test_mask.to(device)
            y_tensor = data.y.to(device)

            # Move necessary results to CPU for scikit-learn
            y_true_test_cpu = y_tensor[test_mask_tensor].cpu().numpy()
            y_pred_test_cpu = pred[test_mask_tensor].cpu().numpy()
            # Get predictions for all nodes (needed for visualization mapping)
            all_preds_np = pred.cpu().numpy()

            print("Inference complete.")
            del out, pred # Clean up large tensors

    except RuntimeError as e:
         if "out of memory" in str(e).lower():
              print("Error: Ran out of memory during full graph inference for evaluation.")
              print("Consider implementing sampled inference for evaluation on very large graphs.")
              return None, None # Indicate failure
         else:
              print(f"An error occurred during evaluation inference: {e}")
              return None, None
    finally:
         # Clean up GPU memory
         data.to('cpu') # Move data back to CPU
         model.to(device) # Keep model on device
         if torch.cuda.is_available(): torch.cuda.empty_cache()

    if y_true_test_cpu is None or y_pred_test_cpu is None:
        print("Evaluation failed before metric calculation.")
        return None, None

    # Calculate metrics
    print("Calculating evaluation metrics...")
    test_acc = accuracy_score(y_true_test_cpu, y_pred_test_cpu)
    test_f1_macro = f1_score(y_true_test_cpu, y_pred_test_cpu, average='macro', zero_division=0)
    test_f1_weighted = f1_score(y_true_test_cpu, y_pred_test_cpu, average='weighted', zero_division=0)

    # Per-class metrics
    labels_all = np.arange(num_classes) # Use all defined class labels
    target_names_all = [class_names.get(i, f'Class {i}') for i in labels_all]

    precision_per_class = precision_score(y_true_test_cpu, y_pred_test_cpu, average=None, labels=labels_all, zero_division=0)
    recall_per_class = recall_score(y_true_test_cpu, y_pred_test_cpu, average=None, labels=labels_all, zero_division=0)
    f1_per_class = f1_score(y_true_test_cpu, y_pred_test_cpu, average=None, labels=labels_all, zero_division=0)

    # Classification report text
    class_report_text = classification_report(
        y_true_test_cpu, y_pred_test_cpu,
        labels=labels_all, # Report for all classes
        target_names=target_names_all,
        zero_division=0
    )

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test F1 Macro: {test_f1_macro:.4f}")
    print(f"Test F1 Weighted: {test_f1_weighted:.4f}")
    print("\nClassification Report:")
    print(class_report_text)

    # Confusion Matrix
    cm = confusion_matrix(y_true_test_cpu, y_pred_test_cpu, labels=labels_all)
    # Normalize by row (true class) only if sum is not zero
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    with np.errstate(divide='ignore', invalid='ignore'): # Ignore warnings for zero division
        cm_normalized = np.where(cm_sum > 0, cm.astype('float') / cm_sum, 0.0)


    # Plot Confusion Matrix
    try:
        fig_cm, ax_cm = plt.subplots(figsize=(max(8, num_classes*0.8), max(6, num_classes*0.6)))
        im = ax_cm.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
        ax_cm.figure.colorbar(im, ax=ax_cm)
        tick_labels = target_names_all
        ax_cm.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
                  xticklabels=tick_labels, yticklabels=tick_labels,
                  title='Matriz de Confusão Normalizada (Conjunto de Teste)',
                  ylabel='Rótulo Verdadeiro', xlabel='Rótulo Previsto')
        plt.setp(ax_cm.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        fmt = '.2f'
        thresh = cm_normalized.max() / 2. if cm_normalized.max() > 0 else 0.5
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                ax_cm.text(j, i, format(cm_normalized[i, j], fmt),
                           ha="center", va="center",
                           color="white" if cm_normalized[i, j] > thresh else "black",
                           fontsize=8)
        fig_cm.tight_layout()
        confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix_test.png')
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusão salva em {confusion_matrix_path}")
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
    finally:
        plt.close(fig_cm) # Ensure figure is closed

    # Plot per-class metrics
    try:
        fig_metrics, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        x = np.arange(num_classes)
        width = 0.35
        bars1 = ax1.bar(x - width/2, precision_per_class, width, label='Precision')
        bars2 = ax1.bar(x + width/2, recall_per_class, width, label='Recall')
        ax1.set_ylabel('Score')
        ax1.set_title('Precision e Recall por Classe')
        ax1.set_xticks(x)
        ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax1.set_ylim(0, 1.05) # Set y-limit for scores

        bars3 = ax2.bar(x, f1_per_class, width, label='F1-Score', color='green')
        ax2.set_ylabel('F1-Score')
        ax2.set_title('F1-Score por Classe')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax2.axhline(y=test_f1_macro, color='r', linestyle='--', label=f'Macro Avg F1: {test_f1_macro:.3f}')
        ax2.legend()
        ax2.grid(True, axis='y', linestyle='--', alpha=0.6)
        ax2.set_ylim(0, 1.05) # Set y-limit for scores

        plt.tight_layout()
        class_metrics_path = os.path.join(output_dir, 'class_metrics_test.png')
        plt.savefig(class_metrics_path, dpi=300, bbox_inches='tight')
        print(f"Visualização de métricas por classe salva em {class_metrics_path}")
    except Exception as e:
        print(f"Error plotting per-class metrics: {e}")
    finally:
        plt.close(fig_metrics) # Ensure figure is closed

    # Generate Report Dictionary and Files
    report = {
        'test_accuracy': float(test_acc),
        'test_f1_macro': float(test_f1_macro),
        'test_f1_weighted': float(test_f1_weighted),
        'per_class_precision': [float(p) for p in precision_per_class],
        'per_class_recall': [float(r) for r in recall_per_class],
        'per_class_f1': [float(f) for f in f1_per_class],
        'confusion_matrix': cm.tolist(),
        'classification_report_text': class_report_text,
        'class_distribution': {}
    }
    # Add class distributions (ensure data is on CPU)
    data.to('cpu')
    if hasattr(data, 'train_mask'): report['class_distribution']['train'] = torch.bincount(data.y[data.train_mask], minlength=num_classes).tolist()
    if hasattr(data, 'val_mask'): report['class_distribution']['val'] = torch.bincount(data.y[data.val_mask], minlength=num_classes).tolist()
    if hasattr(data, 'test_mask'): report['class_distribution']['test'] = torch.bincount(data.y[data.test_mask], minlength=num_classes).tolist()

    # Save Text Report
    report_path_txt = os.path.join(output_dir, 'evaluation_report.txt')
    try:
        with open(report_path_txt, 'w') as f:
            f.write(f"=== RELATÓRIO DE AVALIAÇÃO DO MODELO ===\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device Utilizado: {device}\n\n")
            f.write(f"--- Métricas Gerais (Conjunto de Teste) ---\n")
            f.write(f"Acurácia: {test_acc:.4f}\n")
            f.write(f"F1 Score (Macro): {test_f1_macro:.4f}\n")
            f.write(f"F1 Score (Ponderado): {test_f1_weighted:.4f}\n\n")
            f.write(f"--- Métricas por Classe ---\n")
            f.write(f"{'Classe':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}\n")
            f.write("-" * 50 + "\n")
            for i in range(num_classes):
                class_name = class_names.get(i, f'Classe {i}')
                f.write(f"{class_name:<15} {precision_per_class[i]:<10.4f} {recall_per_class[i]:<10.4f} {f1_per_class[i]:<10.4f}\n")
            f.write("\n--- Relatório de Classificação Detalhado ---\n")
            f.write(class_report_text)
            f.write("\n\n--- Distribuição das Classes ---\n")
            for split, counts in report['class_distribution'].items():
                f.write(f"  {split.capitalize()}: {counts}\n")
            f.write("\n--- Matriz de Confusão (Contagens) ---\n")
            header = "Prev -> | " + " | ".join([class_names.get(i, str(i)).ljust(8) for i in range(num_classes)]) + " |"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for i, row in enumerate(cm):
                row_str = f"Verd {class_names.get(i, str(i)).ljust(4)} | " + " | ".join([str(count).ljust(8) for count in row]) + " |"
                f.write(row_str + "\n")
        print(f"Relatório de avaliação salvo em: {report_path_txt}")
    except Exception as e:
        print(f"Error saving text evaluation report: {e}")


    # Save JSON Report
    report_path_json = os.path.join(output_dir, 'evaluation_report.json')
    try:
        with open(report_path_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Métricas em JSON salvas em: {report_path_json}")
    except Exception as e:
        print(f"Error saving JSON evaluation report: {e}")

    # Return all_preds_np which contains predictions for *all* nodes in the evaluated data object
    return report, all_preds_np


# --- Main Execution ---

def main():
    print("--- Iniciando Script de Análise Morfológica GNN Otimizado para L4 ---")
    # --- Otimizações Iniciais ---
    optimize_for_l4_gpu()

    # --- Configuração de Diretórios (Colab/Local) ---
    # Usar pasta específica dentro do MyDrive
    gdrive_base_folder = 'geoprocessamento_gnn/results_gnn_l4'  # Nome específico para versão L4
    save_path_base, models_dir, results_dir_base, is_colab = setup_colab_integration(gdrive_base_folder)

    # --- Configuração de Caminhos de Dados ---
    if is_colab:
        # Assume data is in a specific folder relative to MyDrive
        data_base_dir = '/content/drive/MyDrive/geoprocessamento_gnn/data'
    else:
        # Attempt relative path if not in Colab
        script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        # Adjust relative path based on your local project structure
        data_base_dir = os.path.join(script_dir, 'data_grapho')  # Pasta local
        print(f"Google Drive não detectado. Usando diretório de dados local: {os.path.abspath(data_base_dir)}")

    # Verificar diretório de dados
    if not os.path.isdir(data_base_dir):
        print(f"Erro Crítico: Diretório de dados não encontrado em '{os.path.abspath(data_base_dir)}'. Verifique o caminho.")
        return

    # Encontrar o arquivo .pt mais recente
    try:
        # Procurar por arquivos .pt que contenham 'road_graph_pyg'
        pt_files = [f for f in os.listdir(data_base_dir) if f.endswith('.pt') and 'road_graph_pyg' in f]
        if not pt_files:
            print(f"Erro: Nenhum arquivo PyTorch Geometric (.pt contendo 'road_graph_pyg') encontrado em {data_base_dir}")
            return
        # Sort files by modification time to get the latest
        latest_pt_file = max(pt_files, key=lambda f: os.path.getmtime(os.path.join(data_base_dir, f)))
        data_path = os.path.join(data_base_dir, latest_pt_file)
        print(f"Utilizando arquivo de dados: {data_path}")
    except FileNotFoundError:
        print(f"Erro: Diretório de dados não encontrado em {data_base_dir}")
        return
    except Exception as e:
        print(f"Erro ao encontrar arquivo de dados: {e}")
        return

    # Criar diretório de saída único para esta execução
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(results_dir_base, f"run_L4_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    # Certificar que o diretório de modelos para checkpoints também existe
    run_models_dir = os.path.join(models_dir, f"run_L4_{timestamp}_checkpoints")
    os.makedirs(run_models_dir, exist_ok=True)
    print(f"Resultados serão salvos em: {output_dir}")
    print(f"Checkpoints serão salvos em: {run_models_dir}")

    # --- Carregar Dados ---
    print("\n--- Carregando e Preparando Dados ---")
    data = load_data(data_path)
    if data is None: return

    # --- Pré-processamento de Dados ---
    # 1. Criar ou Verificar Máscaras Estratificadas 
    if not all(hasattr(data, m) for m in ['train_mask', 'val_mask', 'test_mask']):
        print("Máscaras não encontradas. Criando máscaras estratificadas...")
        data = create_stratified_masks(data, train_ratio=0.7, val_ratio=0.15)
    else:
        print("Utilizando máscaras existentes do arquivo.")
        # Ensure masks are boolean type
        for m in ['train_mask', 'val_mask', 'test_mask']:
            if hasattr(data, m) and getattr(data, m).dtype != torch.bool:
                print(f"Convertendo máscara '{m}' para boolean.")
                setattr(data, m, getattr(data, m).bool())
        
        # Imprimir estatísticas das máscaras
        print(f"  Nós de treino: {data.train_mask.sum().item()}")
        print(f"  Nós de validação: {data.val_mask.sum().item()}")
        print(f"  Nós de teste: {data.test_mask.sum().item()}")

    # 2. Guardar Grafo Original (NetworkX) para Visualização ANTES de filtrar componentes
    print("Criando grafo NetworkX original para visualização...")
    original_graph_nx = None 
    try:
        # Incluir atributos 'x', 'y' se existirem para posições
        node_attrs_nx = []
        if hasattr(data, 'x'): node_attrs_nx.append('x')
        if hasattr(data, 'y'): node_attrs_nx.append('y')
        if hasattr(data, 'pos'): node_attrs_nx.append('pos')

        original_graph_nx = to_networkx(data, node_attrs=node_attrs_nx, to_undirected=True)
        print(f"Grafo NetworkX criado com {original_graph_nx.number_of_nodes()} nós e {original_graph_nx.number_of_edges()} arestas.")
        
        # Criar posições para o grafo NetworkX se disponíveis
        if hasattr(data, 'pos') and data.pos is not None:
            print("Criando coordenadas x, y a partir do atributo 'pos'...")
            for i, node_id in enumerate(original_graph_nx.nodes()):
                if i < data.pos.shape[0] and data.pos.shape[1] >= 2:
                    original_graph_nx.nodes[node_id]['x'] = float(data.pos[i, 0].item())
                    original_graph_nx.nodes[node_id]['y'] = float(data.pos[i, 1].item())
        # Tentar extrair coordenadas de features se pos não existir
        elif hasattr(data, 'x') and data.x.shape[1] >= 2:
            print("Tentando extrair coordenadas x, y das primeiras 2 colunas de 'data.x'...")
            for i, node_id in enumerate(original_graph_nx.nodes()):
                original_graph_nx.nodes[node_id]['x'] = float(data.x[i, 0].item())
                original_graph_nx.nodes[node_id]['y'] = float(data.x[i, 1].item())
    except Exception as e:
        print(f"Erro ao criar grafo NetworkX: {e}")
        original_graph_nx = None

    # 3. Processar Fragmentação do Grafo 
    print("\n--- Processando Fragmentação do Grafo ---")
    data_processed, kept_node_indices = preprocess_fragmented_graph(data, strategy='largest_component')
    if kept_node_indices is None:
        print("Aviso: Não foi possível processar fragmentação. Usando grafo original.")
        kept_node_indices = torch.arange(data.num_nodes)
        data_processed = data
    else:
        print(f"Processamento de fragmentação concluído. {data_processed.num_nodes} nós mantidos.")

    # --- Definição do Modelo ---
    print("\n--- Configurando Modelo Otimizado para L4 ---")
    if not hasattr(data_processed, 'x') or not hasattr(data_processed, 'y'):
        print("Erro: Não é possível definir o modelo sem data_processed.x e data_processed.y.")
        return

    input_dim = data_processed.num_node_features
    # Inferir output_dim robustamente do conjunto de dados processado
    output_dim = int(data_processed.y.max().item() + 1) if data_processed.y.numel() > 0 else 0
    if output_dim == 0:
        print("Erro: Não foi possível determinar o número de classes. O tensor de rótulos está vazio?")
        return
    print(f"Dimensões: Input={input_dim}, Output={output_dim} classes")

    # Definir Nomes das Classes
    class_names = {i: f'Classe {i}' for i in range(output_dim)}
    # Nomes específicos se conhecidos:
    # class_names = {
    #    0: 'Motorway', 1: 'Trunk', 2: 'Primary', 3: 'Secondary', 4: 'Tertiary', 5: 'Residential'
    # }
    print(f"Classes: {class_names}")

    # Instanciar Modelo Otimizado
    # Parâmetros otimizados para L4
    model = ImprovedGNN(
        input_dim=input_dim,
        hidden_dim=384,    # Dimensionalidade aumentada para L4
        output_dim=output_dim,
        num_layers=5,      # Mais camadas para maior capacidade
        heads=6,           # Mais cabeças de atenção para paralelismo 
        dropout=0.25       # Dropout ajustado
    ).to(device)

    print("\n--- Arquitetura do Modelo ---")
    print(model)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total de parâmetros treináveis: {num_params:,}")

    # --- Hiperparâmetros de Treinamento Otimizados para L4 ---
    print("\n--- Iniciando Treinamento Otimizado para L4 ---")
    # Configuração específica para GPU L4
    train_config = {
        # Tamanhos de lote maiores
        'batch_size': 4096,             # Aproveitar a maior memória da L4
        'batch_size_validation': 8192,  # Validação com lotes grandes
        
        # Estratégia de vizinhança otimizada para model.num_gnn_layers=5
        'num_neighbors': [20, 15, 10, 8, 5],  # Corresponde às 5 camadas GNN
        
        # Otimização
        'lr': 0.0015,                   # Taxa de aprendizado inicial maior
        'weight_decay': 5e-5,           # Regularização L2
        'label_smoothing': 0.15,        # Suavização de rótulos para classes desbalanceadas
        
        # Técnicas avançadas
        'gradient_accumulation_steps': 2,  # Acumular gradientes (efeito de batch maior)
        'lr_warmup_epochs': 10,         # Aquecer taxa de aprendizado por 10 épocas
        
        # Controle de execução
        'num_epochs': 300,              # Mais épocas para convergência completa
        'patience': 40,                 # Maior paciência para early stopping
        'min_delta': 0.0008,            # Limiar menor para contar como melhoria
        
        # Outros parâmetros
        'use_amp': (device.type == 'cuda'),  # Usar precisão mista se disponível
        'checkpoint_dir': run_models_dir,
        'run_id': timestamp
    }

    # --- Treinamento com Configuração Otimizada ---
    training_start = time.time()
    
    # Monitoramento de memória inicial
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"Memória GPU inicial: {initial_mem:.2f} MB")
    
    # Treinamento com amostragem de vizinhança
    model, history = train_with_sampling(
        model=model,
        data=data_processed,
        **train_config
    )
    
    training_time = time.time() - training_start
    print(f"Tempo total de treinamento: {training_time:.2f} segundos ({training_time/60:.2f} minutos)")

    # --- Salvar Modelo Final ---
    model_path = os.path.join(run_models_dir, f'model_final_L4_{timestamp}.pt')
    try:
        torch.save({
            'model_state_dict': model.state_dict(),
            'hyperparameters': train_config,
            'class_names': class_names,
            'input_dim': input_dim,
            'output_dim': output_dim,
            'timestamp': timestamp,
            'train_time': training_time
        }, model_path)
        print(f"Modelo final salvo em {model_path}")
    except Exception as e:
        print(f"Erro ao salvar modelo final: {e}")

    # --- Análise Pós-Treinamento ---
    print("\n--- Iniciando Análise Pós-Treinamento ---")
    
    # 1. Visualizar Histórico de Treinamento
    history_path = os.path.join(output_dir, f'training_history_L4_{timestamp}.png')
    visualize_training_history(history, history_path)

    # 2. Avaliar no Conjunto de Teste
    print("\n--- Avaliando Modelo no Conjunto de Teste ---")
    # Usar CPU para avaliação para liberar memória GPU
    evaluation_report, test_predictions = evaluate_model(
        model, data_processed.to('cpu'), output_dir, class_names
    )
    if evaluation_report is None:
        print("Avaliação falhou. Pulando análises restantes.")
        return

    # 3. Análise de Embeddings (t-SNE)
    print("\n--- Analisando Embeddings com t-SNE ---")
    tsne_path = os.path.join(output_dir, f'embeddings_tsne_L4_{timestamp}.png')
    analyze_node_embeddings(
        model, 
        data_processed.to('cpu'), 
        tsne_path,
        use_intermediate=True, 
        n_nodes_tsne=5000  # Amostragem de 5000 nós para t-SNE
    )

    # 4. Visualização Morfológica (se grafo NetworkX estiver disponível)
    if original_graph_nx is not None and test_predictions is not None:
        print("\n--- Gerando Visualização Morfológica ---")
        morphology_path = os.path.join(output_dir, f'morphology_classification_L4_{timestamp}.png')
        memory_efficient_visualization(
            G=original_graph_nx,
            node_indices=kept_node_indices.cpu().numpy(),
            predictions=test_predictions,
            class_names=class_names,
            output_path=morphology_path,
            max_nodes_vis=100000  # Aumentar limite para visualização
        )
    else:
        if original_graph_nx is None:
            print("Visualização morfológica indisponível: grafo NetworkX não criado corretamente.")
        if test_predictions is None:
            print("Visualização morfológica indisponível: predições de teste não disponíveis.")

    # --- Resumo Final ---
    print("\n--- Análise GNN Otimizada para L4 Concluída ---")
    print(f"Resultados salvos em: {output_dir}")
    print(f"Modelos e checkpoints salvos em: {run_models_dir}")
    
    # Monitoramento de memória final
    if torch.cuda.is_available():
        final_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"Memória GPU final: {final_mem:.2f} MB")


if __name__ == "__main__":
    main()
