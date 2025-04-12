#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import folium
from folium import plugins
from folium.plugins import MarkerCluster, HeatMap
import networkx as nx
from shapely.geometry import LineString, Point, Polygon, MultiLineString
import contextily as ctx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import warnings
import traceback
from shapely.ops import linemerge
warnings.filterwarnings('ignore')

# Configurar diretórios
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualizations', 'roads')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definir arquivos de entrada
ROADS_FILE = os.path.join(INPUT_DIR, 'roads_processed.gpkg')
SOROCABA_FILE = os.path.join(WORKSPACE_DIR, 'data', 'raw', 'sorocaba.gpkg')

def load_data():
    """Carrega os dados processados da rede viária."""
    print("Carregando dados da rede viária...")
    
    data = {}
    
    try:
        # Carregar dados da rede viária
        print(f"Tentando carregar arquivo: {ROADS_FILE}")
        if not os.path.exists(ROADS_FILE):
            print(f"ERRO: Arquivo não encontrado: {ROADS_FILE}")
            data['roads'] = None
        else:
            data['roads'] = gpd.read_file(ROADS_FILE)
            print(f"Rede viária carregada:")
            print(f"- Número de registros: {len(data['roads'])}")
            print(f"- Colunas disponíveis: {', '.join(data['roads'].columns)}")
            print(f"- Tipos de geometria: {data['roads'].geometry.type.unique()}")
            print(f"- CRS: {data['roads'].crs}")
            
            # Verificar se há geometrias válidas
            if 'geometry' in data['roads'].columns:
                null_geoms = data['roads'].geometry.isna().sum()
                invalid_geoms = (~data['roads'].geometry.is_valid).sum() if not data['roads'].geometry.isna().all() else 0
                print(f"- Geometrias nulas: {null_geoms}")
                print(f"- Geometrias inválidas: {invalid_geoms}")
            
            # Verificar colunas essenciais
            essential_cols = ['road_class', 'length_km', 'highway']
            missing_cols = [col for col in essential_cols if col not in data['roads'].columns]
            if missing_cols:
                print(f"AVISO: Colunas essenciais faltando: {', '.join(missing_cols)}")
    except Exception as e:
        print(f"Erro ao carregar rede viária: {str(e)}")
        print(traceback.format_exc())
        data['roads'] = None
    
    try:
        # Carregar área de estudo
        print(f"\nTentando carregar arquivo: {SOROCABA_FILE}")
        if not os.path.exists(SOROCABA_FILE):
            print(f"ERRO: Arquivo não encontrado: {SOROCABA_FILE}")
            data['sorocaba'] = None
        else:
            data['sorocaba'] = gpd.read_file(SOROCABA_FILE)
            print(f"Área de estudo carregada:")
            print(f"- Número de registros: {len(data['sorocaba'])}")
            print(f"- Colunas disponíveis: {', '.join(data['sorocaba'].columns)}")
            print(f"- Tipo de geometria: {data['sorocaba'].geometry.type.unique()}")
            print(f"- CRS: {data['sorocaba'].crs}")
            
            # Verificar se há um único polígono válido
            if 'geometry' in data['sorocaba'].columns:
                null_geoms = data['sorocaba'].geometry.isna().sum()
                invalid_geoms = (~data['sorocaba'].geometry.is_valid).sum() if not data['sorocaba'].geometry.isna().all() else 0
                print(f"- Geometrias nulas: {null_geoms}")
                print(f"- Geometrias inválidas: {invalid_geoms}")
    except Exception as e:
        print(f"Erro ao carregar área de estudo: {str(e)}")
        print(traceback.format_exc())
        data['sorocaba'] = None
    
    # Verificar CRS e garantir que todos estejam no mesmo sistema
    print("\nVerificando sistemas de coordenadas...")
    for key, gdf in data.items():
        if gdf is not None:
            print(f"CRS de {key}: {gdf.crs}")
    
    # Padronizar CRS para SIRGAS 2000 (EPSG:4674)
    target_crs = "EPSG:4674"
    for key, gdf in data.items():
        if gdf is not None and gdf.crs != target_crs:
            print(f"Reprojetando {key} de {gdf.crs} para {target_crs}")
            try:
                data[key] = gdf.to_crs(target_crs)
            except Exception as e:
                print(f"ERRO ao reprojetar {key}: {str(e)}")
    
    return data

def create_interactive_road_map(data, output_path):
    """Cria um mapa interativo da rede viária usando Folium."""
    print("Criando mapa interativo da rede viária...")
    
    # Verificar e converter dados para EPSG:4326 (WGS84) que é requerido pelo Folium
    if data['sorocaba'] is not None:
        sorocaba = data['sorocaba'].to_crs(epsg=4326)
        # Usar o centroide da área de estudo como centro do mapa
        center_lat = sorocaba.geometry.centroid.y.mean()
        center_lon = sorocaba.geometry.centroid.x.mean()
    elif data['roads'] is not None:
        # Usar o centro da rede viária
        roads = data['roads'].to_crs(epsg=4326)
        center_lat = roads.geometry.centroid.y.mean()
        center_lon = roads.geometry.centroid.x.mean()
    else:
        print("Dados insuficientes para criar o mapa")
        return
    
    # Criar mapa base
    m = folium.Map(location=[center_lat, center_lon], 
                  zoom_start=12,
                  tiles='CartoDB positron')
    
    # Adicionar mini mapa
    minimap = folium.plugins.MiniMap()
    m.add_child(minimap)
    
    # Adicionar escala
    folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)
    
    # Adicionar área de estudo (Sorocaba)
    if data['sorocaba'] is not None:
        sorocaba_json = sorocaba.to_json()
        folium.GeoJson(
            data=sorocaba_json,
            name='Área de Estudo',
            style_function=lambda x: {
                'fillColor': '#ffff00',
                'color': '#000000',
                'weight': 2,
                'fillOpacity': 0.1
            }
        ).add_to(m)
    
    # Adicionar rede viária
    if data['roads'] is not None:
        roads = data['roads'].to_crs(epsg=4326)
        
        # Criar grupos para diferentes classes de vias
        road_groups = {}
        for road_class in roads['road_class'].unique():
            road_groups[road_class] = folium.FeatureGroup(name=f'Vias {road_class.title()}')
        
        # Adicionar vias por classe
        for road_class, group in road_groups.items():
            class_roads = roads[roads['road_class'] == road_class]
            
            # Definir estilo por classe
            style = get_road_style(road_class)
            
            folium.GeoJson(
                data=class_roads.to_json(),
                style_function=lambda x, style=style: style,
                tooltip=folium.GeoJsonTooltip(
                    fields=['name', 'highway', 'length_km'],
                    aliases=['Nome:', 'Tipo:', 'Comprimento (km):'],
                    localize=True,
                    sticky=False
                )
            ).add_to(group)
            
            group.add_to(m)
    
    # Adicionar legenda
    legend_html = create_road_legend_html()
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Adicionar controle de camadas
    folium.LayerControl().add_to(m)
    
    # Salvar mapa
    m.save(output_path)
    print(f"Mapa interativo salvo em: {output_path}")
    return output_path

def get_road_style(road_class):
    """Retorna estilo para cada classe de via."""
    styles = {
        'arterial': {
            'color': '#e41a1c',  # Vermelho
            'weight': 4,
            'opacity': 0.8
        },
        'collector': {
            'color': '#377eb8',  # Azul
            'weight': 3,
            'opacity': 0.7
        },
        'local': {
            'color': '#4daf4a',  # Verde
            'weight': 2,
            'opacity': 0.6
        },
        'pedestrian': {
            'color': '#984ea3',  # Roxo
            'weight': 1,
            'opacity': 0.5,
            'dashArray': '5, 5'
        },
        'cycleway': {
            'color': '#ff7f00',  # Laranja
            'weight': 1,
            'opacity': 0.5,
            'dashArray': '5, 5'
        }
    }
    return styles.get(road_class, {
        'color': '#999999',  # Cinza para classes não mapeadas
        'weight': 1,
        'opacity': 0.5
    })

def create_road_legend_html():
    """Cria HTML para a legenda do mapa."""
    return """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px;
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                opacity: 0.8;">
    <b>Legenda</b><br>
    <i style="background: #ffff00; opacity: 0.3; border: 1px solid black; display: inline-block; width: 18px; height: 18px;"></i> Área de Estudo<br>
    <i style="background: #e41a1c; height: 4px; width: 18px; display: inline-block;"></i> Vias Arteriais<br>
    <i style="background: #377eb8; height: 3px; width: 18px; display: inline-block;"></i> Vias Coletoras<br>
    <i style="background: #4daf4a; height: 2px; width: 18px; display: inline-block;"></i> Vias Locais<br>
    <i style="background: #984ea3; height: 1px; width: 18px; display: inline-block;"></i> Vias de Pedestres<br>
    <i style="background: #ff7f00; height: 1px; width: 18px; display: inline-block;"></i> Ciclovias<br>
    </div>
    """

def plot_road_class_distribution(data, output_path):
    """Plota a distribuição das classes de vias na rede viária."""
    print("Criando gráfico de distribuição das classes de vias...")
    
    if data['roads'] is None:
        print("Dados da rede viária não disponíveis")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Criar dois subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Distribuição por número de vias
    road_counts = data['roads']['road_class'].value_counts()
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    bars1 = ax1.bar(road_counts.index, road_counts.values, color=colors[:len(road_counts)])
    ax1.set_title('Distribuição por Número de Vias', fontsize=12)
    ax1.set_xlabel('Classe de Via')
    ax1.set_ylabel('Número de Vias')
    
    # Adicionar valores sobre as barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # 2. Distribuição por comprimento
    length_by_class = data['roads'].groupby('road_class')['length_km'].sum()
    
    bars2 = ax2.bar(length_by_class.index, length_by_class.values, color=colors[:len(length_by_class)])
    ax2.set_title('Distribuição por Comprimento Total', fontsize=12)
    ax2.set_xlabel('Classe de Via')
    ax2.set_ylabel('Comprimento Total (km)')
    
    # Adicionar valores sobre as barras
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:,.1f}',
                ha='center', va='bottom')
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de distribuição das classes de vias salvo em: {output_path}")

def get_line_endpoints(geometry):
    """Retorna os pontos inicial e final de uma geometria linear."""
    if isinstance(geometry, LineString):
        return tuple(geometry.coords[0]), tuple(geometry.coords[-1])
    elif isinstance(geometry, MultiLineString):
        # Para MultiLineString, tentar mesclar em uma única linha
        merged = linemerge(geometry)
        if isinstance(merged, LineString):
            return tuple(merged.coords[0]), tuple(merged.coords[-1])
        else:
            # Se não puder mesclar, usar a primeira linha
            first_line = geometry.geoms[0]
            return tuple(first_line.coords[0]), tuple(first_line.coords[-1])
    return None, None

def create_network_analysis(data, output_path):
    """Realiza análise de rede na rede viária usando NetworkX."""
    print("Realizando análise de rede viária...")
    
    if data['roads'] is None:
        print("Dados da rede viária não disponíveis")
        return
    
    try:
        # Criar grafo
        G = nx.Graph()
        
        # Adicionar nós e arestas
        edges_added = 0
        for idx, row in data['roads'].iterrows():
            start, end = get_line_endpoints(row.geometry)
            if start and end:
                # Adicionar aresta com atributos
                G.add_edge(start, end,
                          length=row['length_km'],
                          road_class=row['road_class'],
                          name=row.get('name', ''))
                edges_added += 1
        
        print(f"Grafo criado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas")
        print(f"Total de arestas adicionadas: {edges_added}")
        
        # Verificar se o grafo tem nós suficientes para análise
        if G.number_of_nodes() < 2:
            print("Grafo muito pequeno para análise de rede")
            return
        
        # Calcular métricas de centralidade
        print("Calculando métricas de centralidade...")
        
        # Betweenness centrality
        betweenness = nx.betweenness_centrality(G, weight='length')
        
        # Closeness centrality
        closeness = nx.closeness_centrality(G, distance='length')
        
        # Degree centrality
        degree = nx.degree_centrality(G)
        
        # Criar visualização
        plt.figure(figsize=(15, 10))
        
        # Criar subplots para diferentes métricas
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
        
        # 1. Grafo completo com betweenness
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.2)
        nodes1 = nx.draw_networkx_nodes(G, pos, ax=ax1,
                                      node_color=list(betweenness.values()),
                                      node_size=20,
                                      cmap=plt.cm.viridis)
        ax1.set_title('Centralidade de Intermediação (Betweenness)', pad=20)
        plt.colorbar(nodes1, ax=ax1)
        
        # 2. Grafo com closeness
        nx.draw_networkx_edges(G, pos, ax=ax2, alpha=0.2)
        nodes2 = nx.draw_networkx_nodes(G, pos, ax=ax2,
                                      node_color=list(closeness.values()),
                                      node_size=20,
                                      cmap=plt.cm.viridis)
        ax2.set_title('Centralidade de Proximidade (Closeness)', pad=20)
        plt.colorbar(nodes2, ax=ax2)
        
        # 3. Grafo com degree
        nx.draw_networkx_edges(G, pos, ax=ax3, alpha=0.2)
        nodes3 = nx.draw_networkx_nodes(G, pos, ax=ax3,
                                      node_color=list(degree.values()),
                                      node_size=20,
                                      cmap=plt.cm.viridis)
        ax3.set_title('Centralidade de Grau (Degree)', pad=20)
        plt.colorbar(nodes3, ax=ax3)
        
        # 4. Estatísticas da rede
        ax4.axis('off')
        
        # Calcular componentes conectados
        connected_components = list(nx.connected_components(G))
        if connected_components:
            largest_component = max(connected_components, key=len)
            largest_subgraph = G.subgraph(largest_component)
            try:
                diameter = nx.diameter(largest_subgraph)
            except nx.NetworkXError:
                diameter = 0
        else:
            diameter = 0
        
        stats_text = f"""Estatísticas da Rede:
        
        Nós: {G.number_of_nodes():,}
        Arestas: {G.number_of_edges():,}
        Densidade: {nx.density(G):.4f}
        
        Componentes Conectados: {len(connected_components)}
        Diâmetro do Maior Componente: {diameter:.1f}
        
        Grau Médio: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}
        Coeficiente de Clustering: {nx.average_clustering(G):.4f}
        """
        ax4.text(0.1, 0.5, stats_text, fontsize=12, va='center')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Análise de rede salva em: {output_path}")
        
    except Exception as e:
        print(f"Erro na análise de rede: {str(e)}")
        print(traceback.format_exc())

def create_static_road_map(data, output_path):
    """Cria um mapa estático da rede viária com camada base."""
    print("Criando mapa estático da rede viária...")
    
    if data['roads'] is None:
        print("Dados da rede viária não disponíveis")
        return
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Reprojetar para WebMercator (EPSG:3857) para compatibilidade com camadas base
    if data['sorocaba'] is not None:
        area_bounds = data['sorocaba'].to_crs(epsg=3857).total_bounds
        area = data['sorocaba'].to_crs(epsg=3857)
        area.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5, alpha=0.8)
    else:
        area_bounds = data['roads'].to_crs(epsg=3857).total_bounds
    
    # Plotar vias por classe
    roads = data['roads'].to_crs(epsg=3857)
    
    # Definir ordem de plotagem (das menos importantes para as mais importantes)
    road_classes = ['local', 'collector', 'arterial']
    colors = ['#4daf4a', '#377eb8', '#e41a1c']
    
    for road_class, color in zip(road_classes, colors):
        subset = roads[roads['road_class'] == road_class]
        if not subset.empty:
            subset.plot(ax=ax, color=color, 
                       linewidth=1 if road_class == 'local' else 2 if road_class == 'collector' else 3,
                       alpha=0.8, zorder=10 if road_class == 'local' else 20 if road_class == 'collector' else 30,
                       label=f'Vias {road_class.title()}')
    
    # Adicionar camada base do OpenStreetMap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)
    
    # Configurar limites do mapa
    ax.set_xlim([area_bounds[0], area_bounds[2]])
    ax.set_ylim([area_bounds[1], area_bounds[3]])
    
    # Remover eixos
    ax.set_axis_off()
    
    # Adicionar título e legenda
    plt.title('Rede Viária - Sorocaba', fontsize=16, pad=20)
    plt.legend(loc='lower right')
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa estático salvo em: {output_path}")

def analyze_road_connectivity(data, output_path):
    """Analisa e visualiza a conectividade da rede viária."""
    print("Analisando conectividade da rede viária...")
    
    if data['roads'] is None:
        print("Dados da rede viária não disponíveis")
        return
    
    try:
        # Criar figura com subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
        
        # 1. Distribuição de grau dos nós
        G = nx.Graph()
        roads = data['roads']
        
        # Adicionar arestas ao grafo
        edges_added = 0
        for idx, row in roads.iterrows():
            start, end = get_line_endpoints(row.geometry)
            if start and end:
                G.add_edge(start, end)
                edges_added += 1
        
        print(f"Grafo criado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas")
        print(f"Total de arestas adicionadas: {edges_added}")
        
        if edges_added == 0:
            print("Nenhuma aresta válida para análise de conectividade")
            plt.close()
            return
        
        degrees = [d for n, d in G.degree()]
        if degrees:
            sns.histplot(degrees, ax=ax1, bins=30, kde=True)
            ax1.set_title('Distribuição de Grau dos Nós', fontsize=14)
            ax1.set_xlabel('Grau do Nó')
            ax1.set_ylabel('Frequência')
        else:
            ax1.text(0.5, 0.5, 'Sem dados de grau disponíveis', 
                    ha='center', va='center')
        
        # 2. Comprimento das vias por classe
        if 'length_km' in roads.columns and 'road_class' in roads.columns:
            sns.boxplot(data=roads, x='road_class', y='length_km', ax=ax2)
            ax2.set_title('Distribuição de Comprimento por Classe', fontsize=14)
            ax2.set_xlabel('Classe da Via')
            ax2.set_ylabel('Comprimento (km)')
        else:
            ax2.text(0.5, 0.5, 'Dados de comprimento ou classe não disponíveis', 
                    ha='center', va='center')
        
        # 3. Mapa de calor de densidade viária
        try:
            roads_proj = roads.to_crs(epsg=3857)  # Projetar para sistema métrico
            xmin, ymin, xmax, ymax = roads_proj.total_bounds
            
            # Criar grid
            cell_size = 1000  # 1km
            nx_cells = max(1, int((xmax - xmin) / cell_size))
            ny_cells = max(1, int((ymax - ymin) / cell_size))
            
            grid = np.zeros((ny_cells, nx_cells))
            
            for idx, row in roads_proj.iterrows():
                if not isinstance(row.geometry, (LineString, MultiLineString)):
                    continue
                    
                # Se for MultiLineString, usar todas as partes
                if isinstance(row.geometry, MultiLineString):
                    geometries = row.geometry.geoms
                else:
                    geometries = [row.geometry]
                
                for geom in geometries:
                    # Calcular células que a via atravessa
                    line_coords = np.array(geom.coords)
                    x_cells = ((line_coords[:, 0] - xmin) / cell_size).astype(int)
                    y_cells = ((line_coords[:, 1] - ymin) / cell_size).astype(int)
                    
                    # Incrementar células
                    for x, y in zip(x_cells, y_cells):
                        if 0 <= x < nx_cells and 0 <= y < ny_cells:
                            grid[y, x] += row.get('length_km', 1) / len(geometries)
            
            if grid.any():  # Verificar se há dados no grid
                im = ax3.imshow(grid, cmap='YlOrRd', 
                              extent=[xmin, xmax, ymin, ymax])
                ax3.set_title('Densidade da Rede Viária', fontsize=14)
                plt.colorbar(im, ax=ax3, label='Comprimento total (km)')
            else:
                ax3.text(0.5, 0.5, 'Dados insuficientes para mapa de calor', 
                        ha='center', va='center')
                
        except Exception as e:
            print(f"Erro ao criar mapa de calor: {str(e)}")
            ax3.text(0.5, 0.5, 'Erro ao criar mapa de calor', 
                    ha='center', va='center')
        
        # 4. Estatísticas de conectividade
        ax4.axis('off')
        
        if G.number_of_nodes() > 0:
            stats_text = f"""Estatísticas de Conectividade:
            
            Número de Nós: {G.number_of_nodes():,}
            Número de Arestas: {G.number_of_edges():,}
            
            Grau Médio: {np.mean(degrees):.2f}
            Grau Máximo: {max(degrees)}
            
            Componentes Conectados: {nx.number_connected_components(G)}
            Densidade do Grafo: {nx.density(G):.4f}
            
            Comprimento Total da Rede: {roads['length_km'].sum():.1f} km
            Comprimento Médio das Vias: {roads['length_km'].mean():.1f} km
            """
        else:
            stats_text = "Dados insuficientes para análise de conectividade"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, va='center')
        
        plt.tight_layout()
        
        # Salvar
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Análise de conectividade salva em: {output_path}")
        
    except Exception as e:
        print(f"Erro na análise de conectividade: {str(e)}")
        print(traceback.format_exc())

def main():
    """Função principal para criar visualizações."""
    print("\n--- Criando visualizações para rede viária ---\n")
    
    # Carregar dados
    data = load_data()
    
    # Verificar se dados foram carregados corretamente
    if all(gdf is None for gdf in data.values()):
        print("Nenhum dado viário pôde ser carregado. Verifique os arquivos de entrada.")
        return
    
    # Criar visualizações
    
    # 1. Mapa interativo da rede viária
    interactive_map_path = os.path.join(OUTPUT_DIR, 'mapa_interativo_vias.html')
    create_interactive_road_map(data, interactive_map_path)
    
    # 2. Distribuição das classes de vias
    road_dist_path = os.path.join(OUTPUT_DIR, 'distribuicao_classes_vias.png')
    plot_road_class_distribution(data, road_dist_path)
    
    # 3. Análise de rede
    network_analysis_path = os.path.join(OUTPUT_DIR, 'analise_rede_viaria.png')
    create_network_analysis(data, network_analysis_path)
    
    # 4. Mapa estático com camada base
    static_map_path = os.path.join(OUTPUT_DIR, 'mapa_estatico_vias.png')
    create_static_road_map(data, static_map_path)
    
    # 5. Análise de conectividade
    connectivity_path = os.path.join(OUTPUT_DIR, 'analise_conectividade.png')
    analyze_road_connectivity(data, connectivity_path)
    
    print(f"\nVisualizações salvas em: {OUTPUT_DIR}")
    print("Todas as visualizações foram criadas com sucesso!")

if __name__ == "__main__":
    main() 