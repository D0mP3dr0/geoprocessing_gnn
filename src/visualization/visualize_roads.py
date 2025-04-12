#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo simplificado para visualização de dados geoespaciais
com foco em resultados de alta qualidade para teses acadêmicas.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
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

# Configurar diretórios
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualizations', 'roads')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definir arquivos de entrada
ROADS_FILE = os.path.join(INPUT_DIR, 'roads_processed.gpkg')
SOROCABA_FILE = os.path.join(WORKSPACE_DIR, 'data', 'raw', 'sorocaba.gpkg')

def load_data():
    """Carrega os dados processados de rede viária."""
    print("\n--- Criando visualizações para rede viária ---\n")
    
    data = {}
    
    # Carregar rede viária
    try:
        roads_file = ROADS_FILE
        print(f"Tentando carregar arquivo: {roads_file}")
        data['roads'] = gpd.read_file(roads_file)
        print("Rede viária carregada:")
        print(f"- Número de registros: {len(data['roads'])}")
        print(f"- Colunas disponíveis: {', '.join(data['roads'].columns[:15])}")
        print(f"- Tipos de geometria: {data['roads'].geometry.type.unique()}")
        print(f"- CRS: {data['roads'].crs}")
        print(f"- Geometrias nulas: {data['roads'].geometry.isna().sum()}")
        print(f"- Geometrias inválidas: {(~data['roads'].geometry.is_valid).sum()}")
    except Exception as e:
        print(f"Erro ao carregar dados da rede viária: {str(e)}")
        data['roads'] = None
    
    # Carregar área de estudo (Sorocaba)
    try:
        sorocaba_file = SOROCABA_FILE
        print(f"Tentando carregar arquivo: {sorocaba_file}")
        data['sorocaba'] = gpd.read_file(sorocaba_file)
        print(f"- Geometrias inválidas: {(~data['sorocaba'].geometry.is_valid).sum()}")
    except Exception as e:
        print(f"Erro ao carregar dados da área de estudo: {str(e)}")
        data['sorocaba'] = None
    
    # Verificar sistemas de coordenadas
    print("Verificando sistemas de coordenadas...")
    for key, gdf in data.items():
        if gdf is not None:
            print(f"CRS de {key}: {gdf.crs}")
    
    # Padronizar CRS para SIRGAS 2000 (EPSG:4674)
    target_crs = "EPSG:4674"
    for key, gdf in data.items():
        if gdf is not None and gdf.crs != target_crs:
            data[key] = gdf.to_crs(target_crs)
            print(f"Reprojetado {key} para {target_crs}")
    
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
        # Usar o centro das vias
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
    
    # Adicionar escala e ferramentas de medição
    folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)
    
    # Adicionar área de estudo (Sorocaba)
    if data['sorocaba'] is not None:
        # Converter para GeoJSON para o Folium
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
        
        # Criar grupos de camadas por classe de via
        arterial_group = folium.FeatureGroup(name='Vias Arteriais')
        collector_group = folium.FeatureGroup(name='Vias Coletoras')
        local_group = folium.FeatureGroup(name='Vias Locais')
        
        if 'road_class' in roads.columns:
            # Filtrar por classe de via
            arterial_roads = roads[roads['road_class'] == 'arterial']
            collector_roads = roads[roads['road_class'] == 'collector']
            local_roads = roads[roads['road_class'] == 'local']
            
            # Adicionar vias arteriais
            if not arterial_roads.empty:
                arterial_json = arterial_roads.to_json()
                folium.GeoJson(
                    data=arterial_json,
                    name='Vias Arteriais',
                    style_function=lambda x: {
                        'color': '#e41a1c',
                        'weight': 4,
                        'opacity': 0.8
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'length_km', 'road_class'],
                        aliases=['Nome:', 'Comprimento (km):', 'Classe:'],
                        localize=True,
                        sticky=False
                    )
                ).add_to(arterial_group)
            
            # Adicionar vias coletoras
            if not collector_roads.empty:
                collector_json = collector_roads.to_json()
                folium.GeoJson(
                    data=collector_json,
                    name='Vias Coletoras',
                    style_function=lambda x: {
                        'color': '#377eb8',
                        'weight': 3,
                        'opacity': 0.7
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'length_km', 'road_class'],
                        aliases=['Nome:', 'Comprimento (km):', 'Classe:'],
                        localize=True,
                        sticky=False
                    )
                ).add_to(collector_group)
            
            # Adicionar vias locais
            if not local_roads.empty:
                local_json = local_roads.to_json()
                folium.GeoJson(
                    data=local_json,
                    name='Vias Locais',
                    style_function=lambda x: {
                        'color': '#4daf4a',
                        'weight': 2,
                        'opacity': 0.6
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'length_km', 'road_class'],
                        aliases=['Nome:', 'Comprimento (km):', 'Classe:'],
                        localize=True,
                        sticky=False
                    )
                ).add_to(local_group)
        else:
            # Se não tiver classificação, usar toda a rede
            roads_json = roads.to_json()
            folium.GeoJson(
                data=roads_json,
                name='Rede Viária',
                style_function=lambda x: {
                    'color': '#377eb8',
                    'weight': 2,
                    'opacity': 0.7
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['name', 'highway'],
                    aliases=['Nome:', 'Tipo:'],
                    localize=True,
                    sticky=False
                )
            ).add_to(m)
        
        # Adicionar grupos ao mapa
        arterial_group.add_to(m)
        collector_group.add_to(m)
        local_group.add_to(m)
    
    # Adicionar legenda
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 130px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                opacity: 0.8;
                ">
    <b>Legenda</b><br>
    <i style="background: #ffff00; opacity: 0.3; border: 1px solid black; display: inline-block; width: 18px; height: 18px;"></i> Área de Estudo<br>
    <i style="background: none; border: 4px solid #e41a1c; display: inline-block; width: 18px; height: 4px;"></i> Vias Arteriais<br>
    <i style="background: none; border: 3px solid #377eb8; display: inline-block; width: 18px; height: 3px;"></i> Vias Coletoras<br>
    <i style="background: none; border: 2px solid #4daf4a; display: inline-block; width: 18px; height: 2px;"></i> Vias Locais<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Adicionar controle de camadas
    folium.LayerControl().add_to(m)
    
    # Salvar mapa
    m.save(output_path)
    print(f"Mapa interativo salvo em: {output_path}")
    return output_path

def create_static_road_map(data, output_path):
    """Cria um mapa estático da rede viária com camada base usando Contextily."""
    print("Criando mapa estático da rede viária com camada base...")
    
    if data['roads'] is None:
        print("Dados da rede viária não disponíveis")
        return
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Reprojetar para WebMercator (EPSG:3857) para compatibilidade com camadas base
    if data['sorocaba'] is not None:
        area_bounds = data['sorocaba'].to_crs(epsg=3857).total_bounds
        area = data['sorocaba'].to_crs(epsg=3857)
        area.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5, alpha=0.8, zorder=1)
    else:
        area_bounds = data['roads'].to_crs(epsg=3857).total_bounds
    
    # Adicionar rede viária por classe
    roads = data['roads'].to_crs(epsg=3857)
    
    if 'road_class' in roads.columns:
        # Filtrar por classe de via e plotar em ordem (locais primeiro, arteriais por último)
        local_roads = roads[roads['road_class'] == 'local']
        collector_roads = roads[roads['road_class'] == 'collector']
        arterial_roads = roads[roads['road_class'] == 'arterial']
        
        # Plotar vias locais
        if not local_roads.empty:
            local_roads.plot(ax=ax, color='#4daf4a', linewidth=1.5, alpha=0.6, zorder=2)
        
        # Plotar vias coletoras
        if not collector_roads.empty:
            collector_roads.plot(ax=ax, color='#377eb8', linewidth=2.5, alpha=0.7, zorder=3)
        
        # Plotar vias arteriais
        if not arterial_roads.empty:
            arterial_roads.plot(ax=ax, color='#e41a1c', linewidth=3.5, alpha=0.8, zorder=4)
    else:
        # Se não tiver classificação, usar toda a rede com uma cor única
        roads.plot(ax=ax, color='#377eb8', linewidth=1.5, alpha=0.7, zorder=2)
    
    # Adicionar camada base do OpenStreetMap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Configurar limites do mapa baseados na área de estudo
    ax.set_xlim([area_bounds[0], area_bounds[2]])
    ax.set_ylim([area_bounds[1], area_bounds[3]])
    
    # Remover eixos
    ax.set_axis_off()
    
    # Adicionar título
    plt.title('Rede Viária - Sorocaba', fontsize=16, pad=20)
    
    # Adicionar legenda
    if 'road_class' in roads.columns:
        legend_elements = [
            Line2D([0], [0], color='#e41a1c', linewidth=3.5, label='Vias Arteriais'),
            Line2D([0], [0], color='#377eb8', linewidth=2.5, label='Vias Coletoras'),
            Line2D([0], [0], color='#4daf4a', linewidth=1.5, label='Vias Locais')
        ]
        ax.legend(handles=legend_elements, loc='lower right', title='Rede Viária')
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa estático salvo em: {output_path}")

def plot_road_class_distribution(data, output_path):
    """Plota a distribuição das classes de vias na rede viária."""
    print("Criando gráfico de distribuição das classes de vias...")
    
    if data['roads'] is None:
        print("Dados da rede viária não disponíveis")
        return
    
    if 'road_class' not in data['roads'].columns:
        print("Coluna 'road_class' não encontrada nos dados")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Contar frequência de cada classe de via
    road_class_counts = data['roads']['road_class'].value_counts().sort_values(ascending=False)
    
    # Definir cores
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    
    # Criar barras
    bars = road_class_counts.plot(
        kind='bar', 
        color=colors[:len(road_class_counts)],
        edgecolor='black',
        linewidth=1.2
    )
    
    # Adicionar valores acima das barras
    for i, v in enumerate(road_class_counts):
        plt.text(i, v + 0.1, f"{v:,}", ha='center', fontweight='bold')
    
    # Adicionar rótulos
    plt.title('Distribuição das Classes de Vias na Rede Viária', fontsize=14)
    plt.xlabel('Classe da Via', fontsize=12)
    plt.ylabel('Número de Segmentos', fontsize=12)
    
    # Adicionar grid
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar estatísticas
    total_roads = len(data['roads'])
    total_length = data['roads']['length_km'].sum()
    
    stats_text = f"Total de segmentos: {total_roads:,}\nComprimento total: {total_length:.2f} km"
    plt.figtext(0.75, 0.82, stats_text, fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.5, edgecolor='gray'))
    
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
    # Carregar dados
    data = load_data()
    
    # Verificar se dados foram carregados corretamente
    if all(gdf is None for gdf in data.values()):
        print("Nenhum dado da rede viária pôde ser carregado. Verifique os arquivos de entrada.")
        return
    
    # Criar visualizações
    
    # 1. Mapa interativo da rede viária
    interactive_map_path = os.path.join(OUTPUT_DIR, 'mapa_interativo_vias.html')
    create_interactive_road_map(data, interactive_map_path)
    
    # 2. Mapa estático da rede viária
    static_map_path = os.path.join(OUTPUT_DIR, 'mapa_estatico_vias.png')
    create_static_road_map(data, static_map_path)
    
    # 3. Distribuição das classes de vias
    road_class_dist_path = os.path.join(OUTPUT_DIR, 'distribuicao_classes_vias.png')
    plot_road_class_distribution(data, road_class_dist_path)
    
    # As análises de rede foram desabilitadas conforme solicitado
    # 4. Análise de rede
    # network_analysis_path = os.path.join(OUTPUT_DIR, 'analise_rede_vias.png')
    # create_network_analysis(data, network_analysis_path)
    
    # 5. Análise de conectividade
    # connectivity_path = os.path.join(OUTPUT_DIR, 'analise_conectividade_vias.png')
    # analyze_road_connectivity(data, connectivity_path)
    
    print(f"\nVisualizações salvas em: {OUTPUT_DIR}")
    print("Visualizações básicas foram criadas com sucesso!")
    print("Observação: As análises de rede (grafos) foram desabilitadas conforme solicitado.")

if __name__ == "__main__":
    main()