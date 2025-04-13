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
from shapely.geometry import LineString, Point, Polygon
import contextily as ctx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import warnings
import logging
from datetime import datetime
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurar diretórios
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualizations', 'railway')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definir arquivos de entrada
RAILWAY_FILE = os.path.join(INPUT_DIR, 'railway_processed.gpkg')
SOROCABA_FILE = os.path.join(WORKSPACE_DIR, 'data', 'raw', 'sorocaba.gpkg')

def load_data():
    """Carrega os dados processados da rede ferroviária."""
    print("Carregando dados da rede ferroviária...")
    
    data = {}
    
    try:
        data['railway'] = gpd.read_file(RAILWAY_FILE)
        print(f"Ferrovias: {len(data['railway'])} registros")
    except Exception as e:
        print(f"Erro ao carregar ferrovias: {str(e)}")
        data['railway'] = None
    
    try:
        data['sorocaba'] = gpd.read_file(SOROCABA_FILE)
        print(f"Área de estudo: {len(data['sorocaba'])} registros")
    except Exception as e:
        print(f"Erro ao carregar área de estudo: {str(e)}")
        data['sorocaba'] = None
    
    # Verificar CRS e garantir que todos estejam no mesmo sistema
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

def create_railway_network_map(data, output_path):
    """Cria um mapa interativo da rede ferroviária usando Folium."""
    print("Criando mapa interativo da rede ferroviária...")
    
    # Verificar e converter dados para EPSG:4326 (WGS84) que é requerido pelo Folium
    if data['sorocaba'] is not None:
        sorocaba = data['sorocaba'].to_crs(epsg=4326)
        # Usar o centroide da área de estudo como centro do mapa
        center_lat = sorocaba.geometry.centroid.y.mean()
        center_lon = sorocaba.geometry.centroid.x.mean()
    elif data['railway'] is not None:
        # Usar o centro da rede ferroviária
        railway = data['railway'].to_crs(epsg=4326)
        center_lat = railway.geometry.centroid.y.mean()
        center_lon = railway.geometry.centroid.x.mean()
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
    
    # Adicionar estações ferroviárias como marcadores (se existissem estações nos dados)
    # Este é um exemplo, que pode ser implementado se tivermos dados de estações
    """
    if data['railway_stations'] is not None:
        stations = data['railway_stations'].to_crs(epsg=4326)
        for idx, row in stations.iterrows():
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=row['name'] if 'name' in row and pd.notna(row['name']) else "Estação ferroviária",
                icon=folium.Icon(color='red', icon='train', prefix='fa'),
                tooltip=row['name'] if 'name' in row and pd.notna(row['name']) else "Estação ferroviária"
            ).add_to(m)
    """
    
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
    
    # Adicionar rede ferroviária
    if data['railway'] is not None:
        railway = data['railway'].to_crs(epsg=4326)
        railway_json = railway.to_json()
        
        # Verificar se existem colunas relevantes para estilização
        # Poderia ser usado para diferentes tipos de ferrovia, eletrificada/não-eletrificada etc.
        style_column = None
        if 'railway' in railway.columns:
            style_column = 'railway'
        elif 'type' in railway.columns:
            style_column = 'type'
        
        if style_column:
            # Criar função de estilo com base no tipo de ferrovia
            def get_railway_style(feature):
                railway_type = feature['properties'].get(style_column, '')
                
                # Definir cores diferentes para diferentes tipos de ferrovia
                if railway_type == 'rail':
                    color = '#1f78b4'  # Azul escuro para ferrovia principal
                    weight = 3
                elif railway_type == 'tram':
                    color = '#33a02c'  # Verde para bondes
                    weight = 2
                elif railway_type == 'subway':
                    color = '#e31a1c'  # Vermelho para metrô
                    weight = 3
                elif railway_type == 'light_rail':
                    color = '#ff7f00'  # Laranja para VLT
                    weight = 2
                else:
                    color = '#6a3d9a'  # Roxo para outros tipos
                    weight = 2
                
                # Adicionar estilo de linha tracejada para ferrovias não eletrificadas
                if 'electrified' in feature['properties']:
                    if feature['properties']['electrified'] == 'no':
                        dash_array = '5, 5'  # Linha tracejada
                    else:
                        dash_array = None  # Linha contínua
                else:
                    dash_array = None
                
                return {
                    'color': color,
                    'weight': weight,
                    'opacity': 0.8,
                    'dashArray': dash_array
                }
            
            # Criar GeoJSON com estilo baseado no tipo
            folium.GeoJson(
                data=railway_json,
                name='Rede Ferroviária',
                style_function=get_railway_style,
                tooltip=folium.GeoJsonTooltip(
                    fields=['name', 'railway', 'operator', 'gauge_mm', 'electrified', 'length_km'],
                    aliases=['Nome:', 'Tipo:', 'Operador:', 'Bitola (mm):', 'Eletrificada:', 'Comprimento (km):'],
                    localize=True,
                    sticky=False
                )
            ).add_to(m)
        else:
            # Estilo simples se não houver coluna para diferenciar tipos
            folium.GeoJson(
                data=railway_json,
                name='Rede Ferroviária',
                style_function=lambda x: {
                    'color': '#1f78b4',
                    'weight': 3,
                    'opacity': 0.8
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['name', 'length_km', 'sinuosity'] if all(col in railway.columns for col in ['name', 'length_km', 'sinuosity']) else None,
                    aliases=['Nome:', 'Comprimento (km):', 'Sinuosidade:'] if all(col in railway.columns for col in ['name', 'length_km', 'sinuosity']) else None,
                    localize=True,
                    sticky=False
                )
            ).add_to(m)
    
    # Adicionar legenda
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 100px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                opacity: 0.8;
                ">
    <b>Legenda</b><br>
    <i style="background: #ffff00; opacity: 0.3; border: 1px solid black; display: inline-block; width: 18px; height: 18px;"></i> Área de Estudo<br>
    <i style="background: none; border: 3px solid #1f78b4; display: inline-block; width: 18px; height: 5px;"></i> Ferrovia<br>
    <i style="background: none; border: 3px dashed #1f78b4; display: inline-block; width: 18px; height: 5px;"></i> Ferrovia Não Eletrificada<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Adicionar controle de camadas por último para incluir todas as camadas
    folium.LayerControl().add_to(m)
    
    # Salvar mapa
    m.save(output_path)
    print(f"Mapa interativo salvo em: {output_path}")
    return output_path

def create_railway_map_with_basemap(data, output_path):
    """
    Creates a map with a basemap using contextily
    
    Parameters:
    -----------
    data: dict
        Dictionary containing the railway data
    output_path: str
        Path to save the output map
    """
    logger.info("Creating railway map with basemap")
    
    # Ensure the railway data exists
    if data['railway'] is None:
        logger.error("Railway data not available")
        return
    
    # Convert to projected CRS for proper visualization with contextily
    railway = data['railway'].to_crs(epsg=3857)
    
    # Create a figure with a larger size and higher DPI
    fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=300)
    
    # Plot the railway data on the axes
    if 'railway' in railway.columns:
        # Use railway type for coloring if available
        railway.plot(
            column='railway',
            categorical=True,
            legend=True,
            linewidth=0.8,
            ax=ax,
            cmap='Set1'
        )
    else:
        # Use default styling if railway type not available
        railway.plot(
            color='blue',
            linewidth=0.8,
            ax=ax
        )
    
    # Add a basemap
    ctx.add_basemap(
        ax, 
        source=ctx.providers.OpenStreetMap.Mapnik,
        zoom=9,
        attribution_size=8
    )
    
    # Add a title
    ax.set_title('Portuguese Railway Network with Basemap', fontsize=16)
    
    # Remove the axis
    ax.set_axis_off()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure with high quality
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.close()
    
    logger.info(f"Railway map with basemap saved to {output_path}")

def plot_railway_types_distribution(data, output_path):
    """Plota a distribuição dos tipos de ferrovia."""
    print("Criando gráfico de distribuição dos tipos de ferrovia...")
    
    if data['railway'] is None:
        print("Dados da rede ferroviária não disponíveis")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Verificar se a coluna do tipo ferroviário existe
    if 'railway' in data['railway'].columns:
        # Contar frequência de cada tipo
        railway_counts = data['railway']['railway'].value_counts().sort_index()
        
        # Definir cores para o gráfico
        colors = plt.cm.tab10(np.linspace(0, 1, len(railway_counts)))
        
        # Criar barras
        ax = railway_counts.plot(kind='bar', color=colors)
        
        # Adicionar valores acima das barras
        for i, v in enumerate(railway_counts):
            ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
        
        # Configurar gráfico
        plt.title('Distribuição dos Tipos de Ferrovia', fontsize=14)
        plt.xlabel('Tipo de Ferrovia', fontsize=12)
        plt.ylabel('Número de Segmentos', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Salvar
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de distribuição dos tipos de ferrovia salvo em: {output_path}")
    else:
        print("Coluna 'railway' não encontrada no dataset")

def plot_railway_length_distribution(data, output_path):
    """Plota a distribuição de comprimentos da rede ferroviária."""
    print("Criando gráfico de distribuição de comprimentos da rede ferroviária...")
    
    if data['railway'] is None:
        print("Dados da rede ferroviária não disponíveis")
        return
    
    if 'length_km' not in data['railway'].columns:
        print("Dados de comprimento não disponíveis")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Filtrar valores válidos
    lengths = data['railway']['length_km'].dropna()
    
    # Criar histograma
    sns.histplot(lengths, bins=20, kde=True, color='steelblue')
    
    # Adicionar linha vertical na média
    mean_length = lengths.mean()
    plt.axvline(x=mean_length, color='r', linestyle='--', alpha=0.7)
    plt.text(mean_length + 0.1, plt.ylim()[1] * 0.9, f'Média: {mean_length:.2f} km', 
             color='r', fontweight='bold')
    
    # Configurar gráfico
    plt.title('Distribuição do Comprimento dos Segmentos Ferroviários', fontsize=14)
    plt.xlabel('Comprimento (km)', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de distribuição de comprimentos salvo em: {output_path}")

def create_network_analysis(data, output_path):
    """Realiza análise de rede na rede ferroviária usando NetworkX."""
    print("Realizando análise de rede ferroviária...")
    
    if data['railway'] is None:
        print("Dados da rede ferroviária não disponíveis")
        return
    
    # Criar grafo a partir dos segmentos ferroviários
    G = nx.Graph()
    
    # Adicionar nós e arestas
    for idx, row in data['railway'].iterrows():
        if isinstance(row.geometry, LineString):
            # Extrair pontos de início e fim como identificadores de nós
            start_point = tuple(row.geometry.coords[0])
            end_point = tuple(row.geometry.coords[-1])
            
            # Adicionar nós com coordenadas
            G.add_node(start_point, pos=start_point)
            G.add_node(end_point, pos=end_point)
            
            # Adicionar aresta com atributos
            attributes = {
                'weight': row['length_km'] if 'length_km' in row and pd.notna(row['length_km']) else 1.0,
                'type': row['railway'] if 'railway' in row and pd.notna(row['railway']) else 'unknown',
                'name': row['name'] if 'name' in row and pd.notna(row['name']) else ''
            }
            
            G.add_edge(start_point, end_point, **attributes)
    
    print(f"Grafo criado com {G.number_of_nodes()} nós e {G.number_of_edges()} arestas")
    
    # Calcular métricas de centralidade
    print("Calculando métricas de centralidade...")
    
    # Betweenness centrality
    try:
        betweenness = nx.betweenness_centrality(G, weight='weight')
    except Exception as e:
        print(f"Erro ao calcular betweenness centrality: {str(e)}")
        betweenness = {node: 0 for node in G.nodes()}
    
    # Closeness centrality
    try:
        closeness = nx.closeness_centrality(G, distance='weight')
    except Exception as e:
        print(f"Erro ao calcular closeness centrality: {str(e)}")
        closeness = {node: 0 for node in G.nodes()}
    
    # Degree centrality
    degree = nx.degree_centrality(G)
    
    # Adicionar métricas aos nós
    nx.set_node_attributes(G, betweenness, 'betweenness')
    nx.set_node_attributes(G, closeness, 'closeness')
    nx.set_node_attributes(G, degree, 'degree')
    
    # Criar visualização da rede
    plt.figure(figsize=(12, 10))
    
    # Obter posições dos nós
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        pos = nx.spring_layout(G)
    
    # Desenhar arestas com espessura baseada no tipo
    edge_width = []
    edge_color = []
    for u, v, attrs in G.edges(data=True):
        if attrs.get('type') == 'rail':
            width = 2.0
            color = '#1f78b4'  # Azul para linha principal
        elif attrs.get('type') == 'tram':
            width = 1.5
            color = '#33a02c'  # Verde para bonde
        else:
            width = 1.0
            color = '#6a3d9a'  # Roxo para outros
        
        edge_width.append(width)
        edge_color.append(color)
    
    # Desenhar nós com tamanho baseado em betweenness
    node_size = [5000 * G.nodes[node].get('betweenness', 0) + 20 for node in G.nodes()]
    
    # Colorir nós com base em closeness
    node_color = [G.nodes[node].get('closeness', 0) for node in G.nodes()]
    
    # Desenhar grafo
    nx.draw_networkx_edges(G, pos, width=edge_width, edge_color=edge_color, alpha=0.6)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                                  node_color=node_color, cmap=plt.cm.viridis, alpha=0.7)
    
    # Adicionar colorbar
    plt.colorbar(nodes, label='Closeness Centrality')
    
    # Adicionar título
    plt.title('Análise de Rede - Ferrovia', fontsize=16)
    
    # Criar legendas para tipos de aresta
    edge_legend_elements = [
        Line2D([0], [0], color='#1f78b4', lw=2, label='Ferrovia Principal'),
        Line2D([0], [0], color='#33a02c', lw=1.5, label='Bonde/VLT'),
        Line2D([0], [0], color='#6a3d9a', lw=1, label='Outros')
    ]
    
    # Adicionar legenda
    plt.legend(handles=edge_legend_elements, loc='upper right', title='Tipo de Ferrovia')
    
    plt.tight_layout()
    plt.axis('off')
    
    # Salvar figura
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de rede salva em: {output_path}")
    
    return G

def analyze_railway_sinuosity(data, output_path):
    """
    Calculates the sinuosity of the railway lines and saves a histogram plot
    
    Parameters:
    -----------
    data: dict
        Dictionary containing the railway data
    output_path: str
        Path to save the output plot
    """
    logger.info("Analyzing railway sinuosity")
    
    # Check if railway data exists
    if 'railway' not in data or data['railway'] is None:
        logger.error("Railway data not available")
        return
    
    # Calculate sinuosity for each railway segment
    railway = data['railway'].copy()
    railway['length_km'] = railway.length / 1000  # Convert to kilometers
    railway['straight_dist'] = railway.geometry.apply(
        lambda g: Point(g.coords[0]).distance(Point(g.coords[-1])) / 1000
    )
    railway['sinuosity'] = railway.apply(
        lambda row: row['length_km'] / row['straight_dist'] if row['straight_dist'] > 0 else 0, 
        axis=1
    )
    
    # Filter out extreme values for better visualization
    railway = railway[railway['sinuosity'] < 5]
    
    # Create a histogram of sinuosity values
    plt.figure(figsize=(10, 6))
    sns.histplot(railway['sinuosity'], bins=20, kde=True)
    plt.title('Railway Sinuosity (Length / Straight Distance)')
    plt.xlabel('Sinuosity Index')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate and display average sinuosity
    avg_sinuosity = railway['sinuosity'].mean()
    logger.info(f"Average railway sinuosity: {avg_sinuosity:.2f}")
    
    return railway[['railway', 'length_km', 'straight_dist', 'sinuosity']]

def analyze_station_accessibility(data, output_path, distance_threshold=5):
    """
    Analyzes the accessibility of railway stations within the network.
    
    Parameters:
    -----------
    data: dict
        Dictionary containing the railway data including stations
    output_path: str
        Path to save the output analysis map
    distance_threshold: float
        Threshold distance in kilometers to consider for accessibility analysis
    
    Returns:
    --------
    GeoDataFrame
        DataFrame containing stations with accessibility metrics
    """
    logger.info(f"Analyzing station accessibility with {distance_threshold}km threshold")
    
    # Check if railway and station data exists
    if 'railway' not in data or data['railway'] is None:
        logger.error("Railway data not available")
        return
    
    # Extract railway and create station points if available
    railway = data['railway'].copy()
    
    # Create sample stations if not available in data
    # This is a placeholder - in a real scenario, you would use actual station data
    if 'stations' not in data or data['stations'] is None:
        logger.warning("No station data available, creating sample stations along the railway")
        # Create sample stations by taking points along the railway
        points = []
        names = []
        
        # Take a point every X kilometers along each segment
        for idx, line in railway.iterrows():
            geom = line.geometry
            length = line.length / 1000  # Convert to km
            
            if length > 10:  # Only create stations for longer segments
                # Place a station at the start, middle, and end of longer segments
                points.append(Point(geom.coords[0]))
                names.append(f"Station_{idx}_Start")
                
                # Middle point
                mid_idx = len(geom.coords) // 2
                points.append(Point(geom.coords[mid_idx]))
                names.append(f"Station_{idx}_Middle")
                
                # End point
                points.append(Point(geom.coords[-1]))
                names.append(f"Station_{idx}_End")
        
        # Create a GeoDataFrame with the stations
        stations = gpd.GeoDataFrame({
            'name': names,
            'geometry': points,
            'station_type': 'sample'
        }, crs=railway.crs)
    else:
        stations = data['stations'].copy()
    
    # Create a buffer around each railway line to find stations near railways
    railway_buffer = railway.copy()
    railway_buffer.geometry = railway.buffer(distance_threshold * 1000)  # Convert km to meters
    
    # Identify stations within the buffer distance of any railway
    stations['near_railway'] = stations.intersects(railway_buffer.unary_union)
    
    # Calculate distance to nearest railway for each station
    stations['distance_to_railway'] = stations.apply(
        lambda x: railway.distance(x.geometry).min() / 1000,  # Convert to km
        axis=1
    )
    
    # Calculate connectivity metrics
    # 1. Number of railway segments within buffer distance
    stations['connected_segments'] = stations.apply(
        lambda x: sum(railway.intersects(x.geometry.buffer(distance_threshold * 1000))),
        axis=1
    )
    
    # Create a visualization of the analysis
    fig, ax = plt.subplots(1, 1, figsize=(15, 15), dpi=300)
    
    # Plot the railway network
    railway.plot(ax=ax, color='blue', linewidth=0.8, alpha=0.7, label='Railway')
    
    # Plot the stations colored by accessibility
    stations.plot(
        ax=ax,
        column='distance_to_railway',
        cmap='RdYlGn_r',  # Red (far) to Green (close)
        markersize=50,
        legend=True,
        legend_kwds={'label': 'Distance to Railway (km)'}
    )
    
    # Add station labels
    for idx, row in stations.iterrows():
        ax.annotate(
            row['name'] if 'name' in stations.columns else f'Station {idx}',
            xy=(row.geometry.x, row.geometry.y),
            xytext=(3, 3),  # Offset text slightly
            textcoords='offset points',
            fontsize=8
        )
    
    # Add title and legend
    ax.set_title(f'Railway Station Accessibility (Threshold: {distance_threshold}km)', fontsize=16)
    
    # Add basemap
    ctx.add_basemap(
        ax, 
        source=ctx.providers.OpenStreetMap.Mapnik,
        zoom=9,
        attribution_size=8
    )
    
    # Remove axes
    ax.set_axis_off()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    # Log summary statistics
    accessible_stations = stations[stations['near_railway']].shape[0]
    total_stations = stations.shape[0]
    logger.info(f"Station accessibility: {accessible_stations}/{total_stations} " 
                f"({accessible_stations/total_stations*100:.1f}%) stations within {distance_threshold}km of railway")
    
    return stations

def create_railway_heatmap(data, output_path):
    """Cria um mapa de calor da rede ferroviária baseado na densidade de linhas."""
    print("Criando mapa de calor da rede ferroviária...")
    
    if data['railway'] is None:
        print("Dados da rede ferroviária não disponíveis")
        return
    
    # Verificar e converter dados para EPSG:4326 (WGS84) que é requerido pelo Folium
    if data['sorocaba'] is not None:
        sorocaba = data['sorocaba'].to_crs(epsg=4326)
        # Usar o centroide da área de estudo como centro do mapa
        center_lat = sorocaba.geometry.centroid.y.mean()
        center_lon = sorocaba.geometry.centroid.x.mean()
    else:
        railway = data['railway'].to_crs(epsg=4326)
        center_lat = railway.geometry.centroid.y.mean()
        center_lon = railway.geometry.centroid.x.mean()
    
    # Criar mapa base
    m = folium.Map(location=[center_lat, center_lon], 
                  zoom_start=13,
                  tiles='CartoDB dark_matter')
    
    # Adicionar mini mapa
    minimap = folium.plugins.MiniMap()
    m.add_child(minimap)
    
    # Gerar pontos para o mapa de calor a partir das linhas ferroviárias
    heat_data = []
    railway = data['railway'].to_crs(epsg=4326)
    
    # Extrair pontos das geometrias para criar o mapa de calor
    for _, row in railway.iterrows():
        if isinstance(row.geometry, LineString):
            # Extrair pontos ao longo da linha
            for i in range(len(row.geometry.coords)):
                # Usar pesos maiores para segmentos principais se tivermos a informação do tipo
                if 'railway' in row and isinstance(row['railway'], str):
                    weight = 3.0 if row['railway'] == 'rail' else 1.0
                else:
                    weight = 1.0
                
                # Adicionar cada ponto com seu peso
                heat_data.append([row.geometry.coords[i][1], 
                                 row.geometry.coords[i][0], 
                                 weight])
    
    # Adicionar mapa de calor
    HeatMap(
        data=heat_data,
        radius=15,
        max_zoom=13,
        blur=10,
        gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}
    ).add_to(m)
    
    # Adicionar contorno da área de estudo
    if data['sorocaba'] is not None:
        folium.GeoJson(
            data=sorocaba.to_json(),
            name='Área de Estudo',
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'white',
                'weight': 2,
                'fillOpacity': 0
            }
        ).add_to(m)
    
    # Adicionar escala
    folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)
    
    # Adicionar título
    title_html = '''
             <h3 align="center" style="font-size:16px"><b>Mapa de Calor - Rede Ferroviária</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Salvar mapa
    m.save(output_path)
    print(f"Mapa de calor ferroviário salvo em: {output_path}")
    return output_path

def main():
    """Função principal para criar visualizações."""
    print("\n--- Criando visualizações para dados ferroviários ---\n")
    
    # Carregar dados
    data = load_data()
    
    # Verificar se dados foram carregados corretamente
    if all(gdf is None for gdf in data.values()):
        print("Nenhum dado ferroviário pôde ser carregado. Verifique os arquivos de entrada.")
        return
    
    # Criar visualizações
    
    # 1. Mapa interativo da rede ferroviária
    interactive_map_path = os.path.join(OUTPUT_DIR, 'mapa_interativo_ferrovias.html')
    create_railway_network_map(data, interactive_map_path)
    
    # 2. Mapa estático com camada base
    static_map_path = os.path.join(OUTPUT_DIR, 'mapa_estatico_ferrovias.png')
    create_railway_map_with_basemap(data, static_map_path)
    
    # 3. Distribuição de tipos ferroviários (se existir a coluna)
    types_dist_path = os.path.join(OUTPUT_DIR, 'distribuicao_tipos_ferrovias.png')
    plot_railway_types_distribution(data, types_dist_path)
    
    # 4. Distribuição de comprimentos
    length_dist_path = os.path.join(OUTPUT_DIR, 'distribuicao_comprimentos_ferrovias.png')
    plot_railway_length_distribution(data, length_dist_path)
    
    # 5. Análise de rede
    network_analysis_path = os.path.join(OUTPUT_DIR, 'analise_rede_ferrovias.png')
    create_network_analysis(data, network_analysis_path)
    
    # 6. Análise de sinuosidade
    sinuosity_path = os.path.join(OUTPUT_DIR, 'analise_sinuosidade_ferrovias.png')
    analyze_railway_sinuosity(data, sinuosity_path)
    
    # 7. Análise de acessibilidade de estações
    accessibility_path = os.path.join(OUTPUT_DIR, 'analise_acessibilidade_estacoes.png')
    analyze_station_accessibility(data, accessibility_path)
    
    # 8. Mapa de calor - comentado temporariamente devido a problemas
    # heatmap_path = os.path.join(OUTPUT_DIR, 'mapa_calor_ferrovias.html')
    # create_railway_heatmap(data, heatmap_path)
    print("Nota: O mapa de calor não foi gerado devido a problemas de compatibilidade com a biblioteca.")

if __name__ == "__main__":
    main() 