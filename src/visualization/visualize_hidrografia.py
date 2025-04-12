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
warnings.filterwarnings('ignore')

# Configurar diretórios
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualizations', 'hidrografia')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definir arquivos de entrada
TRECHO_DRENAGEM_FILE = os.path.join(INPUT_DIR, 'hidrografia_trecho_drenagem_processed.gpkg')
CURSO_DAGUA_FILE = os.path.join(INPUT_DIR, 'hidrografia_curso_dagua_processed.gpkg')
AREA_DRENAGEM_FILE = os.path.join(INPUT_DIR, 'hidrografia_area_drenagem_processed.gpkg')
PONTO_DRENAGEM_FILE = os.path.join(INPUT_DIR, 'hidrografia_ponto_drenagem_processed.gpkg')
SOROCABA_FILE = os.path.join(WORKSPACE_DIR, 'data', 'raw', 'sorocaba.gpkg')

def load_data():
    """Carrega os dados processados de hidrografia."""
    print("Carregando dados de hidrografia...")
    
    data = {}
    
    try:
        data['trecho_drenagem'] = gpd.read_file(TRECHO_DRENAGEM_FILE)
        print(f"Trechos de drenagem: {len(data['trecho_drenagem'])} registros")
    except Exception as e:
        print(f"Erro ao carregar trechos de drenagem: {str(e)}")
        data['trecho_drenagem'] = None
        
    try:
        data['curso_dagua'] = gpd.read_file(CURSO_DAGUA_FILE)
        print(f"Cursos d'água: {len(data['curso_dagua'])} registros")
    except Exception as e:
        print(f"Erro ao carregar cursos d'água: {str(e)}")
        data['curso_dagua'] = None
        
    try:
        data['area_drenagem'] = gpd.read_file(AREA_DRENAGEM_FILE)
        print(f"Áreas de drenagem: {len(data['area_drenagem'])} registros")
    except Exception as e:
        print(f"Erro ao carregar áreas de drenagem: {str(e)}")
        data['area_drenagem'] = None
        
    try:
        data['ponto_drenagem'] = gpd.read_file(PONTO_DRENAGEM_FILE)
        print(f"Pontos de drenagem: {len(data['ponto_drenagem'])} registros")
    except Exception as e:
        print(f"Erro ao carregar pontos de drenagem: {str(e)}")
        data['ponto_drenagem'] = None
    
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

def create_drainage_network_map(data, output_path):
    """Cria um mapa interativo da rede de drenagem usando Folium."""
    print("Criando mapa interativo da rede de drenagem...")
    
    # Verificar e converter dados para EPSG:4326 (WGS84) que é requerido pelo Folium
    if data['sorocaba'] is not None:
        sorocaba = data['sorocaba'].to_crs(epsg=4326)
        # Usar o centroide da área de estudo como centro do mapa
        center_lat = sorocaba.geometry.centroid.y.mean()
        center_lon = sorocaba.geometry.centroid.x.mean()
    elif data['trecho_drenagem'] is not None:
        # Usar o centro dos trechos de drenagem
        trecho_drenagem = data['trecho_drenagem'].to_crs(epsg=4326)
        center_lat = trecho_drenagem.geometry.centroid.y.mean()
        center_lon = trecho_drenagem.geometry.centroid.x.mean()
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
    
    # Adicionar pontos de drenagem como cluster de marcadores
    marker_cluster = None
    if data['ponto_drenagem'] is not None:
        ponto_drenagem = data['ponto_drenagem'].to_crs(epsg=4326)
        marker_cluster = MarkerCluster(name='Pontos de Drenagem').add_to(m)
        
        for idx, row in ponto_drenagem.iterrows():
            if pd.isna(row.geometry.y) or pd.isna(row.geometry.x):
                continue
                
            ponto_nome = row['deponto'] if 'deponto' in row and pd.notna(row['deponto']) else "Ponto de drenagem"
            
            folium.Marker(
                location=[row.geometry.y, row.geometry.x],
                popup=ponto_nome,
                icon=folium.Icon(color='blue', icon='tint', prefix='fa'),
                tooltip=ponto_nome
            ).add_to(marker_cluster)
    
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
    
    # Adicionar áreas de drenagem
    if data['area_drenagem'] is not None:
        area_drenagem = data['area_drenagem'].to_crs(epsg=4326)
        area_drenagem_json = area_drenagem.to_json()
        
        folium.GeoJson(
            data=area_drenagem_json,
            name='Áreas de Drenagem',
            style_function=lambda x: {
                'fillColor': '#0000ff',
                'color': '#000000',
                'weight': 1,
                'fillOpacity': 0.1
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['nunivotto', 'cobacia'],
                aliases=['Nível Otto:', 'Código da Bacia:'],
                localize=True,
                sticky=False
            )
        ).add_to(m)
    
    # Adicionar trechos de drenagem
    if data['trecho_drenagem'] is not None:
        trecho_drenagem = data['trecho_drenagem'].to_crs(epsg=4326)
        trecho_drenagem_json = trecho_drenagem.to_json()
        
        # Criar uma paleta de cores baseada na ordem de Strahler
        strahler_min = trecho_drenagem['nustrahler'].min()
        strahler_max = trecho_drenagem['nustrahler'].max()
        
        folium.GeoJson(
            data=trecho_drenagem_json,
            name='Trechos de Drenagem',
            style_function=lambda feature: {
                'color': get_folium_color_for_strahler(feature['properties'].get('nustrahler', 0), strahler_min, strahler_max),
                'weight': min(feature['properties'].get('nustrahler', 1) or 1, 5),
                'opacity': 0.8
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['noriocomp', 'nustrahler', 'nucomptrec'],
                aliases=['Rio:', 'Ordem Strahler:', 'Comprimento (km):'],
                localize=True,
                sticky=False
            )
        ).add_to(m)
    
    # Adicionar cursos d'água
    if data['curso_dagua'] is not None:
        curso_dagua = data['curso_dagua'].to_crs(epsg=4326)
        curso_dagua_json = curso_dagua.to_json()
        
        folium.GeoJson(
            data=curso_dagua_json,
            name="Cursos d'Água",
            style_function=lambda x: {
                'color': '#0066cc',
                'weight': 3,
                'opacity': 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['cocursodag', 'nucompcda'],
                aliases=['Código:', 'Comprimento (km):'],
                localize=True,
                sticky=False
            )
        ).add_to(m)
    
    # Adicionar legenda
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 120px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                opacity: 0.8;
                ">
    <b>Legenda</b><br>
    <i style="background: #ffff00; opacity: 0.3; border: 1px solid black; display: inline-block; width: 18px; height: 18px;"></i> Área de Estudo<br>
    <i style="background: #0000ff; opacity: 0.1; border: 1px solid black; display: inline-block; width: 18px; height: 18px;"></i> Áreas de Drenagem<br>
    <i style="background: none; border: 3px solid #0066cc; display: inline-block; width: 18px; height: 5px;"></i> Cursos d'Água<br>
    <i class="fa fa-tint fa-lg" style="color:blue"></i> Pontos de Drenagem<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Adicionar controle de camadas por último para incluir todas as camadas
    folium.LayerControl().add_to(m)
    
    # Salvar mapa
    m.save(output_path)
    print(f"Mapa interativo salvo em: {output_path}")
    return output_path

def get_folium_color_for_strahler(strahler_value, min_value, max_value):
    """Retorna cor em formato hexadecimal para valor de Strahler."""
    if pd.isna(strahler_value) or strahler_value is None:
        return '#AAAAAA'  # Cinza para valores nulos
    
    # Normalizar valor
    try:
        strahler_value = float(strahler_value)
        range_value = max_value - min_value
        if range_value == 0:
            normalized = 0.5
        else:
            normalized = (strahler_value - min_value) / range_value
            
        # Garantir que esteja no intervalo [0, 1]
        normalized = max(0, min(1, normalized))
        
        # Usar escala de azul
        # Cores mais escuras para ordens mais altas
        r = int(51 + (1-normalized) * 204)
        g = int(51 + (1-normalized) * 204)
        b = 255
        
        color = f'#{r:02x}{g:02x}{b:02x}'
        return color
    except Exception as e:
        print(f"Erro ao calcular cor para Strahler {strahler_value}: {str(e)}")
        return '#AAAAAA'  # Valor padrão em caso de erro

def get_color_for_strahler(strahler_value, min_value, max_value):
    """Retorna cor para valor de Strahler numa escala de azul (claro->escuro)."""
    if pd.isna(strahler_value):
        return '#AAAAAA'  # Cinza para valores nulos
    
    # Normalizar valor
    range_value = max_value - min_value
    if range_value == 0:
        normalized = 0.5
    else:
        normalized = (strahler_value - min_value) / range_value
    
    # Escala de azul: do mais claro (pequenos) ao mais escuro (grandes)
    return plt.cm.Blues(normalized)

def plot_strahler_order_distribution(data, output_path):
    """Plota a distribuição das ordens de Strahler na rede hidrográfica."""
    print("Criando gráfico de distribuição das ordens de Strahler...")
    
    if data['trecho_drenagem'] is None:
        print("Dados de trechos de drenagem não disponíveis")
        return
    
    plt.figure(figsize=(10, 6))
    
    # Contar frequência de cada ordem de Strahler
    strahler_counts = data['trecho_drenagem']['nustrahler'].value_counts().sort_index()
    
    # Criar barras
    ax = strahler_counts.plot(kind='bar', color=plt.cm.Blues(np.linspace(0.3, 0.9, len(strahler_counts))))
    
    # Adicionar valores acima das barras
    for i, v in enumerate(strahler_counts):
        ax.text(i, v + 0.1, str(v), ha='center', fontweight='bold')
    
    # Configurar gráfico
    plt.title('Distribuição das Ordens de Strahler na Rede Hidrográfica', fontsize=14)
    plt.xlabel('Ordem de Strahler', fontsize=12)
    plt.ylabel('Número de Trechos', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de distribuição das ordens de Strahler salvo em: {output_path}")

def plot_stream_length_by_strahler(data, output_path):
    """Plota o comprimento médio dos rios para cada ordem de Strahler."""
    print("Criando gráfico de comprimento médio dos rios por ordem de Strahler...")
    
    if data['trecho_drenagem'] is None:
        print("Dados de trechos de drenagem não disponíveis")
        return
    
    # Calcular comprimento médio por ordem
    strahler_groups = data['trecho_drenagem'].groupby('nustrahler')['nucomptrec'].agg(['mean', 'std', 'count']).reset_index()
    
    plt.figure(figsize=(10, 6))
    
    # Gráfico de barras com barras de erro
    plt.errorbar(
        strahler_groups['nustrahler'], 
        strahler_groups['mean'],
        yerr=strahler_groups['std'],
        fmt='o',
        capsize=5,
        ecolor='#888888',
        markersize=8,
        color='#1f77b4'
    )
    
    # Adicionar linha de tendência
    plt.plot(strahler_groups['nustrahler'], strahler_groups['mean'], 'b--', alpha=0.7)
    
    # Configurar gráfico
    plt.title('Comprimento Médio dos Trechos por Ordem de Strahler', fontsize=14)
    plt.xlabel('Ordem de Strahler', fontsize=12)
    plt.ylabel('Comprimento Médio (km)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(strahler_groups['nustrahler'])
    
    # Adicionar valores
    for i, row in strahler_groups.iterrows():
        plt.text(row['nustrahler'], row['mean'] + row['std'] + 0.1, 
                 f"{row['mean']:.2f} km", 
                 ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de comprimento médio por ordem de Strahler salvo em: {output_path}")

def create_network_analysis(data, output_path):
    """Realiza análise de rede na rede de drenagem usando NetworkX."""
    print("Realizando análise de rede hidrográfica...")
    
    if data['trecho_drenagem'] is None:
        print("Dados de trechos de drenagem não disponíveis")
        return
    
    # Criar grafo a partir dos trechos
    G = nx.Graph()
    
    # Adicionar nós e arestas
    for idx, row in data['trecho_drenagem'].iterrows():
        if pd.notna(row['noorigem']) and pd.notna(row['nodestino']):
            origem = int(row['noorigem'])
            destino = int(row['nodestino'])
            
            # Adicionar nós com coordenadas se disponíveis
            if isinstance(row.geometry, LineString):
                G.add_node(origem, pos=(row.geometry.coords[0][0], row.geometry.coords[0][1]))
                G.add_node(destino, pos=(row.geometry.coords[-1][0], row.geometry.coords[-1][1]))
                
                # Adicionar aresta com atributos
                G.add_edge(origem, destino, 
                          weight=row['nucomptrec'] if pd.notna(row['nucomptrec']) else 1,
                          strahler=row['nustrahler'] if pd.notna(row['nustrahler']) else 1,
                          name=row['noriocomp'] if pd.notna(row['noriocomp']) else '')
    
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
    
    # Desenhar arestas com espessura baseada em Strahler
    edge_width = [G[u][v].get('strahler', 1) for u, v in G.edges()]
    
    # Desenhar nós com tamanho baseado em betweenness
    node_size = [1000 * G.nodes[node].get('betweenness', 0) + 15 for node in G.nodes()]
    
    # Colorir nós com base em closeness
    node_color = [G.nodes[node].get('closeness', 0) for node in G.nodes()]
    
    # Desenhar grafo
    nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.5, edge_color='blue')
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, 
                                  node_color=node_color, cmap=plt.cm.viridis, alpha=0.7)
    
    # Adicionar colorbar
    plt.colorbar(nodes, label='Closeness Centrality')
    
    # Adicionar título e legendas
    plt.title('Análise de Rede - Hidrografia', fontsize=16)
    
    # Criar legendas personalizadas
    node_legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#6a0dad', label='Nós: Junções de Drenagem',
              markersize=10),
    ]
    
    edge_legend_elements = [
        Line2D([0], [0], color='blue', lw=1, label='Ordem 1'),
        Line2D([0], [0], color='blue', lw=2, label='Ordem 2'),
        Line2D([0], [0], color='blue', lw=3, label='Ordem 3+')
    ]
    
    # Adicionar duas legendas
    plt.legend(handles=node_legend_elements, loc='upper left', title='Nós')
    plt.legend(handles=edge_legend_elements, loc='upper right', title='Trechos (Ordem Strahler)')
    
    plt.tight_layout()
    plt.axis('off')
    
    # Salvar figura
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de rede salva em: {output_path}")
    
    return G

def create_drainage_map_with_basemap(data, output_path):
    """Cria um mapa estático de alta qualidade com camada base usando Contextily."""
    print("Criando mapa estático da rede de drenagem com camada base...")
    
    if data['trecho_drenagem'] is None:
        print("Dados de trechos de drenagem não disponíveis")
        return
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Reprojetar para WebMercator (EPSG:3857) para compatibilidade com camadas base
    if data['sorocaba'] is not None:
        area_bounds = data['sorocaba'].to_crs(epsg=3857).total_bounds
        area = data['sorocaba'].to_crs(epsg=3857)
        area.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5, alpha=0.8)
    else:
        area_bounds = data['trecho_drenagem'].to_crs(epsg=3857).total_bounds
    
    # Adicionar áreas de drenagem
    if data['area_drenagem'] is not None:
        areas = data['area_drenagem'].to_crs(epsg=3857)
        areas.plot(ax=ax, column='nunivotto', cmap='Blues', alpha=0.5, 
                  legend=True, legend_kwds={'label': 'Nível Otto Pfafstetter'})
    
    # Adicionar trechos de drenagem com gradiente baseado na ordem de Strahler
    if data['trecho_drenagem'] is not None:
        trechos = data['trecho_drenagem'].to_crs(epsg=3857)
        # Criar coluna de LineWidth baseada em Strahler
        trechos['linewidth'] = trechos['nustrahler'].fillna(1).apply(lambda x: min(x, 5) * 0.5)
        
        # Plotar por ordem crescente para visualização adequada (menores embaixo, maiores em cima)
        for order in range(int(trechos['nustrahler'].min()), int(trechos['nustrahler'].max()) + 1):
            subset = trechos[trechos['nustrahler'] == order]
            if not subset.empty:
                subset.plot(ax=ax, linewidth=subset['linewidth'], 
                           color=plt.cm.Blues(0.3 + (order * 0.7 / trechos['nustrahler'].max())),
                           alpha=0.8, zorder=order+10)
    
    # Adicionar pontos de drenagem
    if data['ponto_drenagem'] is not None:
        pontos = data['ponto_drenagem'].to_crs(epsg=3857)
        pontos.plot(ax=ax, color='red', markersize=30, marker='o', alpha=0.7, zorder=20)
        
        # Adicionar labels para pontos
        if 'deponto' in pontos.columns:
            for idx, row in pontos.iterrows():
                if pd.notna(row['deponto']):
                    ax.annotate(row['deponto'], 
                               xy=(row.geometry.x, row.geometry.y),
                               xytext=(5, 5),
                               textcoords="offset points",
                               fontsize=7, color='black',
                               backgroundcolor='white', zorder=21)
    
    # Adicionar camada base do OpenStreetMap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)
    
    # Configurar limites do mapa baseados na área de estudo
    ax.set_xlim([area_bounds[0], area_bounds[2]])
    ax.set_ylim([area_bounds[1], area_bounds[3]])
    
    # Remover eixos
    ax.set_axis_off()
    
    # Adicionar título e escala
    plt.title('Rede de Drenagem - Sorocaba', fontsize=16, pad=20)
    
    # Adicionar legenda para ordem de Strahler
    strahler_colors = [plt.cm.Blues(0.3 + (i * 0.7 / int(trechos['nustrahler'].max()))) 
                      for i in range(1, int(trechos['nustrahler'].max()) + 1)]
    
    strahler_patches = [
        Line2D([0], [0], color=color, linewidth=min(order, 5) * 0.5 + 1, label=f'Ordem {order}')
        for order, color in enumerate(strahler_colors, start=1)
    ]
    
    plt.legend(handles=strahler_patches, loc='lower right', title='Rede de Drenagem (Ordem de Strahler)')
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa estático salvo em: {output_path}")

def analyze_river_drainage_areas(data, output_path):
    """Analisa e visualiza a distribuição de áreas de drenagem."""
    print("Analisando áreas de drenagem...")
    
    if data['area_drenagem'] is None:
        print("Dados de áreas de drenagem não disponíveis")
        return
    
    # Calcular área das bacias em km²
    data['area_drenagem']['area_km2'] = data['area_drenagem'].geometry.area / 1_000_000
    
    # Criar figura com subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Histograma de distribuição de áreas
    sns.histplot(data['area_drenagem']['area_km2'], ax=ax1, bins=20, kde=True, color='skyblue')
    ax1.set_title('Distribuição das Áreas de Drenagem', fontsize=14)
    ax1.set_xlabel('Área (km²)', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Boxplot por nível Otto
    if 'nunivotto' in data['area_drenagem'].columns:
        # Agrupar por nível Otto e calcular estatísticas
        otto_stats = data['area_drenagem'].groupby('nunivotto')['area_km2'].agg(['mean', 'median', 'std', 'count']).reset_index()
        
        sns.boxplot(x='nunivotto', y='area_km2', data=data['area_drenagem'], ax=ax2, palette='Blues')
        ax2.set_title('Área por Nível Otto Pfafstetter', fontsize=14)
        ax2.set_xlabel('Nível Otto', fontsize=12)
        ax2.set_ylabel('Área (km²)', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Adicionar estatísticas como tabela
        table_data = otto_stats[['nunivotto', 'mean', 'median', 'count']]
        table_data.columns = ['Nível', 'Média (km²)', 'Mediana (km²)', 'Contagem']
        table_data = table_data.round(2)
        
        # Criar tabela de estatísticas embaixo do boxplot
        plt.figtext(0.5, 0.01, table_data.to_string(index=False), 
                   ha='center', fontsize=10, family='monospace')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Ajustar espaço para a tabela
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de áreas de drenagem salva em: {output_path}")

def create_sinuosity_analysis(data, output_path):
    """Analisa e visualiza a sinuosidade dos rios."""
    print("Analisando sinuosidade dos rios...")
    
    if data['trecho_drenagem'] is None:
        print("Dados de trechos de drenagem não disponíveis")
        return
    
    # Calcular sinuosidade para cada trecho (comprimento real / distância em linha reta)
    trechos = data['trecho_drenagem'].copy()
    
    # Calcular sinuosidade para linhas
    sinuosities = []
    for geom in trechos.geometry:
        if isinstance(geom, LineString) and len(geom.coords) >= 2:
            # Comprimento real do rio
            real_length = geom.length
            
            # Distância em linha reta do ponto inicial ao final
            start_point = Point(geom.coords[0])
            end_point = Point(geom.coords[-1])
            straight_length = start_point.distance(end_point)
            
            # Sinuosidade (valor de 1 = linha reta, maior que 1 = sinuoso)
            if straight_length > 0:
                sinuosity = real_length / straight_length
            else:
                sinuosity = 1.0
        else:
            sinuosity = np.nan
        
        sinuosities.append(sinuosity)
    
    trechos['sinuosity'] = sinuosities
    
    # Figura com subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Histograma de sinuosidade
    sns.histplot(trechos['sinuosity'].dropna(), ax=ax1, bins=20, kde=True, color='forestgreen')
    ax1.set_title('Distribuição de Sinuosidade dos Rios', fontsize=14)
    ax1.set_xlabel('Índice de Sinuosidade', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar linhas de referência e anotações
    ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7)
    ax1.axvline(x=1.5, color='orange', linestyle='--', alpha=0.7)
    ax1.text(1.0, ax1.get_ylim()[1]*0.9, 'Retilíneo', rotation=90, color='red', alpha=0.8)
    ax1.text(1.5, ax1.get_ylim()[1]*0.9, 'Sinuoso', rotation=90, color='orange', alpha=0.8)
    
    # 2. Gráfico de sinuosidade por ordem de Strahler
    if 'nustrahler' in trechos.columns:
        # Calcular estatísticas por ordem de Strahler
        strahler_stats = trechos.groupby('nustrahler')['sinuosity'].agg(['mean', 'median', 'std']).reset_index()
        
        # Plotar médias com barras de erro
        ax2.errorbar(
            strahler_stats['nustrahler'], 
            strahler_stats['mean'],
            yerr=strahler_stats['std'],
            fmt='o-',
            capsize=5,
            ecolor='#888888',
            markersize=8,
            color='forestgreen'
        )
        
        ax2.set_title('Sinuosidade por Ordem de Strahler', fontsize=14)
        ax2.set_xlabel('Ordem de Strahler', fontsize=12)
        ax2.set_ylabel('Sinuosidade Média', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_xticks(strahler_stats['nustrahler'].values)
        
        # Adicionar linhas de referência
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=1.5, color='orange', linestyle='--', alpha=0.7)
        
        # Adicionar valores
        for i, row in strahler_stats.iterrows():
            ax2.text(row['nustrahler'], row['mean'] + 0.05, 
                    f"{row['mean']:.2f}", 
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de sinuosidade salva em: {output_path}")

def main():
    """Função principal para criar visualizações."""
    print("\n--- Criando visualizações para dados hidrográficos ---\n")
    
    # Carregar dados
    data = load_data()
    
    # Verificar se dados foram carregados corretamente
    if all(gdf is None for gdf in data.values()):
        print("Nenhum dado hidrográfico pôde ser carregado. Verifique os arquivos de entrada.")
        return
    
    # Criar visualizações
    
    # 1. Mapa interativo da rede de drenagem
    interactive_map_path = os.path.join(OUTPUT_DIR, 'mapa_interativo_hidrografia.html')
    create_drainage_network_map(data, interactive_map_path)
    
    # 2. Distribuição das ordens de Strahler
    strahler_dist_path = os.path.join(OUTPUT_DIR, 'distribuicao_strahler.png')
    plot_strahler_order_distribution(data, strahler_dist_path)
    
    # 3. Comprimento por ordem de Strahler
    length_by_strahler_path = os.path.join(OUTPUT_DIR, 'comprimento_por_strahler.png')
    plot_stream_length_by_strahler(data, length_by_strahler_path)
    
    # 4. Análise de rede
    network_analysis_path = os.path.join(OUTPUT_DIR, 'analise_rede_hidrografia.png')
    create_network_analysis(data, network_analysis_path)
    
    # 5. Mapa estático com camada base
    static_map_path = os.path.join(OUTPUT_DIR, 'mapa_estatico_hidrografia.png')
    create_drainage_map_with_basemap(data, static_map_path)
    
    # 6. Análise de áreas de drenagem
    drainage_areas_path = os.path.join(OUTPUT_DIR, 'analise_areas_drenagem.png')
    analyze_river_drainage_areas(data, drainage_areas_path)
    
    # 7. Análise de sinuosidade
    sinuosity_path = os.path.join(OUTPUT_DIR, 'analise_sinuosidade.png')
    create_sinuosity_analysis(data, sinuosity_path)
    
    print(f"\nVisualizations salvas em: {OUTPUT_DIR}")
    print("Todas as visualizações foram criadas com sucesso!")

if __name__ == "__main__":
    main() 