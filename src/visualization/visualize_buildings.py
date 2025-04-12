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
from shapely.geometry import Polygon, MultiPolygon, Point
import contextily as ctx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import warnings
from matplotlib.colors import LinearSegmentedColormap
import sklearn.metrics as skm
import warnings
warnings.filterwarnings('ignore')

# Configurar diretórios
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualizations', 'buildings')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definir arquivos de entrada
BUILDINGS_FILE = os.path.join(INPUT_DIR, 'buildings_processed.gpkg')
SOROCABA_FILE = os.path.join(WORKSPACE_DIR, 'data', 'raw', 'sorocaba.gpkg')

def load_data():
    """Carrega os dados processados de edifícios."""
    print("Carregando dados de edifícios...")
    
    data = {}
    
    try:
        data['buildings'] = gpd.read_file(BUILDINGS_FILE)
        print(f"Edifícios: {len(data['buildings'])} registros")
    except Exception as e:
        print(f"Erro ao carregar edifícios: {str(e)}")
        data['buildings'] = None
        
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

def create_interactive_buildings_map(data, output_path):
    """Cria um mapa interativo dos edifícios usando Folium."""
    print("Criando mapa interativo dos edifícios...")
    
    # Verificar e converter dados para EPSG:4326 (WGS84) que é requerido pelo Folium
    if data['sorocaba'] is not None:
        sorocaba = data['sorocaba'].to_crs(epsg=4326)
        # Usar o centroide da área de estudo como centro do mapa
        center_lat = sorocaba.geometry.centroid.y.mean()
        center_lon = sorocaba.geometry.centroid.x.mean()
    elif data['buildings'] is not None:
        # Usar o centro dos edifícios
        buildings = data['buildings'].to_crs(epsg=4326)
        center_lat = buildings.geometry.centroid.y.mean()
        center_lon = buildings.geometry.centroid.x.mean()
    else:
        print("Dados insuficientes para criar o mapa")
        return
    
    # Criar mapa base
    m = folium.Map(location=[center_lat, center_lon], 
                  zoom_start=14,
                  tiles='CartoDB positron')
    
    # Adicionar mini mapa
    minimap = folium.plugins.MiniMap()
    m.add_child(minimap)
    
    # Adicionar escala
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
    
    # Adicionar edifícios
    if data['buildings'] is not None:
        buildings = data['buildings'].to_crs(epsg=4326)
        
        # Criar grupos para diferentes classes de edifícios
        building_groups = {}
        
        if 'building_class' in buildings.columns:
            # Agrupar por classe de edifício
            for building_class in buildings['building_class'].unique():
                if pd.notna(building_class):
                    building_groups[building_class] = folium.FeatureGroup(name=f'Edifícios {building_class.title()}')
            
            # Adicionar edifícios por classe
            for building_class, group in building_groups.items():
                class_buildings = buildings[buildings['building_class'] == building_class]
                
                # Definir estilo por classe
                style = get_building_style(building_class)
                
                # Limitar o número de edifícios para melhor performance
                if len(class_buildings) > 5000:
                    print(f"Amostrando edifícios da classe '{building_class}' (de {len(class_buildings)} para 5000)")
                    class_buildings = class_buildings.sample(5000)
                
                folium.GeoJson(
                    data=class_buildings.to_json(),
                    name=f'Edifícios {building_class.title()}',
                    style_function=lambda x, style=style: style,
                    tooltip=folium.GeoJsonTooltip(
                        fields=['name', 'building', 'height', 'levels', 'area_m2'],
                        aliases=['Nome:', 'Tipo:', 'Altura (m):', 'Andares:', 'Área (m²):'],
                        localize=True,
                        sticky=False
                    )
                ).add_to(group)
                
                group.add_to(m)
        else:
            # Se não houver classificação, adicionar todos os edifícios
            building_sample = buildings
            
            # Limitar para evitar problemas de performance
            if len(building_sample) > 10000:
                print(f"Amostrando edifícios (de {len(buildings)} para 10000)")
                building_sample = buildings.sample(10000)
            
            folium.GeoJson(
                data=building_sample.to_json(),
                name='Edifícios',
                style_function=lambda x: {
                    'fillColor': '#3388ff',
                    'color': '#333333',
                    'weight': 1,
                    'fillOpacity': 0.5
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['name', 'building', 'height', 'levels', 'area_m2'],
                    aliases=['Nome:', 'Tipo:', 'Altura (m):', 'Andares:', 'Área (m²):'],
                    localize=True,
                    sticky=False
                )
            ).add_to(m)
    
    # Adicionar legenda
    legend_html = create_building_legend_html()
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Adicionar controle de camadas
    folium.LayerControl().add_to(m)
    
    # Salvar mapa
    m.save(output_path)
    print(f"Mapa interativo salvo em: {output_path}")
    return output_path

def get_building_style(building_class):
    """Retorna estilo para cada classe de edifício."""
    styles = {
        'residential': {
            'fillColor': '#ff8080',  # Vermelho claro
            'color': '#800000',  # Bordas em vermelho escuro
            'weight': 1,
            'fillOpacity': 0.5
        },
        'commercial': {
            'fillColor': '#80b3ff',  # Azul claro
            'color': '#000080',  # Bordas em azul escuro
            'weight': 1,
            'fillOpacity': 0.5
        },
        'industrial': {
            'fillColor': '#b366ff',  # Roxo
            'color': '#4d0099',  # Bordas em roxo escuro
            'weight': 1,
            'fillOpacity': 0.5
        },
        'office': {
            'fillColor': '#66ffcc',  # Verde água
            'color': '#009973',  # Bordas em verde escuro
            'weight': 1,
            'fillOpacity': 0.5
        },
        'educational': {
            'fillColor': '#ffdb4d',  # Amarelo
            'color': '#806600',  # Bordas em amarelo escuro
            'weight': 1,
            'fillOpacity': 0.5
        },
        'healthcare': {
            'fillColor': '#ff3333',  # Vermelho
            'color': '#800000',  # Bordas em vermelho escuro
            'weight': 1,
            'fillOpacity': 0.5
        },
        'religious': {
            'fillColor': '#ffffff',  # Branco
            'color': '#000000',  # Bordas em preto
            'weight': 1,
            'fillOpacity': 0.5
        },
        'leisure': {
            'fillColor': '#ff80ff',  # Rosa
            'color': '#800080',  # Bordas em roxo escuro
            'weight': 1,
            'fillOpacity': 0.5
        },
        'utility': {
            'fillColor': '#bfbfbf',  # Cinza
            'color': '#4d4d4d',  # Bordas em cinza escuro
            'weight': 1,
            'fillOpacity': 0.5
        },
        'accommodation': {
            'fillColor': '#ff9933',  # Laranja
            'color': '#804d00',  # Bordas em laranja escuro
            'weight': 1,
            'fillOpacity': 0.5
        },
        'construction': {
            'fillColor': '#ffff80',  # Amarelo claro
            'color': '#808000',  # Bordas em amarelo esverdeado
            'weight': 1,
            'fillOpacity': 0.5
        },
        'abandoned': {
            'fillColor': '#bfbfbf',  # Cinza
            'color': '#4d4d4d',  # Bordas em cinza escuro
            'weight': 1,
            'fillOpacity': 0.5
        }
    }
    
    return styles.get(building_class, {
        'fillColor': '#3388ff',  # Azul padrão
        'color': '#333333',  # Bordas em cinza escuro
        'weight': 1,
        'fillOpacity': 0.5
    })

def create_building_legend_html():
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
    <i style="background: #ff8080; opacity: 0.5; border: 1px solid #800000; display: inline-block; width: 18px; height: 18px;"></i> Edifícios Residenciais<br>
    <i style="background: #80b3ff; opacity: 0.5; border: 1px solid #000080; display: inline-block; width: 18px; height: 18px;"></i> Edifícios Comerciais<br>
    <i style="background: #b366ff; opacity: 0.5; border: 1px solid #4d0099; display: inline-block; width: 18px; height: 18px;"></i> Edifícios Industriais<br>
    <i style="background: #ffdb4d; opacity: 0.5; border: 1px solid #806600; display: inline-block; width: 18px; height: 18px;"></i> Edifícios Educacionais<br>
    <i style="background: #ff3333; opacity: 0.5; border: 1px solid #800000; display: inline-block; width: 18px; height: 18px;"></i> Edifícios de Saúde<br>
    <i style="background: #ffffff; opacity: 0.5; border: 1px solid #000000; display: inline-block; width: 18px; height: 18px;"></i> Edifícios Religiosos<br>
    </div>
    """

def plot_building_class_distribution(data, output_path):
    """Plota a distribuição das classes de edifícios."""
    print("Criando gráfico de distribuição das classes de edifícios...")
    
    if data['buildings'] is None:
        print("Dados de edifícios não disponíveis")
        return
    
    if 'building_class' not in data['buildings'].columns:
        print("Coluna 'building_class' não disponível")
        return
    
    # Contar frequência de cada classe
    buildings = data['buildings']
    class_counts = buildings['building_class'].value_counts()
    
    # Calcular área por classe
    class_areas = buildings.groupby('building_class')['area_m2'].sum() / 1000  # Em milhares de m²
    
    # Criar figura com dois subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Gráfico de contagem de edifícios por classe
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_counts)))
    bars1 = ax1.bar(class_counts.index, class_counts.values, color=colors)
    ax1.set_title('Distribuição por Número de Edifícios', fontsize=14)
    ax1.set_xlabel('Classe de Edifício')
    ax1.set_ylabel('Número de Edifícios')
    ax1.tick_params(axis='x', rotation=45)
    
    # Adicionar valores sobre as barras
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}',
                 ha='center', va='bottom')
    
    # 2. Gráfico de área total por classe
    bars2 = ax2.bar(class_areas.index, class_areas.values, color=colors)
    ax2.set_title('Distribuição por Área Total (milhares de m²)', fontsize=14)
    ax2.set_xlabel('Classe de Edifício')
    ax2.set_ylabel('Área Total (x1000 m²)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Adicionar valores sobre as barras
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}',
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de distribuição de classes de edifícios salvo em: {output_path}")

def analyze_building_heights(data, output_path):
    """Analisa e visualiza a distribuição de alturas dos edifícios."""
    print("Analisando altura dos edifícios...")
    
    if data['buildings'] is None:
        print("Dados de edifícios não disponíveis")
        return
    
    buildings = data['buildings']
    
    # Verificar se existem dados de altura
    if 'height' not in buildings.columns or 'levels' not in buildings.columns:
        print("Dados de altura ou andares não disponíveis")
        return
    
    # Remover valores nulos
    buildings_with_height = buildings.dropna(subset=['height'])
    buildings_with_levels = buildings.dropna(subset=['levels'])
    
    if len(buildings_with_height) == 0 and len(buildings_with_levels) == 0:
        print("Não há dados suficientes de altura ou andares")
        return
    
    # Criar figura com subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Histograma de alturas
    if len(buildings_with_height) > 0:
        sns.histplot(buildings_with_height['height'], bins=30, kde=True, ax=ax1)
        ax1.set_title('Distribuição de Altura dos Edifícios', fontsize=14)
        ax1.set_xlabel('Altura (m)')
        ax1.set_ylabel('Frequência')
        
        # Adicionar linhas verticais para quartis
        height_q1 = buildings_with_height['height'].quantile(0.25)
        height_median = buildings_with_height['height'].median()
        height_q3 = buildings_with_height['height'].quantile(0.75)
        
        ax1.axvline(height_median, color='r', linestyle='-', label=f'Mediana: {height_median:.1f}m')
        ax1.axvline(height_q1, color='g', linestyle='--', label=f'Q1: {height_q1:.1f}m')
        ax1.axvline(height_q3, color='g', linestyle='--', label=f'Q3: {height_q3:.1f}m')
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'Dados de altura não disponíveis', ha='center', va='center')
    
    # 2. Histograma de número de andares
    if len(buildings_with_levels) > 0:
        sns.histplot(buildings_with_levels['levels'], bins=range(1, int(buildings_with_levels['levels'].max())+2),
                    kde=False, discrete=True, ax=ax2)
        ax2.set_title('Distribuição do Número de Andares', fontsize=14)
        ax2.set_xlabel('Número de Andares')
        ax2.set_ylabel('Frequência')
        
        # Mostrar números inteiros no eixo x
        ax2.set_xticks(range(1, int(buildings_with_levels['levels'].max())+1))
    else:
        ax2.text(0.5, 0.5, 'Dados de andares não disponíveis', ha='center', va='center')
    
    # 3. Boxplot de altura por classe de edifício
    if 'building_class' in buildings.columns and len(buildings_with_height) > 0:
        sns.boxplot(x='building_class', y='height', data=buildings_with_height, ax=ax3)
        ax3.set_title('Altura por Classe de Edifício', fontsize=14)
        ax3.set_xlabel('Classe de Edifício')
        ax3.set_ylabel('Altura (m)')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Dados de altura por classe não disponíveis', ha='center', va='center')
    
    # 4. Relação entre número de andares e altura
    if len(buildings_with_height) > 0 and len(buildings_with_levels) > 0:
        # Filtra apenas edifícios que têm ambos os valores
        buildings_complete = buildings.dropna(subset=['height', 'levels'])
        
        if len(buildings_complete) > 0:
            sns.scatterplot(x='levels', y='height', data=buildings_complete, 
                           alpha=0.5, ax=ax4)
            
            # Adicionar linha de tendência
            try:
                # Calcular regressão linear
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    buildings_complete['levels'], buildings_complete['height'])
                
                # Plotar linha de regressão
                x = np.array([buildings_complete['levels'].min(), buildings_complete['levels'].max()])
                ax4.plot(x, intercept + slope * x, 'r', 
                        label=f'y = {slope:.2f}x + {intercept:.2f} (R² = {r_value**2:.2f})')
                ax4.legend()
            except:
                pass
            
            ax4.set_title('Relação entre Número de Andares e Altura', fontsize=14)
            ax4.set_xlabel('Número de Andares')
            ax4.set_ylabel('Altura (m)')
            
            # Mostrar números inteiros no eixo x
            ax4.set_xticks(range(1, int(buildings_complete['levels'].max())+1))
        else:
            ax4.text(0.5, 0.5, 'Dados insuficientes para correlação', ha='center', va='center')
    else:
        ax4.text(0.5, 0.5, 'Dados de altura e andares não disponíveis', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de altura dos edifícios salva em: {output_path}")

def create_building_density_heatmap(data, output_path):
    """Cria um mapa de calor da densidade de edifícios."""
    print("Criando mapa de calor de densidade de edifícios...")
    
    if data['buildings'] is None or data['sorocaba'] is None:
        print("Dados de edifícios ou área de estudo não disponíveis")
        return
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Reprojetar para WebMercator (EPSG:3857) para compatibilidade com camadas base
    sorocaba = data['sorocaba'].to_crs(epsg=3857)
    buildings = data['buildings'].to_crs(epsg=3857)
    
    # Plotar área de estudo
    sorocaba.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5)
    
    # Extrair centroides dos edifícios
    centroids = buildings.copy()
    centroids['geometry'] = buildings.geometry.centroid
    
    # Criar mapa de calor usando KDE (Kernel Density Estimation)
    x = centroids.geometry.x
    y = centroids.geometry.y
    
    # Limites para o mapa de calor
    xmin, ymin, xmax, ymax = sorocaba.total_bounds
    
    # Criar grade para densidade
    h = (ymax - ymin) / 100  # Tamanho da célula
    xi = np.arange(xmin, xmax, h)
    yi = np.arange(ymin, ymax, h)
    xi, yi = np.meshgrid(xi, yi)
    
    # Calcular densidade usando statsmodels KDE
    try:
        from scipy.stats import gaussian_kde
        
        # Criar KDE
        positions = np.vstack([xi.ravel(), yi.ravel()])
        values = np.vstack([x, y])
        kernel = gaussian_kde(values)
        
        # Calcular densidade
        z = np.reshape(kernel(positions).T, xi.shape)
        
        # Plotar mapa de calor
        heat = ax.imshow(z, cmap=plt.cm.hot_r, alpha=0.6,
                       extent=[xmin, xmax, ymin, ymax],
                       origin='lower', aspect='auto')
        
        # Adicionar barra de cores
        plt.colorbar(heat, ax=ax, label='Densidade de Edifícios')
        
    except Exception as e:
        print(f"Erro ao criar mapa de calor: {str(e)}")
        # Alternativa: plotar pontos simples
        centroids.plot(ax=ax, markersize=1, color='red', alpha=0.5)
    
    # Adicionar camada base do OpenStreetMap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Configurar limites do mapa
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    
    # Remover eixos
    ax.set_axis_off()
    
    # Adicionar título
    plt.title('Densidade de Edifícios em Sorocaba', fontsize=16, pad=20)
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa de calor de densidade salvo em: {output_path}")

def analyze_building_morphology(data, output_path):
    """Analisa a morfologia dos edifícios (forma, compacidade, área)."""
    print("Analisando morfologia dos edifícios...")
    
    if data['buildings'] is None:
        print("Dados de edifícios não disponíveis")
        return
    
    buildings = data['buildings']
    
    # Verificar se existem dados de área e compacidade
    if 'area_m2' not in buildings.columns or 'compactness_index' not in buildings.columns:
        print("Dados de área ou compacidade não disponíveis")
        return
    
    # Remover valores nulos
    buildings_clean = buildings.dropna(subset=['area_m2', 'compactness_index'])
    
    if len(buildings_clean) == 0:
        print("Não há dados suficientes de área ou compacidade")
        return
    
    # Criar figura com subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Histograma de áreas
    sns.histplot(buildings_clean['area_m2'], bins=30, kde=True, ax=ax1)
    ax1.set_title('Distribuição de Área dos Edifícios', fontsize=14)
    ax1.set_xlabel('Área (m²)')
    ax1.set_ylabel('Frequência')
    
    # Adicionar linhas verticais para quartis
    area_q1 = buildings_clean['area_m2'].quantile(0.25)
    area_median = buildings_clean['area_m2'].median()
    area_q3 = buildings_clean['area_m2'].quantile(0.75)
    
    ax1.axvline(area_median, color='r', linestyle='-', label=f'Mediana: {area_median:.1f}m²')
    ax1.axvline(area_q1, color='g', linestyle='--', label=f'Q1: {area_q1:.1f}m²')
    ax1.axvline(area_q3, color='g', linestyle='--', label=f'Q3: {area_q3:.1f}m²')
    ax1.legend()
    
    # 2. Histograma de índice de compacidade
    sns.histplot(buildings_clean['compactness_index'], bins=30, kde=True, ax=ax2)
    ax2.set_title('Distribuição de Índice de Compacidade', fontsize=14)
    ax2.set_xlabel('Índice de Compacidade (0-1)')
    ax2.set_ylabel('Frequência')
    ax2.set_xlim(0, 1)
    
    # Adicionar linhas verticais para valores de referência
    ax2.axvline(0.25, color='b', linestyle='--', label='Baixa Compacidade (0.25)')
    ax2.axvline(0.5, color='g', linestyle='--', label='Média Compacidade (0.5)')
    ax2.axvline(0.8, color='r', linestyle='--', label='Alta Compacidade (0.8)')
    ax2.legend()
    
    # 3. Boxplot de área por classe de edifício
    if 'building_class' in buildings.columns:
        sns.boxplot(x='building_class', y='area_m2', data=buildings_clean, ax=ax3)
        ax3.set_title('Área por Classe de Edifício', fontsize=14)
        ax3.set_xlabel('Classe de Edifício')
        ax3.set_ylabel('Área (m²)')
        ax3.tick_params(axis='x', rotation=45)
    else:
        ax3.text(0.5, 0.5, 'Dados de classe não disponíveis', ha='center', va='center')
    
    # 4. Boxplot de compacidade por classe de edifício
    if 'building_class' in buildings.columns:
        sns.boxplot(x='building_class', y='compactness_index', data=buildings_clean, ax=ax4)
        ax4.set_title('Compacidade por Classe de Edifício', fontsize=14)
        ax4.set_xlabel('Classe de Edifício')
        ax4.set_ylabel('Índice de Compacidade')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1)
    else:
        ax4.text(0.5, 0.5, 'Dados de classe não disponíveis', ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de morfologia dos edifícios salva em: {output_path}")

def create_static_buildings_map(data, output_path):
    """Cria um mapa estático dos edifícios com camada base."""
    print("Criando mapa estático de edifícios...")
    
    if data['buildings'] is None:
        print("Dados de edifícios não disponíveis")
        return
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Reprojetar para WebMercator (EPSG:3857) para compatibilidade com camadas base
    if data['sorocaba'] is not None:
        sorocaba = data['sorocaba'].to_crs(epsg=3857)
        # Plotar área de estudo
        sorocaba.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5)
        area_bounds = sorocaba.total_bounds
    else:
        area_bounds = data['buildings'].to_crs(epsg=3857).total_bounds
    
    # Plotar edifícios por classe (se disponível)
    buildings = data['buildings'].to_crs(epsg=3857)
    
    if 'building_class' in buildings.columns:
        # Definir mapa de cores para classes de edifícios
        class_colors = {
            'residential': '#ff8080',  # Vermelho claro
            'commercial': '#80b3ff',   # Azul claro
            'industrial': '#b366ff',   # Roxo
            'office': '#66ffcc',       # Verde água
            'educational': '#ffdb4d',  # Amarelo
            'healthcare': '#ff3333',   # Vermelho
            'religious': '#ffffff',    # Branco
            'leisure': '#ff80ff',      # Rosa
            'utility': '#bfbfbf',      # Cinza
            'accommodation': '#ff9933', # Laranja
            'construction': '#ffff80',  # Amarelo claro
            'abandoned': '#bfbfbf',     # Cinza
            'unclassified': '#3388ff'   # Azul padrão
        }
        
        # Plotar cada classe separadamente
        for building_class, color in class_colors.items():
            subset = buildings[buildings['building_class'] == building_class]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor='#333333', linewidth=0.3, 
                          alpha=0.7, label=building_class)
    else:
        # Plotar todos os edifícios com a mesma cor
        buildings.plot(ax=ax, color='#3388ff', edgecolor='#333333', linewidth=0.3, 
                     alpha=0.7, label='Edifícios')
    
    # Adicionar camada base do OpenStreetMap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    
    # Configurar limites do mapa
    ax.set_xlim([area_bounds[0], area_bounds[2]])
    ax.set_ylim([area_bounds[1], area_bounds[3]])
    
    # Remover eixos
    ax.set_axis_off()
    
    # Adicionar título e legenda
    plt.title('Edifícios em Sorocaba', fontsize=16, pad=20)
    plt.legend(loc='lower right', title='Classes de Edifícios')
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa estático salvo em: {output_path}")

def main():
    """Função principal para criar visualizações."""
    print("\n--- Criando visualizações para edifícios ---\n")
    
    # Carregar dados
    data = load_data()
    
    # Verificar se dados foram carregados corretamente
    if all(gdf is None for gdf in data.values()):
        print("Nenhum dado de edifícios pôde ser carregado. Verifique os arquivos de entrada.")
        return
    
    # Criar visualizações
    
    # 1. Mapa interativo de edifícios
    interactive_map_path = os.path.join(OUTPUT_DIR, 'mapa_interativo_edificios.html')
    create_interactive_buildings_map(data, interactive_map_path)
    
    # 2. Distribuição das classes de edifícios
    building_dist_path = os.path.join(OUTPUT_DIR, 'distribuicao_classes_edificios.png')
    plot_building_class_distribution(data, building_dist_path)
    
    # 3. Análise de altura dos edifícios
    height_analysis_path = os.path.join(OUTPUT_DIR, 'analise_altura_edificios.png')
    analyze_building_heights(data, height_analysis_path)
    
    # 4. Mapa de calor de densidade
    density_map_path = os.path.join(OUTPUT_DIR, 'mapa_calor_densidade.png')
    create_building_density_heatmap(data, density_map_path)
    
    # 5. Análise de morfologia dos edifícios
    morphology_path = os.path.join(OUTPUT_DIR, 'analise_morfologia.png')
    analyze_building_morphology(data, morphology_path)
    
    # 6. Mapa estático com camada base
    static_map_path = os.path.join(OUTPUT_DIR, 'mapa_estatico_edificios.png')
    create_static_buildings_map(data, static_map_path)
    
    print(f"\nVisualizações salvas em: {OUTPUT_DIR}")
    print("Todas as visualizações foram criadas com sucesso!")

if __name__ == "__main__":
    main()