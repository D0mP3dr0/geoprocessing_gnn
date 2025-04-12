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
from folium.plugins import MarkerCluster, HeatMap, MiniMap, MeasureControl
from branca.colormap import linear
import contextily as ctx
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
import warnings
import json
from typing import Dict, List, Optional, Tuple
import random
warnings.filterwarnings('ignore')

# Configurar diretórios
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualizations', 'landuse')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definir arquivos de entrada
LANDUSE_FILE = os.path.join(INPUT_DIR, 'landuse_processed.gpkg')
SOROCABA_FILE = os.path.join(WORKSPACE_DIR, 'data', 'raw', 'sorocaba.gpkg')

# Definir paleta de cores para categorias de uso do solo
LANDUSE_COLORS = {
    'urban': '#FD8D3C',         # Laranja (áreas urbanas)
    'institutional': '#E31A1C',  # Vermelho (institucionais)
    'green': '#41AB5D',          # Verde (áreas verdes)
    'agriculture': '#FFEDA0',    # Amarelo claro (áreas agrícolas)
    'forest': '#005A32',         # Verde escuro (florestas)
    'extraction': '#8C510A',     # Marrom (áreas de extração)
    'water': '#4292C6',          # Azul (corpos d'água)
    'other': '#969696'           # Cinza (outros usos)
}

# Mapeamento para categorias específicas do OSM
LANDUSE_TYPE_COLORS = {
    'residential': '#FC9272',    # Vermelho claro
    'commercial': '#FD8D3C',     # Laranja claro
    'retail': '#F16913',         # Laranja escuro
    'industrial': '#A6761D',     # Marrom escuro
    'construction': '#FDAE6B',   # Laranja muito claro
    'education': '#E31A1C',      # Vermelho
    'government': '#BD0026',     # Vermelho escuro
    'religious': '#FABEBE',      # Rosa claro
    'recreation_ground': '#41AB5D', # Verde médio
    'park': '#238B45',           # Verde médio escuro
    'grass': '#74C476',          # Verde claro
    'meadow': '#A1D99B',         # Verde muito claro
    'farmland': '#FFEDA0',       # Amarelo claro
    'farmyard': '#FED976',       # Amarelo
    'forest': '#005A32',         # Verde escuro
    'quarry': '#8C510A',         # Marrom
    'basin': '#4292C6',          # Azul
    'reservoir': '#2171B5',      # Azul médio
    'cemetery': '#7A0177',       # Roxo
    'military': '#C51B8A'        # Magenta
}

def load_data():
    """Carrega os dados processados de uso do solo."""
    print("Carregando dados de uso do solo...")
    
    data = {}
    
    try:
        # Carregar dados de uso do solo
        print(f"Tentando carregar arquivo: {LANDUSE_FILE}")
        if not os.path.exists(LANDUSE_FILE):
            print(f"ERRO: Arquivo não encontrado: {LANDUSE_FILE}")
            data['landuse'] = None
        else:
            data['landuse'] = gpd.read_file(LANDUSE_FILE)
            print(f"Dados de uso do solo carregados:")
            print(f"- Número de registros: {len(data['landuse'])}")
            print(f"- Colunas disponíveis: {', '.join(data['landuse'].columns)}")
            print(f"- Tipos de geometria: {data['landuse'].geometry.type.unique()}")
            print(f"- CRS: {data['landuse'].crs}")
            
            # Verificar se há geometrias válidas
            if 'geometry' in data['landuse'].columns:
                null_geoms = data['landuse'].geometry.isna().sum()
                invalid_geoms = (~data['landuse'].geometry.is_valid).sum() if not data['landuse'].geometry.isna().all() else 0
                print(f"- Geometrias nulas: {null_geoms}")
                print(f"- Geometrias inválidas: {invalid_geoms}")
            
            # Verificar colunas essenciais
            essential_cols = ['landuse', 'area_km2', 'land_category']
            missing_cols = [col for col in essential_cols if col not in data['landuse'].columns]
            if missing_cols:
                print(f"AVISO: Colunas essenciais faltando: {', '.join(missing_cols)}")
    except Exception as e:
        print(f"Erro ao carregar dados de uso do solo: {str(e)}")
        import traceback
        print(traceback.format_exc())
        data['landuse'] = None
    
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
        import traceback
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

def create_interactive_landuse_map(data, output_path):
    """Cria um mapa interativo com as categorias de uso do solo."""
    print("Criando mapa interativo de uso do solo...")
    
    if data['landuse'] is None:
        print("Dados de uso do solo não disponíveis")
        return None
    
    # Criar mapa base centrado na área de estudo
    if data['sorocaba'] is not None:
        bounds = data['sorocaba'].to_crs(epsg=4326).total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
    else:
        bounds = data['landuse'].to_crs(epsg=4326).total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2
        center_lon = (bounds[0] + bounds[2]) / 2
    
    # Criar mapa base
    m = folium.Map(location=[center_lat, center_lon], 
                  zoom_start=11, 
                  control_scale=True,
                  tiles='CartoDB positron')
    
    # Adicionar controle de minimapa
    MiniMap().add_to(m)
    
    # Adicionar controle de medição
    MeasureControl(position='topleft', primary_length_unit='kilometers').add_to(m)
    
    # Adicionar área de estudo se disponível
    if data['sorocaba'] is not None:
        # Converter para EPSG:4326 (WGS84) para compatibilidade com Folium
        sorocaba_geojson = data['sorocaba'].to_crs(epsg=4326).__geo_interface__
        
        folium.GeoJson(
            sorocaba_geojson,
            name='Área de Estudo',
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'black',
                'weight': 2
            }
        ).add_to(m)
    
    try:
        # Verificar qual coluna usar para categorização
        category_column = 'land_category'
        if category_column not in data['landuse'].columns:
            print(f"Aviso: Coluna '{category_column}' não encontrada. Usando 'landuse' como alternativa.")
            category_column = 'landuse'
            
        # Obter categorias únicas de uso do solo
        categories = data['landuse'][category_column].unique()
        
        # Definir cores para cada categoria
        colors = {
            'Agricultura': '#E5AB17',
            'Área Urbanizada': '#FF0000',
            'Floresta': '#008000',
            'Pastagem': '#90EE90',
            'Silvicultura': '#006400',
            'Corpo d\'água': '#1E90FF',
            'Solo Exposto': '#A52A2A',
            'Vegetação Natural': '#228B22',
            # Adicionar cores para categorias específicas do OSM
            'farmland': '#E5AB17',
            'residential': '#FF0000',
            'forest': '#008000',
            'grass': '#90EE90',
            'meadow': '#90EE90',
            'water': '#1E90FF',
        }
        
        print(f"Adicionando {len(categories)} categorias de uso do solo ao mapa...")
        
        # Adicionar cada categoria como uma camada separada
        for category in categories:
            if pd.notna(category):
                # Filtrar dados para a categoria atual
                category_data = data['landuse'][data['landuse'][category_column] == category]
                
                if len(category_data) == 0:
                    continue
                    
                # Escolher cor para a categoria
                if category in colors:
                    color = colors[category]
                else:
                    color = '#' + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)])
                
                # Converter para GeoJSON para melhor desempenho
                category_geojson = category_data.to_crs(epsg=4326).__geo_interface__
                
                # Adicionar camada GeoJSON
                folium.GeoJson(
                    category_geojson,
                    name=f"{category} ({len(category_data)} áreas)",
                    style_function=lambda x, color=color: {
                        'fillColor': color,
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.7
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=[category_column, 'area_km2', 'perimeter_km'],
                        aliases=['Tipo de Uso', 'Área (km²)', 'Perímetro (km)'],
                        localize=True,
                        sticky=False,
                        labels=True
                    )
                ).add_to(m)
    except Exception as e:
        print(f"Erro ao adicionar camadas de uso do solo: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    # Adicionar controle de camadas (sempre por último para incluir todas as camadas)
    folium.LayerControl().add_to(m)
    
    # Ajustar zoom para cobrir toda a área
    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    
    # Salvar mapa
    try:
        m.save(output_path)
        print(f"Mapa interativo salvo em: {output_path}")
        return output_path
    except Exception as e:
        print(f"Erro ao salvar mapa interativo: {str(e)}")
        return None

def create_landuse_legend_html(data):
    """Cria HTML para a legenda do mapa."""
    if data['landuse'] is None:
        return ""
    
    legend_items = []
    
    if 'land_category' in data['landuse'].columns:
        # Usar categorias simplificadas
        categories = data['landuse']['land_category'].unique()
        
        for category in sorted(categories):
            color = LANDUSE_COLORS.get(category, LANDUSE_COLORS['other'])
            legend_items.append(
                f'<i style="background: {color}; opacity: 0.7; border: 1px solid black; display: inline-block; width: 18px; height: 18px;"></i> {category.title()}'
            )
    else:
        # Usar tipos específicos do OSM (mostrar apenas os mais comuns para não sobrecarregar a legenda)
        landuse_types = data['landuse']['landuse'].value_counts().head(10).index.tolist()
        
        for landuse_type in landuse_types:
            color = LANDUSE_TYPE_COLORS.get(landuse_type, '#969696')
            legend_items.append(
                f'<i style="background: {color}; opacity: 0.7; border: 1px solid black; display: inline-block; width: 18px; height: 18px;"></i> {landuse_type}'
            )
    
    # Adicionar área de estudo na legenda
    legend_items.append(
        '<i style="background: #ffff00; opacity: 0.1; border: 1px solid black; display: inline-block; width: 18px; height: 18px;"></i> Área de Estudo'
    )
    
    legend_html = f"""
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 200px;
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                opacity: 0.8;">
    <b>Legenda</b><br>
    {"<br>".join(legend_items)}
    </div>
    """
    
    return legend_html

def plot_landuse_distribution(data, output_path):
    """Plota a distribuição de uso do solo por categoria."""
    print("Criando gráfico de distribuição de uso do solo...")
    
    if data['landuse'] is None:
        print("Dados de uso do solo não disponíveis")
        return None
    
    try:
        # Criar dois subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 1. Distribuição por número de polígonos
        if 'land_category' in data['landuse'].columns:
            # Usar categorias simplificadas
            category_counts = data['landuse']['land_category'].value_counts()
            colors = [LANDUSE_COLORS.get(cat, LANDUSE_COLORS['other']) for cat in category_counts.index]
            
            bars1 = ax1.bar(category_counts.index, category_counts.values, color=colors)
            ax1.set_title('Distribuição por Número de Polígonos', fontsize=12)
            ax1.set_xlabel('Categoria de Uso do Solo')
            ax1.set_ylabel('Número de Polígonos')
            ax1.tick_params(axis='x', rotation=45)
            
            # Adicionar valores sobre as barras
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom')
        else:
            # Usar tipos de uso do solo (show top 10)
            landuse_counts = data['landuse']['landuse'].value_counts().head(10)
            colors = [LANDUSE_TYPE_COLORS.get(lu, '#969696') for lu in landuse_counts.index]
            
            bars1 = ax1.bar(landuse_counts.index, landuse_counts.values, color=colors)
            ax1.set_title('Top 10 Tipos de Uso do Solo (por contagem)', fontsize=12)
            ax1.set_xlabel('Tipo de Uso do Solo')
            ax1.set_ylabel('Número de Polígonos')
            ax1.tick_params(axis='x', rotation=45)
            
            # Adicionar valores sobre as barras
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height):,}',
                        ha='center', va='bottom')
        
        # 2. Distribuição por área
        if 'land_category' in data['landuse'].columns and 'area_km2' in data['landuse'].columns:
            # Calcular área por categoria
            area_by_category = data['landuse'].groupby('land_category')['area_km2'].sum()
            colors = [LANDUSE_COLORS.get(cat, LANDUSE_COLORS['other']) for cat in area_by_category.index]
            
            bars2 = ax2.bar(area_by_category.index, area_by_category.values, color=colors)
            ax2.set_title('Distribuição por Área Total', fontsize=12)
            ax2.set_xlabel('Categoria de Uso do Solo')
            ax2.set_ylabel('Área Total (km²)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Adicionar valores sobre as barras
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom')
        elif 'area_km2' in data['landuse'].columns:
            # Calcular área por tipo de uso do solo (top 10)
            area_by_landuse = data['landuse'].groupby('landuse')['area_km2'].sum().sort_values(ascending=False).head(10)
            colors = [LANDUSE_TYPE_COLORS.get(lu, '#969696') for lu in area_by_landuse.index]
            
            bars2 = ax2.bar(area_by_landuse.index, area_by_landuse.values, color=colors)
            ax2.set_title('Top 10 Tipos de Uso do Solo (por área)', fontsize=12)
            ax2.set_xlabel('Tipo de Uso do Solo')
            ax2.set_ylabel('Área Total (km²)')
            ax2.tick_params(axis='x', rotation=45)
            
            # Adicionar valores sobre as barras
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}',
                        ha='center', va='bottom')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de distribuição de uso do solo salvo em: {output_path}")
        return output_path
    except Exception as e:
        print(f"Erro ao criar gráfico de distribuição: {str(e)}")
        import traceback
        print(traceback.format_exc())
        plt.close()
        return None

def create_static_landuse_map(data, output_path):
    """Cria um mapa estático de uso do solo com camada base."""
    print("Criando mapa estático de uso do solo...")
    
    if data['landuse'] is None:
        print("Dados de uso do solo não disponíveis")
        return None
    
    try:
        # Criar figura
        fig, ax = plt.subplots(figsize=(15, 15))
        
        # Reprojetar para WebMercator (EPSG:3857) para compatibilidade com camadas base
        if data['sorocaba'] is not None:
            area_bounds = data['sorocaba'].to_crs(epsg=3857).total_bounds
            area = data['sorocaba'].to_crs(epsg=3857)
            area.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5, alpha=0.8)
        else:
            area_bounds = data['landuse'].to_crs(epsg=3857).total_bounds
        
        # Plotar uso do solo
        landuse = data['landuse'].to_crs(epsg=3857)
        
        if 'land_category' in landuse.columns:
            # Criar mapa de cores por categoria
            categories = landuse['land_category'].unique()
            cmap = {cat: LANDUSE_COLORS.get(cat, LANDUSE_COLORS['other']) for cat in categories}
            
            # Plotar por categoria
            for category, color in cmap.items():
                subset = landuse[landuse['land_category'] == category]
                if not subset.empty:
                    subset.plot(ax=ax, color=color, 
                               edgecolor='black', linewidth=0.5, 
                               alpha=0.7,
                               label=category.title())
        else:
            # Usar a coluna 'landuse' diretamente
            unique_types = landuse['landuse'].unique()
            cmap = {lu: LANDUSE_TYPE_COLORS.get(lu, '#969696') for lu in unique_types}
            
            # Plotar apenas os 15 tipos mais comuns para evitar sobrecarga visual
            top_types = landuse['landuse'].value_counts().head(15).index.tolist()
            for lu in top_types:
                subset = landuse[landuse['landuse'] == lu]
                if not subset.empty:
                    subset.plot(ax=ax, color=cmap[lu], 
                               edgecolor='black', linewidth=0.5, 
                               alpha=0.7,
                               label=lu)
        
        # Adicionar camada base do OpenStreetMap
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)
        
        # Configurar limites do mapa
        ax.set_xlim([area_bounds[0], area_bounds[2]])
        ax.set_ylim([area_bounds[1], area_bounds[3]])
        
        # Remover eixos
        ax.set_axis_off()
        
        # Adicionar título e legenda
        plt.title('Mapa de Uso do Solo - Sorocaba', fontsize=16, pad=20)
        plt.legend(loc='lower right', title="Uso do Solo")
        
        # Salvar
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Mapa estático salvo em: {output_path}")
        return output_path
    except Exception as e:
        print(f"Erro ao criar mapa estático: {str(e)}")
        import traceback
        print(traceback.format_exc())
        plt.close()
        return None

def create_landuse_pie_chart(data, output_path):
    """Cria um gráfico de pizza mostrando a distribuição de uso do solo por área."""
    print("Criando gráfico de pizza de uso do solo...")
    
    if data['landuse'] is None or 'area_km2' not in data['landuse'].columns:
        print("Dados de uso do solo não disponíveis ou sem informação de área")
        return None
    
    try:
        # Configurar figura
        plt.figure(figsize=(12, 8))
        
        # Definir categorias para agrupar
        if 'land_category' in data['landuse'].columns:
            # Calcular área por categoria
            area_by_category = data['landuse'].groupby('land_category')['area_km2'].sum()
            
            # Definir cores
            colors = [LANDUSE_COLORS.get(cat, LANDUSE_COLORS['other']) for cat in area_by_category.index]
            
            # Criar gráfico de pizza
            plt.pie(area_by_category, labels=None, colors=colors, autopct='%1.1f%%', startangle=90, shadow=False)
            
            # Criar legenda
            plt.legend(
                title='Uso do Solo', 
                labels=[f"{cat.title()} ({area:.1f} km²)" for cat, area in area_by_category.items()],
                loc='center left', 
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            
            plt.title('Distribuição de Área por Categoria de Uso do Solo', fontsize=16)
        else:
            # Calcular área por tipo de uso do solo (top 10)
            area_by_landuse = data['landuse'].groupby('landuse')['area_km2'].sum().sort_values(ascending=False).head(10)
            
            # Calcular outros
            total_area = data['landuse']['area_km2'].sum()
            other_area = total_area - area_by_landuse.sum()
            if other_area > 0:
                area_by_landuse['outros'] = other_area
            
            # Definir cores
            colors = [LANDUSE_TYPE_COLORS.get(lu, '#969696') for lu in area_by_landuse.index[:-1]]
            colors.append('#969696')  # cor para 'outros'
            
            # Criar gráfico de pizza
            plt.pie(area_by_landuse, labels=None, colors=colors, autopct='%1.1f%%', startangle=90, shadow=False)
            
            # Criar legenda
            plt.legend(
                title='Uso do Solo', 
                labels=[f"{lu} ({area:.1f} km²)" for lu, area in area_by_landuse.items()],
                loc='center left', 
                bbox_to_anchor=(1, 0, 0.5, 1)
            )
            
            plt.title('Distribuição de Área por Tipo de Uso do Solo (Top 10)', fontsize=16)
        
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Salvar figura
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Gráfico de pizza salvo em: {output_path}")
        return output_path
    except Exception as e:
        print(f"Erro ao criar gráfico de pizza: {str(e)}")
        import traceback
        print(traceback.format_exc())
        plt.close()
        return None

def create_land_fragmentation_analysis(data, output_path):
    """Analisa a fragmentação do uso do solo usando o índice de compacidade."""
    print("Analisando fragmentação do uso do solo...")
    
    if data['landuse'] is None or 'compactness' not in data['landuse'].columns:
        print("Dados de uso do solo não disponíveis ou sem informação de compacidade")
        return None
    
    try:
        # Criar figura com subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # 1. Histograma de compacidade
        sns.histplot(data['landuse']['compactness'].dropna(), kde=True, ax=ax1)
        ax1.set_title('Distribuição do Índice de Compacidade', fontsize=14)
        ax1.set_xlabel('Índice de Compacidade (0-1)')
        ax1.set_ylabel('Frequência')
        ax1.grid(True, alpha=0.3)
        
        # Adicionar referência do círculo perfeito
        ax1.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Círculo (compacidade=1)')
        ax1.legend()
        
        # 2. Boxplot de compacidade por categoria
        if 'land_category' in data['landuse'].columns:
            # Boxplot por categoria
            sns.boxplot(x='land_category', y='compactness', data=data['landuse'], ax=ax2)
            ax2.set_title('Compacidade por Categoria de Uso do Solo', fontsize=14)
            ax2.set_xlabel('Categoria')
            ax2.set_ylabel('Índice de Compacidade')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, axis='y', alpha=0.3)
        else:
            # Boxplot por tipo (top 8)
            top_types = data['landuse']['landuse'].value_counts().head(8).index.tolist()
            df_top = data['landuse'][data['landuse']['landuse'].isin(top_types)]
            
            sns.boxplot(x='landuse', y='compactness', data=df_top, ax=ax2)
            ax2.set_title('Compacidade por Tipo de Uso do Solo (Top 8)', fontsize=14)
            ax2.set_xlabel('Tipo de Uso do Solo')
            ax2.set_ylabel('Índice de Compacidade')
            ax2.tick_params(axis='x', rotation=45)
            ax2.grid(True, axis='y', alpha=0.3)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Análise de fragmentação salva em: {output_path}")
        return output_path
    except Exception as e:
        print(f"Erro ao criar análise de fragmentação: {str(e)}")
        import traceback
        print(traceback.format_exc())
        plt.close()
        return None

def create_landuse_heatmap(data, output_path):
    """Cria um mapa de calor mostrando a densidade de áreas por tipo de uso do solo."""
    print("Criando mapa de calor de uso do solo...")
    
    if data['landuse'] is None or 'area_km2' not in data['landuse'].columns:
        print("Dados de uso do solo não disponíveis ou sem informação de área")
        return None
    
    try:
        # Criar figura
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Reprojetar para WebMercator para compatibilidade com camadas base
        landuse_proj = data['landuse'].to_crs(epsg=3857)
        
        # Determinar limites da área
        if data['sorocaba'] is not None:
            area_proj = data['sorocaba'].to_crs(epsg=3857)
            xmin, ymin, xmax, ymax = area_proj.total_bounds
        else:
            xmin, ymin, xmax, ymax = landuse_proj.total_bounds
        
        # Configurar grid para mapa de calor
        resolution = 100  # número de células na dimensão x
        
        # Calcular tamanho da célula com base nos limites
        cell_size_x = (xmax - xmin) / resolution
        cell_size_y = (ymax - ymin) / resolution * (xmax - xmin) / (ymax - ymin)
        
        # Criar grid de células
        x_edges = np.linspace(xmin, xmax, resolution+1)
        y_edges = np.linspace(ymin, ymax, resolution+1)
        
        # Inicializar matriz para armazenar áreas
        density_grid = np.zeros((len(y_edges)-1, len(x_edges)-1))
        
        # Filtrar geometrias nulas ou vazias e valores NaN de área
        valid_data = landuse_proj[~landuse_proj.geometry.isna() & 
                                ~landuse_proj.geometry.is_empty & 
                                ~np.isnan(landuse_proj['area_km2'])]
        
        print(f"Processando {len(valid_data)} polígonos válidos para o mapa de calor...")
        
        # Calcular centroide de cada geometria e adicionar sua área à célula correspondente
        for idx, row in valid_data.iterrows():
            try:
                centroid = row.geometry.centroid
                x, y = centroid.x, centroid.y
                
                # Verificar se o centroide está dentro dos limites
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    # Encontrar índice da célula
                    i = min(int((y - ymin) / cell_size_y), len(y_edges)-2)
                    j = min(int((x - xmin) / cell_size_x), len(x_edges)-2)
                    
                    # Adicionar área à célula
                    if 0 <= i < len(y_edges)-1 and 0 <= j < len(x_edges)-1:
                        density_grid[i, j] += row['area_km2']
            except Exception as e:
                print(f"Erro ao processar geometria {idx}: {str(e)}")
        
        # Aplicar suavização ao grid (opcional)
        from scipy.ndimage import gaussian_filter
        density_grid = gaussian_filter(density_grid, sigma=1.0)
        
        # Plotar mapa de calor com mascaramento de valores zero
        masked_grid = np.ma.masked_where(density_grid == 0, density_grid)
        im = ax.imshow(masked_grid, origin='lower', extent=[xmin, xmax, ymin, ymax], 
                    cmap='YlOrRd', alpha=0.7, interpolation='bilinear')
        
        # Adicionar barra de cores
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Área (km²)')
        
        # Adicionar contorno da área de estudo se disponível
        if data['sorocaba'] is not None:
            area_proj.boundary.plot(ax=ax, edgecolor='black', linewidth=2)
        
        # Adicionar mapa base
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zoom=12)
        
        # Configurar limites e remover eixos
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()
        
        # Adicionar título
        plt.title('Mapa de Calor de Áreas de Uso do Solo', fontsize=16)
        
        # Salvar figura
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Mapa de calor salvo em: {output_path}")
        return output_path
    except Exception as e:
        print(f"Erro ao criar mapa de calor: {str(e)}")
        import traceback
        print(traceback.format_exc())
        plt.close()
        return None

def main():
    """Função principal para criar visualizações."""
    print("\n--- Criando visualizações para dados de uso do solo ---\n")
    
    # Carregar dados
    data = load_data()
    
    # Verificar se dados foram carregados corretamente
    if data['landuse'] is None:
        print("Dados de uso do solo não puderam ser carregados. Verifique os arquivos de entrada.")
        return
    
    # Lista para armazenar resultados
    resultados = {}
    
    # Criar visualizações
    
    # 1. Mapa interativo de uso do solo
    interactive_map_path = os.path.join(OUTPUT_DIR, 'mapa_interativo_landuse.html')
    resultados['mapa_interativo'] = create_interactive_landuse_map(data, interactive_map_path)
    
    # 2. Mapa estático com camada base
    static_map_path = os.path.join(OUTPUT_DIR, 'mapa_estatico_landuse.png')
    resultados['mapa_estatico'] = create_static_landuse_map(data, static_map_path)
    
    # 3. Gráfico de distribuição de uso do solo
    distribution_path = os.path.join(OUTPUT_DIR, 'distribuicao_landuse.png')
    resultados['grafico_distribuicao'] = plot_landuse_distribution(data, distribution_path)
    
    # 4. Gráfico de pizza de distribuição de área
    pie_chart_path = os.path.join(OUTPUT_DIR, 'distribuicao_area_landuse.png')
    resultados['grafico_pizza'] = create_landuse_pie_chart(data, pie_chart_path)
    
    # 5. Análise de fragmentação
    fragmentation_path = os.path.join(OUTPUT_DIR, 'analise_fragmentacao_landuse.png')
    resultados['analise_fragmentacao'] = create_land_fragmentation_analysis(data, fragmentation_path)
    
    # 6. Mapa de calor de densidade
    heatmap_path = os.path.join(OUTPUT_DIR, 'mapa_calor_landuse.png')
    resultados['mapa_calor'] = create_landuse_heatmap(data, heatmap_path)
    
    # Verificar resultados
    sucesso = sum(1 for res in resultados.values() if res is not None)
    total = len(resultados)
    
    print(f"\nVisualizações salvas em: {OUTPUT_DIR}")
    print(f"Completado: {sucesso}/{total} visualizações")
    
    if sucesso == total:
        print("Todas as visualizações foram criadas com sucesso!")
    else:
        print("Algumas visualizações não puderam ser criadas. Verifique as mensagens de erro acima.")

if __name__ == "__main__":
    main()