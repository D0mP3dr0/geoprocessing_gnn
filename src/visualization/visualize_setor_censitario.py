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
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualizations', 'setor_censitario')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Definir arquivos de entrada
SETOR_CENSITARIO_FILE = os.path.join(INPUT_DIR, 'setor_censitario_processed.gpkg')
SOROCABA_FILE = os.path.join(WORKSPACE_DIR, 'data', 'raw', 'sorocaba.gpkg')

# Arquivos alternativos (caso os processados não existam)
SETOR_CENSITARIO_ALT_FILE = os.path.join(WORKSPACE_DIR, 'data', 'raw', 'setores_censitarios.gpkg')
if not os.path.exists(SETOR_CENSITARIO_FILE) and not os.path.exists(SETOR_CENSITARIO_ALT_FILE):
    print(f"AVISO: Arquivo de setores censitários não encontrado em: {SETOR_CENSITARIO_FILE}")
    print("Verifique se o arquivo existe ou se o nome está correto.")
    print("Tentando buscar outras alternativas...")

def load_data():
    """Carrega os dados processados de setores censitários."""
    print("Carregando dados de setores censitários...")
    
    data = {}
    
    # Tentar carregar o arquivo processado primeiro
    try:
        data['setor_censitario'] = gpd.read_file(SETOR_CENSITARIO_FILE)
        print(f"Setores censitários: {len(data['setor_censitario'])} registros")
    except Exception as e:
        print(f"Erro ao carregar setores censitários processados: {str(e)}")
        
        # Tentar carregar o arquivo alternativo
        try:
            data['setor_censitario'] = gpd.read_file(SETOR_CENSITARIO_ALT_FILE)
            print(f"Setores censitários (arquivo alternativo): {len(data['setor_censitario'])} registros")
        except Exception as e2:
            print(f"Erro ao carregar arquivo alternativo: {str(e2)}")
            
            # Buscar qualquer arquivo gpkg ou shp que possa conter setores censitários
            try:
                possible_files = []
                for dirpath, dirnames, filenames in os.walk(os.path.join(WORKSPACE_DIR, 'data')):
                    for filename in filenames:
                        if 'setor' in filename.lower() and ('gpkg' in filename.lower() or 'shp' in filename.lower()):
                            possible_files.append(os.path.join(dirpath, filename))
                
                if possible_files:
                    print(f"Encontrados possíveis arquivos de setores censitários: {possible_files}")
                    data['setor_censitario'] = gpd.read_file(possible_files[0])
                    print(f"Carregado arquivo alternativo: {possible_files[0]}")
                    print(f"Setores censitários: {len(data['setor_censitario'])} registros")
                else:
                    data['setor_censitario'] = None
            except Exception as e3:
                print(f"Não foi possível encontrar ou carregar nenhum arquivo de setores censitários: {str(e3)}")
                data['setor_censitario'] = None
    
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

def create_census_sectors_map(data, output_path):
    """Cria um mapa interativo dos setores censitários usando Folium."""
    print("Criando mapa interativo dos setores censitários...")
    
    # Verificar e converter dados para EPSG:4326 (WGS84) que é requerido pelo Folium
    if data['sorocaba'] is not None:
        sorocaba = data['sorocaba'].to_crs(epsg=4326)
        # Usar o centroide da área de estudo como centro do mapa
        center_lat = sorocaba.geometry.centroid.y.mean()
        center_lon = sorocaba.geometry.centroid.x.mean()
    elif data['setor_censitario'] is not None:
        # Usar o centro dos setores censitários
        setor_censitario = data['setor_censitario'].to_crs(epsg=4326)
        center_lat = setor_censitario.geometry.centroid.y.mean()
        center_lon = setor_censitario.geometry.centroid.x.mean()
    else:
        print("Dados insuficientes para criar o mapa")
        # Usar coordenadas de Sorocaba como fallback
        center_lat = -23.4691011
        center_lon = -47.4416311
    
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
    
    # Adicionar aviso se não houver dados de setores censitários
    if data['setor_censitario'] is None:
        warning_html = """
        <div style="position: fixed; 
                    top: 50px; left: 50px; width: 300px; 
                    border:2px solid red; z-index:9999; font-size:14px;
                    background-color:white;
                    padding: 10px;
                    opacity: 0.9;
                    ">
        <b style="color:red;">AVISO:</b><br>
        Os dados de setores censitários não foram encontrados.<br>
        Verifique se o arquivo existe em data/processed/setor_censitario_processed.gpkg
        </div>
        """
        m.get_root().html.add_child(folium.Element(warning_html))
        
        # Adicionar uma camada vazia para os setores censitários
        folium.GeoJson(
            data={"type": "FeatureCollection", "features": []},
            name='Setores Censitários (dados não disponíveis)',
        ).add_to(m)
    else:
        # Se temos os dados dos setores censitários, adicioná-los ao mapa
        setor_censitario = data['setor_censitario'].to_crs(epsg=4326)
        
        # Verificar se há uma coluna 'densidade' ou similar para colorir os setores
        choropleth_column = None
        if 'densidade' in setor_censitario.columns:
            choropleth_column = 'densidade'
        elif 'DENSIDADE' in setor_censitario.columns:
            choropleth_column = 'DENSIDADE'
        elif 'POP' in setor_censitario.columns:
            choropleth_column = 'POP'
        
        # Verificar se há a coluna de código do setor
        id_column = None
        for col in ['CD_SETOR', 'cod_setor', 'codigo', 'id']:
            if col in setor_censitario.columns:
                id_column = col
                break
        
        if id_column is None and len(setor_censitario) > 0:
            # Criar uma coluna de ID se não existir
            setor_censitario['id_setor'] = range(1, len(setor_censitario) + 1)
            id_column = 'id_setor'
        
        if choropleth_column and id_column:
            # Adicionar choropleth para setores
            folium.Choropleth(
                geo_data=setor_censitario,
                name='Setores Censitários',
                data=setor_censitario,
                columns=[id_column, choropleth_column],
                key_on=f'feature.properties.{id_column}',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=f'Setores Censitários - {choropleth_column}'
            ).add_to(m)
        else:
            # Adicionar apenas os polígonos
            setor_censitario_json = setor_censitario.to_json()
            folium.GeoJson(
                data=setor_censitario_json,
                name='Setores Censitários',
                style_function=lambda x: {
                    'fillColor': '#ff7800',
                    'color': '#000000',
                    'weight': 1,
                    'fillOpacity': 0.5
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=[id_column] if id_column else [],
                    aliases=['Código do Setor:'] if id_column else [],
                    localize=True,
                    sticky=False
                )
            ).add_to(m)
        
    # Adicionar legenda
    legend_html = """
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 180px; height: 80px; 
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white;
                padding: 10px;
                opacity: 0.8;
                ">
    <b>Legenda</b><br>
    <i style="background: #ffff00; opacity: 0.3; border: 1px solid black; display: inline-block; width: 18px; height: 18px;"></i> Área de Estudo<br>
    <i style="background: #ff7800; opacity: 0.5; border: 1px solid black; display: inline-block; width: 18px; height: 18px;"></i> Setores Censitários<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Adicionar título ao mapa
    title_html = '''
             <h3 align="center" style="font-size:20px"><b>Mapa de Setores Censitários - Sorocaba</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Adicionar controle de camadas por último para incluir todas as camadas
    folium.LayerControl().add_to(m)
    
    # Salvar mapa
    m.save(output_path)
    print(f"Mapa interativo salvo em: {output_path}")
    return output_path

def create_population_density_map(data, output_path):
    """Cria um mapa estático de densidade populacional com camada base usando Contextily."""
    print("Criando mapa de densidade populacional...")
    
    if data['setor_censitario'] is None:
        print("Dados de setores censitários não disponíveis")
        return
    
    # Criar figura
    fig, ax = plt.subplots(figsize=(15, 15))
    
    # Reprojetar para WebMercator (EPSG:3857) para compatibilidade com camadas base
    if data['sorocaba'] is not None:
        area_bounds = data['sorocaba'].to_crs(epsg=3857).total_bounds
        area = data['sorocaba'].to_crs(epsg=3857)
        area.plot(ax=ax, color='none', edgecolor='black', linewidth=1.5, alpha=0.8)
    else:
        area_bounds = data['setor_censitario'].to_crs(epsg=3857).total_bounds
    
    # Preparar dados dos setores censitários
    setores = data['setor_censitario'].to_crs(epsg=3857).copy()
    
    # Verificar se há informações de população e área para calcular densidade
    if 'POP' in setores.columns and not 'DENSIDADE' in setores.columns:
        # Calcular área em km²
        setores['AREA_KM2'] = setores.geometry.area / 1_000_000
        # Calcular densidade populacional
        setores['DENSIDADE'] = setores['POP'] / setores['AREA_KM2']
    
    # Plotar densidade populacional se disponível
    if 'DENSIDADE' in setores.columns:
        # Remover outliers para melhor visualização
        q1 = setores['DENSIDADE'].quantile(0.01)
        q3 = setores['DENSIDADE'].quantile(0.99)
        filtered_data = setores[(setores['DENSIDADE'] >= q1) & (setores['DENSIDADE'] <= q3)]
        
        # Plotar mapa de densidade
        filtered_data.plot(
            column='DENSIDADE',
            cmap='viridis',
            scheme='quantiles',
            k=5,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.3,
            ax=ax,
            legend=True,
            legend_kwds={'title': 'Densidade Pop. (hab/km²)'}
        )
    else:
        # Plotar apenas os setores sem coloração específica
        setores.plot(ax=ax, color='lightblue', alpha=0.7, edgecolor='black', linewidth=0.3)
    
    # Adicionar camada base do OpenStreetMap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)
    
    # Configurar limites do mapa baseados na área de estudo
    ax.set_xlim([area_bounds[0], area_bounds[2]])
    ax.set_ylim([area_bounds[1], area_bounds[3]])
    
    # Remover eixos
    ax.set_axis_off()
    
    # Adicionar título
    plt.title('Densidade Populacional por Setor Censitário', fontsize=16, pad=20)
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Mapa de densidade populacional salvo em: {output_path}")

def analyze_population_distribution(data, output_path):
    """Analisa e visualiza a distribuição de população nos setores censitários."""
    print("Analisando distribuição populacional...")
    
    if data['setor_censitario'] is None:
        print("Dados de setores censitários não disponíveis")
        return
    
    # Verificar se há dados de população
    setores = data['setor_censitario'].copy()
    
    if 'POP' not in setores.columns:
        print("Dados de população não encontrados nos setores censitários")
        return
    
    # Calcular área dos setores em km²
    setores['AREA_KM2'] = setores.geometry.area / 1_000_000
    
    # Calcular densidade populacional se não existir
    if 'DENSIDADE' not in setores.columns:
        setores['DENSIDADE'] = setores['POP'] / setores['AREA_KM2']
    
    # Criar figura com subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Histograma de distribuição populacional
    sns.histplot(setores['POP'], ax=ax1, bins=20, kde=True, color='darkblue')
    ax1.set_title('Distribuição da População por Setor Censitário', fontsize=14)
    ax1.set_xlabel('População (habitantes)', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Histograma de densidade populacional
    sns.histplot(setores['DENSIDADE'], ax=ax2, bins=20, kde=True, color='darkgreen')
    ax2.set_title('Distribuição da Densidade Populacional', fontsize=14)
    ax2.set_xlabel('Densidade (hab/km²)', fontsize=12)
    ax2.set_ylabel('Frequência', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar estatísticas descritivas como texto
    pop_stats = setores['POP'].describe()
    dense_stats = setores['DENSIDADE'].describe()
    
    stats_text = f"""
    Estatísticas de População:
    - Total: {setores['POP'].sum():.0f} habitantes
    - Média: {pop_stats['mean']:.1f} hab/setor
    - Mediana: {pop_stats['50%']:.1f} hab/setor
    - Min: {pop_stats['min']:.0f}, Max: {pop_stats['max']:.0f}
    
    Estatísticas de Densidade:
    - Média: {dense_stats['mean']:.1f} hab/km²
    - Mediana: {dense_stats['50%']:.1f} hab/km²
    - Min: {dense_stats['min']:.1f}, Max: {dense_stats['max']:.1f}
    """
    
    plt.figtext(0.5, 0.01, stats_text, ha='center', bbox=dict(facecolor='white', alpha=0.8),
               fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Ajustar espaço para o texto de estatísticas
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de distribuição populacional salva em: {output_path}")

def create_sectors_area_analysis(data, output_path):
    """Analisa e visualiza a distribuição de áreas dos setores censitários."""
    print("Analisando áreas dos setores censitários...")
    
    if data['setor_censitario'] is None:
        print("Dados de setores censitários não disponíveis")
        return
    
    # Calcular áreas em km²
    setores = data['setor_censitario'].copy()
    setores['AREA_KM2'] = setores.geometry.area / 1_000_000
    
    # Classificar setores por tamanho
    setores['CLASSE_AREA'] = pd.cut(
        setores['AREA_KM2'],
        bins=[0, 0.1, 0.5, 1, 5, float('inf')],
        labels=['< 0.1 km²', '0.1-0.5 km²', '0.5-1 km²', '1-5 km²', '> 5 km²']
    )
    
    # Contagem por classe
    counts = setores['CLASSE_AREA'].value_counts().sort_index()
    
    # Criar figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Histograma de áreas
    sns.histplot(setores['AREA_KM2'].clip(upper=setores['AREA_KM2'].quantile(0.99)), 
                bins=20, kde=True, color='purple', ax=ax1)
    ax1.set_title('Distribuição das Áreas dos Setores Censitários', fontsize=14)
    ax1.set_xlabel('Área (km²)', fontsize=12)
    ax1.set_ylabel('Frequência', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 2. Gráfico de barras por classe de área
    counts.plot(kind='bar', color='mediumpurple', ax=ax2)
    ax2.set_title('Setores por Classe de Área', fontsize=14)
    ax2.set_xlabel('Classe de Área', fontsize=12)
    ax2.set_ylabel('Número de Setores', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adicionar valores acima das barras
    for i, v in enumerate(counts):
        ax2.text(i, v + 0.5, str(v), ha='center', fontweight='bold')
    
    # Adicionar estatísticas descritivas
    area_stats = setores['AREA_KM2'].describe()
    stats_text = f"""
    Estatísticas de Área dos Setores Censitários:
    - Número de Setores: {len(setores)}
    - Área Total: {setores['AREA_KM2'].sum():.2f} km²
    - Média: {area_stats['mean']:.2f} km²
    - Mediana: {area_stats['50%']:.2f} km²
    - Min: {area_stats['min']:.4f} km²
    - Max: {area_stats['max']:.2f} km²
    """
    
    plt.figtext(0.5, 0.01, stats_text, ha='center', bbox=dict(facecolor='white', alpha=0.8),
               fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)  # Ajustar espaço para o texto de estatísticas
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de áreas dos setores censitários salva em: {output_path}")

def analyze_urban_rural_distribution(data, output_path):
    """Analisa e visualiza a distribuição de setores urbanos e rurais."""
    print("Analisando distribuição urbano/rural...")
    
    if data['setor_censitario'] is None:
        print("Dados de setores censitários não disponíveis")
        return
    
    setores = data['setor_censitario'].copy()
    
    # Verificar se há informação sobre zona urbana/rural
    if not any(col in setores.columns for col in ['SITUACAO', 'ZONA', 'TIPO']):
        print("Dados de classificação urbano/rural não encontrados")
        return
    
    # Identificar coluna com informação urbano/rural
    zona_col = None
    for col in ['SITUACAO', 'ZONA', 'TIPO']:
        if col in setores.columns:
            zona_col = col
            break
    
    # Criar figura com subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # 1. Gráfico de pizza para proporção urbano/rural
    counts = setores[zona_col].value_counts()
    ax1.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['skyblue', 'lightgreen', 'salmon', 'khaki'],
           startangle=90, explode=[0.05] * len(counts))
    ax1.set_title(f'Distribuição de Setores por {zona_col}', fontsize=14)
    
    # 2. Mapa coroplético de setores por tipo
    sorocaba_gdf = data['sorocaba'].to_crs(epsg=3857) if data['sorocaba'] is not None else None
    setores_3857 = setores.to_crs(epsg=3857)
    
    if sorocaba_gdf is not None:
        sorocaba_gdf.boundary.plot(ax=ax2, color='black', linewidth=1)
    
    setores_3857.plot(
        column=zona_col,
        categorical=True,
        cmap='viridis',
        linewidth=0.3,
        edgecolor='black',
        alpha=0.7,
        ax=ax2,
        legend=True,
        legend_kwds={'title': zona_col}
    )
    
    # Adicionar basemap
    ctx.add_basemap(ax2, source=ctx.providers.OpenStreetMap.Mapnik, zoom=11)
    ax2.set_axis_off()
    ax2.set_title(f'Mapa de Setores por {zona_col}', fontsize=14)
    
    # Adicionar estatísticas descritivas
    if 'POP' in setores.columns:
        # Calcular população por tipo de zona
        pop_by_zona = setores.groupby(zona_col)['POP'].sum()
        total_pop = setores['POP'].sum()
        
        stats_text = f"População total: {total_pop:.0f} habitantes\n"
        for zona, pop in pop_by_zona.items():
            stats_text += f"- {zona}: {pop:.0f} hab. ({pop/total_pop*100:.1f}%)\n"
        
        plt.figtext(0.5, 0.01, stats_text, ha='center', bbox=dict(facecolor='white', alpha=0.8),
                   fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Ajustar espaço para o texto de estatísticas
    
    # Salvar
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de distribuição urbano/rural salva em: {output_path}")

def create_population_heatmap(data, output_path):
    """Cria um mapa de calor de população dos setores censitários."""
    print("Criando mapa de calor populacional...")
    
    if data['setor_censitario'] is None:
        print("Dados de setores censitários não disponíveis")
        return
    
    if 'POP' not in data['setor_censitario'].columns:
        print("Dados de população não encontrados")
        return
    
    # Verificar e converter dados para EPSG:4326 (WGS84) para o Folium
    if data['sorocaba'] is not None:
        sorocaba = data['sorocaba'].to_crs(epsg=4326)
        # Usar o centroide da área de estudo como centro do mapa
        center_lat = sorocaba.geometry.centroid.y.mean()
        center_lon = sorocaba.geometry.centroid.x.mean()
    else:
        setor_censitario = data['setor_censitario'].to_crs(epsg=4326)
        center_lat = setor_censitario.geometry.centroid.y.mean()
        center_lon = setor_censitario.geometry.centroid.x.mean()
    
    # Criar mapa base
    m = folium.Map(location=[center_lat, center_lon], 
                  zoom_start=12,
                  tiles='CartoDB dark_matter')
    
    # Converter os setores para pontos com a população
    setores = data['setor_censitario'].to_crs(epsg=4326).copy()
    
    # Criar pontos a partir dos centroides dos setores
    heat_data = []
    
    for idx, row in setores.iterrows():
        centroid = row.geometry.centroid
        if pd.notna(row['POP']) and row['POP'] > 0:
            pop = float(row['POP'])
            heat_data.append([centroid.y, centroid.x, pop])
    
    # Adicionar mapa de calor
    HeatMap(
        data=heat_data,
        radius=15,
        max_zoom=13,
        blur=10,
        gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}
    ).add_to(m)
    
    # Adicionar limites da área de estudo
    if data['sorocaba'] is not None:
        sorocaba_json = sorocaba.to_json()
        folium.GeoJson(
            data=sorocaba_json,
            name='Área de Estudo',
            style_function=lambda x: {
                'fillColor': 'transparent',
                'color': 'white',
                'weight': 2,
                'fillOpacity': 0
            }
        ).add_to(m)
    
    # Adicionar mini mapa
    minimap = folium.plugins.MiniMap()
    m.add_child(minimap)
    
    # Adicionar escala
    folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)
    
    # Adicionar título
    title_html = '''
             <h3 align="center" style="font-size:16px"><b>Mapa de Calor - População</b></h3>
             '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Salvar mapa
    m.save(output_path)
    print(f"Mapa de calor populacional salvo em: {output_path}")
    return output_path

def main():
    """Função principal para criar visualizações."""
    print("\n--- Criando visualizações para dados de setores censitários ---\n")
    
    # Carregar dados
    data = load_data()
    
    # Verificar se dados foram carregados corretamente
    if all(gdf is None for gdf in data.values()):
        print("Nenhum dado de setor censitário pôde ser carregado. Verifique os arquivos de entrada.")
        return
    
    # Criar visualizações
    
    # 1. Mapa interativo dos setores censitários
    interactive_map_path = os.path.join(OUTPUT_DIR, 'mapa_interativo_setores_censitarios.html')
    create_census_sectors_map(data, interactive_map_path)
    
    # 2. Mapa de densidade populacional
    density_map_path = os.path.join(OUTPUT_DIR, 'mapa_densidade_populacional.png')
    create_population_density_map(data, density_map_path)
    
    # 3. Análise de distribuição populacional
    pop_distribution_path = os.path.join(OUTPUT_DIR, 'analise_distribuicao_populacional.png')
    analyze_population_distribution(data, pop_distribution_path)
    
    # 4. Análise de áreas dos setores
    areas_analysis_path = os.path.join(OUTPUT_DIR, 'analise_areas_setores.png')
    create_sectors_area_analysis(data, areas_analysis_path)
    
    # 5. Análise urbano/rural
    urban_rural_path = os.path.join(OUTPUT_DIR, 'analise_urbano_rural.png')
    analyze_urban_rural_distribution(data, urban_rural_path)
    
    # 6. Mapa de calor populacional
    heatmap_path = os.path.join(OUTPUT_DIR, 'mapa_calor_populacional.html')
    create_population_heatmap(data, heatmap_path)

if __name__ == "__main__":
    main()
