"""
Script para visualização de dados geoespaciais de áreas naturais de Sorocaba.

Este script cria visualizações interativas e estáticas para análise das áreas
naturais de Sorocaba, incluindo:
1. Mapa interativo com Folium
2. Mapas estáticos com diferentes classificações
3. Gráficos de distribuição de áreas naturais
4. Análises de padrões espaciais
5. Gráficos comparativos de métricas ambientais

Autor: Claude
Data: Abril 2025
"""

import os
import sys
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap, MeasureControl, MiniMap
import contextily as ctx
from shapely.geometry import Point
import warnings
from branca.element import Template, MacroElement
from matplotlib.colors import LinearSegmentedColormap
import json
warnings.filterwarnings('ignore')

# Configurar diretórios
INPUT_FILE = r"F:\TESE_MESTRADO\geoprocessing\data\processed\natural_areas_processed.gpkg"
CITY_BOUNDARY = r"F:\TESE_MESTRADO\geoprocessing\data\raw\sorocaba.gpkg"
OUTPUT_DIR = r"F:\TESE_MESTRADO\geoprocessing\outputs\visualizations\natural_areas"

# Criar diretório de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configurações gerais para visualizações
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.style.use('ggplot')

# Definir paletas de cores
GREEN_PALETTE = sns.light_palette("green", as_cmap=True)
AREA_PALETTE = sns.color_palette("YlGn", 10)
DIST_PALETTE = sns.color_palette("Blues", 10)
INDEX_PALETTE = sns.color_palette("RdYlBu", 10)

def load_data(input_file, city_boundary_file):
    """
    Carrega dados de áreas naturais e limites da cidade.
    
    Args:
        input_file (str): Caminho para o arquivo GPKG com áreas naturais processadas
        city_boundary_file (str): Caminho para o arquivo GPKG com limite da cidade
        
    Returns:
        GeoDataFrame: GeoDataFrame com áreas naturais e limite da cidade
    """
    try:
        # Carregar dados das áreas naturais
        natural_areas = gpd.read_file(input_file)
        print(f"Carregados {len(natural_areas)} registros de áreas naturais")
        print(f"Colunas disponíveis: {natural_areas.columns.tolist()}")
        
        # Carregar limites da cidade
        city_boundary = gpd.read_file(city_boundary_file)
        print(f"Limites da cidade carregados: {len(city_boundary)} polígonos")
        
        # Verificar e padronizar CRS
        if natural_areas.crs != city_boundary.crs:
            print(f"Padronizando CRS: natural_areas={natural_areas.crs}, city_boundary={city_boundary.crs}")
            city_boundary = city_boundary.to_crs(natural_areas.crs)
        
        # Adicionar limite da cidade ao GeoDataFrame
        natural_areas['city_boundary'] = None
        natural_areas.loc[0, 'city_boundary'] = city_boundary.unary_union
        
        # Calcular área em hectares se não existir
        if 'area_hectares' not in natural_areas.columns:
            print("Calculando área em hectares...")
            natural_areas['area_hectares'] = natural_areas.geometry.area / 10000  # m² para hectares
        
        # Garantir que temos a coluna de importância ecológica
        if 'ecological_importance' not in natural_areas.columns:
            print("AVISO: Coluna 'ecological_importance' não encontrada. Criando valores padrão.")
            # Tentar derivar de outras colunas ou usar um valor padrão
            if 'importance' in natural_areas.columns:
                natural_areas['ecological_importance'] = natural_areas['importance']
            else:
                natural_areas['ecological_importance'] = 'Média'
        
        return natural_areas
    
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_choropleth_colors(gdf, column, cmap='viridis'):
    """
    Cria mapeamento de cores para valores em uma coluna.
    
    Args:
        gdf (GeoDataFrame): GeoDataFrame a ser visualizado
        column (str): Nome da coluna para mapeamento de cores
        cmap (str): Nome do mapa de cores matplotlib
        
    Returns:
        dict: Mapeamento de valores para cores
    """
    if column not in gdf.columns:
        print(f"AVISO: Coluna {column} não encontrada no GeoDataFrame")
        return {}
    
    # Remover valores nulos para o mapeamento
    valid_data = gdf[gdf[column].notna()]
    if len(valid_data) == 0:
        print(f"AVISO: Coluna {column} contém apenas valores nulos")
        return {}
        
    # Para colunas categóricas
    if gdf[column].dtype == 'object' or gdf[column].dtype.name == 'category':
        unique_values = valid_data[column].unique()
        # Obter mapa de cores com número correto de categorias
        cmap_obj = plt.cm.get_cmap(cmap, len(unique_values))
        colors = [mcolors.rgb2hex(cmap_obj(i)) for i in range(len(unique_values))]
        color_dict = dict(zip(unique_values, colors))
        return color_dict
    
    # Para colunas numéricas
    else:
        min_val = valid_data[column].min()
        max_val = valid_data[column].max()
        
        # Verificar se min e max são iguais (evitar divisão por zero)
        if min_val == max_val:
            # Usar um valor único como média
            single_color = mcolors.rgb2hex(plt.cm.get_cmap(cmap)(0.5))
            return {min_val: single_color}
            
        cmap_obj = plt.cm.get_cmap(cmap)
        
        # Normalizar valores para intervalo [0, 1]
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
        
        # Criar função que mapeia valores para cores
        def get_color(value):
            if pd.isna(value):
                return '#CCCCCC'  # Cor cinza para valores nulos
            normalized = norm(value)
            color = cmap_obj(normalized)
            return mcolors.rgb2hex(color)
            
        return {val: get_color(val) for val in gdf[column].unique() if not pd.isna(val)}

def create_choropleth_map(natural_areas, variable, output_file):
    """
    Cria um mapa coroplético para a variável especificada.
    
    Parameters:
    ----------
    natural_areas : GeoDataFrame
        DataFrame com dados das áreas naturais
    variable : str
        Nome da variável para mapeamento
    output_file : str
        Caminho para salvar o mapa
    """
    if variable not in natural_areas.columns:
        print(f"AVISO: Variável {variable} não encontrada nos dados.")
        return
    
    # Criar figura e eixos
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Escolher paleta de cores apropriada
    cmap = GREEN_PALETTE
    if 'area' in variable.lower():
        cmap = 'YlGn'
    elif 'dist' in variable.lower():
        cmap = 'Blues'
    elif 'index' in variable.lower() or 'compact' in variable.lower():
        cmap = 'RdYlBu'
    
    # Plotar mapa
    natural_areas.plot(
        column=variable,
        cmap=cmap,
        linewidth=0.5,
        edgecolor='black',
        alpha=0.7,
        legend=True,
        ax=ax
    )
    
    # Adicionar contexto
    try:
        ctx.add_basemap(
            ax, 
            crs=natural_areas.crs,
            source=ctx.providers.CartoDB.Positron
        )
    except Exception as e:
        print(f"Aviso: Não foi possível adicionar basemap: {e}")
    
    # Configurar mapa
    ax.set_title(f'Distribuição de {variable.replace("_", " ").title()}', fontsize=16)
    ax.set_axis_off()
    
    # Salvar figura
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_interactive_map(natural_areas, output_file):
    """
    Cria um mapa interativo com Folium.
    
    Parameters:
    ----------
    natural_areas : GeoDataFrame
        DataFrame com dados das áreas naturais
    output_file : str
        Caminho para salvar o mapa HTML
    """
    # Converter para WGS84 (requerido pelo Folium)
    natural_areas_wgs84 = natural_areas.copy()
    
    # Remover colunas não serializáveis para JSON
    if 'city_boundary' in natural_areas_wgs84.columns:
        natural_areas_wgs84 = natural_areas_wgs84.drop(columns=['city_boundary'])
    
    if natural_areas_wgs84.crs != 'EPSG:4326':
        natural_areas_wgs84 = natural_areas_wgs84.to_crs('EPSG:4326')
    
    # Calcular centroide para o mapa
    center_lat = natural_areas_wgs84.unary_union.centroid.y
    center_lon = natural_areas_wgs84.unary_union.centroid.x
    
    # Criar mapa base
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='CartoDB positron',
        control_scale=True
    )
    
    # Adicionar ferramentas de medição
    MeasureControl(
        position='topright',
        primary_length_unit='kilometers',
        secondary_length_unit='miles',
        primary_area_unit='hectares',
        secondary_area_unit='acres'
    ).add_to(m)
    
    # Adicionar minimapa
    MiniMap(
        toggle_display=True,
        position='bottomright'
    ).add_to(m)
    
    # Identificar colunas categóricas e numéricas para popup
    categorical_cols = natural_areas_wgs84.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = natural_areas_wgs84.select_dtypes(include=[np.number]).columns.tolist()
    popup_fields = [col for col in natural_areas_wgs84.columns if col not in ['geometry', 'city_boundary'] and not col.startswith('__')]
    
    # Determinar variável para cor se disponível
    color_var = None
    if 'ecological_importance' in natural_areas.columns:
        color_var = 'ecological_importance'
    elif 'protection_level' in natural_areas.columns:
        color_var = 'protection_level'
    elif 'area_hectares' in natural_areas.columns:
        color_var = 'area_hectares'
    
    # Função para determinar a cor com base na variável
    def get_color(value):
        colors = ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
        if value is None or pd.isna(value):
            return colors[0]
        
        if isinstance(value, (int, float)):
            idx = min(int(value * len(colors) / 5), len(colors) - 1)
            return colors[idx]
        else:
            # Mapeamento para valores categóricos
            category_map = {
                'alta': colors[-1],
                'média': colors[-3],
                'baixa': colors[-5],
                'high': colors[-1],
                'medium': colors[-3],
                'low': colors[-5]
            }
            value_lower = str(value).lower()
            return category_map.get(value_lower, colors[0])
    
    # Adicionar camada de áreas naturais
    if color_var:
        # Criar style_function com a variável de cor
        def style_function(feature):
            value = feature['properties'].get(color_var)
            return {
                'fillColor': get_color(value),
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            }
        
        # Adicionar áreas como GeoJSON com estilo
        folium.GeoJson(
            natural_areas_wgs84,
            name='Áreas Naturais',
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=popup_fields[:5],  # Limitar para 5 campos no tooltip
                aliases=[f.replace('_', ' ').title() for f in popup_fields[:5]],
                sticky=False
            ),
            popup=folium.GeoJsonPopup(
                fields=popup_fields,
                aliases=[f.replace('_', ' ').title() for f in popup_fields],
                localize=True
            )
        ).add_to(m)
    else:
        # Adicionar sem variável de cor
        folium.GeoJson(
            natural_areas_wgs84,
            name='Áreas Naturais',
            style_function=lambda x: {
                'fillColor': '#2ca25f',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=popup_fields[:5],
                aliases=[f.replace('_', ' ').title() for f in popup_fields[:5]],
                sticky=False
            ),
            popup=folium.GeoJsonPopup(
                fields=popup_fields,
                aliases=[f.replace('_', ' ').title() for f in popup_fields],
                localize=True
            )
        ).add_to(m)
    
    # Adicionar controle de camadas
    folium.LayerControl().add_to(m)
    
    # Adicionar legenda se houver variável de cor
    if color_var:
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                    padding: 10px; border: 2px solid grey; border-radius: 5px">
        <p><b>Legenda - {}</b></p>
        <div style="display: flex; flex-direction: column;">
        """.format(color_var.replace('_', ' ').title())
        
        colors = ['#ffffcc', '#c7e9b4', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84']
        
        if color_var == 'ecological_importance' or color_var == 'protection_level':
            values = ['Baixa', 'Média-Baixa', 'Média', 'Média-Alta', 'Alta']
            for i, val in enumerate(values):
                color_idx = min(i, len(colors) - 1)
                legend_html += f"""
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="background-color: {colors[color_idx]}; width: 20px; height: 20px; margin-right: 5px;"></div>
                    <span>{val}</span>
                </div>
                """
        else:
            # Assumindo variável numérica (como área)
            max_val = natural_areas[color_var].max() if color_var in natural_areas.columns else 100
            intervals = np.linspace(0, max_val, len(colors))
            for i in range(len(colors) - 1):
                legend_html += f"""
                <div style="display: flex; align-items: center; margin-bottom: 5px;">
                    <div style="background-color: {colors[i]}; width: 20px; height: 20px; margin-right: 5px;"></div>
                    <span>{intervals[i]:.1f} - {intervals[i+1]:.1f}</span>
                </div>
                """
        
        legend_html += "</div></div>"
        
        # Adicionar a legenda ao mapa
        m.get_root().html.add_child(folium.Element(legend_html))
    
    # Salvar mapa
    m.save(output_file)

def create_distribution_chart(natural_areas, variable, output_file):
    """
    Cria um gráfico de distribuição para variáveis categóricas.
    
    Parameters:
    ----------
    natural_areas : GeoDataFrame
        DataFrame com dados das áreas naturais
    variable : str
        Nome da variável para análise
    output_file : str
        Caminho para salvar o gráfico
    """
    if variable not in natural_areas.columns:
        print(f"AVISO: Variável {variable} não encontrada nos dados.")
        return
    
    # Verificar se há dados válidos
    valid_data = natural_areas.dropna(subset=[variable])
    if len(valid_data) == 0:
        print(f"AVISO: Não há dados válidos para a variável {variable}.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Contar valores
    value_counts = natural_areas[variable].value_counts().sort_values(ascending=False)
    
    # Limitar a 10 categorias se houver muitas
    if len(value_counts) > 10:
        other_count = value_counts.iloc[9:].sum()
        value_counts = value_counts.iloc[:9]
        value_counts['Outros'] = other_count
    
    # Criar gráfico de barras
    ax = sns.barplot(x=value_counts.index, y=value_counts.values, palette='viridis')
    
    # Configurar gráfico
    plt.title(f'Distribuição de {variable.replace("_", " ").title()}', fontsize=16)
    plt.xlabel(variable.replace('_', ' ').title(), fontsize=14)
    plt.ylabel('Contagem', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    
    # Adicionar rótulos
    for i, v in enumerate(value_counts.values):
        ax.text(i, v + 0.1, str(v), ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_histogram(natural_areas, variable, output_file):
    """
    Cria um histograma para variáveis numéricas.
    
    Parameters:
    ----------
    natural_areas : GeoDataFrame
        DataFrame com dados das áreas naturais
    variable : str
        Nome da variável para análise
    output_file : str
        Caminho para salvar o gráfico
    """
    if variable not in natural_areas.columns:
        print(f"AVISO: Variável {variable} não encontrada nos dados.")
        return
    
    # Verificar se há dados válidos
    valid_data = natural_areas.dropna(subset=[variable])
    if len(valid_data) == 0:
        print(f"AVISO: Não há dados válidos para a variável {variable}.")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Criar histograma
    sns.histplot(
        data=natural_areas,
        x=variable,
        kde=True,
        palette='viridis',
        edgecolor='black',
        alpha=0.7
    )
    
    # Adicionar estatísticas descritivas
    mean_val = natural_areas[variable].mean()
    median_val = natural_areas[variable].median()
    
    # Adicionar linhas para média e mediana
    plt.axvline(mean_val, color='red', linestyle='--', label=f'Média: {mean_val:.2f}')
    plt.axvline(median_val, color='green', linestyle=':', label=f'Mediana: {median_val:.2f}')
    
    # Configurar gráfico
    plt.title(f'Distribuição de {variable.replace("_", " ").title()}', fontsize=16)
    plt.xlabel(variable.replace('_', ' ').title(), fontsize=14)
    plt.ylabel('Frequência', fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def create_statistical_summary(natural_areas, output_file):
    """
    Cria um resumo estatístico das áreas naturais.
    
    Parameters:
    ----------
    natural_areas : GeoDataFrame
        DataFrame com dados das áreas naturais
    output_file : str
        Caminho para salvar o resumo estatístico
    """
    # Preparar resumo
    summary = []
    summary.append("RESUMO ESTATÍSTICO - ÁREAS NATURAIS")
    summary.append("=" * 50)
    
    # Informações gerais
    summary.append(f"\nTotal de áreas naturais: {len(natural_areas)}")
    
    # Área total
    if 'area_hectares' in natural_areas.columns:
        total_area = natural_areas['area_hectares'].sum()
        summary.append(f"Área total: {total_area:.2f} hectares")
        
        # Estatísticas de área
        summary.append("\nESTATÍSTICAS DE ÁREA (hectares):")
        summary.append(f"  Mínima: {natural_areas['area_hectares'].min():.2f}")
        summary.append(f"  Máxima: {natural_areas['area_hectares'].max():.2f}")
        summary.append(f"  Média: {natural_areas['area_hectares'].mean():.2f}")
        summary.append(f"  Mediana: {natural_areas['area_hectares'].median():.2f}")
        summary.append(f"  Desvio padrão: {natural_areas['area_hectares'].std():.2f}")
    
    # Distribuição por tipo (se existir)
    type_columns = [col for col in natural_areas.columns if 'type' in col.lower() or 'category' in col.lower()]
    for col in type_columns:
        if col in natural_areas.columns:
            summary.append(f"\nDISTRIBUIÇÃO POR {col.replace('_', ' ').upper()}:")
            type_counts = natural_areas[col].value_counts()
            for type_name, count in type_counts.items():
                summary.append(f"  {type_name}: {count} áreas")
    
    # Importância ecológica (se existir)
    if 'ecological_importance' in natural_areas.columns:
        summary.append("\nDISTRIBUIÇÃO POR IMPORTÂNCIA ECOLÓGICA:")
        importance_counts = natural_areas['ecological_importance'].value_counts()
        for importance, count in importance_counts.items():
            summary.append(f"  {importance}: {count} áreas")
    
    # Nível de proteção (se existir)
    if 'protection_level' in natural_areas.columns:
        summary.append("\nDISTRIBUIÇÃO POR NÍVEL DE PROTEÇÃO:")
        protection_counts = natural_areas['protection_level'].value_counts()
        for protection, count in protection_counts.items():
            summary.append(f"  {protection}: {count} áreas")
    
    # Estatísticas adicionais para outras variáveis numéricas
    numeric_cols = natural_areas.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col not in ['area_hectares', 'geometry'] and not col.startswith('__'):
            summary.append(f"\nESTATÍSTICAS DE {col.replace('_', ' ').upper()}:")
            summary.append(f"  Mínima: {natural_areas[col].min():.2f}")
            summary.append(f"  Máxima: {natural_areas[col].max():.2f}")
            summary.append(f"  Média: {natural_areas[col].mean():.2f}")
            summary.append(f"  Mediana: {natural_areas[col].median():.2f}")
            summary.append(f"  Desvio padrão: {natural_areas[col].std():.2f}")
    
    # Salvar o resumo em arquivo de texto
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary))
    
    return '\n'.join(summary)

def main():
    """
    Função principal para geração de visualizações de áreas naturais.
    Carrega dados processados, gera mapas, gráficos e relatórios estatísticos.
    """
    # Definir caminhos de arquivo
    input_file = r"F:\TESE_MESTRADO\geoprocessing\data\processed\natural_areas_processed.gpkg"
    city_boundary_file = r"F:\TESE_MESTRADO\geoprocessing\data\raw\sorocaba.gpkg"
    output_dir = r"F:\TESE_MESTRADO\geoprocessing\outputs\visualizations\natural_areas"
    
    # Verificar e criar diretório de saída
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Diretório de saída verificado: {output_dir}")
    except Exception as e:
        print(f"Erro ao criar diretório de saída: {e}")
        return
    
    try:
        # Carregar dados
        natural_areas = load_data(input_file, city_boundary_file)
        if natural_areas is None or len(natural_areas) == 0:
            print("ERRO: Não foi possível carregar dados ou nenhuma área natural encontrada.")
            return
        
        print(f"Dados carregados com sucesso: {len(natural_areas)} áreas naturais")
        
        # 1. Gerar mapas coropletos com matplotlib
        print("\nGerando mapas coropletos...")
        try:
            # Mapas para as principais colunas numéricas
            numeric_columns = natural_areas.select_dtypes(include=np.number).columns.tolist()
            for col in numeric_columns:
                if col not in ['geometry', 'city_boundary'] and not col.startswith('__'):
                    output_file = os.path.join(output_dir, f"mapa_coropletico_{col}.png")
                    create_choropleth_map(natural_areas, col, output_file)
                    print(f"Mapa coroplético gerado para {col}: {output_file}")
        except Exception as e:
            print(f"Erro ao gerar mapas coropletos: {e}")
        
        # 2. Criar mapa interativo
        print("\nGerando mapa interativo...")
        try:
            interactive_map_file = os.path.join(output_dir, "mapa_interativo.html")
            create_interactive_map(natural_areas, interactive_map_file)
            print(f"Mapa interativo gerado: {interactive_map_file}")
        except Exception as e:
            print(f"Erro ao gerar mapa interativo: {e}")
        
        # 3. Gerar gráficos de distribuição
        print("\nGerando gráficos de distribuição...")
        try:
            # Para colunas categóricas
            categorical_columns = natural_areas.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_columns:
                if col not in ['geometry', 'city_boundary'] and not col.startswith('__'):
                    output_file = os.path.join(output_dir, f"distribuicao_{col}.png")
                    create_distribution_chart(natural_areas, col, output_file)
                    print(f"Gráfico de distribuição gerado para {col}: {output_file}")
            
            # Para variáveis numéricas
            for col in numeric_columns:
                if col not in ['geometry', 'city_boundary'] and not col.startswith('__'):
                    output_file = os.path.join(output_dir, f"histograma_{col}.png")
                    create_histogram(natural_areas, col, output_file)
                    print(f"Histograma gerado para {col}: {output_file}")
        except Exception as e:
            print(f"Erro ao gerar gráficos de distribuição: {e}")
        
        # 4. Gerar resumo estatístico
        print("\nGerando resumo estatístico...")
        try:
            stats_file = os.path.join(output_dir, "resumo_estatistico.txt")
            create_statistical_summary(natural_areas, stats_file)
            print(f"Resumo estatístico gerado: {stats_file}")
        except Exception as e:
            print(f"Erro ao gerar resumo estatístico: {e}")
        
        print("\nProcesso de visualização concluído com sucesso!")
        print(f"Todos os arquivos gerados em: {output_dir}")
    
    except Exception as e:
        print(f"Erro durante o processo de visualização: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()