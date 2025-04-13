import os
import sys
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, MultiLineString
from tqdm import tqdm
import matplotlib.pyplot as plt
import contextily as ctx
from scipy.stats import percentileofscore
import logging
import json
from datetime import datetime
from pathlib import Path
import argparse

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('nature_elevation_enrichment')

# Função para extrair estatísticas de elevação para feições
def extract_elevation_statistics(gdf, dem_path, sample_distance=None):
    """
    Extrai estatísticas de elevação do DEM para cada geometria no GeoDataFrame.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame contendo geometrias
        dem_path (str): Caminho para o arquivo DEM (GeoTIFF)
        sample_distance (float, optional): Distância de amostragem para linhas. Se None, usa resolução do DEM.
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame com estatísticas de elevação adicionadas
    """
    logger.info("Extraindo estatísticas de elevação do DEM...")
    
    # Verificar se o arquivo DEM existe
    if not os.path.exists(dem_path):
        logger.error(f"Arquivo DEM não encontrado: {dem_path}")
        return gdf
    
    result = gdf.copy()
    
    try:
        # Abrir o DEM com rasterio
        with rasterio.open(dem_path) as dem:
            logger.info(f"DEM carregado: dimensões {dem.width}x{dem.height}, CRS {dem.crs}")
            
            # Verificar se os CRS são compatíveis, se não, reprojetar
            if result.crs != dem.crs:
                logger.info(f"Reprojetando geometrias de {result.crs} para {dem.crs}")
                result = result.to_crs(dem.crs)
            
            # Determinar distância de amostragem se não fornecida
            if sample_distance is None:
                sample_distance = min(abs(dem.res[0]), abs(dem.res[1])) * 2
                logger.info(f"Distância de amostragem definida como {sample_distance} unidades")
            
            # Preparar colunas para estatísticas de elevação
            result['elevation_min'] = np.nan
            result['elevation_max'] = np.nan
            result['elevation_mean'] = np.nan
            result['elevation_median'] = np.nan
            result['elevation_std'] = np.nan
            result['elevation_range'] = np.nan
            result['elevation_percentile_10'] = np.nan
            result['elevation_percentile_90'] = np.nan
            
            # Processar cada geometria
            for idx, row in tqdm(result.iterrows(), total=len(result), desc="Processando geometrias"):
                try:
                    geom = row.geometry
                    
                    # Abordagem diferente dependendo do tipo de geometria
                    if isinstance(geom, (Polygon, MultiPolygon)):
                        # Para polígonos, usar máscara para extrair todos os pixels dentro da geometria
                        try:
                            # Transformar geometria para formato GeoJSON
                            geom_json = [json.loads(gpd.GeoSeries([geom]).to_json())['features'][0]['geometry']]
                            
                            # Aplicar máscara ao raster
                            out_image, out_transform = mask(dem, geom_json, crop=True, nodata=dem.nodata)
                            
                            # Filtrar valores válidos
                            valid_pixels = out_image[0][out_image[0] != dem.nodata]
                            
                            if len(valid_pixels) > 0:
                                result.at[idx, 'elevation_min'] = float(np.min(valid_pixels))
                                result.at[idx, 'elevation_max'] = float(np.max(valid_pixels))
                                result.at[idx, 'elevation_mean'] = float(np.mean(valid_pixels))
                                result.at[idx, 'elevation_median'] = float(np.median(valid_pixels))
                                result.at[idx, 'elevation_std'] = float(np.std(valid_pixels))
                                result.at[idx, 'elevation_range'] = float(np.max(valid_pixels) - np.min(valid_pixels))
                                result.at[idx, 'elevation_percentile_10'] = float(np.percentile(valid_pixels, 10))
                                result.at[idx, 'elevation_percentile_90'] = float(np.percentile(valid_pixels, 90))
                        except Exception as e:
                            logger.warning(f"Erro ao processar polígono {idx}: {str(e)}")
                    
                    elif isinstance(geom, (LineString, MultiLineString)):
                        # Para linhas, amostrar pontos ao longo da linha
                        try:
                            # Lidar com MultiLineString
                            if isinstance(geom, MultiLineString):
                                lines = list(geom.geoms)
                            else:
                                lines = [geom]
                            
                            all_elevations = []
                            
                            for line in lines:
                                # Calcular comprimento da linha
                                length = line.length
                                
                                # Determinar número de pontos para amostragem
                                num_points = max(10, int(length / sample_distance))
                                
                                # Amostrar pontos ao longo da linha
                                points = [line.interpolate(i / (num_points - 1), normalized=True) 
                                         for i in range(num_points)]
                                
                                # Extrair elevação para cada ponto
                                for point in points:
                                    # Converter coordenadas do ponto para índices de pixel
                                    x, y = point.x, point.y
                                    py, px = dem.index(x, y)
                                    
                                    # Verificar se os índices estão dentro dos limites do raster
                                    if 0 <= py < dem.height and 0 <= px < dem.width:
                                        # Ler valor do pixel
                                        elevation = dem.read(1, window=((py, py+1), (px, px+1)))
                                        
                                        # Adicionar à lista se não for valor nulo
                                        if elevation[0][0] != dem.nodata:
                                            all_elevations.append(float(elevation[0][0]))
                            
                            if all_elevations:
                                result.at[idx, 'elevation_min'] = float(np.min(all_elevations))
                                result.at[idx, 'elevation_max'] = float(np.max(all_elevations))
                                result.at[idx, 'elevation_mean'] = float(np.mean(all_elevations))
                                result.at[idx, 'elevation_median'] = float(np.median(all_elevations))
                                result.at[idx, 'elevation_std'] = float(np.std(all_elevations))
                                result.at[idx, 'elevation_range'] = float(np.max(all_elevations) - np.min(all_elevations))
                                result.at[idx, 'elevation_percentile_10'] = float(np.percentile(all_elevations, 10))
                                result.at[idx, 'elevation_percentile_90'] = float(np.percentile(all_elevations, 90))
                        except Exception as e:
                            logger.warning(f"Erro ao processar linha {idx}: {str(e)}")
                    
                    elif isinstance(geom, Point):
                        # Para pontos, simplesmente obter o valor do pixel correspondente
                        try:
                            x, y = geom.x, geom.y
                            py, px = dem.index(x, y)
                            
                            if 0 <= py < dem.height and 0 <= px < dem.width:
                                elevation = dem.read(1, window=((py, py+1), (px, px+1)))
                                
                                if elevation[0][0] != dem.nodata:
                                    elev_value = float(elevation[0][0])
                                    result.at[idx, 'elevation_min'] = elev_value
                                    result.at[idx, 'elevation_max'] = elev_value
                                    result.at[idx, 'elevation_mean'] = elev_value
                                    result.at[idx, 'elevation_median'] = elev_value
                                    result.at[idx, 'elevation_std'] = 0.0
                                    result.at[idx, 'elevation_range'] = 0.0
                                    result.at[idx, 'elevation_percentile_10'] = elev_value
                                    result.at[idx, 'elevation_percentile_90'] = elev_value
                        except Exception as e:
                            logger.warning(f"Erro ao processar ponto {idx}: {str(e)}")
                
                except Exception as e:
                    logger.error(f"Erro ao processar geometria {idx}: {str(e)}")
            
            # Calcular estatísticas gerais
            elevation_mean_values = result['elevation_mean'].dropna()
            if not elevation_mean_values.empty:
                logger.info(f"Estatísticas de elevação: média={elevation_mean_values.mean():.2f}m, "
                          f"min={elevation_mean_values.min():.2f}m, max={elevation_mean_values.max():.2f}m")
    
    except Exception as e:
        logger.error(f"Erro ao processar DEM: {str(e)}")
    
    return result

# Função para calcular métricas derivadas da altimetria
def calculate_elevation_derived_metrics(gdf):
    """
    Calcula métricas derivadas das estatísticas de elevação.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame com estatísticas de elevação
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame com métricas derivadas adicionadas
    """
    logger.info("Calculando métricas derivadas da altimetria...")
    
    result = gdf.copy()
    
    # Verificar se as colunas necessárias existem
    if not all(col in result.columns for col in ['elevation_mean', 'elevation_range', 'elevation_std']):
        logger.error("Colunas de elevação necessárias não encontradas")
        return result
    
    # Filtrar apenas registros com dados de elevação válidos
    valid_records = result[result['elevation_mean'].notna()]
    if valid_records.empty:
        logger.warning("Nenhum registro com dados de elevação válidos")
        return result
    
    # 1. Índice de Rugosidade do Terreno (TRI)
    # TRI é uma medida da heterogeneidade do terreno
    # Usamos o desvio padrão da elevação como proxy
    result['terrain_roughness_index'] = result['elevation_std']
    
    # 2. Rugosidade relativa
    # Normaliza a rugosidade em relação ao valor máximo do conjunto de dados
    max_roughness = result['terrain_roughness_index'].max()
    if max_roughness > 0:
        result['relative_roughness'] = result['terrain_roughness_index'] / max_roughness
    
    # 3. Classificação de Relevo
    # Com base nas diferenças de elevação (amplitude)
    bins = [0, 5, 20, 50, 100, float('inf')]
    labels = ['Muito Plano', 'Suave', 'Moderado', 'Acidentado', 'Montanhoso']
    result['relief_class'] = pd.cut(result['elevation_range'], bins=bins, labels=labels)
    
    # 4. Perfil do Terreno (Convexidade/Concavidade)
    # Se o valor mediano estiver mais próximo do mínimo, o terreno tende a ser côncavo
    # Se estiver mais próximo do máximo, tende a ser convexo
    valid_mask = (result['elevation_min'].notna() & 
                 result['elevation_max'].notna() & 
                 result['elevation_median'].notna() &
                 (result['elevation_max'] > result['elevation_min']))
    
    result['terrain_profile'] = np.nan
    
    for idx in result[valid_mask].index:
        e_min = result.at[idx, 'elevation_min']
        e_max = result.at[idx, 'elevation_max']
        e_median = result.at[idx, 'elevation_median']
        
        # Calcular posição relativa do valor mediano entre min e max (0 a 1)
        if e_max > e_min:
            rel_pos = (e_median - e_min) / (e_max - e_min)
            
            # Valores < 0.4: côncavo, > 0.6: convexo, entre: plano/misto
            if rel_pos < 0.4:
                result.at[idx, 'terrain_profile'] = -1  # Côncavo
            elif rel_pos > 0.6:
                result.at[idx, 'terrain_profile'] = 1   # Convexo
            else:
                result.at[idx, 'terrain_profile'] = 0   # Plano/misto
    
    # 5. Classificação por quantil de elevação (em relação a todas as áreas)
    # Isso ajuda a identificar áreas relativamente altas ou baixas
    elevation_mean = result['elevation_mean'].dropna()
    for idx in result[result['elevation_mean'].notna()].index:
        elev = result.at[idx, 'elevation_mean']
        percentil = percentileofscore(elevation_mean, elev)
        result.at[idx, 'elevation_percentile'] = percentil
    
    # 6. Classificar em zonas de elevação
    bins_elev = [0, 600, 800, 1000, 1200, float('inf')]
    labels_elev = ['Baixada', 'Colinas Baixas', 'Colinas Altas', 'Montanhas Baixas', 'Montanhas Altas']
    result['elevation_zone'] = pd.cut(result['elevation_mean'], bins=bins_elev, labels=labels_elev)
    
    # 7. Índice de Exposição
    # Baseado na diferença entre os percentis 90 e 10, relativizada pela média
    # Altos valores indicam alta exposição a diferentes elevações
    result['exposure_index'] = np.nan
    for idx in result[result['elevation_mean'].notna()].index:
        p90 = result.at[idx, 'elevation_percentile_90']
        p10 = result.at[idx, 'elevation_percentile_10']
        mean = result.at[idx, 'elevation_mean']
        
        if mean > 0 and not pd.isna(p90) and not pd.isna(p10):
            result.at[idx, 'exposure_index'] = (p90 - p10) / mean
    
    # 8. Calcular relevância hidrológica
    # Áreas em baixadas têm maior potencial para acúmulo de água
    result['hydrological_relevance'] = np.nan
    percentile_25 = np.nanpercentile(result['elevation_mean'], 25)
    
    for idx in result[result['elevation_mean'].notna()].index:
        elev = result.at[idx, 'elevation_mean']
        roughness = result.at[idx, 'terrain_roughness_index']
        profile = result.at[idx, 'terrain_profile']
        
        # Fatores que aumentam relevância hidrológica:
        # 1. Elevação baixa (abaixo do percentil 25)
        # 2. Baixa rugosidade (terreno plano)
        # 3. Perfil côncavo (acúmulo de água)
        
        hydr_score = 0.0
        
        # Fator elevação
        if elev <= percentile_25:
            hydr_score += 0.5
        else:
            # Decai linearmente com aumento da elevação
            max_elev = result['elevation_mean'].max()
            hydr_score += 0.5 * (1 - ((elev - percentile_25) / (max_elev - percentile_25)))
        
        # Fator rugosidade
        max_roughness = result['terrain_roughness_index'].max()
        if max_roughness > 0:
            hydr_score += 0.25 * (1 - (roughness / max_roughness))
        
        # Fator perfil
        if profile == -1:  # Côncavo
            hydr_score += 0.25
        elif profile == 0:  # Plano/misto
            hydr_score += 0.125
        
        result.at[idx, 'hydrological_relevance'] = hydr_score
    
    # 9. Índice de Proteção Ecológica
    # Considera rugosidade, elevação e intervalo de elevação
    # Áreas mais rugosas, mais altas e com maior variação de elevação tendem a ser 
    # mais protegidas de atividades humanas e preservar biodiversidade
    result['ecological_protection_index'] = np.nan
    
    max_elevation = result['elevation_mean'].max()
    max_roughness = result['terrain_roughness_index'].max()
    max_range = result['elevation_range'].max()
    
    for idx in result[result['elevation_mean'].notna()].index:
        elev = result.at[idx, 'elevation_mean']
        roughness = result.at[idx, 'terrain_roughness_index']
        range_val = result.at[idx, 'elevation_range']
        
        # Normalizar cada componente (0 a 1)
        elev_norm = elev / max_elevation if max_elevation > 0 else 0
        roughness_norm = roughness / max_roughness if max_roughness > 0 else 0
        range_norm = range_val / max_range if max_range > 0 else 0
        
        # Calcular índice como média ponderada
        epi = (elev_norm * 0.4) + (roughness_norm * 0.4) + (range_norm * 0.2)
        result.at[idx, 'ecological_protection_index'] = epi
    
    # 10. Categoria de proteção ecológica
    bins_eco = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    labels_eco = ['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta']
    result['ecological_protection_category'] = pd.cut(
        result['ecological_protection_index'].fillna(0), 
        bins=bins_eco, 
        labels=labels_eco
    )
    
    # Exibir estatísticas dos novos atributos
    logger.info(f"Métricas derivadas calculadas. Classificação de relevo: "
              f"{result['relief_class'].value_counts().to_dict()}")
    
    return result

# Função para criar visualizações altimétricas
def create_elevation_visualizations(gdf, output_dir, prefix="nature_elevation"):
    """
    Cria visualizações com base nas métricas de elevação.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame com métricas de elevação
        output_dir (str): Diretório para salvar as visualizações
        prefix (str): Prefixo para os nomes dos arquivos
        
    Returns:
        dict: Dicionário com caminhos das visualizações geradas
    """
    logger.info("Gerando visualizações altimétricas...")
    
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar se há dados válidos
    if gdf.empty or not any(col in gdf.columns for col in ['elevation_mean', 'terrain_roughness_index']):
        logger.error("Dados insuficientes para gerar visualizações")
        return {}
    
    # Filtrar apenas registros com dados de elevação válidos
    valid_gdf = gdf[gdf['elevation_mean'].notna()].copy()
    if valid_gdf.empty:
        logger.warning("Nenhum registro com dados de elevação válidos para visualização")
        return {}
    
    # Dicionário para armazenar caminhos das visualizações
    visualization_paths = {}
    
    # Definir paletas de cores específicas para cada tipo de visualização
    elevation_cmap = 'terrain'
    roughness_cmap = 'YlOrRd'
    hydro_cmap = 'Blues'
    eco_protection_cmap = 'Greens'
    
    # Timestamp para nomes de arquivos
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # 1. Mapa de elevação média
        plt.figure(figsize=(14, 10))
        ax = plt.subplot(111)
        
        # Plotar as geometrias coloridas por elevação média
        valid_gdf.plot(column='elevation_mean', 
                    cmap=elevation_cmap, 
                    legend=True, 
                    ax=ax,
                    legend_kwds={'label': 'Elevação Média (m)',
                                'orientation': 'horizontal',
                                'shrink': 0.6,
                                'pad': 0.01})
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, crs=valid_gdf.crs)
        except Exception as e:
            logger.warning(f"Não foi possível adicionar mapa base: {str(e)}")
        
        # Adicionar informações ao mapa
        plt.title('Elevação Média das Áreas Naturais')
        plt.tight_layout()
        
        # Salvar figura
        elev_map_path = os.path.join(output_dir, f"{prefix}_elevation_map_{timestamp}.png")
        plt.savefig(elev_map_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['elevation_map'] = elev_map_path
        logger.info(f"Mapa de elevação salvo em {elev_map_path}")
        
        # 2. Mapa de rugosidade do terreno
        plt.figure(figsize=(14, 10))
        ax = plt.subplot(111)
        
        # Plotar as geometrias coloridas por rugosidade
        valid_gdf.plot(column='terrain_roughness_index', 
                    cmap=roughness_cmap, 
                    legend=True, 
                    ax=ax,
                    legend_kwds={'label': 'Rugosidade do Terreno (m)',
                                'orientation': 'horizontal',
                                'shrink': 0.6,
                                'pad': 0.01})
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, crs=valid_gdf.crs)
        except Exception as e:
            logger.warning(f"Não foi possível adicionar mapa base: {str(e)}")
        
        # Adicionar informações ao mapa
        plt.title('Rugosidade do Terreno nas Áreas Naturais')
        plt.tight_layout()
        
        # Salvar figura
        roughness_map_path = os.path.join(output_dir, f"{prefix}_roughness_map_{timestamp}.png")
        plt.savefig(roughness_map_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['roughness_map'] = roughness_map_path
        logger.info(f"Mapa de rugosidade salvo em {roughness_map_path}")
        
        # 3. Mapa de relevância hidrológica
        plt.figure(figsize=(14, 10))
        ax = plt.subplot(111)
        
        # Plotar as geometrias coloridas por relevância hidrológica
        valid_gdf.plot(column='hydrological_relevance', 
                    cmap=hydro_cmap, 
                    legend=True, 
                    ax=ax,
                    legend_kwds={'label': 'Relevância Hidrológica',
                                'orientation': 'horizontal',
                                'shrink': 0.6,
                                'pad': 0.01})
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, crs=valid_gdf.crs)
        except Exception as e:
            logger.warning(f"Não foi possível adicionar mapa base: {str(e)}")
        
        # Adicionar informações ao mapa
        plt.title('Relevância Hidrológica das Áreas Naturais')
        plt.tight_layout()
        
        # Salvar figura
        hydro_map_path = os.path.join(output_dir, f"{prefix}_hydrological_map_{timestamp}.png")
        plt.savefig(hydro_map_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['hydrological_map'] = hydro_map_path
        logger.info(f"Mapa de relevância hidrológica salvo em {hydro_map_path}")
        
        # 4. Mapa de proteção ecológica
        plt.figure(figsize=(14, 10))
        ax = plt.subplot(111)
        
        # Plotar as geometrias coloridas por índice de proteção ecológica
        valid_gdf.plot(column='ecological_protection_index', 
                    cmap=eco_protection_cmap, 
                    legend=True, 
                    ax=ax,
                    legend_kwds={'label': 'Índice de Proteção Ecológica',
                                'orientation': 'horizontal',
                                'shrink': 0.6,
                                'pad': 0.01})
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, crs=valid_gdf.crs)
        except Exception as e:
            logger.warning(f"Não foi possível adicionar mapa base: {str(e)}")
        
        # Adicionar informações ao mapa
        plt.title('Índice de Proteção Ecológica das Áreas Naturais')
        plt.tight_layout()
        
        # Salvar figura
        eco_map_path = os.path.join(output_dir, f"{prefix}_ecological_protection_map_{timestamp}.png")
        plt.savefig(eco_map_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['ecological_protection_map'] = eco_map_path
        logger.info(f"Mapa de proteção ecológica salvo em {eco_map_path}")
        
        # 5. Gráfico de distribuição de elevação
        plt.figure(figsize=(12, 8))
        
        # Histograma da elevação média
        valid_gdf['elevation_mean'].plot.hist(bins=20, 
                                           color='skyblue', 
                                           edgecolor='black', 
                                           grid=True,
                                           alpha=0.7)
        
        plt.title('Distribuição da Elevação Média nas Áreas Naturais')
        plt.xlabel('Elevação (m)')
        plt.ylabel('Frequência')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Salvar figura
        hist_path = os.path.join(output_dir, f"{prefix}_elevation_histogram_{timestamp}.png")
        plt.savefig(hist_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['elevation_histogram'] = hist_path
        logger.info(f"Histograma de elevação salvo em {hist_path}")
        
        # 6. Gráfico de barras - Distribuição das classes de relevo
        if 'relief_class' in valid_gdf.columns:
            plt.figure(figsize=(12, 8))
            
            # Contar ocorrências de cada classe de relevo
            relief_counts = valid_gdf['relief_class'].value_counts().sort_index()
            
            # Criar gráfico de barras
            ax = relief_counts.plot.bar(color='lightgreen', edgecolor='black')
            
            # Adicionar rótulos sobre as barras
            for i, v in enumerate(relief_counts):
                ax.text(i, v + 0.5, str(v), ha='center', va='bottom')
            
            plt.title('Distribuição das Classes de Relevo')
            plt.xlabel('Classe de Relevo')
            plt.ylabel('Número de Áreas')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Salvar figura
            relief_dist_path = os.path.join(output_dir, f"{prefix}_relief_distribution_{timestamp}.png")
            plt.savefig(relief_dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths['relief_distribution'] = relief_dist_path
            logger.info(f"Distribuição de classes de relevo salva em {relief_dist_path}")
        
        # 7. Mapa de classificação por perfil de terreno
        plt.figure(figsize=(14, 10))
        ax = plt.subplot(111)
        
        # Criar mapeamento de cores para perfil do terreno
        profile_colors = {
            -1: 'blue',    # Côncavo
            0: 'green',    # Plano/misto
            1: 'red'       # Convexo
        }
        
        # Lista para legenda
        legend_patches = []
        
        # Plotar cada perfil com sua cor específica
        for profile, color in profile_colors.items():
            subset = valid_gdf[valid_gdf['terrain_profile'] == profile]
            if not subset.empty:
                subset.plot(color=color, ax=ax, alpha=0.7)
                legend_patches.append(plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.7))
        
        # Adicionar mapa base
        try:
            ctx.add_basemap(ax, crs=valid_gdf.crs)
        except Exception as e:
            logger.warning(f"Não foi possível adicionar mapa base: {str(e)}")
        
        # Adicionar legenda manual
        plt.legend(legend_patches, ['Côncavo', 'Plano/Misto', 'Convexo'], 
                 loc='lower right', title='Perfil do Terreno')
        
        # Adicionar informações ao mapa
        plt.title('Perfil do Terreno nas Áreas Naturais')
        plt.tight_layout()
        
        # Salvar figura
        profile_map_path = os.path.join(output_dir, f"{prefix}_terrain_profile_map_{timestamp}.png")
        plt.savefig(profile_map_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['terrain_profile_map'] = profile_map_path
        logger.info(f"Mapa de perfil do terreno salvo em {profile_map_path}")
        
        # 8. Gráfico de dispersão - Elevação vs Proteção Ecológica
        plt.figure(figsize=(12, 8))
        
        plt.scatter(valid_gdf['elevation_mean'], 
                  valid_gdf['ecological_protection_index'], 
                  c=valid_gdf['terrain_roughness_index'], 
                  cmap='viridis', 
                  alpha=0.7,
                  s=50,
                  edgecolor='black')
        
        plt.colorbar(label='Rugosidade do Terreno')
        plt.title('Relação entre Elevação e Proteção Ecológica')
        plt.xlabel('Elevação Média (m)')
        plt.ylabel('Índice de Proteção Ecológica')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Salvar figura
        scatter_path = os.path.join(output_dir, f"{prefix}_elevation_vs_protection_{timestamp}.png")
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths['elevation_vs_protection'] = scatter_path
        logger.info(f"Gráfico de dispersão salvo em {scatter_path}")
        
    except Exception as e:
        logger.error(f"Erro ao gerar visualizações: {str(e)}")
    
    return visualization_paths

# Função para gerar relatório de qualidade dos dados enriquecidos
def generate_enriched_quality_report(gdf, output_dir, original_gdf=None, prefix="nature_elevation"):
    """
    Gera um relatório de qualidade para os dados naturais enriquecidos com altimetria.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame enriquecido
        output_dir (str): Diretório para salvar o relatório
        original_gdf (geopandas.GeoDataFrame, optional): GeoDataFrame original para comparação
        prefix (str): Prefixo para o nome do arquivo
        
    Returns:
        str: Caminho para o arquivo de relatório gerado
    """
    logger.info("Gerando relatório de qualidade para dados enriquecidos...")
    
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Timestamp para nome do arquivo
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Inicializar relatório
    report = {
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "report_type": "nature_elevation_enrichment_quality",
        "data_summary": {
            "total_features": len(gdf),
            "crs": str(gdf.crs),
            "geometry_types": list(gdf.geometry.geom_type.unique())
        },
        "elevation_statistics": {},
        "derived_metrics": {},
        "data_quality": {},
        "recommendations": []
    }
    
    # Adicionar informações sobre dados originais, se fornecidos
    if original_gdf is not None:
        report["data_summary"]["original_features"] = len(original_gdf)
        report["data_summary"]["original_columns"] = list(original_gdf.columns)
        report["data_summary"]["new_columns"] = list(set(gdf.columns) - set(original_gdf.columns))
    
    # Adicionar estatísticas de elevação
    elevation_cols = [col for col in gdf.columns if col.startswith('elevation_')]
    for col in elevation_cols:
        if col in gdf.columns:
            valid_values = gdf[col].dropna()
            if not valid_values.empty:
                report["elevation_statistics"][col] = {
                    "count": int(valid_values.count()),
                    "mean": float(valid_values.mean()),
                    "median": float(valid_values.median()),
                    "std": float(valid_values.std()),
                    "min": float(valid_values.min()),
                    "max": float(valid_values.max()),
                    "missing_values": int(gdf[col].isna().sum()),
                    "missing_percentage": float(gdf[col].isna().mean() * 100)
                }
    
    # Adicionar estatísticas de métricas derivadas
    derived_cols = ['terrain_roughness_index', 'relative_roughness', 'terrain_profile',
                  'elevation_percentile', 'hydrological_relevance', 'ecological_protection_index']
    
    for col in derived_cols:
        if col in gdf.columns:
            valid_values = gdf[col].dropna()
            if not valid_values.empty:
                report["derived_metrics"][col] = {
                    "count": int(valid_values.count()),
                    "mean": float(valid_values.mean()),
                    "median": float(valid_values.median()),
                    "std": float(valid_values.std()),
                    "min": float(valid_values.min()),
                    "max": float(valid_values.max()),
                    "missing_values": int(gdf[col].isna().sum()),
                    "missing_percentage": float(gdf[col].isna().mean() * 100)
                }
    
    # Adicionar distribuições de categorias
    for col in ['relief_class', 'elevation_zone', 'ecological_protection_category']:
        if col in gdf.columns:
            value_counts = gdf[col].value_counts().to_dict()
            # Converter as chaves para string (necessário para categorias do pandas)
            report["derived_metrics"][f"{col}_distribution"] = {str(k): int(v) for k, v in value_counts.items()}
    
    # Avaliar qualidade dos dados
    report["data_quality"]["total_missing_values"] = int(gdf.isna().sum().sum())
    report["data_quality"]["columns_with_nulls"] = list(gdf.columns[gdf.isna().any()].values)
    report["data_quality"]["completeness_score"] = float(1 - (gdf.isna().sum().sum() / (len(gdf) * len(gdf.columns))))
    
    # Detecção de valores extremos (outliers potenciais) em métricas relevantes
    outliers_info = {}
    
    for col in ['elevation_mean', 'terrain_roughness_index', 'ecological_protection_index']:
        if col in gdf.columns:
            values = gdf[col].dropna()
            if len(values) > 10:  # Verificar se há dados suficientes
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - (1.5 * iqr)
                upper_bound = q3 + (1.5 * iqr)
                
                outliers = values[(values < lower_bound) | (values > upper_bound)]
                outliers_info[col] = {
                    "count": int(len(outliers)),
                    "percentage": float(len(outliers) / len(values) * 100),
                    "min_value": float(outliers.min()) if not outliers.empty else None,
                    "max_value": float(outliers.max()) if not outliers.empty else None
                }
    
    report["data_quality"]["potential_outliers"] = outliers_info
    
    # Gerar recomendações baseadas na análise
    recommendations = []
    
    # Verificar completude dos dados de elevação
    elevation_completeness = gdf['elevation_mean'].notna().mean() * 100
    if elevation_completeness < 95:
        recommendations.append(f"Melhorar a cobertura dos dados de elevação (atualmente {elevation_completeness:.1f}%).")
    
    # Verificar áreas com alta relevância hidrológica
    if 'hydrological_relevance' in gdf.columns:
        high_hydro = gdf[gdf['hydrological_relevance'] > 0.7]
        if not high_hydro.empty:
            recommendations.append(f"Priorizar conservação de {len(high_hydro)} áreas com alta relevância hidrológica.")
    
    # Verificar áreas com alto índice de proteção ecológica
    if 'ecological_protection_index' in gdf.columns:
        high_eco = gdf[gdf['ecological_protection_index'] > 0.8]
        if not high_eco.empty:
            recommendations.append(f"Considerar {len(high_eco)} áreas com alto índice de proteção ecológica para preservação prioritária.")
    
    # Verificar diversidade de classes de relevo
    if 'relief_class' in gdf.columns:
        relief_classes = gdf['relief_class'].nunique()
        if relief_classes < 3:
            recommendations.append("Baixa diversidade de classes de relevo. Considerar expandir a cobertura da análise.")
    
    # Adicionar recomendações ao relatório
    report["recommendations"] = recommendations
    
    # Salvar relatório como JSON
    report_path = os.path.join(output_dir, f"{prefix}_quality_report_{timestamp}.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Relatório de qualidade salvo em {report_path}")
    
    return report_path

# Função principal para coordenar o processo de enriquecimento
def enrich_natural_areas_with_elevation(input_file, dem_file, output_dir=None, visualization_dir=None, report_dir=None):
    """
    Função principal para enriquecer dados naturais com informações altimétricas.
    
    Args:
        input_file (str): Caminho para o arquivo de dados naturais processados
        dem_file (str): Caminho para o arquivo DEM (GeoTIFF)
        output_dir (str, optional): Diretório para salvar dados enriquecidos
        visualization_dir (str, optional): Diretório para salvar visualizações
        report_dir (str, optional): Diretório para salvar relatórios
        
    Returns:
        tuple: (GeoDataFrame enriquecido, caminhos das visualizações, caminho do relatório)
    """
    # Registrar início do processo
    start_time = time.time()
    logger.info(f"Iniciando processo de enriquecimento com dados altimétricos: {datetime.now()}")
    
    # Definir diretórios padrão se não fornecidos
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_file), "enriched")
    
    if visualization_dir is None:
        visualization_dir = os.path.join(os.path.dirname(input_file), "visualizations")
    
    if report_dir is None:
        report_dir = os.path.join(os.path.dirname(input_file), "reports")
    
    # Criar diretórios se não existirem
    for directory in [output_dir, visualization_dir, report_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Verificar se os arquivos existem
    if not os.path.exists(input_file):
        logger.error(f"Arquivo de entrada não encontrado: {input_file}")
        return None, {}, None
    
    if not os.path.exists(dem_file):
        logger.error(f"Arquivo DEM não encontrado: {dem_file}")
        return None, {}, None
    
    try:
        # 1. Carregar dados naturais
        logger.info(f"Carregando dados naturais de {input_file}")
        natural_gdf = gpd.read_file(input_file)
        logger.info(f"Carregados {len(natural_gdf)} registros de áreas naturais")
        
        # Guardar cópia dos dados originais para comparação no relatório
        original_gdf = natural_gdf.copy()
        
        # 2. Extrair estatísticas de elevação
        logger.info(f"Extraindo estatísticas de elevação do DEM: {dem_file}")
        enriched_gdf = extract_elevation_statistics(natural_gdf, dem_file)
        
        # 3. Calcular métricas derivadas
        logger.info("Calculando métricas derivadas da altimetria")
        enriched_gdf = calculate_elevation_derived_metrics(enriched_gdf)
        
        # 4. Gerar visualizações
        logger.info(f"Gerando visualizações em {visualization_dir}")
        viz_paths = create_elevation_visualizations(enriched_gdf, visualization_dir)
        
        # 5. Gerar relatório de qualidade
        logger.info(f"Gerando relatório de qualidade em {report_dir}")
        report_path = generate_enriched_quality_report(
            enriched_gdf, 
            report_dir, 
            original_gdf=original_gdf
        )
        
        # 6. Salvar dados enriquecidos
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"natural_areas_enriched_{timestamp}.gpkg")
        logger.info(f"Salvando dados enriquecidos em {output_file}")
        enriched_gdf.to_file(output_file, driver="GPKG")
        
        # Mostrar resumo do processo
        elapsed_time = time.time() - start_time
        logger.info(f"Processo de enriquecimento concluído em {elapsed_time:.2f} segundos")
        logger.info(f"Dados enriquecidos salvos em: {output_file}")
        logger.info(f"Visualizações geradas: {len(viz_paths)}")
        logger.info(f"Relatório de qualidade: {report_path}")
        
        return enriched_gdf, viz_paths, report_path
    
    except Exception as e:
        logger.error(f"Erro durante o processo de enriquecimento: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, {}, None

# Função para adicionar função de enriquecimento ao módulo nature.py existente
def integrate_with_nature_module(module_path, output_path=None, backup=True):
    """
    Integra as funções de enriquecimento altimétrico ao módulo nature.py existente.
    
    Args:
        module_path (str): Caminho para o módulo nature.py existente
        output_path (str, optional): Caminho para salvar o módulo modificado. Se None, sobrescreve o original.
        backup (bool): Se True, cria um backup do arquivo original
        
    Returns:
        bool: True se a integração foi bem-sucedida, False caso contrário
    """
    try:
        # Verificar se o arquivo existe
        if not os.path.exists(module_path):
            logger.error(f"Arquivo do módulo não encontrado: {module_path}")
            return False
        
        # Criar backup se necessário
        if backup:
            backup_path = f"{module_path}.bak"
            import shutil
            shutil.copy2(module_path, backup_path)
            logger.info(f"Backup criado em {backup_path}")
        
        # Ler o conteúdo do arquivo
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Definir caminho de saída
        if output_path is None:
            output_path = module_path
        
        # Definir funções a serem adicionadas
        new_functions = """
# Funções para enriquecimento com dados altimétricos
def extract_elevation_statistics(gdf, dem_path, sample_distance=None):
    \"\"\"
    Extrai estatísticas de elevação do DEM para cada geometria no GeoDataFrame.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame contendo geometrias
        dem_path (str): Caminho para o arquivo DEM (GeoTIFF)
        sample_distance (float, optional): Distância de amostragem para linhas. Se None, usa resolução do DEM.
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame com estatísticas de elevação adicionadas
    \"\"\"
    print("Extraindo estatísticas de elevação do DEM...")
    
    # Verificar se o arquivo DEM existe
    if not os.path.exists(dem_path):
        print(f"Arquivo DEM não encontrado: {dem_path}")
        return gdf
    
    result = gdf.copy()
    
    try:
        # Abrir o DEM com rasterio
        with rasterio.open(dem_path) as dem:
            print(f"DEM carregado: dimensões {dem.width}x{dem.height}, CRS {dem.crs}")
            
            # Verificar se os CRS são compatíveis, se não, reprojetar
            if result.crs != dem.crs:
                print(f"Reprojetando geometrias de {result.crs} para {dem.crs}")
                result = result.to_crs(dem.crs)
            
            # Determinar distância de amostragem se não fornecida
            if sample_distance is None:
                sample_distance = min(abs(dem.res[0]), abs(dem.res[1])) * 2
                print(f"Distância de amostragem definida como {sample_distance} unidades")
            
            # Preparar colunas para estatísticas de elevação
            result['elevation_min'] = np.nan
            result['elevation_max'] = np.nan
            result['elevation_mean'] = np.nan
            result['elevation_median'] = np.nan
            result['elevation_std'] = np.nan
            result['elevation_range'] = np.nan
            
            # Processar cada geometria
            for idx, row in tqdm(result.iterrows(), total=len(result), desc="Processando geometrias"):
                try:
                    geom = row.geometry
                    
                    # Abordagem diferente dependendo do tipo de geometria
                    if isinstance(geom, (Polygon, MultiPolygon)):
                        # Para polígonos, usar máscara para extrair todos os pixels dentro da geometria
                        try:
                            # Transformar geometria para formato GeoJSON
                            geom_json = [json.loads(gpd.GeoSeries([geom]).to_json())['features'][0]['geometry']]
                            
                            # Aplicar máscara ao raster
                            out_image, out_transform = mask(dem, geom_json, crop=True, nodata=dem.nodata)
                            
                            # Filtrar valores válidos
                            valid_pixels = out_image[0][out_image[0] != dem.nodata]
                            
                            if len(valid_pixels) > 0:
                                result.at[idx, 'elevation_min'] = float(np.min(valid_pixels))
                                result.at[idx, 'elevation_max'] = float(np.max(valid_pixels))
                                result.at[idx, 'elevation_mean'] = float(np.mean(valid_pixels))
                                result.at[idx, 'elevation_median'] = float(np.median(valid_pixels))
                                result.at[idx, 'elevation_std'] = float(np.std(valid_pixels))
                                result.at[idx, 'elevation_range'] = float(np.max(valid_pixels) - np.min(valid_pixels))
                        except Exception as e:
                            print(f"Erro ao processar polígono {idx}: {str(e)}")
                    
                    elif isinstance(geom, (LineString, MultiLineString)):
                        # Para linhas, amostrar pontos ao longo da linha
                        try:
                            # Lidar com MultiLineString
                            if isinstance(geom, MultiLineString):
                                lines = list(geom.geoms)
                            else:
                                lines = [geom]
                            
                            all_elevations = []
                            
                            for line in lines:
                                # Calcular comprimento da linha
                                length = line.length
                                
                                # Determinar número de pontos para amostragem
                                num_points = max(10, int(length / sample_distance))
                                
                                # Amostrar pontos ao longo da linha
                                points = [line.interpolate(i / (num_points - 1), normalized=True) 
                                         for i in range(num_points)]
                                
                                # Extrair elevação para cada ponto
                                for point in points:
                                    # Converter coordenadas do ponto para índices de pixel
                                    x, y = point.x, point.y
                                    py, px = dem.index(x, y)
                                    
                                    # Verificar se os índices estão dentro dos limites do raster
                                    if 0 <= py < dem.height and 0 <= px < dem.width:
                                        # Ler valor do pixel
                                        elevation = dem.read(1, window=((py, py+1), (px, px+1)))
                                        
                                        # Adicionar à lista se não for valor nulo
                                        if elevation[0][0] != dem.nodata:
                                            all_elevations.append(float(elevation[0][0]))
                            
                            if all_elevations:
                                result.at[idx, 'elevation_min'] = float(np.min(all_elevations))
                                result.at[idx, 'elevation_max'] = float(np.max(all_elevations))
                                result.at[idx, 'elevation_mean'] = float(np.mean(all_elevations))
                                result.at[idx, 'elevation_median'] = float(np.median(all_elevations))
                                result.at[idx, 'elevation_std'] = float(np.std(all_elevations))
                                result.at[idx, 'elevation_range'] = float(np.max(all_elevations) - np.min(all_elevations))
                        except Exception as e:
                            print(f"Erro ao processar linha {idx}: {str(e)}")
                    
                    elif isinstance(geom, Point):
                        # Para pontos, simplesmente obter o valor do pixel correspondente
                        try:
                            x, y = geom.x, geom.y
                            py, px = dem.index(x, y)
                            
                            if 0 <= py < dem.height and 0 <= px < dem.width:
                                elevation = dem.read(1, window=((py, py+1), (px, px+1)))
                                
                                if elevation[0][0] != dem.nodata:
                                    elev_value = float(elevation[0][0])
                                    result.at[idx, 'elevation_min'] = elev_value
                                    result.at[idx, 'elevation_max'] = elev_value
                                    result.at[idx, 'elevation_mean'] = elev_value
                                    result.at[idx, 'elevation_median'] = elev_value
                                    result.at[idx, 'elevation_std'] = 0.0
                                    result.at[idx, 'elevation_range'] = 0.0
                        except Exception as e:
                            print(f"Erro ao processar ponto {idx}: {str(e)}")
                
                except Exception as e:
                    print(f"Erro ao processar geometria {idx}: {str(e)}")
            
            # Calcular estatísticas gerais
            elevation_mean_values = result['elevation_mean'].dropna()
            if not elevation_mean_values.empty:
                print(f"Estatísticas de elevação: média={elevation_mean_values.mean():.2f}m, "
                      f"min={elevation_mean_values.min():.2f}m, max={elevation_mean_values.max():.2f}m")
    
    except Exception as e:
        print(f"Erro ao processar DEM: {str(e)}")
    
    return result

def calculate_elevation_derived_metrics(gdf):
    \"\"\"
    Calcula métricas derivadas das estatísticas de elevação.
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame com estatísticas de elevação
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame com métricas derivadas adicionadas
    \"\"\"
    print("Calculando métricas derivadas da altimetria...")
    
    result = gdf.copy()
    
    # Verificar se as colunas necessárias existem
    if not all(col in result.columns for col in ['elevation_mean', 'elevation_range']):
        print("Colunas de elevação necessárias não encontradas")
        return result
    
    # Filtrar apenas registros com dados de elevação válidos
    valid_records = result[result['elevation_mean'].notna()]
    if valid_records.empty:
        print("Nenhum registro com dados de elevação válidos")
        return result
    
    # 1. Índice de Rugosidade do Terreno (TRI)
    if 'elevation_std' in result.columns:
        result['terrain_roughness_index'] = result['elevation_std']
    
    # 2. Classificação de Relevo
    # Com base nas diferenças de elevação (amplitude)
    bins = [0, 5, 20, 50, 100, float('inf')]
    labels = ['Muito Plano', 'Suave', 'Moderado', 'Acidentado', 'Montanhoso']
    result['relief_class'] = pd.cut(result['elevation_range'], bins=bins, labels=labels)
    
    # 3. Calcular relevância hidrológica (simplificada)
    # Áreas em baixadas têm maior potencial para acúmulo de água
    result['hydrological_relevance'] = np.nan
    percentile_25 = np.nanpercentile(result['elevation_mean'], 25)
    
    for idx in result[result['elevation_mean'].notna()].index:
        elev = result.at[idx, 'elevation_mean']
        
        # Fatores que aumentam relevância hidrológica:
        # 1. Elevação baixa (abaixo do percentil 25)
        # 2. Baixa rugosidade (terreno plano) - se disponível
        
        hydr_score = 0.0
        
        # Fator elevação
        if elev <= percentile_25:
            hydr_score += 0.7
        else:
            # Decai linearmente com aumento da elevação
            max_elev = result['elevation_mean'].max()
            hydr_score += 0.7 * (1 - ((elev - percentile_25) / (max_elev - percentile_25)))
        
        # Fator rugosidade (se disponível)
        if 'terrain_roughness_index' in result.columns:
            roughness = result.at[idx, 'terrain_roughness_index']
            max_roughness = result['terrain_roughness_index'].max()
            if not pd.isna(roughness) and max_roughness > 0:
                hydr_score += 0.3 * (1 - (roughness / max_roughness))
        
        result.at[idx, 'hydrological_relevance'] = hydr_score
    
    # 4. Classificação por zona de elevação
    mean_elev = result['elevation_mean'].mean()
    std_elev = result['elevation_mean'].std()
    
    bins_elev = [
        0,  # Mínimo
        mean_elev - std_elev,  # -1 desvio padrão
        mean_elev,             # Média
        mean_elev + std_elev,  # +1 desvio padrão
        float('inf')           # Máximo
    ]
    labels_elev = ['Muito Baixa', 'Baixa', 'Média', 'Alta', 'Muito Alta']
    result['elevation_zone'] = pd.cut(result['elevation_mean'], bins=bins_elev, labels=labels_elev)
    
    # 5. Índice de Proteção Ecológica (versão simplificada)
    result['ecological_protection_index'] = np.nan
    
    for idx in result[result['elevation_mean'].notna()].index:
        elev = result.at[idx, 'elevation_mean']
        elev_range = result.at[idx, 'elevation_range']
        
        # Normalizar componentes
        elev_norm = (elev - result['elevation_mean'].min()) / (result['elevation_mean'].max() - result['elevation_mean'].min())
        range_norm = elev_range / result['elevation_range'].max() if result['elevation_range'].max() > 0 else 0
        
        # Áreas mais altas e com maior variação tendem a ser mais preservadas (menos acessíveis)
        epi = (elev_norm * 0.7) + (range_norm * 0.3)
        result.at[idx, 'ecological_protection_index'] = epi
    
    return result

def process_natural_areas_with_elevation(input_file, dem_file, output_file=None):
    \"\"\"
    Processa áreas naturais com enriquecimento altimétrico.
    
    Args:
        input_file (str): Caminho para o arquivo de áreas naturais
        dem_file (str): Caminho para o arquivo DEM
        output_file (str, optional): Caminho para salvar o resultado
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame enriquecido
    \"\"\"
    print(f"Processando áreas naturais com enriquecimento altimétrico: {input_file}")
    
    # Verificar se os arquivos existem
    if not os.path.exists(input_file):
        print(f"Arquivo de entrada não encontrado: {input_file}")
        return None
    
    if not os.path.exists(dem_file):
        print(f"Arquivo DEM não encontrado: {dem_file}")
        return None
    
    # Se não for especificado arquivo de saída, criar um no mesmo diretório
    if output_file is None:
        output_dir = os.path.dirname(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"natural_areas_enriched_{timestamp}.gpkg")
    
    # Garantir que o diretório de saída existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    try:
        # Carregar dados
        print(f"Carregando dados de {input_file}")
        gdf = gpd.read_file(input_file)
        print(f"Carregados {len(gdf)} registros")
        
        # Extrair estatísticas de elevação
        print(f"Extraindo estatísticas de elevação de {dem_file}")
        gdf = extract_elevation_statistics(gdf, dem_file)
        
        # Calcular métricas derivadas
        print("Calculando métricas derivadas")
        gdf = calculate_elevation_derived_metrics(gdf)
        
        # Salvar resultado
        print(f"Salvando resultado em {output_file}")
        gdf.to_file(output_file, driver="GPKG")
        
        return gdf
    
    except Exception as e:
        print(f"Erro ao processar áreas naturais: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
\"\"\"

# Adicionar importação de rasterio e mask após as importações existentes
import_str = "from rasterio.mask import mask"
if import_str not in content:
    content = content.replace("import rasterio", "import rasterio\nfrom rasterio.mask import mask")

# Adicionar importação de json se não existir
if "import json" not in content:
    # Adicionar após pandas
    content = content.replace("import pandas as pd", "import pandas as pd\nimport json")

# Verificar se já existe uma função principal (process_natural_areas ou main)
main_exists = "def main(" in content
process_exists = "def process_natural_areas(" in content

# Adicionar chamada para enriquecimento com elevação na função principal existente
if main_exists or process_exists:
    # Encontrar o final da função
    target_function = "def main(" if main_exists else "def process_natural_areas("
    function_start = content.find(target_function)
    
    # Encontrar o final da função usando indentação
    lines = content[function_start:].split('\n')
    function_body = []
    for i, line in enumerate(lines):
        function_body.append(line)
        # Verificar se estamos no final da função
        if i > 0 and not line.strip() and not lines[i+1].startswith((' ', '\t')):
            break
    
    # Construir a chamada para enriquecimento
    elevation_call = """
    # Enriquecimento com dados altimétricos, se DEM estiver disponível
    dem_file = "F:/TESE_MESTRADO/geoprocessing/data/raw/dem.tif"
    if os.path.exists(dem_file):
        print("Aplicando enriquecimento com dados altimétricos...")
        enriched_gdf = extract_elevation_statistics(gdf, dem_file)
        enriched_gdf = calculate_elevation_derived_metrics(enriched_gdf)
        print("Enriquecimento altimétrico concluído.")

        # Usar o GeoDataFrame enriquecido para o resto do processamento
        gdf = enriched_gdf
    else:
        print("Arquivo DEM não encontrado. Pulando enriquecimento altimétrico.")
    """
    
    # Encontrar um bom lugar para inserir (após carregar/processar dados, antes de salvar)
    insertion_point = 0
    for i, line in enumerate(function_body):
        if "to_file" in line or "save" in line.lower():
            insertion_point = i
            break
    
    # Se não encontrou um bom ponto, usar o final da função
    if insertion_point == 0:
        insertion_point = len(function_body) - 1
    
    # Inserir chamada
    function_body.insert(insertion_point, elevation_call)
    
    # Reconstruir a função
    new_function = '\n'.join(function_body)
    
    # Substituir a função antiga pela nova
    content = content.replace('\n'.join(lines[:len(function_body)]), new_function)

# Adicionar as novas funções ao final do arquivo
content += new_functions

# Escrever o conteúdo modificado
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(content)

logger.info(f"Módulo atualizado com funções de enriquecimento altimétrico: {output_path}")

return True

if __name__ == "__main__":
    # Configuração de argumentos da linha de comando
    parser = argparse.ArgumentParser(description='Enriquecer áreas naturais com dados de elevação')
    parser.add_argument('--input', '-i', required=True, help='Arquivo de entrada (GPKG) com áreas naturais')
    parser.add_argument('--dem', '-d', required=True, help='Arquivo DEM (GeoTIFF) para extração de elevação')
    parser.add_argument('--output', '-o', help='Arquivo de saída (GPKG) para dados enriquecidos')
    parser.add_argument('--visualization-dir', '-v', help='Diretório para salvar visualizações')
    parser.add_argument('--report-dir', '-r', help='Diretório para salvar relatórios de qualidade')

    args = parser.parse_args()

    # Executar o enriquecimento
    enrich_natural_areas_with_elevation(
        args.input,
        args.dem,
        output_dir=os.path.dirname(args.output) if args.output else None,
        visualization_dir=args.visualization_dir,
        report_dir=args.report_dir
    ) 