# Análise e Enriquecimento de Dados de ERBs - Parte 2
# Implementação de Diagramas de Voronoi e Grade Hexagonal de Cobertura

# Funções para cálculo de diagrama de Voronoi
import numpy as np
import geopandas as gpd
from scipy.spatial import Voronoi
from shapely.geometry import Polygon, Point
import pandas as pd
import logging
import h3
from tqdm import tqdm
import matplotlib.pyplot as plt
import contextily as ctx
import os

# Usar as mesmas configurações de caminhos da Parte 1
# Certifique-se de executar a Parte 1 antes desta célula
from colab_enriched_rbs_part1 import (
    BASE_PATH, DATA_PATH, PROCESSED_DATA_PATH, ENRICHED_DATA_PATH, 
    VISUALIZATION_DIR, REPORT_DIR, logger, setup_logging
)

# Se o logger não estiver configurado, configure-o
if logger is None:
    logger = setup_logging()

def calculate_voronoi(gdf):
    """
    Calcula o diagrama de Voronoi para as ERBs.

    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB

    Returns:
        tuple: (gdf atualizado, gdf do diagrama de Voronoi)
    """
    logger.info("Calculando diagrama de Voronoi para as ERBs")

    # Criar uma cópia projetada para cálculos geométricos
    gdf_proj = gdf.to_crs(epsg=3857)  # Web Mercator

    # Extrair coordenadas das ERBs
    coords = np.array([(point.x, point.y) for point in gdf_proj.geometry])

    # Adicionar pontos de contorno para limitar o diagrama de Voronoi
    boundary_points = np.array([
        [coords[:, 0].min() - 10000, coords[:, 1].min() - 10000],
        [coords[:, 0].min() - 10000, coords[:, 1].max() + 10000],
        [coords[:, 0].max() + 10000, coords[:, 1].min() - 10000],
        [coords[:, 0].max() + 10000, coords[:, 1].max() + 10000]
    ])
    coords_extended = np.vstack([coords, boundary_points])

    # Calcular diagrama de Voronoi
    try:
        vor = Voronoi(coords_extended)

        # Criar polígonos de Voronoi
        polygons = []
        point_indices = []

        for i, region_idx in enumerate(vor.point_region[:len(coords)]):
            region = vor.regions[region_idx]
            if -1 not in region and len(region) > 0:
                polygon_vertices = [vor.vertices[v] for v in region]
                if len(polygon_vertices) >= 3:
                    poly = Polygon(polygon_vertices)
                    polygons.append(poly)
                    point_indices.append(i)
        
        # Criar GeoDataFrame para os polígonos de Voronoi
        voronoi_gdf = gpd.GeoDataFrame(
            {'erb_id': point_indices,
             'NomeEntidade': gdf.iloc[point_indices]['NomeEntidade'].values,
             'geometry': polygons},
            crs=gdf_proj.crs
        )

        # Voltar para CRS original
        voronoi_gdf = voronoi_gdf.to_crs(gdf.crs)

        # Calcular área dos polígonos em km²
        voronoi_gdf['area_voronoi_km2'] = voronoi_gdf.to_crs('+proj=utm +zone=23 +south').geometry.area / 1_000_000
        
        # Adicionar informações sobre vizinhos
        voronoi_gdf['num_vizinhos'] = 0

        for i, row in voronoi_gdf.iterrows():
            # Contar polígonos que compartilham fronteira
            vizinhos = voronoi_gdf[voronoi_gdf.geometry.touches(row.geometry)]
            voronoi_gdf.at[i, 'num_vizinhos'] = len(vizinhos)

        # Associar informações de voronoi com ERBs originais
        gdf_resultado = gdf.copy()
        gdf_resultado['voronoi_area_km2'] = np.nan
        gdf_resultado['voronoi_vizinhos'] = np.nan

        for i, row in voronoi_gdf.iterrows():
            erb_id = row['erb_id']
            gdf_resultado.iloc[erb_id, gdf_resultado.columns.get_loc('voronoi_area_km2')] = row['area_voronoi_km2']
            gdf_resultado.iloc[erb_id, gdf_resultado.columns.get_loc('voronoi_vizinhos')] = row['num_vizinhos']

        logger.info(f"Diagrama de Voronoi calculado com {len(voronoi_gdf)} polígonos")

        return gdf_resultado, voronoi_gdf
    
    except Exception as e:
        logger.error(f"Erro ao calcular Voronoi: {e}")
        return gdf, gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

def plot_voronoi_map(gdf, voronoi_gdf):
    """
    Gera um mapa com o diagrama de Voronoi das ERBs.

    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB
        voronoi_gdf (geopandas.GeoDataFrame): Diagrama de Voronoi
    """
    logger.info("Gerando mapa de Voronoi")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)

    # Criar paleta de cores para operadoras
    operadoras = gdf['NomeEntidade'].unique()
    colors_operadoras = dict(zip(operadoras, plt.cm.tab10.colors[:len(operadoras)]))

    # Plotar polígonos de Voronoi por operadora
    for op in operadoras:
        subset = voronoi_gdf[voronoi_gdf['NomeEntidade'] == op]
        if len(subset) > 0:
            color = colors_operadoras[op]
            subset.plot(ax=ax, color=color, alpha=0.5, label=op)

    # Adicionar pontos das ERBs
    gdf.plot(ax=ax, markersize=15, color='black', alpha=0.7)

    # Adicionar mapa base
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logger.warning(f"Não foi possível adicionar mapa base: {e}")

    # Adicionar título e legenda
    plt.title('Áreas de Domínio das ERBs (Diagrama de Voronoi)', fontsize=16)
    plt.legend(title='Operadora', loc='upper right')

    # Salvar figura
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_voronoi.png'), dpi=300, bbox_inches='tight')
    plt.close()

# --- Função para cacheamento de hexágonos H3 ---
hex_boundary_cache = {}
def cached_hex_boundary(hex_id):
    if hex_id not in hex_boundary_cache:
        hex_boundary_cache[hex_id] = h3.h3_to_geo_boundary(hex_id)
    return hex_boundary_cache[hex_id]

def create_hexagon_coverage_grid(gdf, resolution=9):
    """
    Cria uma grade hexagonal para análise de cobertura.
    
    Usa o sistema H3 para criar uma grade de hexágonos e analisa a cobertura das ERBs em cada hexágono.
    
    Args:
        gdf (geopandas.GeoDataFrame): Dados de ERB com geometria e atributos
        resolution (int): Resolução dos hexágonos H3 (7-11 recomendado)
        
    Returns:
        tuple: (gdf atualizado, hex_gdf com grade hexagonal)
    """
    logger.info(f"Criando grade hexagonal com resolução H3 = {resolution}")
    
    # Verificar se temos setores de cobertura
    if 'setor_geometria' not in gdf.columns:
        logger.warning("Setores de cobertura não encontrados, executando cálculo de setores primeiro")
        # Você precisa implementar essa função na parte 1
        # gdf = create_coverage_sectors(gdf)
    
    # Extrair limites da área dos dados
    min_lon, min_lat, max_lon, max_lat = gdf.total_bounds
    logger.info(f"Limites da área: ({min_lon:.4f}, {min_lat:.4f}) - ({max_lon:.4f}, {max_lat:.4f})")
    
    # Ampliar a área em 10% para criar uma margem
    lon_margin = (max_lon - min_lon) * 0.1
    lat_margin = (max_lat - min_lat) * 0.1
    min_lon -= lon_margin
    min_lat -= lat_margin
    max_lon += lon_margin
    max_lat += lat_margin
    
    # Obter hexágonos que cobrem a área
    # Polígono da área total
    area_poly = Polygon([
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat)
    ])
    area_centroid = area_poly.centroid
    
    # Estimar tamanho de hexágono (km²) para esta resolução
    hex_area_km2 = h3.hex_area(resolution, 'km^2')
    logger.info(f"Área do hexágono para resolução {resolution}: {hex_area_km2:.4f} km²")
    
    # Calcular raio aproximado em km
    # A = π * r²  =>  r = sqrt(A / π)
    hex_radius_km = np.sqrt(hex_area_km2 / np.pi)
    
    # Estimar número de hexágonos necessários
    area_total_km2 = gdf.to_crs('+proj=utm +zone=23 +south').unary_union.convex_hull.area / 1_000_000
    est_num_hexagons = int(area_total_km2 / hex_area_km2) * 1.5  # Adiciona 50% para margem
    logger.info(f"Estimativa de hexágonos necessários: {est_num_hexagons}")
    
    # Criar grade hexagonal
    # Obter ID do hexágono central
    center_hex = h3.geo_to_h3(area_centroid.y, area_centroid.x, resolution)
    
    # Obter k-anéis ao redor do hexágono central para cobrir toda a área
    k_rings = 1
    hex_ids = set([center_hex])
    
    # Expandir anéis até cobrir a área ou atingir limite
    while len(hex_ids) < est_num_hexagons and k_rings < 50:
        ring = h3.k_ring(center_hex, k_rings)
        hex_ids.update(ring)
        k_rings += 1
    
    logger.info(f"Gerados {len(hex_ids)} hexágonos com {k_rings-1} anéis")
    
    # Criar geometrias para os hexágonos
    hex_geometries = []
    for hex_id in hex_ids:
        boundary = cached_hex_boundary(hex_id)
        # Converter para Polygon
        polygon = Polygon(boundary)
        hex_geometries.append((hex_id, polygon))
    
    # Criar GeoDataFrame para os hexágonos
    hex_gdf = gpd.GeoDataFrame(
        {'hex_id': [h[0] for h in hex_geometries]},
        geometry=[h[1] for h in hex_geometries],
        crs="EPSG:4326"  # Sistema H3 usa WGS84
    )
    
    # Filtar hexágonos para manter apenas os que estão dentro ou tocam a área de interesse
    hex_gdf = hex_gdf[hex_gdf.intersects(area_poly)]
    
    logger.info(f"Grade hexagonal filtrada para {len(hex_gdf)} hexágonos na área de interesse")
    
    # Analisar cobertura por operadora em cada hexágono
    operadoras = gdf['NomeEntidade'].unique()
    
    # Adicionar colunas para contagem de ERBs por operadora
    for operadora in operadoras:
        col_name = f"count_{operadora.replace(' ', '_')}"
        hex_gdf[col_name] = 0
    
    # Coluna para número total de ERBs
    hex_gdf['num_erbs'] = 0
    # Coluna para número de operadoras distintas
    hex_gdf['num_operadoras'] = 0
    # Coluna para EIRP médio
    hex_gdf['eirp_medio'] = 0.0
    # Coluna para percentual de cobertura
    hex_gdf['perc_cobertura'] = 0.0
    
    # Processar intersecções com setores
    if 'setor_geometria' in gdf.columns and gdf['setor_geometria'].notna().any():
        # Criar um GeoDataFrame apenas com os setores válidos
        setores_gdf = gpd.GeoDataFrame(
            {
                'NomeEntidade': gdf.loc[gdf['setor_geometria'].notna(), 'NomeEntidade'],
                'EIRP_dBm': gdf.loc[gdf['setor_geometria'].notna(), 'EIRP_dBm']
            },
            geometry=gdf.loc[gdf['setor_geometria'].notna(), 'setor_geometria'],
            crs=gdf.crs
        )
        
        logger.info(f"Analisando cobertura de {len(setores_gdf)} setores em {len(hex_gdf)} hexágonos")
        
        # Para cada hexágono, calcular intersecção com setores
        for idx, row in tqdm(hex_gdf.iterrows(), total=len(hex_gdf), desc="Processando hexágonos"):
            # Hexágono atual
            hex_geom = row.geometry
            
            # Operadoras presentes neste hexágono
            ops_presentes = set()
            count_erbs = 0
            sum_eirp = 0.0
            
            # Para cada operadora
            for operadora in operadoras:
                # Filtrar setores desta operadora
                setores_op = setores_gdf[setores_gdf['NomeEntidade'] == operadora]
                
                # Encontrar intersecções
                intersections = setores_op[setores_op.intersects(hex_geom)]
                
                # Contagem de ERBs desta operadora no hexágono
                count_op = len(intersections)
                if count_op > 0:
                    ops_presentes.add(operadora)
                    count_erbs += count_op
                    sum_eirp += intersections['EIRP_dBm'].sum()
                
                # Atualizar coluna de contagem
                col_name = f"count_{operadora.replace(' ', '_')}"
                hex_gdf.at[idx, col_name] = count_op
            
            # Atualizar estatísticas do hexágono
            hex_gdf.at[idx, 'num_erbs'] = count_erbs
            hex_gdf.at[idx, 'num_operadoras'] = len(ops_presentes)
            
            # Calcular EIRP médio
            if count_erbs > 0:
                hex_gdf.at[idx, 'eirp_medio'] = sum_eirp / count_erbs
            
            # Calcular percentual de cobertura (simplificado)
            # Um hexágono tem cobertura se ao menos uma ERB o cobre
            hex_gdf.at[idx, 'perc_cobertura'] = 100.0 if count_erbs > 0 else 0.0
    
    # Classificar vulnerabilidade de cobertura
    hex_gdf['vulnerabilidade'] = 'Sem cobertura'
    
    # Critérios de classificação
    # 1. Sem cobertura: nenhuma operadora
    # 2. Alta vulnerabilidade: apenas 1 operadora
    # 3. Média vulnerabilidade: 2 operadoras
    # 4. Baixa vulnerabilidade: 3+ operadoras
    
    hex_gdf.loc[hex_gdf['num_operadoras'] == 1, 'vulnerabilidade'] = 'Alta vulnerabilidade'
    hex_gdf.loc[hex_gdf['num_operadoras'] == 2, 'vulnerabilidade'] = 'Média vulnerabilidade'
    hex_gdf.loc[hex_gdf['num_operadoras'] >= 3, 'vulnerabilidade'] = 'Baixa vulnerabilidade'
    
    # Contar hexágonos por categoria de vulnerabilidade
    vuln_counts = hex_gdf['vulnerabilidade'].value_counts()
    logger.info("\nDistribuição de vulnerabilidade:")
    for cat, count in vuln_counts.items():
        logger.info(f"  {cat}: {count} hexágonos ({count/len(hex_gdf)*100:.1f}%)")
    
    # Estatísticas de cobertura
    hex_cobertos = (hex_gdf['num_erbs'] > 0).sum()
    perc_cobertos = hex_cobertos / len(hex_gdf) * 100
    logger.info(f"\nEstatísticas de cobertura:")
    logger.info(f"  Total de hexágonos: {len(hex_gdf)}")
    logger.info(f"  Hexágonos com cobertura: {hex_cobertos} ({perc_cobertos:.1f}%)")
    logger.info(f"  Média de ERBs por hexágono: {hex_gdf['num_erbs'].mean():.2f}")
    logger.info(f"  Média de operadoras por hexágono: {hex_gdf['num_operadoras'].mean():.2f}")
    
    # Retornar GeoDataFrame atualizado e grade hexagonal
    return gdf, hex_gdf

def plot_hex_vulnerability_map(hex_gdf):
    """
    Gera um mapa de vulnerabilidade baseado na grade hexagonal.

    Args:
        hex_gdf (geopandas.GeoDataFrame): Grade hexagonal com análise de vulnerabilidade
    """
    logger.info("Gerando mapa de vulnerabilidade em grade hexagonal")
    plt.figure(figsize=(15, 12))
    ax = plt.subplot(111)

    # Definir cores para categorias de vulnerabilidade
    colors_vulnerabilidade = {
        'Sem cobertura': '#d73027',
        'Alta vulnerabilidade': '#fc8d59',
        'Média vulnerabilidade': '#fee090',
        'Baixa vulnerabilidade': '#e0f3f8'
    }

    # Plotar hexágonos por categoria de vulnerabilidade
    for categoria, color in colors_vulnerabilidade.items():
        subset = hex_gdf[hex_gdf['vulnerabilidade'] == categoria]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, alpha=0.7, label=categoria)

    # Adicionar mapa base
    try:
        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as e:
        logger.warning(f"Não foi possível adicionar mapa base: {e}")

    # Adicionar título e legenda
    plt.title('Análise de Vulnerabilidade de Cobertura por Hexágonos', fontsize=16)
    plt.legend(title='Vulnerabilidade', loc='upper right')

    # Salvar figura
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'erb_hex_vulnerability.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Função para demonstração com dados de exemplo
def run_voronoi_and_hexgrid_demo(data):
    """
    Demonstração de análise Voronoi e grade hexagonal com dados de amostra.
    
    Args:
        data: GeoDataFrame com dados de ERBs (de parte 1)
    """
    print("Iniciando análise de Voronoi e grade hexagonal...")
    
    # Calcular diagrama de Voronoi
    data, voronoi_gdf = calculate_voronoi(data)
    
    # Criar grade hexagonal
    data, hex_gdf = create_hexagon_coverage_grid(data, resolution=8)
    
    # Visualizar diagrama de Voronoi
    plot_voronoi_map(data, voronoi_gdf)
    
    # Visualizar mapa de vulnerabilidade
    plot_hex_vulnerability_map(hex_gdf)
    
    print("\nEstatísticas da análise de Voronoi:")
    print(f"Total de ERBs: {len(data)}")
    print(f"Total de polígonos Voronoi: {len(voronoi_gdf)}")
    print(f"Área média dos polígonos: {voronoi_gdf['area_voronoi_km2'].mean():.2f} km²")
    print(f"Média de vizinhos por ERB: {voronoi_gdf['num_vizinhos'].mean():.2f}")
    
    print("\nEstatísticas da grade hexagonal:")
    print(f"Total de hexágonos: {len(hex_gdf)}")
    print(f"Hexágonos com cobertura: {(hex_gdf['num_erbs'] > 0).sum()} ({(hex_gdf['num_erbs'] > 0).sum()/len(hex_gdf)*100:.1f}%)")
    
    # Distribuição de vulnerabilidade
    vulnerability_dist = hex_gdf['vulnerabilidade'].value_counts(normalize=True) * 100
    print("\nDistribuição de vulnerabilidade:")
    for cat, perc in vulnerability_dist.items():
        print(f"  {cat}: {perc:.1f}%")
    
    return data, voronoi_gdf, hex_gdf

print("Módulo de análise Voronoi e grade hexagonal carregado com sucesso!")
print("Para executar a demonstração, use a função run_voronoi_and_hexgrid_demo() com seus dados.")
print("Veja a próxima célula para continuar com a análise de rede e clustering espacial.") 