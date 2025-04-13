#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import geopandas as gpd
import json
import fiona
from shapely.geometry import box
from datetime import datetime
import matplotlib.pyplot as plt
import contextily as ctx
import seaborn as sns
from tqdm import tqdm
import numpy as np

# Diretório de dados enriquecidos
DATA_DIR = r"F:\TESE_MESTRADO\geoprocessing\data\enriched_data"
OUTPUT_DIR = r"F:\TESE_MESTRADO\geoprocessing\outputs\reports"

# Garantir que o diretório de saída exista
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Função para extrair metadados de uma camada GeoDataFrame
def extract_layer_metadata(gdf, layer_name, file_path):
    """Extrai metadados de uma camada GeoDataFrame."""
    # Informações básicas
    metadata = {
        "layer_name": layer_name,
        "file_path": file_path,
        "crs": str(gdf.crs),
        "feature_count": len(gdf),
        "geometry_types": list(gdf.geometry.type.unique()),
        "columns": list(gdf.columns),
        "attribute_samples": {},
        "bounds": {
            "minx": float(gdf.total_bounds[0]),
            "miny": float(gdf.total_bounds[1]), 
            "maxx": float(gdf.total_bounds[2]),
            "maxy": float(gdf.total_bounds[3])
        }
    }
    
    # Amostras de atributos (primeiros 5 valores únicos de cada coluna)
    for col in gdf.columns:
        if col != 'geometry':
            try:
                unique_values = gdf[col].dropna().unique()
                # Limitar a 5 valores para não sobrecarregar o relatório
                sample_values = [str(v) for v in unique_values[:5]]
                metadata["attribute_samples"][col] = sample_values
                
                # Adicionar estatísticas para colunas numéricas
                if pd.api.types.is_numeric_dtype(gdf[col]):
                    metadata["attribute_samples"][f"{col}_stats"] = {
                        "min": float(gdf[col].min()) if not pd.isna(gdf[col].min()) else None,
                        "max": float(gdf[col].max()) if not pd.isna(gdf[col].max()) else None,
                        "mean": float(gdf[col].mean()) if not pd.isna(gdf[col].mean()) else None,
                        "std": float(gdf[col].std()) if not pd.isna(gdf[col].std()) else None
                    }
            except Exception as e:
                metadata["attribute_samples"][col] = f"Erro ao extrair amostra: {str(e)}"
    
    return metadata

# Função para gerar um mapa rápido da camada
def generate_quick_map(gdf, layer_name, file_name, output_dir):
    """Gera um mapa rápido da camada para visualização."""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Verificar tipo de geometria e ajustar visualização
        geom_types = gdf.geometry.type.unique()
        
        if 'Point' in geom_types:
            gdf.plot(ax=ax, markersize=5, alpha=0.7)
        elif 'LineString' in geom_types or 'MultiLineString' in geom_types:
            gdf.plot(ax=ax, linewidth=1, alpha=0.7)
        else:  # Polygon ou MultiPolygon
            # Se tiver muitos polígonos, usar transparência maior
            alpha = 0.3 if len(gdf) > 100 else 0.7
            gdf.plot(ax=ax, alpha=alpha)
        
        # Adicionar mapa base se possível
        try:
            ctx.add_basemap(ax, crs=gdf.crs)
        except Exception:
            pass  # Ignorar erros ao adicionar mapa base
        
        ax.set_title(f"{file_name} - {layer_name}")
        plt.tight_layout()
        
        # Salvar figura
        map_path = os.path.join(output_dir, f"{file_name}_{layer_name}_map.png")
        plt.savefig(map_path, dpi=150)
        plt.close(fig)
        
        return map_path
    except Exception as e:
        print(f"Erro ao gerar mapa para {layer_name}: {str(e)}")
        return None

# Função principal para escanear arquivos e gerar relatório
def scan_enriched_data():
    """Escaneia todos os arquivos geoespaciais e gera um relatório detalhado."""
    print("Iniciando escaneamento de dados enriquecidos...")
    
    # Timestamp para o relatório
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Diretório para mapas
    maps_dir = os.path.join(OUTPUT_DIR, f"maps_{timestamp}")
    os.makedirs(maps_dir, exist_ok=True)
    
    # Dicionário para armazenar metadados
    data_catalog = {
        "scan_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_directory": DATA_DIR,
        "files": []
    }
    
    # Listar todos os arquivos no diretório
    all_files = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith(('.gpkg', '.shp', '.geojson')):
                all_files.append(os.path.join(root, file))
    
    print(f"Encontrados {len(all_files)} arquivos geoespaciais.")
    
    # Processar cada arquivo
    for file_path in tqdm(all_files, desc="Processando arquivos"):
        file_name = os.path.basename(file_path)
        relative_path = os.path.relpath(file_path, DATA_DIR)
        
        file_info = {
            "file_name": file_name,
            "file_path": relative_path,
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
            "layers": []
        }
        
        try:
            # Listar camadas no arquivo
            layers = fiona.listlayers(file_path)
            file_info["layer_count"] = len(layers)
            
            # Processar cada camada
            for layer_name in layers:
                try:
                    # Abrir camada como GeoDataFrame
                    gdf = gpd.read_file(file_path, layer=layer_name)
                    
                    # Extrair metadados da camada
                    layer_metadata = extract_layer_metadata(gdf, layer_name, file_path)
                    
                    # Gerar mapa para visualização rápida
                    map_path = generate_quick_map(gdf, layer_name, file_name.split('.')[0], maps_dir)
                    if map_path:
                        layer_metadata["quick_map"] = os.path.basename(map_path)
                    
                    # Adicionar metadados da camada ao catálogo
                    file_info["layers"].append(layer_metadata)
                    
                except Exception as e:
                    file_info["layers"].append({
                        "layer_name": layer_name,
                        "error": f"Erro ao processar camada: {str(e)}"
                    })
            
        except Exception as e:
            file_info["error"] = f"Erro ao processar arquivo: {str(e)}"
        
        # Adicionar informações do arquivo ao catálogo
        data_catalog["files"].append(file_info)
    
    # Salvar catálogo como JSON
    catalog_path = os.path.join(OUTPUT_DIR, f"enriched_data_catalog_{timestamp}.json")
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(data_catalog, f, indent=2, ensure_ascii=False)
    
    # Gerar relatório HTML mais amigável
    generate_html_report(data_catalog, timestamp)
    
    # Gerar resumo consolidado em CSV
    generate_summary_csv(data_catalog, timestamp)
    
    print(f"Escaneamento concluído! Relatórios salvos em {OUTPUT_DIR}")
    print(f"Catálogo JSON: {catalog_path}")
    print(f"Relatório HTML: {os.path.join(OUTPUT_DIR, f'enriched_data_report_{timestamp}.html')}")
    print(f"Resumo CSV: {os.path.join(OUTPUT_DIR, f'enriched_data_summary_{timestamp}.csv')}")
    
    return data_catalog

# Função para gerar relatório HTML
def generate_html_report(data_catalog, timestamp):
    """Gera um relatório HTML a partir do catálogo de metadados."""
    html_path = os.path.join(OUTPUT_DIR, f"enriched_data_report_{timestamp}.html")
    
    # Iniciar HTML
    html = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Relatório de Dados Enriquecidos</title>
        <style>
            body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }
            h1, h2, h3 { color: #333; }
            .container { max-width: 1200px; margin: 0 auto; }
            .file-card { border: 1px solid #ddd; border-radius: 5px; margin-bottom: 20px; padding: 15px; background-color: #f9f9f9; }
            .layer-card { border: 1px solid #eee; border-radius: 5px; margin: 10px 0; padding: 10px; background-color: white; }
            .metadata-table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            .metadata-table th, .metadata-table td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            .metadata-table th { background-color: #f2f2f2; }
            .error { color: red; }
            .map-img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }
            .columns-list { column-count: 3; column-gap: 20px; }
            .stats-table { width: 100%; margin-top: 5px; font-size: 0.9em; }
            .summary { background-color: #e9f7ef; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Dados Geoespaciais Enriquecidos</h1>
            <p>Data de escaneamento: """ + data_catalog["scan_date"] + """</p>
            <p>Diretório: """ + data_catalog["data_directory"] + """</p>
            
            <div class="summary">
                <h2>Resumo</h2>
                <p>Total de arquivos: """ + str(len(data_catalog["files"])) + """</p>
            </div>
    """
    
    # Adicionar informações de cada arquivo
    for file_info in data_catalog["files"]:
        html += f"""
            <div class="file-card">
                <h2>{file_info['file_name']}</h2>
                <p>Caminho: {file_info['file_path']}</p>
                <p>Tamanho: {file_info['file_size_mb']} MB</p>
                <p>Número de camadas: {file_info.get('layer_count', 0)}</p>
        """
        
        if "error" in file_info:
            html += f'<p class="error">Erro: {file_info["error"]}</p>'
        
        # Adicionar informações de cada camada
        for layer in file_info.get("layers", []):
            html += f"""
                <div class="layer-card">
                    <h3>Camada: {layer.get('layer_name', 'Desconhecido')}</h3>
            """
            
            if "error" in layer:
                html += f'<p class="error">Erro: {layer["error"]}</p>'
                continue
            
            html += f"""
                    <table class="metadata-table">
                        <tr><th>Propriedade</th><th>Valor</th></tr>
                        <tr><td>Sistema de Coordenadas</td><td>{layer.get('crs', 'Desconhecido')}</td></tr>
                        <tr><td>Número de Feições</td><td>{layer.get('feature_count', 0)}</td></tr>
                        <tr><td>Tipos de Geometria</td><td>{', '.join(layer.get('geometry_types', []))}</td></tr>
                    </table>
                    
                    <h4>Colunas:</h4>
                    <div class="columns-list">
            """
            
            # Listar colunas
            for col in layer.get('columns', []):
                if col != 'geometry':
                    html += f'<p>{col}</p>'
            
            html += '</div>'
            
            # Adicionar visualização da camada se disponível
            if "quick_map" in layer:
                html += f'<img src="maps_{timestamp}/{layer["quick_map"]}" alt="Mapa da camada {layer["layer_name"]}" class="map-img">'
            
            # Adicionar exemplos de valores de atributos
            html += '<h4>Exemplos de valores de atributos:</h4>'
            
            for attr, values in layer.get('attribute_samples', {}).items():
                if attr.endswith('_stats'):
                    # Exibir estatísticas para colunas numéricas
                    base_attr = attr.replace('_stats', '')
                    stats = values
                    
                    html += f"""
                        <table class="stats-table">
                            <tr>
                                <th>{base_attr}</th>
                                <th>Mínimo</th>
                                <th>Máximo</th>
                                <th>Média</th>
                                <th>Desvio Padrão</th>
                            </tr>
                            <tr>
                                <td>Estatísticas</td>
                                <td>{stats.get('min')}</td>
                                <td>{stats.get('max')}</td>
                                <td>{stats.get('mean')}</td>
                                <td>{stats.get('std')}</td>
                            </tr>
                        </table>
                    """
                elif not attr.endswith('_stats'):  # Ignorar as entradas de estatísticas já processadas
                    html += f'<p><strong>{attr}:</strong> {", ".join([str(v) for v in values])}</p>'
            
            html += '</div>'
        
        html += '</div>'
    
    # Finalizar HTML
    html += """
        </div>
    </body>
    </html>
    """
    
    # Salvar HTML
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

# Função para gerar resumo em CSV
def generate_summary_csv(data_catalog, timestamp):
    """Gera um arquivo CSV de resumo com as principais informações dos dados."""
    csv_path = os.path.join(OUTPUT_DIR, f"enriched_data_summary_{timestamp}.csv")
    
    # Preparar dados para o CSV
    rows = []
    
    for file_info in data_catalog["files"]:
        file_name = file_info['file_name']
        file_path = file_info['file_path']
        
        for layer in file_info.get("layers", []):
            if "error" in layer:
                continue
                
            row = {
                "arquivo": file_name,
                "caminho": file_path,
                "camada": layer.get("layer_name", ""),
                "crs": layer.get("crs", ""),
                "num_feicoes": layer.get("feature_count", 0),
                "tipos_geometria": ", ".join(layer.get("geometry_types", [])),
                "colunas": ", ".join([c for c in layer.get("columns", []) if c != "geometry"]),
                "minx": layer.get("bounds", {}).get("minx", ""),
                "miny": layer.get("bounds", {}).get("miny", ""),
                "maxx": layer.get("bounds", {}).get("maxx", ""),
                "maxy": layer.get("bounds", {}).get("maxy", "")
            }
            
            rows.append(row)
    
    # Criar DataFrame e salvar como CSV
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding='utf-8')

# Executar se for o script principal
if __name__ == "__main__":
    scan_enriched_data() 