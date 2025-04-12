#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para gerar uma visualização HTML do relatório de qualidade para áreas naturais.

Este script lê o arquivo JSON do relatório de qualidade e gera um arquivo HTML
com visualizações e tabelas para facilitar a interpretação dos dados.

Autor: Usuário
Data: 2024
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import base64
from io import BytesIO

# Configurar caminhos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
INPUT_PATH = os.path.join(BASE_DIR, "src/preprocessing/quality_reports/nature/quality_report_natural_areas.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs/reports/nature/quality_report.html")

# Cores para gráficos
COLORS = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854', '#ffd92f', '#e5c494', '#b3b3b3']

def load_report(report_path):
    """Carrega o relatório de qualidade do arquivo JSON"""
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        print(f"Relatório carregado com sucesso: {report_path}")
        return report
    except Exception as e:
        print(f"Erro ao carregar o relatório: {e}")
        return None

def fig_to_base64(fig):
    """Converte uma figura matplotlib para base64 para incluir no HTML"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

def create_area_chart(report):
    """Cria um gráfico de distribuição de áreas"""
    area_stats = report.get('area_statistics', {})
    if not area_stats:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['min_area_ha', 'median_area_ha', 'mean_area_ha', 'max_area_ha']
    values = [area_stats.get(metric, 0) for metric in metrics]
    labels = ['Mínima', 'Mediana', 'Média', 'Máxima']
    
    ax.bar(labels, values, color=COLORS)
    ax.set_title('Estatísticas de Área (hectares)', fontsize=14)
    ax.set_ylabel('Área (hectares)', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar rótulos nas barras
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=10)
    
    # Adicionar valor total como texto no gráfico
    total_area = area_stats.get('total_area_ha', 0)
    ax.text(0.5, 0.9, f'Área Total: {total_area:.2f} hectares', 
            horizontalalignment='center', verticalalignment='center',
            transform=ax.transAxes, fontsize=12, 
            bbox=dict(facecolor='#f0f0f0', alpha=0.5, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    return fig

def create_type_chart(report):
    """Cria um gráfico de distribuição por tipo"""
    type_dist = report.get('type_distribution', {})
    if not type_dist:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    types = list(type_dist.keys())
    counts = [type_dist.get(t, 0) for t in types]
    
    # Ordenar por contagem
    sorted_indices = np.argsort(counts)[::-1]
    types = [types[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]
    
    ax.bar(types, counts, color=COLORS[:len(types)])
    ax.set_title('Distribuição por Tipo de Área Natural', fontsize=14)
    ax.set_ylabel('Número de Áreas', fontsize=12)
    ax.set_xlabel('Tipo', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar rótulos nas barras
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def create_importance_chart(report):
    """Cria um gráfico de importância ecológica"""
    eco_importance = report.get('ecological_importance', {})
    if not eco_importance:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    importance_levels = ['very_high', 'high', 'medium', 'low', 'very_low']
    counts = [eco_importance.get(level, 0) for level in importance_levels]
    labels = ['Muito Alta', 'Alta', 'Média', 'Baixa', 'Muito Baixa']
    
    # Filtrar zeros
    valid_indices = [i for i, count in enumerate(counts) if count > 0]
    labels = [labels[i] for i in valid_indices]
    counts = [counts[i] for i in valid_indices]
    colors = [COLORS[i % len(COLORS)] for i in valid_indices]
    
    ax.bar(labels, counts, color=colors)
    ax.set_title('Distribuição por Importância Ecológica', fontsize=14)
    ax.set_ylabel('Número de Áreas', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adicionar rótulos nas barras
    for i, v in enumerate(counts):
        ax.text(i, v + 0.5, str(v), ha='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def create_html_report(report, output_path):
    """Gera um relatório HTML com visualizações"""
    if not report:
        print("Relatório não disponível para gerar HTML.")
        return False
    
    # Gerar gráficos
    area_chart = create_area_chart(report)
    area_chart_base64 = fig_to_base64(area_chart) if area_chart else None
    
    type_chart = create_type_chart(report)
    type_chart_base64 = fig_to_base64(type_chart) if type_chart else None
    
    importance_chart = create_importance_chart(report)
    importance_chart_base64 = fig_to_base64(importance_chart) if importance_chart else None
    
    # Formatar timestamp
    timestamp = report.get('timestamp', '')
    try:
        timestamp_dt = datetime.fromisoformat(timestamp)
        formatted_timestamp = timestamp_dt.strftime('%d/%m/%Y %H:%M:%S')
    except:
        formatted_timestamp = timestamp
    
    # Construir HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt-BR">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Relatório de Qualidade - Áreas Naturais</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                background-color: #f5f5f5;
            }}
            .container {{
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                margin-bottom: 20px;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                padding-bottom: 10px;
                border-bottom: 2px solid #3498db;
                margin-top: 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px 15px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: bold;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            .chart-container {{
                margin: 20px 0;
                text-align: center;
            }}
            .chart {{
                max-width: 100%;
                height: auto;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                font-size: 0.9em;
                color: #7f8c8d;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .stats-card {{
                background-color: #f8f9fa;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }}
            .stats-card h3 {{
                margin-top: 0;
                font-size: 16px;
                color: #7f8c8d;
            }}
            .stats-card p {{
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0 0 0;
                color: #2c3e50;
            }}
            .validation-grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
            @media screen and (max-width: 768px) {{
                .validation-grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Relatório de Qualidade - Áreas Naturais</h1>
            <p><strong>Data do Relatório:</strong> {formatted_timestamp}</p>
            <p><strong>Sistema de Coordenadas:</strong> {report.get('crs', 'Não especificado')}</p>
            
            <div class="stats-grid">
                <div class="stats-card">
                    <h3>Total de Áreas Naturais</h3>
                    <p>{report.get('total_features', 0)}</p>
                </div>
                <div class="stats-card">
                    <h3>Área Total (hectares)</h3>
                    <p>{report.get('area_statistics', {}).get('total_area_ha', 0):.2f}</p>
                </div>
                <div class="stats-card">
                    <h3>Geometrias Válidas</h3>
                    <p>{report.get('validation_results', {}).get('valid_records', 0)}</p>
                </div>
                <div class="stats-card">
                    <h3>Área Média (hectares)</h3>
                    <p>{report.get('area_statistics', {}).get('mean_area_ha', 0):.2f}</p>
                </div>
            </div>
        </div>
        
        <div class="container">
            <h2>Visualizações</h2>
            
            <div class="chart-container">
                <h3>Estatísticas de Área</h3>
                {f'<img class="chart" src="data:image/png;base64,{area_chart_base64}" alt="Estatísticas de Área">' if area_chart_base64 else '<p>Não há dados suficientes para gerar este gráfico.</p>'}
            </div>
            
            <div class="chart-container">
                <h3>Distribuição por Tipo</h3>
                {f'<img class="chart" src="data:image/png;base64,{type_chart_base64}" alt="Distribuição por Tipo">' if type_chart_base64 else '<p>Não há dados suficientes para gerar este gráfico.</p>'}
            </div>
            
            <div class="chart-container">
                <h3>Importância Ecológica</h3>
                {f'<img class="chart" src="data:image/png;base64,{importance_chart_base64}" alt="Importância Ecológica">' if importance_chart_base64 else '<p>Não há dados suficientes para gerar este gráfico.</p>'}
            </div>
        </div>
        
        <div class="container">
            <h2>Validação e Geometria</h2>
            
            <div class="validation-grid">
                <div>
                    <h3>Resultados de Validação</h3>
                    <table>
                        <tr>
                            <th>Métrica</th>
                            <th>Valor</th>
                        </tr>
                        <tr>
                            <td>Total de Registros</td>
                            <td>{report.get('validation_results', {}).get('total_records', 0)}</td>
                        </tr>
                        <tr>
                            <td>Registros Válidos</td>
                            <td>{report.get('validation_results', {}).get('valid_records', 0)}</td>
                        </tr>
                        <tr>
                            <td>Geometrias Inválidas</td>
                            <td>{report.get('validation_results', {}).get('invalid_geometries', 0)}</td>
                        </tr>
                        <tr>
                            <td>Colunas Ausentes</td>
                            <td>{', '.join(report.get('validation_results', {}).get('missing_columns', [])) or 'Nenhuma'}</td>
                        </tr>
                    </table>
                </div>
                
                <div>
                    <h3>Relatório de Geometria</h3>
                    <table>
                        <tr>
                            <th>Métrica</th>
                            <th>Valor</th>
                        </tr>
                        <tr>
                            <td>Total de Geometrias</td>
                            <td>{report.get('geometry_report', {}).get('total_geometries', 0)}</td>
                        </tr>
                        <tr>
                            <td>Inicialmente Inválidas</td>
                            <td>{report.get('geometry_report', {}).get('initially_invalid', 0)}</td>
                        </tr>
                        <tr>
                            <td>Geometrias Corrigidas</td>
                            <td>{report.get('geometry_report', {}).get('fixed_geometries', 0)}</td>
                        </tr>
                        <tr>
                            <td>Geometrias Não Corrigíveis</td>
                            <td>{report.get('geometry_report', {}).get('unfixable_geometries', 0)}</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
        
        <div class="container">
            <h2>Estatísticas Detalhadas</h2>
            
            <h3>Estatísticas de Área</h3>
            <table>
                <tr>
                    <th>Métrica</th>
                    <th>Valor (hectares)</th>
                </tr>
                <tr>
                    <td>Área Mínima</td>
                    <td>{report.get('area_statistics', {}).get('min_area_ha', 0):.4f}</td>
                </tr>
                <tr>
                    <td>Área Máxima</td>
                    <td>{report.get('area_statistics', {}).get('max_area_ha', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Área Média</td>
                    <td>{report.get('area_statistics', {}).get('mean_area_ha', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Área Mediana</td>
                    <td>{report.get('area_statistics', {}).get('median_area_ha', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Área Total</td>
                    <td>{report.get('area_statistics', {}).get('total_area_ha', 0):.2f}</td>
                </tr>
                <tr>
                    <td>Desvio Padrão</td>
                    <td>{report.get('area_statistics', {}).get('std_dev_area_ha', 0):.2f}</td>
                </tr>
            </table>
            
            <h3>Distribuição por Importância Ecológica</h3>
            <table>
                <tr>
                    <th>Nível de Importância</th>
                    <th>Número de Áreas</th>
                </tr>
                {''.join([f"<tr><td>{level.replace('_', ' ').title()}</td><td>{count}</td></tr>" for level, count in report.get('ecological_importance', {}).items()])}
            </table>
            
            <h3>Distribuição por Tipo</h3>
            <table>
                <tr>
                    <th>Tipo</th>
                    <th>Número de Áreas</th>
                </tr>
                {''.join([f"<tr><td>{type_name}</td><td>{count}</td></tr>" for type_name, count in report.get('type_distribution', {}).items()])}
            </table>
        </div>
        
        <div class="footer">
            <p>Relatório gerado automaticamente em {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            <p>Geoprocessing - Análise de Áreas Naturais</p>
        </div>
    </body>
    </html>
    """
    
    try:
        # Criar diretório de saída se não existir
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Salvar arquivo HTML
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Relatório HTML gerado com sucesso: {output_path}")
        return True
    except Exception as e:
        print(f"Erro ao gerar relatório HTML: {e}")
        return False

def main():
    """Função principal"""
    print(f"Gerando relatório de qualidade para áreas naturais...")
    print(f"Arquivo de entrada: {INPUT_PATH}")
    print(f"Arquivo de saída: {OUTPUT_PATH}")
    
    # Carregar relatório
    report = load_report(INPUT_PATH)
    if not report:
        print("Não foi possível carregar o relatório. Abortando.")
        return False
    
    # Gerar relatório HTML
    success = create_html_report(report, OUTPUT_PATH)
    
    if success:
        print(f"Processo concluído com sucesso. Relatório salvo em: {OUTPUT_PATH}")
    else:
        print("Erro ao gerar relatório HTML.")
    
    return success

if __name__ == "__main__":
    main() 