import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
import plotly.graph_objects as go
import os
import webbrowser
from branca.colormap import linear
import numpy as np
from datetime import datetime
import calendar
from typing import Optional, Tuple, List

# Configurar diretórios
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_GPKG = os.path.join(WORKSPACE_DIR, 'data', 'processed', 'inmet_processed.gpkg')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualizations', 'inmet')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_season(date: datetime) -> str:
    """Determina a estação do ano para uma data específica no hemisfério sul."""
    month = date.month
    day = date.day
    
    # Datas de início das estações no hemisfério sul
    seasons = {
        'Verão': (12, 22),
        'Outono': (3, 20),
        'Inverno': (6, 21),
        'Primavera': (9, 23)
    }
    
    current_season = 'Verão'  # Default
    for season, (start_month, start_day) in seasons.items():
        if (month > start_month) or (month == start_month and day >= start_day):
            current_season = season
    
    return current_season

def create_folium_weather_map(gdf, output_path):
    """Cria um mapa interativo das estações meteorológicas usando Folium."""
    print("Criando mapa interativo das estações meteorológicas...")
    
    # Verificar dados válidos
    gdf_valid = gdf[~gdf.geometry.isna()].copy()
    if len(gdf_valid) == 0:
        print("Não há geometrias válidas para criar o mapa")
        return None
    
    # Calcular centro do mapa
    center_lat = gdf_valid.geometry.y.mean()
    center_lon = gdf_valid.geometry.x.mean()
    
    # Criar mapa base
    m = folium.Map(location=[center_lat, center_lon], 
                   zoom_start=4,
                   tiles='CartoDB positron')
    
    # Adicionar cluster de marcadores
    marker_cluster = MarkerCluster().add_to(m)
    
    # Adicionar marcadores para cada estação
    for idx, row in gdf_valid.iterrows():
        # Preparar informações para o popup
        station_name = str(row['station']) if 'station' in row and pd.notna(row['station']) else "Estação"
        region = str(row['region']) if 'region' in row and pd.notna(row['region']) else "N/A"
        state = str(row['state']) if 'state' in row and pd.notna(row['state']) else "N/A"
        altitude = str(row['altitude_m']) if 'altitude_m' in row and pd.notna(row['altitude_m']) else "N/A"
        
        popup_text = f"""
        <b>{station_name}</b><br>
        Região: {region}<br>
        Estado: {state}<br>
        Altitude: {altitude} m<br>
        Lat: {row.geometry.y:.4f}°<br>
        Lon: {row.geometry.x:.4f}°
        """
        
        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='blue', icon='cloud', prefix='fa'),
            tooltip=station_name
        ).add_to(marker_cluster)
    
    # Adicionar controle de camadas
    folium.LayerControl().add_to(m)
    
    # Salvar mapa
    m.save(output_path)
    print(f"Mapa interativo salvo em: {output_path}")
    return output_path

def plot_temperature_distribution(df, output_dir):
    """Cria gráficos de distribuição de temperatura."""
    print("Criando gráficos de distribuição de temperatura...")
    
    # Boxplot de temperatura por região
    if 'region' in df.columns and 'temperature_c' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='region', y='temperature_c', data=df)
        plt.title('Distribuição de Temperatura por Região')
        plt.xlabel('Região')
        plt.ylabel('Temperatura (°C)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temperatura_por_regiao.png'))
        plt.close()

def plot_rainfall_analysis(df, output_dir):
    """Cria gráficos de análise de precipitação."""
    print("Criando gráficos de análise de precipitação...")
    
    if 'precipitation_mm' in df.columns and 'date' in df.columns:
        # Agregar precipitação por data
        daily_rain = df.groupby('date')['precipitation_mm'].sum().reset_index()
        
        # Gráfico de linha de precipitação ao longo do tempo
        plt.figure(figsize=(15, 5))
        plt.plot(daily_rain['date'], daily_rain['precipitation_mm'])
        plt.title('Precipitação Total Diária')
        plt.xlabel('Data')
        plt.ylabel('Precipitação (mm)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precipitacao_diaria.png'))
        plt.close()

def create_static_wind_rose(df: pd.DataFrame, output_dir: str):
    """Cria uma versão estática da rosa dos ventos usando matplotlib."""
    print("Criando rosa dos ventos estática...")
    
    if 'wind_direction_deg' not in df.columns or 'wind_speed_ms' not in df.columns:
        print("Dados de vento não disponíveis")
        return
    
    # Criar figura com subplot polar
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
    
    # Definir bins para direção e velocidade
    dir_bins = np.arange(0, 361, 10)
    speed_bins = np.arange(0, df['wind_speed_ms'].max() + 2, 2)
    
    # Calcular histograma 2D
    hist, _, _ = np.histogram2d(
        df['wind_direction_deg'],
        df['wind_speed_ms'],
        bins=[dir_bins, speed_bins]
    )
    
    # Normalizar os dados
    hist = hist / hist.sum() * 100
    
    # Plotar cada faixa de velocidade
    bottom = np.zeros(len(dir_bins) - 1)
    for i in range(len(speed_bins) - 1):
        values = hist[:, i]
        ax.bar(
            np.deg2rad(dir_bins[:-1]),
            values,
            width=np.deg2rad(10),
            bottom=bottom,
            label=f'{speed_bins[i]:.1f}-{speed_bins[i+1]:.1f} m/s'
        )
        bottom += values
    
    # Configurar o gráfico
    ax.set_theta_direction(-1)  # Sentido horário
    ax.set_theta_zero_location('N')  # 0° no Norte
    ax.set_title('Rosa dos Ventos (Frequência %)', pad=20)
    plt.legend(bbox_to_anchor=(1.2, 0.5), loc='center left')
    
    # Salvar
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rosa_dos_ventos_static.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_seasonal_analysis(df: pd.DataFrame, output_dir: str):
    """Cria análises sazonais dos dados meteorológicos."""
    print("Criando análises sazonais...")
    
    # Adicionar coluna de estação
    df['season'] = df['date'].apply(get_season)
    
    # 1. Temperatura por estação
    if 'temperature_c' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='season', y='temperature_c', data=df, 
                    order=['Verão', 'Outono', 'Inverno', 'Primavera'])
        plt.title('Distribuição de Temperatura por Estação')
        plt.xlabel('Estação')
        plt.ylabel('Temperatura (°C)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temperatura_por_estacao.png'))
        plt.close()
    
    # 2. Precipitação por estação
    if 'precipitation_mm' in df.columns:
        plt.figure(figsize=(12, 6))
        seasonal_rain = df.groupby('season')['precipitation_mm'].mean().reindex(['Verão', 'Outono', 'Inverno', 'Primavera'])
        seasonal_rain.plot(kind='bar')
        plt.title('Precipitação Média por Estação')
        plt.xlabel('Estação')
        plt.ylabel('Precipitação Média (mm)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precipitacao_por_estacao.png'))
        plt.close()
    
    # 3. Radiação por estação
    if 'radiation_kjm2' in df.columns:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='season', y='radiation_kjm2', data=df,
                    order=['Verão', 'Outono', 'Inverno', 'Primavera'])
        plt.title('Distribuição de Radiação Solar por Estação')
        plt.xlabel('Estação')
        plt.ylabel('Radiação (kJ/m²)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radiacao_por_estacao.png'))
        plt.close()

def plot_temporal_analysis(df: pd.DataFrame, output_dir: str):
    """Cria análises temporais dos dados meteorológicos."""
    print("Criando análises temporais...")
    
    # Definir colunas disponíveis para agregação
    agg_dict = {}
    if 'precipitation_mm' in df.columns:
        agg_dict['precipitation_mm'] = 'sum'
    if 'temperature_c' in df.columns:
        agg_dict['temperature_c'] = 'mean'
    if 'radiation_kjm2' in df.columns:
        agg_dict['radiation_kjm2'] = 'mean'
    
    if not agg_dict:
        print("Nenhuma coluna disponível para análise temporal")
        return
    
    # Agregar dados por data
    daily_data = df.groupby('date').agg(agg_dict).reset_index()
    
    # 1. Precipitação acumulada
    if 'precipitation_mm' in daily_data.columns:
        plt.figure(figsize=(15, 6))
        cumulative_rain = daily_data['precipitation_mm'].cumsum()
        plt.plot(daily_data['date'], cumulative_rain)
        plt.title('Precipitação Acumulada ao Longo do Tempo')
        plt.xlabel('Data')
        plt.ylabel('Precipitação Acumulada (mm)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'precipitacao_acumulada.png'))
        plt.close()
    
    # 2. Radiação solar ao longo do tempo
    if 'radiation_kjm2' in daily_data.columns:
        plt.figure(figsize=(15, 6))
        plt.plot(daily_data['date'], daily_data['radiation_kjm2'])
        plt.title('Radiação Solar Média Diária')
        plt.xlabel('Data')
        plt.ylabel('Radiação (kJ/m²)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'radiacao_temporal.png'))
        plt.close()
    
    # 3. Gráfico de calor mensal de temperatura
    if 'temperature_c' in daily_data.columns:
        # Criar colunas de ano e mês
        monthly_data = df.copy()
        monthly_data['year'] = monthly_data['date'].dt.year
        monthly_data['month'] = monthly_data['date'].dt.month
        
        # Agregar por ano e mês
        monthly_agg = monthly_data.groupby(['year', 'month'])['temperature_c'].mean().reset_index()
        monthly_agg['month_name'] = monthly_agg['month'].apply(lambda x: calendar.month_abbr[x])
        
        # Criar matriz para o heatmap
        pivot_temp = monthly_agg.pivot(
            index='year',
            columns='month_name',
            values='temperature_c'
        )
        
        # Reordenar as colunas para seguir a ordem dos meses
        month_order = [calendar.month_abbr[i] for i in range(1, 13)]
        pivot_temp = pivot_temp.reindex(columns=month_order)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_temp, cmap='RdYlBu_r', annot=True, fmt='.1f')
        plt.title('Temperatura Média Mensal')
        plt.xlabel('Mês')
        plt.ylabel('Ano')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temperatura_mensal_heatmap.png'))
        plt.close()

def create_wind_rose(df: pd.DataFrame, output_dir: str):
    """Cria um gráfico interativo de rosa dos ventos usando Plotly."""
    print("Criando rosa dos ventos interativa...")
    
    if 'wind_direction_deg' not in df.columns or 'wind_speed_ms' not in df.columns:
        print("Dados de vento não disponíveis")
        return
    
    # Criar bins para direção do vento
    dir_bins = np.arange(0, 361, 10)
    # Criar bins para velocidade do vento
    speed_bins = np.arange(0, df['wind_speed_ms'].max() + 2, 2)
    
    # Calcular histograma 2D
    wind_hist, _, _ = np.histogram2d(
        df['wind_direction_deg'],
        df['wind_speed_ms'],
        bins=[dir_bins, speed_bins]
    )
    
    # Criar gráfico polar
    fig = go.Figure()
    
    for i in range(len(speed_bins)-1):
        fig.add_trace(go.Barpolar(
            r=wind_hist[:, i],
            theta=dir_bins[:-1],
            name=f'{speed_bins[i]:.1f}-{speed_bins[i+1]:.1f} m/s',
            width=10
        ))
    
    fig.update_layout(
        title='Rosa dos Ventos',
        showlegend=True,
        polar=dict(
            radialaxis=dict(showticklabels=False, ticks=''),
            angularaxis=dict(direction="clockwise", rotation=90)
        )
    )
    
    fig.write_html(os.path.join(output_dir, 'rosa_dos_ventos.html'))

def main():
    """Função principal para criar todas as visualizações."""
    print(f"Criando visualizações para os dados meteorológicos do INMET...")
    print(f"Usando dados do arquivo: {INPUT_GPKG}")
    print(f"As visualizações serão salvas em: {OUTPUT_DIR}")
    
    try:
        # Carregar dados
        gdf = gpd.read_file(INPUT_GPKG)
        print(f"Dados carregados: {len(gdf)} registros")
        print(f"Colunas disponíveis: {gdf.columns.tolist()}")
        
        # Converter para DataFrame para análises
        df = pd.DataFrame(gdf.drop(columns='geometry'))
        
        # Criar coluna de data (dados começam em 01/01/2024)
        start_date = pd.Timestamp('2024-01-01')
        df['date'] = pd.date_range(start=start_date, periods=len(df), freq='h')  # Usando 'h' em vez de 'H'
        
        # Criar visualizações
        create_wind_rose(df, OUTPUT_DIR)  # Versão interativa
        create_static_wind_rose(df, OUTPUT_DIR)  # Versão estática
        plot_seasonal_analysis(df, OUTPUT_DIR)  # Análises sazonais
        plot_temporal_analysis(df, OUTPUT_DIR)  # Análises temporais
        
        print("Todas as visualizações foram criadas com sucesso!")
        print(f"As visualizações foram salvas em: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Erro durante a criação das visualizações: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 