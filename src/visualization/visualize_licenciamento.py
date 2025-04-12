import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import MarkerCluster, HeatMap
import plotly.express as px
import os
import webbrowser
from branca.colormap import linear
import numpy as np

# Configurar diretórios
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_GPKG = os.path.join(WORKSPACE_DIR, 'data', 'processed', 'licenciamento_processed.gpkg')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'outputs', 'visualizations', 'rbs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Criando visualizações para os dados de licenciamento de telecomunicações...")
print(f"Usando dados do arquivo: {INPUT_GPKG}")
print(f"As visualizações serão salvas em: {OUTPUT_DIR}")

# Carregar os dados geoespaciais
print("Carregando dados do GeoPackage...")
gdf = gpd.read_file(INPUT_GPKG)
print(f"Dados carregados: {len(gdf)} registros")

# Verificação rápida dos dados
print("\nInformações do DataFrame geoespacial:")
print(f"Sistema de coordenadas: {gdf.crs}")
print(f"Colunas disponíveis: {', '.join(gdf.columns)}")

# Preparação para visualizações - converter campos relevantes para tipos numéricos
numeric_columns = ['FreqTxMHz', 'FreqRxMHz', 'AlturaAntena', 'GanhoAntena', 'PotenciaTransmissorWatts']
for col in numeric_columns:
    if col in gdf.columns:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce')

# Funções para criar as visualizações
def create_folium_map(gdf, output_path):
    """Cria um mapa interativo usando Folium com clusters e popups."""
    print("Criando mapa interativo com Folium...")
    
    # Verificar se o DataFrame tem pontos válidos
    if gdf is None or len(gdf) == 0:
        print("Não há dados disponíveis para criar o mapa Folium")
        return None
    
    # Verificar se as geometrias são válidas e filtrar registros inválidos
    gdf_valid = gdf[~gdf.geometry.isna()].copy()
    if len(gdf_valid) == 0:
        print("Não há geometrias válidas para criar o mapa Folium")
        return None
    
    print(f"Criando mapa com {len(gdf_valid)} pontos válidos")
    
    # Calcular o centro aproximado para o mapa
    try:
        # Tentar usar o centro real dos dados
        center_lat = gdf_valid.geometry.y.mean()
        center_lon = gdf_valid.geometry.x.mean()
    except Exception as e:
        print(f"Erro ao calcular centro do mapa: {str(e)}")
        # Valores padrão para o Brasil
        center_lat = -15.77972
        center_lon = -47.92972  # Coordenadas de Brasília
    
    # Inicializar o mapa
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9,
                 tiles='CartoDB positron', control_scale=True)
    
    # Adicionar camada de cluster
    marker_cluster = MarkerCluster().add_to(m)
    
    # Limitar o número de pontos para evitar sobrecarga
    max_points = 1000  # Limite para número de marcadores para evitar navegadores travarem
    if len(gdf_valid) > max_points:
        print(f"O conjunto de dados é muito grande ({len(gdf_valid)} pontos). Limitando a {max_points} pontos para marcadores.")
        # Amostragem sistemática para manter distribuição espacial
        step = len(gdf_valid) // max_points
        gdf_sample = gdf_valid.iloc[::step].copy()
    else:
        gdf_sample = gdf_valid.copy()
    
    # Adicionar marcadores com informações
    markers_added = 0
    for idx, row in gdf_sample.iterrows():
        try:
            # Preparar popup com informações relevantes (tratando valores nulos)
            nome_entidade = str(row['NomeEntidade']) if pd.notna(row['NomeEntidade']) else "Desconhecida"
            tecnologia = str(row['Tecnologia']) if pd.notna(row['Tecnologia']) else "Desconhecida"
            municipio = str(row['Municipio.NomeMunicipio']) if pd.notna(row['Municipio.NomeMunicipio']) else "Desconhecido"
            altura = str(row['AlturaAntena']) if pd.notna(row['AlturaAntena']) else "N/A"
            freq_tx = str(row['FreqTxMHz']) if pd.notna(row['FreqTxMHz']) else "N/A"
            potencia = str(row['PotenciaTransmissorWatts']) if pd.notna(row['PotenciaTransmissorWatts']) else "N/A"
            
            popup_text = f"""
            <b>{nome_entidade}</b><br>
            Tecnologia: {tecnologia}<br>
            Município: {municipio}<br>
            Altura da Antena: {altura} m<br>
            Frequência TX: {freq_tx} MHz<br>
            Potência: {potencia} W<br>
            """
            
            # Definir cores por tecnologia
            color = 'blue'  # padrão
            if pd.notna(row['Tecnologia']):
                tech = str(row['Tecnologia']).upper() if isinstance(row['Tecnologia'], str) else str(row['Tecnologia'])
                if 'LTE' in tech:
                    color = 'red'
                elif 'GSM' in tech:
                    color = 'green'
                elif 'WCDMA' in tech:
                    color = 'purple'
                elif 'NR' in tech or '5G' in tech:
                    color = 'orange'
            
            # Verificar se as coordenadas são válidas
            lat = float(row.geometry.y)
            lon = float(row.geometry.x)
            if not (np.isnan(lat) or np.isnan(lon)):
                # Adicionar marcador ao cluster
                folium.Marker(
                    location=[lat, lon],
                    popup=folium.Popup(popup_text, max_width=300),
                    icon=folium.Icon(color=color, icon='signal', prefix='fa'),
                    tooltip=f"{nome_entidade[:20]}"  # Limitar tamanho para evitar problemas
                ).add_to(marker_cluster)
                markers_added += 1
        except Exception as e:
            print(f"Erro ao adicionar marcador #{idx}: {str(e)}")
            continue
    
    print(f"Adicionados {markers_added} marcadores ao mapa")
    
    # Adicionar controle de camadas
    folium.LayerControl().add_to(m)
    
    # Adicionar mapa de calor como camada adicional
    try:
        heat_data = []
        heat_max = 2000  # Limitar pontos no mapa de calor para melhor desempenho
        sample_step = max(1, len(gdf_valid) // heat_max)
        
        for idx, row in gdf_valid.iloc[::sample_step].iterrows():
            try:
                lat = float(row.geometry.y)
                lon = float(row.geometry.x)
                if not (np.isnan(lat) or np.isnan(lon)):
                    heat_data.append([lat, lon])
            except Exception:
                continue
                
        if heat_data:
            print(f"Adicionando mapa de calor com {len(heat_data)} pontos")
            HeatMap(heat_data, radius=15, blur=10, gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
    except Exception as e:
        print(f"Erro ao criar mapa de calor: {str(e)}")
    
    # Adicionar título e legenda - usando HTML simples para evitar erros
    title_html = '''
        <div style="position: fixed; top: 10px; left: 50px; width: 250px; height: 120px; 
                 background-color: white; padding: 10px; border: 2px solid gray; z-index: 9999;">
            <h3 style="margin: 0;">Estações de Telecomunicações</h3>
            <p style="margin-top: 5px;">Legenda:</p>
            <ul style="margin: 0; padding-left: 20px;">
                <li><span style="color: red;">●</span> LTE</li>
                <li><span style="color: green;">●</span> GSM</li>
                <li><span style="color: purple;">●</span> WCDMA</li>
                <li><span style="color: orange;">●</span> 5G (NR)</li>
                <li><span style="color: blue;">●</span> Outros</li>
            </ul>
        </div>
    '''
    
    # Adicionar legenda de forma mais segura
    try:
        m.get_root().html.add_child(folium.Element(title_html))
    except Exception as e:
        print(f"Erro ao adicionar legenda: {str(e)}")
        # Continuar sem a legenda
    
    # Salvar mapa
    try:
        m.save(output_path)
        print(f"Mapa interativo Folium salvo em: {output_path}")
        return output_path
    except Exception as e:
        print(f"Erro ao salvar mapa Folium: {str(e)}")
        # Tentar salvar versão simplificada
        try:
            # Criar mapa mais simples
            simple_map = folium.Map(location=[center_lat, center_lon], zoom_start=9)
            folium.TileLayer('CartoDB positron').add_to(simple_map)
            
            # Adicionar pontos diretamente sem popup
            for i, (idx, row) in enumerate(gdf_valid.iloc[::50].iterrows()):  # Usar apenas 1 a cada 50 pontos
                if i > 100:  # Limitar para apenas 100 pontos
                    break
                try:
                    lat = float(row.geometry.y)
                    lon = float(row.geometry.x)
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=3,
                        color='blue',
                        fill=True
                    ).add_to(simple_map)
                except:
                    continue
                    
            simple_output = output_path.replace('.html', '_simple.html')
            simple_map.save(simple_output)
            print(f"Mapa simples salvo em: {simple_output}")
            return simple_output
        except Exception as e2:
            print(f"Erro ao salvar mapa simples: {str(e2)}")
            return None

def create_matplotlib_maps(gdf, output_dir):
    """Cria mapas estáticos usando Matplotlib e Geopandas."""
    print("Criando mapas estáticos com Matplotlib...")
    
    # 1. Mapa de distribuição por tecnologia
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Verificar se há tecnologias únicas
    tecnologias_unicas = gdf['Tecnologia'].dropna().unique()
    print(f"Tecnologias únicas encontradas: {len(tecnologias_unicas)}")
    print(tecnologias_unicas)
    
    # Verificar se há algum valor nulo na coluna Tecnologia
    if gdf['Tecnologia'].isna().any():
        # Substituir NaN por 'Desconhecida'
        gdf['Tecnologia'] = gdf['Tecnologia'].fillna('Desconhecida')
        print("Valores nulos em Tecnologia substituídos por 'Desconhecida'")
    
    # Criar uma paleta de cores categórica
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    if len(tecnologias_unicas) > 0:
        # Mapear tecnologias para cores
        color_dict = {tech: colors[i % len(colors)] for i, tech in enumerate(gdf['Tecnologia'].unique())}
        
        # Plotar por tecnologia
        for tech, data in gdf.groupby('Tecnologia'):
            data.plot(ax=ax, marker='o', markersize=10, 
                      label=tech, color=color_dict[tech], alpha=0.7)
        
        ax.set_title('Distribuição de Estações de Telecomunicações por Tecnologia', fontsize=15)
        ax.legend(title="Tecnologia")
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Salvar figura
        output_path = os.path.join(output_dir, 'mapa_tecnologias.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Mapa de tecnologias salvo em: {output_path}")
    else:
        print("Não foi possível criar mapa por tecnologia - dados insuficientes")
    
    # 2. Mapa de potência dos transmissores
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Usar tamanho variável com base na potência
    if 'PotenciaTransmissorWatts' in gdf.columns:
        # Verificar se há valores válidos
        potencia_valida = gdf['PotenciaTransmissorWatts'].dropna()
        if len(potencia_valida) > 0:
            # Normalizar potência para tamanho de marcadores
            potencia_norm = (gdf['PotenciaTransmissorWatts'] / gdf['PotenciaTransmissorWatts'].max()) * 150
            potencia_norm = potencia_norm.fillna(10)  # Valor padrão para NaN
            
            scatter = ax.scatter(
                gdf.geometry.x, gdf.geometry.y,
                s=potencia_norm,
                c=gdf['PotenciaTransmissorWatts'],
                cmap='viridis',
                alpha=0.7,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Adicionar barra de cores
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Potência do Transmissor (Watts)')
            
            ax.set_title('Potência dos Transmissores de Telecomunicações', fontsize=15)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Salvar figura
            output_path = os.path.join(output_dir, 'mapa_potencia.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Mapa de potência salvo em: {output_path}")
        else:
            print("Não foi possível criar mapa de potência - dados insuficientes")
    
    # 3. Mapa de altura das antenas
    if 'AlturaAntena' in gdf.columns:
        # Verificar se há valores válidos
        altura_valida = gdf['AlturaAntena'].dropna()
        if len(altura_valida) > 0:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            scatter = ax.scatter(
                gdf.geometry.x, gdf.geometry.y,
                s=50,
                c=gdf['AlturaAntena'],
                cmap='plasma',
                alpha=0.7,
                edgecolor='white',
                linewidth=0.5
            )
            
            # Adicionar barra de cores
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Altura da Antena (m)')
            
            ax.set_title('Altura das Antenas de Telecomunicações', fontsize=15)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Salvar figura
            output_path = os.path.join(output_dir, 'mapa_altura.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300)
            plt.close()
            print(f"Mapa de altura salvo em: {output_path}")
        else:
            print("Não foi possível criar mapa de altura - dados insuficientes")
    
    return True

def create_plotly_map(gdf, output_path):
    """Cria um mapa interativo usando Plotly Express."""
    print("Criando mapa interativo com Plotly...")
    
    # Criar uma cópia do DataFrame para manipulação
    plot_df = pd.DataFrame()
    
    # Adicionar coordenadas geométricas
    plot_df['lat'] = gdf.geometry.y
    plot_df['lon'] = gdf.geometry.x
    
    # Adicionar campos com tratamento para valores nulos
    plot_df['Operadora'] = gdf['NomeEntidade'].fillna('Desconhecida')
    plot_df['Tecnologia'] = gdf['Tecnologia'].fillna('Desconhecida')
    plot_df['Município'] = gdf['Municipio.NomeMunicipio'].fillna('Desconhecido')
    
    # Converter campos numéricos, tratando valores nulos
    try:
        plot_df['Potência (W)'] = pd.to_numeric(gdf['PotenciaTransmissorWatts'], errors='coerce').fillna(0)
        plot_df['Altura (m)'] = pd.to_numeric(gdf['AlturaAntena'], errors='coerce').fillna(0)
        plot_df['Frequência TX (MHz)'] = pd.to_numeric(gdf['FreqTxMHz'], errors='coerce').fillna(0)
        plot_df['Ganho (dB)'] = pd.to_numeric(gdf['GanhoAntena'], errors='coerce').fillna(0)
    except Exception as e:
        print(f"Aviso ao converter campos numéricos: {str(e)}")
    
    # Verificar e filtrar coordenadas inválidas
    plot_df = plot_df.dropna(subset=['lat', 'lon'])
    print(f"Dados para mapa Plotly: {len(plot_df)} registros com coordenadas válidas")
    
    if len(plot_df) == 0:
        print("Não há dados válidos para criar o mapa Plotly")
        return None
    
    # Criar figura interativa
    try:
        fig = px.scatter_mapbox(
            plot_df, 
            lat='lat', 
            lon='lon', 
            color='Tecnologia',
            size='Potência (W)' if 'Potência (W)' in plot_df.columns else None,
            size_max=15,
            zoom=9,
            hover_name='Operadora',
            hover_data={
                'lat': False,
                'lon': False,
                'Município': True,
                'Altura (m)': True if 'Altura (m)' in plot_df.columns else False,
                'Frequência TX (MHz)': True if 'Frequência TX (MHz)' in plot_df.columns else False,
                'Ganho (dB)': True if 'Ganho (dB)' in plot_df.columns else False
            },
            color_discrete_sequence=px.colors.qualitative.Bold,
            title='Mapa Interativo de Estações de Telecomunicações'
        )
        
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r":0,"t":50,"l":0,"b":0},
            height=800
        )
        
        # Salvar como HTML
        fig.write_html(output_path)
        print(f"Mapa Plotly salvo em: {output_path}")
        return output_path
    except Exception as e:
        print(f"Erro ao criar mapa Plotly: {str(e)}")
        return None

# Executar criação de visualizações
try:
    # Verificar se os dados estão corretos
    if gdf is None or len(gdf) == 0:
        raise ValueError("Não foi possível carregar dados do GeoPackage ou o arquivo está vazio")
    
    # Converter geometria para EPSG:4326 (WGS84) se necessário
    if gdf.crs and gdf.crs != "EPSG:4326":
        print(f"Convertendo CRS de {gdf.crs} para EPSG:4326 (WGS84)...")
        try:
            gdf = gdf.to_crs("EPSG:4326")
            print("Conversão CRS concluída com sucesso")
        except Exception as e:
            print(f"Aviso na conversão CRS: {str(e)} - continuando com o CRS original")
    
    # Criar diretório de saída
    print(f"\nCriando diretório para visualizações: {OUTPUT_DIR}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Inicializar contadores
    mapas_gerados = 0
    mapas_interativos = 0
    mapas_estaticos = 0
    
    # 1. Criar mapa interativo com Folium
    print("\n--- Criando Mapa Folium ---")
    folium_output = os.path.join(OUTPUT_DIR, 'mapa_folium_interativo.html')
    try:
        folium_path = create_folium_map(gdf, folium_output)
        if folium_path:
            mapas_gerados += 1
            mapas_interativos += 1
    except Exception as e:
        print(f"Erro ao criar mapa Folium: {str(e)}")
    
    # 2. Criar mapas estáticos com Matplotlib
    print("\n--- Criando Mapas Matplotlib ---")
    try:
        if create_matplotlib_maps(gdf, OUTPUT_DIR):
            mapas_gerados += 3  # Adiciona 3 mapas estáticos
            mapas_estaticos += 3
    except Exception as e:
        print(f"Erro ao criar mapas Matplotlib: {str(e)}")
    
    # 3. Criar mapa interativo com Plotly
    print("\n--- Criando Mapa Plotly ---")
    plotly_output = os.path.join(OUTPUT_DIR, 'mapa_plotly_interativo.html')
    try:
        plotly_path = create_plotly_map(gdf, plotly_output)
        if plotly_path:
            mapas_gerados += 1
            mapas_interativos += 1
    except Exception as e:
        print(f"Erro ao criar mapa Plotly: {str(e)}")
    
    # Resumo das visualizações criadas
    print("\nVisualização concluída!")
    print(f"Total de mapas gerados: {mapas_gerados}")
    print(f"- {mapas_interativos} Mapas interativos")
    print(f"- {mapas_estaticos} Mapas estáticos")
    
    # Abrir um mapa interativo no navegador, se disponível
    if plotly_path and os.path.exists(plotly_path):
        print("\nAbrindo mapa Plotly no navegador...")
        webbrowser.open('file://' + os.path.realpath(plotly_path))
    elif folium_path and os.path.exists(folium_path):
        print("\nAbrindo mapa Folium no navegador...")
        webbrowser.open('file://' + os.path.realpath(folium_path))
    
except Exception as e:
    print(f"Erro ao criar visualizações: {str(e)}") 