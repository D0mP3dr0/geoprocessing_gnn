import pandas as pd
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import geopandas as gpd
from shapely.geometry import Point
import seaborn as sns

# Classe personalizada para serialização JSON de tipos NumPy
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if pd.isna(obj):
            return None
        return super(NpEncoder, self).default(obj)

# Configuração - Usar caminhos absolutos para evitar problemas de referência
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
INPUT_FILE = os.path.join(WORKSPACE_DIR, 'data', 'raw', 'licenciamento', 'csv_licenciamento.csv')
OUTPUT_DIR = os.path.join(WORKSPACE_DIR, 'data', 'processed')
ANALYSIS_REPORT = os.path.join(WORKSPACE_DIR, 'analysis_reports', 'csv_licenciamento_c55c1dea01bd184e27df233da8ac28a2.csv_analysis.json')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'licenciamento_processed.csv')
OUTPUT_GPKG = os.path.join(OUTPUT_DIR, 'licenciamento_processed.gpkg')
REPORT_DIR = os.path.join(os.path.dirname(__file__), 'quality_reports', 'rbs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print("Iniciando pré-processamento de dados de licenciamento...")
print(f"Diretório de trabalho: {WORKSPACE_DIR}")
print(f"Usando arquivo de entrada: {INPUT_FILE}")
print(f"Usando relatório de análise: {ANALYSIS_REPORT}")

# Verificar se os arquivos existem
if not os.path.exists(INPUT_FILE):
    # Tentar procurar o arquivo csv_licenciamento na pasta data/raw
    data_dir = os.path.join(WORKSPACE_DIR, 'data', 'raw')
    raw_licenciamento_dir = os.path.join(data_dir, 'licenciamento')
    print(f"Arquivo de entrada não encontrado: {INPUT_FILE}")
    print(f"Verificando arquivos disponíveis em: {raw_licenciamento_dir}")
    
    # Primeiro tentar na pasta específica de licenciamento
    if os.path.exists(raw_licenciamento_dir):
        available_files = [f for f in os.listdir(raw_licenciamento_dir) if f.startswith('csv_licenciamento')]
        if available_files:
            print(f"Encontrados arquivos alternativos em {raw_licenciamento_dir}: {available_files}")
            # Use o primeiro arquivo encontrado
            INPUT_FILE = os.path.join(raw_licenciamento_dir, available_files[0])
            print(f"Usando arquivo alternativo: {INPUT_FILE}")
        else:
            print(f"Nenhum arquivo csv_licenciamento encontrado em {raw_licenciamento_dir}")
            # Tentar na pasta raw geral
            if os.path.exists(data_dir):
                available_files = [f for f in os.listdir(data_dir) if f.startswith('csv_licenciamento')]
                if available_files:
                    print(f"Encontrados arquivos alternativos em {data_dir}: {available_files}")
                    INPUT_FILE = os.path.join(data_dir, available_files[0])
                    print(f"Usando arquivo alternativo: {INPUT_FILE}")
                else:
                    print(f"Nenhum arquivo csv_licenciamento encontrado em {data_dir}")
                    available_files = os.listdir(data_dir)
                    print(f"Arquivos disponíveis em {data_dir}: {available_files}")
                    raise FileNotFoundError(f"Arquivo de dados não encontrado: {INPUT_FILE}")
    else:
        # Se a pasta específica não existir, tentar na pasta raw geral
        if os.path.exists(data_dir):
            available_files = [f for f in os.listdir(data_dir) if f.startswith('csv_licenciamento')]
            if available_files:
                print(f"Encontrados arquivos alternativos em {data_dir}: {available_files}")
                INPUT_FILE = os.path.join(data_dir, available_files[0])
                print(f"Usando arquivo alternativo: {INPUT_FILE}")
            else:
                print(f"Nenhum arquivo csv_licenciamento encontrado em {data_dir}")
                available_files = os.listdir(data_dir)
                print(f"Arquivos disponíveis em {data_dir}: {available_files}")
                raise FileNotFoundError(f"Arquivo de dados não encontrado: {INPUT_FILE}")
        else:
            raise FileNotFoundError(f"Diretório de dados não encontrado: {data_dir}")

# Inicializar analysis_report como dicionário vazio
analysis_report = {"preprocessing_recommendations": []}

# Tentar carregar o relatório de análise
if os.path.exists(ANALYSIS_REPORT):
    try:
        with open(ANALYSIS_REPORT, 'r', encoding='utf-8') as f:
            analysis_report = json.load(f)
    except Exception as e:
        print(f"Erro ao carregar relatório de análise: {str(e)}")
        print("Continuando sem relatório de análise")
else:
    # Tentar encontrar qualquer relatório de análise
    analysis_dir = os.path.join(WORKSPACE_DIR, 'analysis_reports')
    print(f"Arquivo de análise não encontrado: {ANALYSIS_REPORT}")
    if os.path.exists(analysis_dir):
        available_reports = [f for f in os.listdir(analysis_dir) if f.endswith('_analysis.json')]
        if available_reports:
            print(f"Encontrados relatórios alternativos: {available_reports}")
            # Use o primeiro relatório encontrado
            ANALYSIS_REPORT = os.path.join(analysis_dir, available_reports[0])
            print(f"Usando relatório alternativo: {ANALYSIS_REPORT}")
            try:
                with open(ANALYSIS_REPORT, 'r', encoding='utf-8') as f:
                    analysis_report = json.load(f)
            except Exception as e:
                print(f"Erro ao carregar relatório alternativo: {str(e)}")
                print("Continuando sem relatório de análise")
        else:
            print(f"Nenhum relatório de análise encontrado em {analysis_dir}")
    else:
        print("Diretório de relatórios de análise não encontrado")
        print("Continuando sem relatório de análise")

# Extrair recomendações de pré-processamento
recommendations = analysis_report.get('preprocessing_recommendations', [])
print(f"Recomendações identificadas: {len(recommendations)}")
for rec in recommendations:
    print(f"- {rec}")

# Carregar o arquivo CSV
print(f"Carregando dados do arquivo: {INPUT_FILE}")
try:
    # Verificar tamanho do arquivo antes de carregar
    file_size = os.path.getsize(INPUT_FILE)
    print(f"Tamanho do arquivo: {file_size/1024/1024:.2f} MB")
    
    if file_size == 0:
        raise ValueError("O arquivo de entrada está vazio (tamanho 0 bytes)")
    
    # Tentar determinar o delimitador correto
    with open(INPUT_FILE, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline().strip()
        # Verificar possíveis delimitadores
        possible_delimiters = [';', ',', '\t', '|']
        delimiter_counts = {delimiter: first_line.count(delimiter) for delimiter in possible_delimiters}
        best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
        print(f"Delimitador detectado: '{best_delimiter}' (contagem: {delimiter_counts[best_delimiter]})")
        
        # Exibir primeiras 10 linhas do arquivo bruto para debug
        print("\nPrimeiras 10 linhas do arquivo bruto:")
        f.seek(0)  # Voltar ao início do arquivo
        for i in range(10):
            line = f.readline().strip()
            if not line:
                break
            print(f"Linha {i+1}: {line[:150]}..." if len(line) > 150 else f"Linha {i+1}: {line}")
    
    # Carregar com o delimitador detectado
    df = pd.read_csv(INPUT_FILE, delimiter=best_delimiter, encoding='utf-8', low_memory=False)
    
    # Verificar e imprimir nomes de todas as colunas
    print("\nNomes de todas as colunas:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. '{col}'")
        
except UnicodeDecodeError:
    # Tentar com encoding alternativo se utf-8 falhar
    print("Falha no encoding UTF-8, tentando com latin1...")
    try:
        df = pd.read_csv(INPUT_FILE, delimiter=best_delimiter, encoding='latin1', low_memory=False)
    except Exception as e:
        print(f"Erro ao carregar arquivo com latin1: {str(e)}")
        # Tentar outros encodings comuns
        encodings = ['iso-8859-1', 'cp1252', 'utf-16']
        for enc in encodings:
            try:
                print(f"Tentando com encoding {enc}...")
                df = pd.read_csv(INPUT_FILE, delimiter=best_delimiter, encoding=enc, low_memory=False)
                print(f"Sucesso com encoding {enc}")
                break
            except Exception as e:
                print(f"Falha com encoding {enc}: {str(e)}")
        else:
            raise ValueError(f"Não foi possível abrir o arquivo com nenhum encoding testado: {encodings}")
except Exception as e:
    print(f"Erro ao carregar o arquivo: {str(e)}")
    raise

# Exibir as primeiras linhas para verificar se os dados foram carregados corretamente
print("\nPrimeiras 5 linhas do arquivo:")
print(df.head().to_string())
print(f"\nDados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")

# Definir colunas relevantes para o projeto
# Estas são as colunas que são importantes para análise geoespacial e de telecomunicações
colunas_importantes = [
    # Localização e identificação
    'NomeEntidade',           # Nome da entidade proprietária das antenas
    'EnderecoEstacao',        # Endereço físico da instalação
    'SiglaUf',               # Estado (SP)
    'Municipio.NomeMunicipio', # Nome do município
    'Latitude',              # Coordenada geográfica essencial
    'Longitude',             # Coordenada geográfica essencial
    
    # Atributos técnicos de telecomunicações
    'Tecnologia',            # Tipo de tecnologia (LTE, GSM, WCDMA, etc.)
    'FreqTxMHz',             # Frequência de transmissão em MHz
    'FreqRxMHz',             # Frequência de recepção em MHz
    'AlturaAntena',          # Altura da antena (m)
    'GanhoAntena',           # Ganho da antena (dB)
    'Polarizacao',           # Polarização do sinal
    'PotenciaTransmissorWatts', # Potência de transmissão
    
    # Atributos complementares
    'ClassInfraFisica',      # Tipo de infraestrutura física
    'DataLicenciamento',     # Data de concessão da licença
    'DataValidade'           # Data de validade da licença
]

# Mapeamento para lidar com possíveis variações nos nomes das colunas
mapeamento_colunas = {
    'nome_entidade': 'NomeEntidade',
    'nomeentidade': 'NomeEntidade',
    'nome entidade': 'NomeEntidade',
    'nome da entidade': 'NomeEntidade',
    
    'endereco_estacao': 'EnderecoEstacao',
    'enderecoestacao': 'EnderecoEstacao',
    'endereco estacao': 'EnderecoEstacao',
    'endereco da estacao': 'EnderecoEstacao',
    
    'uf': 'SiglaUf',
    'sigla_uf': 'SiglaUf',
    'estado': 'SiglaUf',
    
    'municipio': 'Municipio.NomeMunicipio',
    'nome_municipio': 'Municipio.NomeMunicipio',
    'nomemunicipio': 'Municipio.NomeMunicipio',
    'municipio.nome': 'Municipio.NomeMunicipio',
    
    'lat': 'Latitude',
    'latitude_decimal': 'Latitude',
    
    'long': 'Longitude',
    'lon': 'Longitude',
    'longitude_decimal': 'Longitude',
    
    'tecnologia_telecom': 'Tecnologia',
    'tipo_tecnologia': 'Tecnologia',
    
    'frequencia_tx': 'FreqTxMHz',
    'freq_tx': 'FreqTxMHz',
    'frequenciatx': 'FreqTxMHz',
    'freqtx': 'FreqTxMHz',
    
    'frequencia_rx': 'FreqRxMHz',
    'freq_rx': 'FreqRxMHz',
    'frequenciarx': 'FreqRxMHz',
    'freqrx': 'FreqRxMHz',
    
    'altura_antena': 'AlturaAntena',
    'altura_da_antena': 'AlturaAntena',
    'alturaantena': 'AlturaAntena',
    
    'ganho_antena': 'GanhoAntena',
    'ganho_da_antena': 'GanhoAntena',
    'ganhoantena': 'GanhoAntena',
    
    'polarizacao_antena': 'Polarizacao',
    'pol': 'Polarizacao',
    
    'potencia_transmissor': 'PotenciaTransmissorWatts',
    'potencia_w': 'PotenciaTransmissorWatts',
    'potencia': 'PotenciaTransmissorWatts',
    
    'classe_infra_fisica': 'ClassInfraFisica',
    'classe_infraestrutura': 'ClassInfraFisica',
    'tipo_infraestrutura': 'ClassInfraFisica',
    
    'data_licenciamento': 'DataLicenciamento',
    'datalicenciamento': 'DataLicenciamento',
    'data_licenca': 'DataLicenciamento',
    
    'data_validade': 'DataValidade',
    'datavalidade': 'DataValidade',
    'validade': 'DataValidade',
    'validade_licenca': 'DataValidade'
}

# Verificar quais colunas importantes estão presentes no DataFrame
print("\nVerificando colunas do arquivo de entrada:")
print(f"Colunas totais no arquivo: {len(df.columns)}")
print("Colunas disponíveis:", df.columns.tolist())

# Normalizar nomes de colunas (converter para minúsculas para comparação)
df.columns = [col.strip() for col in df.columns]  # Remover espaços extras
df_colunas_lower = {col.lower(): col for col in df.columns}

# Mapear colunas disponíveis para os nomes padronizados
colunas_mapeadas = {}
for col_padrao in colunas_importantes:
    # Verificar se o nome padrão existe diretamente
    if col_padrao in df.columns:
        colunas_mapeadas[col_padrao] = col_padrao
    else:
        # Verificar variações conhecidas em minúsculas
        col_padrao_lower = col_padrao.lower()
        if col_padrao_lower in df_colunas_lower:
            colunas_mapeadas[col_padrao] = df_colunas_lower[col_padrao_lower]
        else:
            # Verificar mapeamentos de variações
            for variacao, padrao in mapeamento_colunas.items():
                if padrao == col_padrao and variacao in df_colunas_lower:
                    colunas_mapeadas[col_padrao] = df_colunas_lower[variacao]
                    break

# Verificar colunas mapeadas e ausentes
colunas_presentes = list(colunas_mapeadas.keys())
colunas_ausentes = [col for col in colunas_importantes if col not in colunas_presentes]

print(f"\nColunas importantes encontradas: {len(colunas_presentes)}")
print("Colunas presentes:", colunas_presentes)
if colunas_mapeadas:
    print("\nMapeamento realizado:")
    for padrao, original in colunas_mapeadas.items():
        if padrao != original:
            print(f"  - '{original}' → '{padrao}'")

print(f"\nColunas importantes ausentes: {len(colunas_ausentes)}")
if colunas_ausentes:
    print("Colunas ausentes:", colunas_ausentes)

# Criar DataFrame filtrado com as colunas importantes disponíveis
if not colunas_mapeadas:
    print("\nAVISO: Nenhuma coluna importante foi encontrada no arquivo!")
    print("Usando as primeiras 16 colunas disponíveis como alternativa.")
    if len(df.columns) > 0:
        colunas_temp = df.columns[:min(16, len(df.columns))]
        df_filtrado = df[colunas_temp].copy()
        # Criar mapeamento temporário para colunas
        for i, col in enumerate(colunas_temp):
            if i < len(colunas_importantes):
                print(f"  - Mapeando '{col}' para '{colunas_importantes[i]}'")
        print(f"\nDataFrame filtrado (alternativo): {df_filtrado.shape[0]} linhas, {df_filtrado.shape[1]} colunas")
    else:
        print("ERRO CRÍTICO: O arquivo não contém colunas!")
        df_filtrado = pd.DataFrame()
else:
    # Renomear colunas para nomes padronizados e filtrar
    df_temp = df.copy()
    # Criar dicionário de renomeação invertido (original -> padronizado)
    renomear = {original: padrao for padrao, original in colunas_mapeadas.items()}
    df_temp = df_temp.rename(columns=renomear)
    df_filtrado = df_temp[colunas_presentes].copy()
    print(f"\nDataFrame filtrado: {df_filtrado.shape[0]} linhas, {df_filtrado.shape[1]} colunas")
    print(f"Redução de colunas: {df.shape[1]} -> {df_filtrado.shape[1]} ({(1 - df_filtrado.shape[1]/df.shape[1])*100:.1f}% de redução)")

print("\nColunas mantidas no DataFrame filtrado:", df_filtrado.columns.tolist())

# Verificar se o DataFrame está vazio
if df_filtrado.empty:
    print("AVISO: DataFrame filtrado está vazio!")
    print("Verificando DataFrame original...")
    if df.empty:
        print("ERRO CRÍTICO: DataFrame original também está vazio!")
    else:
        print(f"DataFrame original tem {df.shape[0]} linhas e {df.shape[1]} colunas.")
        print("Primeiras 5 linhas do DataFrame original:")
        print(df.head())
        
    # Salvar um arquivo CSV mínimo para evitar erro
    pd.DataFrame({'Erro': ['Nenhum dado válido encontrado']}).to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8')
    print(f"Arquivo CSV de erro criado em: {OUTPUT_FILE}")
    
    # Encerrar o script para evitar processamento inútil
    import sys
    sys.exit("Encerrando script devido a DataFrame vazio")

# Criar diretório para relatórios de distribuição
distribution_dir = os.path.join(REPORT_DIR, 'distributions')
os.makedirs(distribution_dir, exist_ok=True)

# Função para verificar e relatar a distribuição dos dados
def analyze_column_distribution(df, column_name, output_dir):
    """Analisa e visualiza a distribuição dos dados em uma coluna específica."""
    if column_name not in df.columns:
        return
    
    if df[column_name].dtype.kind in 'ifc':  # Numérico
        plt.figure(figsize=(10, 6))
        
        # Histograma original
        plt.subplot(1, 2, 1)
        sns.histplot(df[column_name].dropna(), kde=True)
        plt.title(f'Distribuição Original: {column_name}')
        
        # Boxplot para identificar outliers
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[column_name].dropna())
        plt.title(f'Boxplot: {column_name}')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{column_name}_distribution.png'))
        plt.close()

# Pré-processamento conservador que preserva a precisão
print("\nAplicando etapas de pré-processamento com foco na preservação da precisão...")

# 1. Manter cópias dos dados originais para colunas críticas
spatial_columns = ['Latitude', 'Longitude']
telecom_columns = ['AlturaAntena', 'FreqTxMHz', 'FreqRxMHz', 'GanhoAntena', 'PotenciaTransmissorWatts']

for col in spatial_columns + telecom_columns:
    if col in df_filtrado.columns:
        df_filtrado[f'{col}_original'] = df_filtrado[col].copy()

# 2. Validação de dados geoespaciais
# Verificar coordenadas inválidas ou fora de faixa
if 'Latitude' in df_filtrado.columns and 'Longitude' in df_filtrado.columns:
    print("Validando coordenadas geográficas...")
    # Brasil aproximadamente: Lat -33 a 5, Long -74 a -34
    invalid_coords = ((df_filtrado['Latitude'] < -33) | (df_filtrado['Latitude'] > 5) | 
                      (df_filtrado['Longitude'] < -74) | (df_filtrado['Longitude'] > -34))
    
    invalid_count = invalid_coords.sum()
    print(f"Coordenadas inválidas encontradas: {invalid_count}")
    
    if invalid_count > 0:
        # Criando coluna de flag para coordenadas inválidas em vez de remover
        df_filtrado['coord_valida'] = ~invalid_coords
        print("Coluna 'coord_valida' adicionada para identificar coordenadas fora do intervalo esperado.")

# 3. Tratamento cuidadoso de valores ausentes
missing_values_report = {}
columns_with_nulls = [col for col in df_filtrado.columns if df_filtrado[col].isnull().any()]

print(f"Processando valores ausentes em {len(columns_with_nulls)} colunas...")
for col in columns_with_nulls:
    if col not in df_filtrado.columns:
        continue
        
    null_count = df_filtrado[col].isnull().sum()
    missing_values_report[col] = {'contagem_nulos': null_count, 'percentual': (null_count / len(df_filtrado)) * 100}
    
    # Tratamento baseado no tipo de coluna
    if col in spatial_columns:
        # NÃO imputar valores para coordenadas - adicionar flag em vez disso
        df_filtrado[f'{col}_ausente'] = df_filtrado[col].isnull()
    elif col in telecom_columns:
        # Para atributos técnicos, usar um método mais conservador
        if df_filtrado[col].dtype.kind in 'ifc':  # Numérico
            # Analisar a distribuição antes do preenchimento
            analyze_column_distribution(df_filtrado, col, distribution_dir)
            
            # Usar mediana em vez de média para reduzir o impacto de outliers
            median_value = df_filtrado[col].median()
            df_filtrado[f'{col}_imputado'] = df_filtrado[col].isnull()  # Flag para valores imputados
            df_filtrado[col] = df_filtrado[col].fillna(median_value)
    else:
        # Para outros atributos, usar estratégia adequada ao tipo
        if df_filtrado[col].dtype == 'object':
            # Usar 'Não informado' em vez de 'Unknown' para facilitar filtragem posterior
            df_filtrado[f'{col}_imputado'] = df_filtrado[col].isnull()
            df_filtrado[col] = df_filtrado[col].fillna('Não informado')
        else:
            # Criar flag para valores imputados
            df_filtrado[f'{col}_imputado'] = df_filtrado[col].isnull()
            # Usar mediana para campos numéricos
            if df_filtrado[col].dtype.kind in 'ifc':
                df_filtrado[col] = df_filtrado[col].fillna(df_filtrado[col].median())

# 4. Tratamento criterioso de outliers para atributos técnicos
outlier_report = {}

for col in telecom_columns:
    if col not in df_filtrado.columns or df_filtrado[col].dtype.kind not in 'ifc':
        continue
    
    # Identificar outliers usando IQR (mais robusto que desvio padrão)
    Q1 = df_filtrado[col].quantile(0.25)
    Q3 = df_filtrado[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Limites para detecção de outliers extremos (usando 3*IQR)
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR
    
    # Identificar outliers
    outliers = df_filtrado[(df_filtrado[col] < lower_bound) | (df_filtrado[col] > upper_bound)]
    outlier_count = len(outliers)
    
    # Registrar informações sobre outliers
    outlier_report[col] = {
        'contagem_outliers': outlier_count,
        'percentual': (outlier_count / len(df_filtrado)) * 100,
        'valor_min': df_filtrado[col].min(),
        'valor_max': df_filtrado[col].max(),
        'limite_inferior': lower_bound,
        'limite_superior': upper_bound
    }
    
    if outlier_count > 0:
        print(f"Coluna {col}: {outlier_count} outliers detectados")
        
        # Criar cópias das colunas antes de modificar
        df_filtrado[f'{col}_com_outliers'] = df_filtrado[col].copy()
        
        # Marcar outliers em vez de automaticamente transformá-los
        df_filtrado[f'{col}_outlier'] = (df_filtrado[col] < lower_bound) | (df_filtrado[col] > upper_bound)
        
        # Opcionalmente, limitar valores extremos (mas manter os originais)
        df_filtrado[f'{col}_limitado'] = df_filtrado[col].clip(lower=lower_bound, upper=upper_bound)

# 5. Harmonização de formatos para campos específicos
# Padronização de campos de data
date_columns = [col for col in df_filtrado.columns if 'Data' in col]
for col in date_columns:
    if col in df_filtrado.columns and df_filtrado[col].dtype == 'object':
        try:
            # Tentar converter para datetime preservando o formato original
            df_filtrado[f'{col}_datetime'] = pd.to_datetime(df_filtrado[col], errors='coerce')
            print(f"Coluna {col} convertida para formato datetime")
        except:
            print(f"Não foi possível converter a coluna {col} para datetime")

# 6. Converter para GeoDataFrame para análise espacial
if 'Latitude' in df_filtrado.columns and 'Longitude' in df_filtrado.columns:
    print("\nConvertendo para formato geoespacial...")
    # Filtrar apenas registros com coordenadas válidas para o GeoDataFrame
    valid_coords = df_filtrado.dropna(subset=['Latitude', 'Longitude'])
    
    if not valid_coords.empty:
        # Criar geometria de pontos
        geometry = [Point(lon, lat) for lon, lat in zip(valid_coords['Longitude'], valid_coords['Latitude'])]
        
        # Criar GeoDataFrame
        gdf = gpd.GeoDataFrame(valid_coords, geometry=geometry, crs="EPSG:4674")  # SIRGAS 2000
        
        # Salvar como GeoPackage
        gdf.to_file(OUTPUT_GPKG, driver="GPKG")
        print(f"GeoDataFrame criado com {len(gdf)} registros e salvo em {OUTPUT_GPKG}")

# 7. Salvar dados pré-processados e relatórios
# Salvar CSV pré-processado
print(f"\nSalvando dados processados em: {OUTPUT_FILE}")
try:
    # Verificar se o DataFrame tem dados
    if df_filtrado.empty:
        print("AVISO: O DataFrame está vazio. Não há dados para salvar.")
    else:
        # Salvar com todas as informações
        df_filtrado.to_csv(OUTPUT_FILE, index=False, sep=';', encoding='utf-8')
        print(f"Dados salvos com sucesso: {df_filtrado.shape[0]} linhas, {df_filtrado.shape[1]} colunas")
        
        # Verificar se o arquivo foi criado corretamente
        if os.path.exists(OUTPUT_FILE):
            file_size = os.path.getsize(OUTPUT_FILE)
            print(f"Arquivo criado com tamanho: {file_size/1024/1024:.2f} MB")
            
            # Verificar se o arquivo não está vazio
            if file_size == 0:
                print("ERRO: O arquivo criado está vazio!")
            else:
                print("Arquivo criado com sucesso!")
        else:
            print("ERRO: Falha ao criar o arquivo!")
except Exception as e:
    print(f"ERRO ao salvar o arquivo CSV: {str(e)}")
    # Tentar salvar em formato alternativo
    try:
        alt_output = OUTPUT_FILE.replace('.csv', '_alt.csv')
        print(f"Tentando salvar em formato alternativo: {alt_output}")
        df_filtrado.to_csv(alt_output, index=False, sep=',', encoding='utf-8')
        print(f"Arquivo alternativo salvo com sucesso!")
    except Exception as e2:
        print(f"ERRO ao salvar arquivo alternativo: {str(e2)}")

# Salvar relatórios
with open(os.path.join(REPORT_DIR, 'missing_values_report.json'), 'w', encoding='utf-8') as f:
    json.dump(missing_values_report, f, indent=4, cls=NpEncoder)

with open(os.path.join(REPORT_DIR, 'outlier_report.json'), 'w', encoding='utf-8') as f:
    json.dump(outlier_report, f, indent=4, cls=NpEncoder)

# 8. Gerar relatório de qualidade pós-processamento
quality_metrics = {
    'linhas_originais': len(df),
    'linhas_processadas': len(df_filtrado),
    'colunas_originais': len(df.columns),
    'colunas_importantes': len(colunas_presentes),
    'colunas_finais': len(df_filtrado.columns),
    'colunas_adicionadas': [col for col in df_filtrado.columns if 
                          col.endswith('_original') or 
                          col.endswith('_imputado') or 
                          col.endswith('_outlier') or
                          col.endswith('_limitado') or
                          col.endswith('_ausente') or
                          col.endswith('_datetime')],
    'registros_coords_validas': df_filtrado['coord_valida'].sum() if 'coord_valida' in df_filtrado.columns else "Não verificado",
    'valores_ausentes_tratados': {col: df_filtrado[f'{col}_imputado'].sum() for col in df_filtrado.columns 
                                 if f'{col}_imputado' in df_filtrado.columns},
    'outliers_identificados': {col: df_filtrado[f'{col}_outlier'].sum() for col in df_filtrado.columns 
                              if f'{col}_outlier' in df_filtrado.columns}
}

with open(os.path.join(REPORT_DIR, 'quality_report.json'), 'w', encoding='utf-8') as f:
    json.dump(quality_metrics, f, indent=4, cls=NpEncoder)

print(f"\nPré-processamento concluído com sucesso.")
print(f"Arquivo CSV salvo em: {OUTPUT_FILE}")
if 'Latitude' in df_filtrado.columns and 'Longitude' in df_filtrado.columns:
    print(f"Arquivo GeoPackage salvo em: {OUTPUT_GPKG}")
print(f"Relatórios de qualidade salvos em: {REPORT_DIR}")
print("\nResumo de alterações:")
print(f"- Filtragem de {df.shape[1]} para {len(colunas_presentes)} colunas importantes ({(1 - len(colunas_presentes)/df.shape[1])*100:.1f}% de redução)")
print(f"- Valores ausentes tratados em {len(columns_with_nulls)} colunas")
print(f"- Outliers identificados em {len(outlier_report)} colunas")
print(f"- Dados convertidos para formato geoespacial")
print(f"- {len(quality_metrics['colunas_adicionadas'])} colunas auxiliares adicionadas para controle de qualidade")

# Verificar se há colunas com nomes semelhantes às que buscamos
print("\nVerificando colunas com nomes semelhantes:")
colunas_buscadas_lower = [col.lower() for col in colunas_importantes]
for col in df.columns:
    col_lower = col.lower()
    for busca in colunas_buscadas_lower:
        if busca in col_lower or col_lower in busca:
            print(f"Coluna '{col}' pode corresponder a '{busca}'") 