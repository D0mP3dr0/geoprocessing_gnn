# -*- coding: utf-8 -*-
"""
Execução do Pipeline de Análise de Redes Viárias

Este script inicia a execução do pipeline modular de análise de redes viárias,
mantendo todos os caminhos de arquivos originais do Google Drive.
"""

import os
import sys
import time
from datetime import datetime

# Adicionar o diretório atual ao PATH para importação dos módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importar o pipeline
from pipeline.main import main
from pipeline.data_loading import mount_google_drive
from pipeline.config import (
    DRIVE_PATH, DATA_DIR, OUTPUT_DIR, REPORT_DIR,
    ROADS_PROCESSED_PATH, ROADS_ENRICHED_PATH
)

if __name__ == "__main__":
    # Timestamp de início para o log
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"===== INÍCIO DA EXECUÇÃO DO PIPELINE DE ANÁLISE DE REDES VIÁRIAS =====")
    print(f"Data e hora: {timestamp}")
    
    # Verificar ambiente e montar Google Drive se necessário
    try:
        import google.colab
        print("Ambiente Google Colab detectado. Montando o Google Drive...")
        mounted = mount_google_drive()
        if not mounted:
            print("ERRO: Falha ao montar o Google Drive. Verifique as permissões.")
            sys.exit(1)
    except ImportError:
        print("Ambiente local detectado. Verificando caminhos do Google Drive...")
        if not os.path.exists(DRIVE_PATH):
            print(f"AVISO: Caminho do Google Drive não encontrado: {DRIVE_PATH}")
            print("Continuando com caminhos relativos...")
    
    # Verificar existência do arquivo de entrada
    if not os.path.exists(ROADS_PROCESSED_PATH):
        print(f"ERRO: Arquivo de entrada não encontrado: {ROADS_PROCESSED_PATH}")
        print("Verifique se o Google Drive está montado corretamente e se o arquivo existe.")
        sys.exit(1)
    
    # Iniciar o temporizador
    start_time = time.time()
    
    try:
        # Executar o pipeline principal
        results = main()
        
        # Mostrar resultados
        print("\n===== RESULTADOS DA EXECUÇÃO =====")
        print(f"Acurácia do modelo: {results['test_results']['accuracy']:.4f}")
        print(f"Relatórios gerados em: {REPORT_DIR}")
        print(f"Visualizações salvas em: {os.path.join(OUTPUT_DIR, 'visualizations')}")
        
    except Exception as e:
        # Capturar e exibir qualquer erro
        print(f"\nERRO durante a execução do pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Calcular tempo total de execução
    end_time = time.time()
    execution_time = end_time - start_time
    hours, remainder = divmod(execution_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n===== EXECUÇÃO CONCLUÍDA =====")
    print(f"Tempo total de execução: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print(f"Todos os arquivos foram salvos com os caminhos originais do Google Drive.")
    print("=========================================") 