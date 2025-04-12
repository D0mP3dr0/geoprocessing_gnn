#!/usr/bin/env python
"""
Script auxiliar para executar os componentes do fluxo de trabalho de geoprocessamento.
"""

import sys
import os
import subprocess
from pathlib import Path

def print_header(text):
    """Imprime cabeçalho formatado"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")

def run_data_analysis():
    """Executa análise exploratória de dados"""
    print_header("Executando Análise Exploratória de Dados")
    subprocess.run([sys.executable, "data_analysis.py"])

def run_preprocessing(dataset):
    """Executa pré-processamento de um conjunto de dados específico"""
    if dataset == "licenciamento":
        script = "src/preprocessing/preprocess_csv_licenciamento.py"
        print_header("Pré-processando Dados de Licenciamento")
    elif dataset == "inmet":
        script = "src/preprocessing/inmet.py"
        print_header("Pré-processando Dados INMET")
    else:
        print(f"Dataset desconhecido: {dataset}")
        return

    try:
        subprocess.run([sys.executable, script])
    except FileNotFoundError:
        print(f"Erro: Script {script} não encontrado")

def run_visualization():
    """Executa visualização de dados"""
    print_header("Criando Visualizações")
    subprocess.run([sys.executable, "src/visualization/create_visualizations.py"])

def main():
    """Função principal"""
    # Definir diretório base
    base_dir = Path(__file__).resolve().parent
    
    # Garantir que estamos no diretório correto
    os.chdir(base_dir)
    
    # Mostrar menu se não houver argumentos
    if len(sys.argv) < 2:
        print_header("Fluxo de Trabalho de Geoprocessamento")
        print("Uso: python run_workflow.py [comando]")
        print("\nComandos disponíveis:")
        print("  analyze         - Executar análise exploratória de dados")
        print("  preprocess      - Pré-processar um conjunto de dados específico")
        print("  visualize       - Criar visualizações para dados processados")
        print("\nExemplos:")
        print("  python run_workflow.py analyze")
        print("  python run_workflow.py preprocess licenciamento")
        print("  python run_workflow.py preprocess inmet")
        print("  python run_workflow.py visualize")
        return
    
    # Processar comando
    command = sys.argv[1].lower()
    
    if command == "analyze":
        run_data_analysis()
    elif command == "preprocess":
        if len(sys.argv) < 3:
            print("Erro: Especifique o conjunto de dados a ser pré-processado")
            print("Uso: python run_workflow.py preprocess [dataset]")
            print("\nDatasets disponíveis:")
            print("  licenciamento - Dados de licenciamento")
            print("  inmet         - Dados meteorológicos INMET")
            return
        
        dataset = sys.argv[2].lower()
        run_preprocessing(dataset)
    elif command == "visualize":
        run_visualization()
    else:
        print(f"Comando desconhecido: {command}")
        print("Use 'python run_workflow.py' para ver os comandos disponíveis")

if __name__ == "__main__":
    main() 