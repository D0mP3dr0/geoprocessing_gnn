"""
Preprocessing module for data cleaning and transformation.
"""

# Importar funcionalidades para disponibilizar no nível do pacote
from .inmet import read_inmet_data, process_inmet_files, create_inmet_geodataframe

# Adicionar outros módulos conforme necessário

__all__ = [
    # INMET data processing
    'read_inmet_data',
    'process_inmet_files',
    'create_inmet_geodataframe',
    # Adicionar outras funções conforme necessário
]
