# Geoprocessamento com GNN

Este projeto implementa análises e visualizações de dados geoespaciais para a cidade de Sorocaba, com foco em redes viárias e hidrografia. O projeto utiliza técnicas avançadas de geoprocessamento e análise de redes para extrair insights significativos dos dados.

## Estrutura do Projeto

```
geoprocessing/
├── data/
│   ├── raw/           # Dados brutos
│   └── processed/     # Dados processados
├── src/
│   ├── preprocessing/ # Scripts de pré-processamento
│   └── visualization/ # Scripts de visualização
└── outputs/          # Visualizações e resultados
```

## Funcionalidades

### Pré-processamento
- Limpeza e validação de dados geoespaciais
- Padronização de sistemas de coordenadas
- Análise de qualidade dos dados
- Correção de geometrias

### Visualizações
- Mapas interativos com Folium
- Análise de rede viária
- Análise de conectividade
- Mapas de densidade
- Análises estatísticas

## Requisitos

- Python 3.8+
- GeoPandas
- NetworkX
- Folium
- Matplotlib
- Seaborn
- Contextily

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/D0mP3dr0/geoprocessing_gnn.git
cd geoprocessing_gnn
```

2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

## Uso

### Processamento de Dados

1. Processamento da rede viária:
```bash
python src/preprocessing/roads.py
```

2. Processamento da hidrografia:
```bash
python src/preprocessing/hidrografia.py
```

### Visualizações

1. Visualizações da rede viária:
```bash
python src/visualization/visualize_roads.py
```

2. Visualizações da hidrografia:
```bash
python src/visualization/visualize_hidrografia.py
```

## Resultados

Os resultados incluem:
- Mapas interativos da rede viária e hidrografia
- Análises de conectividade
- Estatísticas da rede
- Mapas de densidade
- Relatórios de qualidade dos dados

## Contribuição

1. Faça um Fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## Contato

Pedro Domingos - [@D0mP3dr0](https://github.com/D0mP3dr0) 