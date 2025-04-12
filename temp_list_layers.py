import fiona

try:
    # Tenta listar as camadas do arquivo
    layers = fiona.listlayers("F:/TESE_MESTRADO/geoprocessing/data/raw/sorocaba_natural.gpkg")
    print("Camadas disponíveis:")
    for layer in layers:
        print(f"- {layer}")
    
    # Se não houver erro, tenta abrir o arquivo para ver mais detalhes
    with fiona.open("F:/TESE_MESTRADO/geoprocessing/data/raw/sorocaba_natural.gpkg") as src:
        print(f"\nEsquema da camada padrão:")
        print(f"Driver: {src.driver}")
        print(f"CRS: {src.crs}")
        print(f"Número de registros: {len(src)}")
        print(f"Esquema: {src.schema}")
        
except Exception as e:
    print(f"ERRO: {e}") 