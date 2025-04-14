try:
    from src.graph.road.pipeline import (
        load_road_data,
        load_contextual_data,
        explode_multilines,
        calculate_sinuosity,
        clean_road_data,
        check_connectivity,
        prepare_node_features,
        prepare_edge_features,
        normalize_features,
        preprocess_road_data,
        run_preprocessing_pipeline
    )
    print("Módulos importados com sucesso!")
    print("Funções disponíveis:")
    print("- load_road_data")
    print("- load_contextual_data")
    print("- explode_multilines")
    print("- calculate_sinuosity")
    print("- clean_road_data")
    print("- check_connectivity")
    print("- prepare_node_features")
    print("- prepare_edge_features")
    print("- normalize_features")
    print("- preprocess_road_data")
    print("- run_preprocessing_pipeline")
except Exception as e:
    print(f"Erro ao importar os módulos: {str(e)}") 