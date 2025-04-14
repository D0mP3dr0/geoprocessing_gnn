# -*- coding: utf-8 -*-
"""
Data Loading Functions

This module contains functions for loading road network data from various sources.
"""

import geopandas as gpd
import pandas as pd
import os
import matplotlib.pyplot as plt
import osmnx as ox
import networkx as nx
from shapely.geometry import Point, LineString, box
import torch

# Base directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# For Google Drive setup
DRIVE_DIR = "/content/drive/MyDrive/geoprocessamento_gnn"
DATA_DIR = os.path.join(DRIVE_DIR, "data")
OUTPUT_DIR = os.path.join(DRIVE_DIR, "results")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

def load_road_data(file_path, crs=None):
    """
    Load road data from a file.
    
    Args:
        file_path: Path to the road data file
        crs: Coordinate reference system to use, or None to use the file's CRS
        
    Returns:
        GeoDataFrame with road data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Road data file not found: {file_path}")
    
    # Load based on file extension
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ['.gpkg', '.gdb']:
        # For GeoPackage or GDB, we need to list layers first
        layers = None
        if ext == '.gpkg':
            layers = gpd.io.fiona.listlayers(file_path)
        
        if layers and len(layers) > 0:
            # If multiple layers, load the first one or one with 'road' in the name
            layer_name = next((l for l in layers if 'road' in l.lower()), layers[0])
            gdf = gpd.read_file(file_path, layer=layer_name)
        else:
            gdf = gpd.read_file(file_path)
    elif ext in ['.shp', '.geojson']:
        gdf = gpd.read_file(file_path)
    elif ext == '.csv':
        # For CSV, try to create geometry from coordinates
        df = pd.read_csv(file_path)
        
        # Check if geometry column exists
        if 'geometry' in df.columns:
            # Try to convert from WKT
            try:
                from shapely import wkt
                df['geometry'] = df['geometry'].apply(wkt.loads)
                gdf = gpd.GeoDataFrame(df, geometry='geometry')
            except:
                raise ValueError("Could not parse geometry column in CSV file")
        # Check for lat/lon columns
        elif all(col in df.columns for col in ['lon', 'lat']):
            geometry = [Point(xy) for xy in zip(df.lon, df.lat)]
            gdf = gpd.GeoDataFrame(df, geometry=geometry)
        else:
            raise ValueError("CSV file does not contain valid geometry information")
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    
    # Set CRS if provided
    if crs is not None:
        gdf = gdf.to_crs(crs)
    
    print(f"Loaded {len(gdf)} road segments from {file_path}")
    
    return gdf

def load_from_osm(place_name=None, bbox=None, network_type='drive'):
    """
    Load road network data from OpenStreetMap.
    
    Args:
        place_name: Name of the place to load data for
        bbox: Bounding box as (min_lat, min_lon, max_lat, max_lon)
        network_type: Type of network to load ('drive', 'walk', 'bike', etc.)
        
    Returns:
        NetworkX graph with road network
    """
    if place_name is None and bbox is None:
        raise ValueError("Either place_name or bbox must be provided")
    
    # Load the graph
    if place_name:
        G = ox.graph_from_place(place_name, network_type=network_type)
    else:
        G = ox.graph_from_bbox(bbox[0], bbox[1], bbox[2], bbox[3], network_type=network_type)
    
    # Project to UTM
    G = ox.project_graph(G)
    
    # Add edge and node attributes
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    
    print(f"Loaded OSM network with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    return G

def convert_nx_to_gdf(G):
    """
    Convert a NetworkX graph to GeoDataFrames for nodes and edges.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Tuple of (nodes_gdf, edges_gdf)
    """
    # Get node and edge GeoDataFrames
    nodes_gdf = ox.graph_to_gdfs(G, edges=False)
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    
    return nodes_gdf, edges_gdf

def convert_gdf_to_nx(nodes_gdf, edges_gdf):
    """
    Convert GeoDataFrames to a NetworkX graph.
    
    Args:
        nodes_gdf: GeoDataFrame with nodes
        edges_gdf: GeoDataFrame with edges
        
    Returns:
        NetworkX graph
    """
    # Create graph
    G = nx.Graph()
    
    # Add nodes with attributes
    for idx, row in nodes_gdf.iterrows():
        attrs = row.to_dict()
        # Remove geometry from attributes to avoid serialization issues
        if 'geometry' in attrs:
            attrs['x'] = attrs['geometry'].x
            attrs['y'] = attrs['geometry'].y
            del attrs['geometry']
        G.add_node(idx, **attrs)
    
    # Add edges with attributes
    for idx, row in edges_gdf.iterrows():
        attrs = row.to_dict()
        # Replace geometry with length
        if 'geometry' in attrs:
            attrs['length'] = attrs['geometry'].length
            del attrs['geometry']
        
        u, v = idx if isinstance(idx, tuple) else (idx[0], idx[1])
        G.add_edge(u, v, **attrs)
    
    return G

def load_pytorch_geometric_data(G, node_features=None, edge_features=None):
    """
    Convert a NetworkX graph to PyTorch Geometric data.
    
    Args:
        G: NetworkX graph
        node_features: List of node attributes to use as features
        edge_features: List of edge attributes to use as features
        
    Returns:
        PyTorch Geometric Data object
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        raise ImportError("PyTorch Geometric not found. Please install with: pip install torch-geometric")
    
    # Default features if not provided
    if node_features is None:
        node_features = ['x', 'y']
    
    if edge_features is None:
        edge_features = ['length']
    
    # Extract edge indices
    edge_index = []
    for u, v in G.edges():
        edge_index.append([u, v])
        # For undirected graph, add the reverse edge
        edge_index.append([v, u])
    
    # Convert to tensor
    edge_index = torch.tensor(edge_index).t().contiguous()
    
    # Extract node features
    x = []
    for node in G.nodes():
        node_data = G.nodes[node]
        # Check if all required features exist
        features = []
        for feat in node_features:
            if feat in node_data:
                features.append(float(node_data[feat]))
            else:
                # Use default value of 0.0 for missing features
                features.append(0.0)
        x.append(features)
    
    # Convert to tensor
    x = torch.tensor(x, dtype=torch.float)
    
    # Create data object
    data = Data(x=x, edge_index=edge_index)
    
    return data

def mount_google_drive():
    """Mount Google Drive for Colab environment."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted successfully")
        return True
    except ImportError:
        print("Not running in Google Colab, skipping Drive mount")
        return False 