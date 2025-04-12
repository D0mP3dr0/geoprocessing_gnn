#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from shapely.ops import unary_union, linemerge
import matplotlib.pyplot as plt
import networkx as nx
import json
from datetime import datetime

# Get the absolute path to the project directory
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir))
workspace_dir = os.path.dirname(src_dir)

# Define input and output directories
INPUT_DIR = os.path.join(workspace_dir, 'data', 'processed')
OUTPUT_DIR = os.path.join(workspace_dir, 'data', 'enriched')
REPORT_DIR = os.path.join(workspace_dir, 'src', 'enriched_data', 'quality_reports', 'hidrografia')

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def load_data():
    """
    Load the processed hydrographic data from the processed directory.
    
    Returns:
        geopandas.GeoDataFrame: The loaded hydrographic data.
    """
    input_file = os.path.join(INPUT_DIR, 'hidrografia_processed.gpkg')
    try:
        gdf = gpd.read_file(input_file)
        print(f"Successfully loaded {len(gdf)} features from {input_file}")
        return gdf
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def calculate_sinuosity(gdf):
    """
    Calculate sinuosity for each stream segment.
    Sinuosity = actual length / straight line distance
    
    Args:
        gdf (geopandas.GeoDataFrame): Hydrographic data
        
    Returns:
        geopandas.GeoDataFrame: Updated with sinuosity column
    """
    # Create a copy to avoid SettingWithCopyWarning
    result = gdf.copy()
    
    # Calculate sinuosity for each LineString
    sinuosities = []
    for geom in result.geometry:
        if isinstance(geom, LineString):
            # Actual length of the line
            actual_length = geom.length
            # Straight line distance from start to end
            start_point = Point(geom.coords[0])
            end_point = Point(geom.coords[-1])
            straight_length = start_point.distance(end_point)
            
            # Avoid division by zero
            if straight_length > 0:
                sinuosity = actual_length / straight_length
            else:
                sinuosity = 1.0
        else:
            sinuosity = None
        sinuosities.append(sinuosity)
    
    result['sinuosity'] = sinuosities
    return result

def build_stream_network(gdf):
    """
    Build a network representation of the stream network using NetworkX.
    
    Args:
        gdf (geopandas.GeoDataFrame): Hydrographic data
        
    Returns:
        tuple: (NetworkX graph, Updated GeoDataFrame with network metrics)
    """
    # Create a network graph
    G = nx.Graph()
    
    # Create a copy to avoid SettingWithCopyWarning
    result = gdf.copy()
    
    # Add edges to the graph
    for idx, row in result.iterrows():
        if isinstance(row.geometry, LineString):
            # Use start and end points as nodes
            start_point = row.geometry.coords[0]
            end_point = row.geometry.coords[-1]
            
            # Add edge with attributes
            G.add_edge(start_point, end_point, 
                       length=row.geometry.length,
                       feature_id=idx)
    
    # Calculate network centrality measures
    print("Calculating network centrality measures...")
    
    # Betweenness centrality (identifies important connecting segments)
    edge_betweenness = nx.edge_betweenness_centrality(G, weight='length')
    
    # Map edge betweenness back to the GeoDataFrame
    betweenness_values = []
    for idx, row in result.iterrows():
        if isinstance(row.geometry, LineString):
            start_point = row.geometry.coords[0]
            end_point = row.geometry.coords[-1]
            try:
                betweenness = edge_betweenness.get((start_point, end_point), 
                                                  edge_betweenness.get((end_point, start_point), 0))
            except:
                betweenness = 0
            betweenness_values.append(betweenness)
        else:
            betweenness_values.append(0)
    
    result['betweenness'] = betweenness_values
    
    return G, result

def calculate_drainage_density(gdf, area_sqkm=None):
    """
    Calculate the drainage density (total stream length / area).
    
    Args:
        gdf (geopandas.GeoDataFrame): Hydrographic data
        area_sqkm (float, optional): Area in square kilometers. If None, uses the bounding box area.
        
    Returns:
        float: Drainage density in km/km²
    """
    # Calculate total stream length in kilometers
    total_length_km = gdf.geometry.length.sum() / 1000
    
    if area_sqkm is None:
        # Calculate bounding box area in square kilometers
        bounds = gdf.total_bounds
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        area_sqkm = (width * height) / 1_000_000
    
    # Calculate drainage density
    drainage_density = total_length_km / area_sqkm
    
    return drainage_density

def enrich_strahler_order(gdf):
    """
    Validate and enrich Strahler stream order data if available.
    
    Args:
        gdf (geopandas.GeoDataFrame): Hydrographic data
        
    Returns:
        geopandas.GeoDataFrame: Updated with validated Strahler orders
    """
    result = gdf.copy()
    
    # Check if Strahler order column exists
    strahler_cols = [col for col in result.columns if 'strahler' in col.lower() or 'order' in col.lower()]
    
    if strahler_cols:
        strahler_col = strahler_cols[0]
        
        # Convert to numeric if not already
        if result[strahler_col].dtype not in ['int64', 'float64']:
            result[strahler_col] = pd.to_numeric(result[strahler_col], errors='coerce')
        
        # Fill missing values with 1 (lowest order)
        result[strahler_col] = result[strahler_col].fillna(1).astype(int)
        
        # Create a standardized strahler_order column
        if strahler_col != 'strahler_order':
            result['strahler_order'] = result[strahler_col]
    else:
        # If no Strahler order information, create a basic one (all set to 1)
        print("No Strahler order information found. Creating placeholder column.")
        result['strahler_order'] = 1
    
    return result

def generate_quality_report(original_gdf, enriched_gdf):
    """
    Generate a detailed quality report for the enrichment process.
    
    Args:
        original_gdf (geopandas.GeoDataFrame): Original hydrographic data
        enriched_gdf (geopandas.GeoDataFrame): Enriched hydrographic data
    """
    # Create report
    report = {
        "report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_features": len(original_gdf),
        "enriched_features": len(enriched_gdf),
        "new_attributes": list(set(enriched_gdf.columns) - set(original_gdf.columns)),
        "statistics": {
            "sinuosity": {
                "mean": float(enriched_gdf['sinuosity'].mean()),
                "median": float(enriched_gdf['sinuosity'].median()),
                "min": float(enriched_gdf['sinuosity'].min()),
                "max": float(enriched_gdf['sinuosity'].max())
            },
            "betweenness": {
                "mean": float(enriched_gdf['betweenness'].mean()),
                "median": float(enriched_gdf['betweenness'].median()),
                "min": float(enriched_gdf['betweenness'].min()),
                "max": float(enriched_gdf['betweenness'].max())
            },
            "strahler_order": {
                "distribution": enriched_gdf['strahler_order'].value_counts().to_dict()
            }
        },
        "drainage_density": calculate_drainage_density(enriched_gdf)
    }
    
    # Save report as JSON
    report_file = os.path.join(REPORT_DIR, 'enrichment_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"Quality report saved to {report_file}")
    
    # Generate visualizations
    generate_visualizations(enriched_gdf)

def generate_visualizations(gdf):
    """
    Generate visualizations to accompany the quality report.
    
    Args:
        gdf (geopandas.GeoDataFrame): Enriched hydrographic data
    """
    # Histogram of sinuosity
    plt.figure(figsize=(10, 6))
    gdf['sinuosity'].hist(bins=20)
    plt.title('Distribution of Stream Sinuosity')
    plt.xlabel('Sinuosity')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(REPORT_DIR, 'sinuosity_histogram.png'), dpi=300)
    plt.close()
    
    # Map of network colored by betweenness centrality
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)
    gdf.plot(column='betweenness', ax=ax, linewidth=0.8, cmap='viridis', 
             legend=True, legend_kwds={'label': 'Betweenness Centrality'})
    plt.title('Stream Network Betweenness Centrality')
    plt.savefig(os.path.join(REPORT_DIR, 'betweenness_network.png'), dpi=300)
    plt.close()
    
    # Map of network colored by Strahler order
    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111)
    gdf.plot(column='strahler_order', ax=ax, linewidth=1.0, cmap='plasma', 
             legend=True, legend_kwds={'label': 'Strahler Order'})
    plt.title('Stream Network Strahler Order')
    plt.savefig(os.path.join(REPORT_DIR, 'strahler_order_network.png'), dpi=300)
    plt.close()

def main():
    """
    Main function to enrich hydrographic data.
    """
    print("Starting hydrographic data enrichment...")
    
    # Load processed data
    original_gdf = load_data()
    
    # Apply enrichment steps
    print("Calculating sinuosity...")
    enriched_gdf = calculate_sinuosity(original_gdf)
    
    print("Building stream network...")
    _, enriched_gdf = build_stream_network(enriched_gdf)
    
    print("Enriching Strahler order data...")
    enriched_gdf = enrich_strahler_order(enriched_gdf)
    
    # Save enriched data
    output_file = os.path.join(OUTPUT_DIR, 'hidrografia_enriched.gpkg')
    enriched_gdf.to_file(output_file, driver='GPKG')
    print(f"Enriched data saved to {output_file}")
    
    # Generate quality report and visualizations
    print("Generating quality report and visualizations...")
    generate_quality_report(original_gdf, enriched_gdf)
    
    print("Hydrographic data enrichment completed successfully!")

if __name__ == "__main__":
    main() 