#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to safely load and process buildings data
"""

import os
import sys
import pickle
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

def main():
    # Define paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT")
    
    # Path to the buildings data file
    results_dir = os.path.join(OUTPUT_DIR, 'intermediate_results')
    os.makedirs(results_dir, exist_ok=True)
    buildings_landuse_pkl = os.path.join(results_dir, 'buildings_with_landuse.pkl')

    print(f"Looking for buildings data at: {buildings_landuse_pkl}")
    
    # Check if the file exists
    if not os.path.exists(buildings_landuse_pkl):
        print(f"File not found: {buildings_landuse_pkl}")
        buildings_gdf = gpd.GeoDataFrame()
    else:
        # Load the data safely
        try:
            with open(buildings_landuse_pkl, 'rb') as f:
                buildings_with_landuse = pickle.load(f)
                
            print(f"Data loaded from {buildings_landuse_pkl}")
            
            # Check the type of data loaded
            if isinstance(buildings_with_landuse, dict):
                print(f"Data loaded as dictionary with {len(buildings_with_landuse)} keys")
                print(f"Available keys: {list(buildings_with_landuse.keys())}")
                
                # Try to find a GeoDataFrame in the dictionary
                buildings_gdf = None
                
                # First, check for common building keys
                building_keys = ['buildings', 'edificios', 'edificações']
                for key in building_keys:
                    if key in buildings_with_landuse and isinstance(buildings_with_landuse[key], gpd.GeoDataFrame):
                        buildings_gdf = buildings_with_landuse[key]
                        print(f"Found buildings data in key '{key}' with {len(buildings_gdf)} features")
                        break
                
                # If not found by key name, check all values for GeoDataFrames
                if buildings_gdf is None:
                    for key, value in buildings_with_landuse.items():
                        if isinstance(value, gpd.GeoDataFrame):
                            # Check if it might be building data based on columns
                            if 'building' in value.columns or any('build' in col.lower() for col in value.columns):
                                buildings_gdf = value
                                print(f"Found potential buildings data in key '{key}' with {len(buildings_gdf)} features")
                                break
                
                # If still not found, use any GeoDataFrame as a fallback
                if buildings_gdf is None:
                    for key, value in buildings_with_landuse.items():
                        if isinstance(value, gpd.GeoDataFrame):
                            buildings_gdf = value
                            print(f"Using GeoDataFrame from key '{key}' as fallback with {len(buildings_gdf)} features")
                            break
                
                # If we still don't have a GeoDataFrame, create an empty one
                if buildings_gdf is None:
                    print("No GeoDataFrame found in the dictionary")
                    buildings_gdf = gpd.GeoDataFrame()
                    
            elif isinstance(buildings_with_landuse, gpd.GeoDataFrame):
                print(f"Data loaded as GeoDataFrame with {len(buildings_with_landuse)} features")
                buildings_gdf = buildings_with_landuse
            else:
                print(f"Unknown data format: {type(buildings_with_landuse)}")
                buildings_gdf = gpd.GeoDataFrame()
        except Exception as e:
            print(f"Error loading data: {e}")
            buildings_gdf = gpd.GeoDataFrame()
    
    # Now we can safely work with buildings_gdf
    if isinstance(buildings_gdf, gpd.GeoDataFrame) and not buildings_gdf.empty:
        print("\nBuildings data summary:")
        print(f"- Number of features: {len(buildings_gdf)}")
        print(f"- Columns: {buildings_gdf.columns.tolist()}")
        print(f"- CRS: {buildings_gdf.crs}")
        
        # Try to show a sample of the data
        print("\nSample of buildings data:")
        print(buildings_gdf.head())
        
        # Only try to access columns if they actually exist
        if 'categoria_funcional' in buildings_gdf.columns:
            print("\nBuilding categories summary:")
            print(buildings_gdf['categoria_funcional'].value_counts())
        else:
            print("\nNo 'categoria_funcional' column found in the data")
    else:
        print("\nNo valid buildings data available for processing")
        
        # Create demo data for testing
        print("\nCreating demo data for testing...")
        from shapely.geometry import Point
        
        # Create some example points
        geometries = [Point(x, y) for x, y in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]]
        
        # Create example data
        demo_data = {
            'geometry': geometries,
            'building': ['residential', 'commercial', 'industrial', 'school', 'hospital'],
            'land_category': ['residential', 'commercial', 'industrial', 'institutional', 'healthcare']
        }
        
        # Create GeoDataFrame
        demo_gdf = gpd.GeoDataFrame(demo_data, crs="EPSG:4326")
        print(demo_gdf)
        
if __name__ == "__main__":
    main() 