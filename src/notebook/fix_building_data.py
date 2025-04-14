#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to demonstrate proper handling of buildings_with_landuse data
whether it's a dictionary or GeoDataFrame
"""

import os
import sys
import pickle
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(BASE_DIR, "DATA")
OUTPUT_DIR = os.path.join(BASE_DIR, "OUTPUT")
REPORT_DIR = os.path.join(BASE_DIR, "QUALITY_REPORT")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

print(f"Base directory: {BASE_DIR}")
print(f"Data directory: {DATA_DIR}")
print(f"Output directory: {OUTPUT_DIR}")

# Path to the buildings data file
results_dir = os.path.join(OUTPUT_DIR, 'intermediate_results')
os.makedirs(results_dir, exist_ok=True)
buildings_landuse_pkl = os.path.join(results_dir, 'buildings_with_landuse.pkl')

def explore_building_attributes(gdf):
    """
    Explores the available attributes for building categorization.
    
    Args:
        gdf: GeoDataFrame with buildings and land use information
    """
    # Check if the GeoDataFrame is empty
    if gdf.empty:
        print("GeoDataFrame is empty. No data to explore.")
        return
        
    # Check available columns
    building_cols = [col for col in gdf.columns if 'build' in col.lower()]
    landuse_cols = [col for col in gdf.columns if 'land' in col.lower() or 'uso' in col.lower()]
    amenity_cols = [col for col in gdf.columns if 'amen' in col.lower()]
    function_cols = [col for col in gdf.columns if 'func' in col.lower() or 'class' in col.lower()]

    print("Available columns for categorization:")
    print(f"- Building columns: {building_cols}")
    print(f"- Land use columns: {landuse_cols}")
    print(f"- Amenity columns: {amenity_cols}")
    print(f"- Function/class columns: {function_cols}")

    # Explore unique values in main columns
    for cols in [building_cols, landuse_cols, amenity_cols, function_cols]:
        for col in cols:
            if col in gdf.columns:
                values = gdf[col].dropna().unique()
                print(f"\nUnique values in '{col}' ({len(values)} values):")
                if len(values) > 20:
                    print(values[:20], "... (more values)")
                else:
                    print(values)

def main():
    """Main function to demonstrate loading and handling building data"""
    
    # Check if the file exists
    if not os.path.exists(buildings_landuse_pkl):
        print(f"File not found: {buildings_landuse_pkl}")
        
        # Create a dummy GeoDataFrame as an example
        from shapely.geometry import Point
        buildings_gdf = gpd.GeoDataFrame(
            {
                'building': ['house', 'commercial', 'apartments', 'industrial'],
                'land_category': ['residential', 'commercial', 'residential', 'industrial'],
                'geometry': [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)]
            }
        )
        
        # Save as a dummy example for future use
        os.makedirs(os.path.dirname(buildings_landuse_pkl), exist_ok=True)
        with open(buildings_landuse_pkl, 'wb') as f:
            pickle.dump(buildings_gdf, f)
        
        print(f"Created a dummy buildings file at {buildings_landuse_pkl}")
    else:
        print(f"Loading data from {buildings_landuse_pkl}")
        
        # Load the data
        try:
            with open(buildings_landuse_pkl, 'rb') as f:
                buildings_with_landuse = pickle.load(f)
                
            print(f"Data loaded from {buildings_landuse_pkl}")
            
            # Check the type of data loaded
            if isinstance(buildings_with_landuse, dict):
                print(f"Data loaded as dictionary with {len(buildings_with_landuse)} keys")
                # If it's a dictionary, extract the buildings GeoDataFrame
                if 'buildings' in buildings_with_landuse:
                    buildings_gdf = buildings_with_landuse['buildings']
                    print(f"Using 'buildings' layer with {len(buildings_gdf)} features")
                else:
                    # Try to find the first key that is a GeoDataFrame
                    first_key = next(iter(buildings_with_landuse.keys()), None)
                    if first_key:
                        buildings_gdf = buildings_with_landuse[first_key]
                        print(f"Using '{first_key}' layer with {len(buildings_gdf)} features")
                    else:
                        buildings_gdf = gpd.GeoDataFrame()
                        print("Could not find buildings data in the dictionary")
            elif isinstance(buildings_with_landuse, gpd.GeoDataFrame):
                print(f"Data loaded as GeoDataFrame with {len(buildings_with_landuse)} features")
                buildings_gdf = buildings_with_landuse
            else:
                print(f"Unknown data format: {type(buildings_with_landuse)}")
                buildings_gdf = gpd.GeoDataFrame()
            
            # Explore the buildings data
            if not buildings_gdf.empty:
                print("\nExploring buildings data attributes:")
                explore_building_attributes(buildings_gdf)
            else:
                print("No buildings data to explore")
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            buildings_gdf = gpd.GeoDataFrame()

if __name__ == "__main__":
    main() 