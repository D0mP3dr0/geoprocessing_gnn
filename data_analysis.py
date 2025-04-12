# -*- coding: utf-8 -*-

import os
import geopandas as gpd
import pandas as pd
import json
import numpy as np
from shapely.geometry import box
from scipy import stats

# Function to harmonize CRS to a standard (e.g., SIRGAS 2000)
def harmonize_crs(gdf, target_crs='EPSG:4674'):
    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

# Function to transform data (e.g., clip to area of interest)
def transform_data(gdf, area_of_interest=None):
    if area_of_interest is not None:
        gdf = gpd.clip(gdf, area_of_interest)
    return gdf

# Function to validate data integrity
def validate_data(gdf):
    # Example validation: Check for null geometries
    if gdf.is_empty.any():
        raise ValueError("Some geometries are empty.")
    return True

# Function to convert geometries to WKT for JSON serialization
def convert_geometries_to_wkt(gdf):
    gdf['geometry_wkt'] = gdf.geometry.apply(lambda geom: geom.wkt if geom else None)
    return gdf

# Function to calculate data quality metrics
def calculate_data_quality(df):
    quality_metrics = {
        'completeness': {
            'percent_missing': df.isnull().mean().to_dict(),
            'total_missing_values': df.isnull().sum().sum(),
            'columns_with_nulls': df.columns[df.isnull().any()].tolist()
        },
        'uniqueness': {
            'duplicate_rows': df.duplicated().sum(),
            'unique_values_by_column': {col: df[col].nunique() for col in df.columns if df[col].dtype != 'geometry'}
        }
    }
    return quality_metrics

# Function to calculate spatial statistics for geospatial data
def calculate_spatial_statistics(gdf):
    try:
        # Basic spatial metrics
        spatial_stats = {
            'bounds': {
                'minx': gdf.total_bounds[0],
                'miny': gdf.total_bounds[1],
                'maxx': gdf.total_bounds[2],
                'maxy': gdf.total_bounds[3]
            },
            'area': gdf.area.describe().to_dict() if 'Polygon' in gdf.geom_type.unique() else None,
            'length': gdf.length.describe().to_dict() if 'Line' in gdf.geom_type.unique() else None,
            'centroid': {
                'x': gdf.centroid.x.mean(),
                'y': gdf.centroid.y.mean()
            }
        }
        
        # Spatial distribution metrics
        if len(gdf) > 1:
            x_coords = gdf.centroid.x
            y_coords = gdf.centroid.y
            
            # Nearest neighbor analysis
            nearest_neighbor = {
                'mean_x': x_coords.mean(),
                'mean_y': y_coords.mean(),
                'std_x': x_coords.std(),
                'std_y': y_coords.std(),
                'spatial_distribution': 'clustered' if (x_coords.std() + y_coords.std()) / 2 < (gdf.total_bounds[2] - gdf.total_bounds[0] + gdf.total_bounds[3] - gdf.total_bounds[1]) / 10 else 'dispersed'
            }
            spatial_stats['distribution'] = nearest_neighbor
        
        return spatial_stats
    except Exception as e:
        return {'error': str(e)}

# Function to analyze topological relationships for graph creation
def analyze_topology(gdf, other_layers=None):
    try:
        import networkx as nx
        
        # Topology metrics dictionary
        topology_metrics = {
            'connectivity': {},
            'adjacency': {},
            'potential_nodes': 0,
            'potential_edges': 0,
            'node_candidates': [],
            'graph_metrics': {}
        }
        
        # Analyze internal connectivity
        if 'Line' in gdf.geom_type.unique() or 'LineString' in gdf.geom_type.unique() or 'MultiLineString' in gdf.geom_type.unique():
            # For linear features (roads, rivers, etc.)
            from shapely.ops import linemerge, unary_union
            
            # Create a graph from lines
            lines = gdf.geometry.unary_union
            try:
                merged = linemerge(lines)
                
                # Create a NetworkX graph
                G = nx.Graph()
                
                if merged.geom_type == 'LineString':
                    coords = list(merged.coords)
                    for i in range(len(coords) - 1):
                        G.add_edge(coords[i], coords[i+1], weight=merged.length)
                else:  # MultiLineString
                    for line in merged.geoms:
                        coords = list(line.coords)
                        for i in range(len(coords) - 1):
                            G.add_edge(coords[i], coords[i+1], weight=line.length)
                
                # Calculate graph metrics
                topology_metrics['graph_metrics'] = {
                    'num_nodes': G.number_of_nodes(),
                    'num_edges': G.number_of_edges(),
                    'connected_components': nx.number_connected_components(G),
                    'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
                    'density': nx.density(G)
                }
                
                # Store potential nodes
                topology_metrics['potential_nodes'] = G.number_of_nodes()
                topology_metrics['potential_edges'] = G.number_of_edges()
                
                # Identify important nodes (intersections or endpoints)
                important_nodes = [node for node, degree in dict(G.degree()).items() if degree > 2 or degree == 1]
                topology_metrics['node_candidates'] = important_nodes[:10]  # Sample of nodes
            except Exception as e:
                topology_metrics['error'] = f"Error creating graph: {str(e)}"
        
        # For polygon layers, analyze adjacency
        elif 'Polygon' in gdf.geom_type.unique() or 'MultiPolygon' in gdf.geom_type.unique():
            from shapely.geometry import MultiPolygon
            
            # Check for adjacent polygons
            for i, row1 in gdf.iterrows():
                if i >= 10:  # Limit the analysis to 10 polygons for performance
                    break
                    
                adjacent_count = 0
                for j, row2 in gdf.iterrows():
                    if i != j and row1.geometry.touches(row2.geometry):
                        adjacent_count += 1
                
                if adjacent_count > 0:
                    topology_metrics['adjacency'][i] = adjacent_count
            
            # Calculate potential nodes (centroids or vertices)
            topology_metrics['potential_nodes'] = len(gdf)
            
            # Sample node candidates (centroids)
            node_candidates = []
            for i, row in gdf.head(5).iterrows():
                node_candidates.append((row.geometry.centroid.x, row.geometry.centroid.y))
            
            topology_metrics['node_candidates'] = node_candidates
        
        # Cross-layer relationships if other layers are provided
        if other_layers is not None:
            topology_metrics['cross_layer_relationships'] = {}
            
            for layer_name, other_gdf in other_layers.items():
                if len(other_gdf) > 0:
                    # Check for intersections between layers
                    intersection_count = 0
                    for i, row1 in gdf.head(5).iterrows():
                        for j, row2 in other_gdf.head(5).iterrows():
                            if row1.geometry.intersects(row2.geometry):
                                intersection_count += 1
                    
                    topology_metrics['cross_layer_relationships'][layer_name] = {
                        'intersections': intersection_count,
                        'potential_graph_links': intersection_count
                    }
        
        return topology_metrics
    except Exception as e:
        return {'error': str(e)}

# Function to generate graph integration recommendations
def generate_graph_recommendations(report):
    recommendations = []
    
    # Check if the data has spatial topology information
    if 'topology' in report:
        topology = report['topology']
        
        # For network-like data (roads, rivers, etc.)
        if 'graph_metrics' in topology and topology['graph_metrics'].get('num_nodes', 0) > 0:
            if topology['graph_metrics'].get('connected_components', 1) > 1:
                recommendations.append("Network appears disconnected. Consider edge linking algorithms to create a fully connected graph.")
            
            if topology['graph_metrics'].get('avg_degree', 0) < 2:
                recommendations.append("Low connectivity in network. May need additional edges to create meaningful paths.")
            
            recommendations.append(f"Create graph with approximately {topology['potential_nodes']} nodes and {topology['potential_edges']} edges based on spatial features.")
        
        # For polygon adjacency data
        if 'adjacency' in topology and len(topology['adjacency']) > 0:
            recommendations.append("Create polygon adjacency graph using shared boundaries as edges.")
            
        # For cross-layer relationships
        if 'cross_layer_relationships' in topology and len(topology['cross_layer_relationships']) > 0:
            for layer, rel in topology['cross_layer_relationships'].items():
                if rel.get('intersections', 0) > 0:
                    recommendations.append(f"Create cross-layer links with {layer} based on spatial intersections (approx. {rel.get('potential_graph_links', 0)} links).")
    
    # Check for point data (possible graph nodes)
    if report.get('geometry_types') and ('Point' in report['geometry_types'] or 'MultiPoint' in report['geometry_types']):
        recommendations.append("Points can serve as graph nodes. Consider attribute-based or distance-based edge creation.")
    
    # Check for attribute-based links
    if 'data_quality' in report and 'columns_with_nulls' in report['data_quality']['completeness']:
        non_null_cols = [col for col in report.get('columns', []) if col not in report['data_quality']['completeness']['columns_with_nulls']]
        categorical_cols = [col for col in non_null_cols if col in report.get('categorical_stats', {})]
        
        if len(categorical_cols) > 0:
            sample_col = categorical_cols[0]
            recommendations.append(f"Consider attribute-based graph links using categorical attributes like '{sample_col}'.")
    
    return recommendations

# Function to generate preprocessing recommendations
def generate_preprocessing_recommendations(report):
    recommendations = []
    
    # Check for CRS consistency
    if 'crs' in report and report['crs'] != 'EPSG:4674':
        recommendations.append("Convert coordinate system to SIRGAS 2000 (EPSG:4674) for consistency with Brazilian standards.")
    
    # Check for missing values
    if 'data_quality' in report and report['data_quality']['completeness']['total_missing_values'] > 0:
        recommendations.append(f"Handle missing values in columns: {', '.join(report['data_quality']['completeness']['columns_with_nulls'])}.")
    
    # Check for potential outliers in numerical columns
    if 'basic_stats' in report:
        for col, stats_dict in report['basic_stats'].items():
            if 'std' in stats_dict and stats_dict['std'] is not None and not np.isnan(stats_dict['std']):
                # Simple outlier detection based on 3-sigma rule
                if stats_dict['std'] > 0 and (stats_dict['max'] - stats_dict['mean']) / stats_dict['std'] > 3:
                    recommendations.append(f"Investigate potential outliers in column '{col}'.")
    
    # Check for spatial issues
    if 'spatial_statistics' in report:
        if 'area' in report['spatial_statistics'] and report['spatial_statistics']['area'] is not None:
            if report['spatial_statistics']['area'].get('min', 0) == 0:
                recommendations.append("Some polygons have zero area, consider removing or fixing them.")
        
        if 'distribution' in report['spatial_statistics']:
            if report['spatial_statistics']['distribution']['spatial_distribution'] == 'clustered':
                recommendations.append("Data appears spatially clustered, consider analyzing this pattern.")
    
    return recommendations

# Function to analyze a geospatial file with enhanced metrics
def analyze_geospatial_file(file_path, area_of_interest=None):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            return [{'error': f"File does not exist: {file_path}"}]
            
        # Check if the file is a GeoPackage with multiple layers
        try:
            import fiona
            # Use fiona directly for more robust layer listing
            with fiona.open(file_path, 'r') as src:
                if hasattr(src, 'listlayers'):
                    layers = src.listlayers()
                else:
                    layers = ['default']
        except Exception as layer_error:
            # For shapefile or single-layer files
            if file_path.endswith('.shp'):
                layers = ['default']
            else:
                try:
                    # Try to open as a single layer
                    temp_gdf = gpd.read_file(file_path)
                    layers = ['default']
                except Exception as inner_error:
                    return [{'error': f"Cannot read file {file_path}: {str(inner_error)}"}]
        
        reports = []
        layer_gdfs = {}  # Store GeoDataFrames for cross-layer topology analysis
        
        for layer in layers:
            try:
                print(f"  Processing layer: {layer}")
                if layer == 'default':
                    gdf = gpd.read_file(file_path)
                else:
                    gdf = gpd.read_file(file_path, layer=layer)
                
                if gdf.empty:
                    reports.append({'error': f"Layer {layer} is empty"})
                    continue
                    
                if 'geometry' not in gdf.columns or gdf.geometry.isna().all():
                    reports.append({'error': f"Layer {layer} has no valid geometries"})
                    continue
                
                # Harmonize CRS
                if gdf.crs is None:
                    # Try to set a default CRS if missing
                    try:
                        gdf.set_crs('EPSG:4674', inplace=True)
                        print(f"  Warning: CRS was missing, set to EPSG:4674 for layer {layer}")
                    except:
                        pass
                else:
                    gdf = harmonize_crs(gdf)
                
                # Transform if area of interest is provided
                if area_of_interest is not None:
                    try:
                        gdf = transform_data(gdf, area_of_interest)
                    except Exception as clip_error:
                        print(f"  Warning: Could not clip to area of interest: {str(clip_error)}")
                
                # Validate data - but don't fail if validation errors occur
                try:
                    validate_data(gdf)
                except Exception as val_error:
                    print(f"  Warning: Validation issue: {str(val_error)}")
                
                # Store GeoDataFrame for later cross-layer analysis
                layer_gdfs[layer] = gdf
                
                # Calculate additional metrics
                data_quality = calculate_data_quality(gdf)
                spatial_stats = calculate_spatial_statistics(gdf)
                
                # Convert geometries to WKT for serialization
                try:
                    gdf = convert_geometries_to_wkt(gdf)
                except Exception as wkt_error:
                    print(f"  Warning: Could not convert geometries to WKT: {str(wkt_error)}")
                    # Create a simplified version without WKT if it fails
                    gdf['geometry_wkt'] = "GEOMETRY"
                
                # Create comprehensive report
                report = {
                    'file_path': file_path,
                    'layer': layer,
                    'crs': str(gdf.crs) if gdf.crs else "Unknown",
                    'columns': list(gdf.columns),
                    'data_types': {str(col): str(gdf[col].dtype) for col in gdf.columns},
                    'num_features': len(gdf),
                    'geometry_types': gdf.geom_type.unique().tolist() if not gdf.geometry.isna().all() else ["None"],
                    'basic_stats': {},
                    'categorical_stats': {},
                    'data_quality': data_quality,
                    'spatial_statistics': spatial_stats,
                    'sample_geometries': gdf['geometry_wkt'].head(5).tolist() if 'geometry_wkt' in gdf.columns else []
                }
                
                # Calculate stats for numerical columns
                for col in gdf.columns:
                    # Skip geometry column
                    if col == 'geometry' or col == 'geometry_wkt':
                        continue
                    
                    # Calculate stats based on data type
                    try:
                        if pd.api.types.is_numeric_dtype(gdf[col]):
                            report['basic_stats'][str(col)] = gdf[col].describe().to_dict()
                        elif gdf[col].dtype == 'object':
                            # Limit categorical stats to top entries to avoid huge reports
                            report['categorical_stats'][str(col)] = gdf[col].value_counts().head(20).to_dict()
                    except Exception as stats_error:
                        print(f"  Warning: Could not calculate statistics for column {col}: {str(stats_error)}")
                
                # Generate preprocessing recommendations
                report['preprocessing_recommendations'] = generate_preprocessing_recommendations(report)
                
                reports.append(report)
            except Exception as e:
                import traceback
                error_traceback = traceback.format_exc()
                reports.append({'error': f"Error analyzing layer {layer}: {str(e)}", 
                               'traceback': error_traceback})
        
        # After all layers are processed, perform cross-layer topology analysis if we have multiple layers
        if len(layer_gdfs) > 1:
            for layer_name, gdf in layer_gdfs.items():
                # Skip if the layer has issues
                if gdf.empty or 'geometry' not in gdf.columns or gdf.geometry.isna().all():
                    continue
                    
                other_layers = {k: v for k, v in layer_gdfs.items() if k != layer_name}
                
                try:
                    topology = analyze_topology(gdf, other_layers)
                    
                    # Find the report for this layer and add topology information
                    for report in reports:
                        if report.get('layer') == layer_name and 'error' not in report:
                            report['topology'] = topology
                            
                            # Add graph integration recommendations
                            graph_recommendations = generate_graph_recommendations(report)
                            if graph_recommendations:
                                report['graph_integration_recommendations'] = graph_recommendations
                except Exception as topo_error:
                    print(f"  Warning: Couldn't analyze topology for layer {layer_name}: {str(topo_error)}")
        else:
            # Single layer topology analysis
            for layer_name, gdf in layer_gdfs.items():
                # Skip if the layer has issues
                if gdf.empty or 'geometry' not in gdf.columns or gdf.geometry.isna().all():
                    continue
                
                try:
                    topology = analyze_topology(gdf)
                    
                    # Find the report for this layer and add topology information
                    for report in reports:
                        if report.get('layer') == layer_name and 'error' not in report:
                            report['topology'] = topology
                            
                            # Add graph integration recommendations
                            graph_recommendations = generate_graph_recommendations(report)
                            if graph_recommendations:
                                report['graph_integration_recommendations'] = graph_recommendations
                except Exception as topo_error:
                    print(f"  Warning: Couldn't analyze topology for layer {layer_name}: {str(topo_error)}")
        
        return reports
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        return [{'error': f"Error analyzing file {file_path}: {str(e)}", 
                'traceback': error_traceback}]

# Function to analyze a CSV file with enhanced metrics
def analyze_csv_file(file_path):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            return {'error': f"File does not exist: {file_path}"}
            
        # Try different approaches to read the CSV
        df = None
        error_messages = []
        
        # Attempt with different encodings and delimiters
        encodings = ['utf-8', 'latin1', 'cp1252', 'ISO-8859-1']
        delimiters = [';', ',', '\t', '|']
        
        for encoding in encodings:
            if df is not None:
                break
                
            for delimiter in delimiters:
                try:
                    print(f"  Trying to read with encoding={encoding}, delimiter={delimiter}")
                    df = pd.read_csv(file_path, delimiter=delimiter, encoding=encoding, low_memory=False)
                    if len(df.columns) > 1:  # Ensure we have more than one column (successful parsing)
                        break
                except Exception as e:
                    error_messages.append(f"Failed with {encoding}, {delimiter}: {str(e)}")
                    continue
            
        if df is None:
            # Try with pandas inference as last resort
            try:
                print("  Attempting to use pandas inference...")
                df = pd.read_csv(file_path, engine='python', low_memory=False)
            except Exception as e:
                error_messages.append(f"Failed with pandas inference: {str(e)}")
        
        if df is None:
            return {'error': f"Failed to read CSV file with all attempted methods: {error_messages}"}
        
        print(f"  Successfully read CSV with {len(df)} rows, {len(df.columns)} columns")
        
        # Convert integer column names to strings
        df.columns = [str(col) for col in df.columns]
        
        # Check for spatial columns (latitude/longitude)
        spatial_columns = []
        for col in df.columns:
            col_lower = str(col).lower()
            if any(term in col_lower for term in ['latitude', 'longitude', 'lat', 'lon', 'coords', 'x', 'y']):
                spatial_columns.append(col)
        
        has_spatial_data = len(spatial_columns) >= 2
        
        # Calculate data quality metrics
        data_quality = calculate_data_quality(df)
        
        # Calculate additional statistical metrics
        stats_dict = {}
        for col in df.columns:
            # Skip non-numeric columns
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
                
            try:
                # Calculate more advanced statistics
                col_stats = df[col].describe().to_dict()
                # Add additional metrics if possible
                if len(df[col].dropna()) > 0:
                    col_stats['skewness'] = stats.skew(df[col].dropna())
                    col_stats['kurtosis'] = stats.kurtosis(df[col].dropna())
                    col_stats['iqr'] = col_stats['75%'] - col_stats['25%']
                stats_dict[str(col)] = col_stats
            except Exception as e:
                print(f"  Warning: Could not calculate all statistics for column {col}: {str(e)}")
                # Try to get basic stats at least
                try:
                    stats_dict[str(col)] = df[col].describe().to_dict()
                except:
                    pass
        
        # Generate report
        report = {
            'file_path': file_path,
            'columns': list(df.columns),
            'data_types': {str(col): str(df[col].dtype) for col in df.columns},
            'num_rows': len(df),
            'basic_stats': stats_dict,
            'categorical_stats': {},
            'data_quality': data_quality,
            'has_spatial_data': has_spatial_data,
            'spatial_columns': spatial_columns if has_spatial_data else None
        }
        
        # Add categorical stats (limit to avoid huge reports)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    report['categorical_stats'][str(col)] = df[col].value_counts().head(20).to_dict()
                except Exception as e:
                    print(f"  Warning: Could not calculate categorical stats for column {col}: {str(e)}")
        
        # Generate preprocessing recommendations
        recommendations = []
        
        # Check for missing values
        if data_quality['completeness']['total_missing_values'] > 0:
            recommendations.append(f"Handle missing values in columns: {', '.join(data_quality['completeness']['columns_with_nulls'])}.")
        
        # Check for potential outliers in numerical columns
        for col, col_stats in stats_dict.items():
            if 'std' in col_stats and col_stats['std'] is not None and not np.isnan(col_stats['std']):
                # Simple outlier detection based on 3-sigma rule
                if col_stats['std'] > 0 and (col_stats['max'] - col_stats['mean']) / col_stats['std'] > 3:
                    recommendations.append(f"Investigate potential outliers in column '{col}'.")
        
        # Check for skewed distributions
        for col, col_stats in stats_dict.items():
            if 'skewness' in col_stats and not np.isnan(col_stats['skewness']) and abs(col_stats['skewness']) > 1:
                recommendations.append(f"Column '{col}' has a skewed distribution (skewness: {col_stats['skewness']:.2f}), consider transformation.")
        
        # Recommend conversion to spatial if lat/long exists
        if has_spatial_data:
            recommendations.append(f"Convert to geospatial format using columns: {spatial_columns}.")
        
        report['preprocessing_recommendations'] = recommendations
        
        return report
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {'error': str(e), 'error_details': error_details}

# Function to analyze a TIFF file with enhanced metrics
def analyze_tiff_file(file_path):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            return {'error': f"File does not exist: {file_path}"}
            
        import rasterio
        from rasterio.features import shapes
        import numpy as np
        
        with rasterio.open(file_path) as src:
            # Basic metadata
            report = {
                'file_path': file_path,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'crs': str(src.crs) if src.crs else "Unknown",
                'transform': src.transform.to_gdal(),
                'resolution': {
                    'x': src.res[0],
                    'y': src.res[1]
                },
                'bounds': {
                    'left': src.bounds.left,
                    'bottom': src.bounds.bottom,
                    'right': src.bounds.right,
                    'top': src.bounds.top
                }
            }
            
            # Read the first band and calculate statistics
            band_stats = []
            for i in range(1, src.count + 1):
                try:
                    band = src.read(i)
                    # Mask no data values
                    if src.nodata is not None:
                        band_data = band[band != src.nodata]
                    else:
                        band_data = band.flatten()
                    
                    if len(band_data) > 0:
                        stats_dict = {
                            'band': i,
                            'min': float(np.min(band_data)),
                            'max': float(np.max(band_data)),
                            'mean': float(np.mean(band_data)),
                            'std': float(np.std(band_data)),
                            'median': float(np.median(band_data)),
                            'nodata_value': src.nodata,
                            'nodata_percentage': 100 * np.sum(band == src.nodata) / band.size if src.nodata is not None else 0
                        }
                        band_stats.append(stats_dict)
                except Exception as band_err:
                    print(f"  Warning: Error processing band {i}: {str(band_err)}")
            
            report['band_statistics'] = band_stats
            
            # Generate preprocessing recommendations
            recommendations = []
            
            # Check for nodata values
            if any(stat.get('nodata_percentage', 0) > 5 for stat in band_stats):
                recommendations.append("High percentage of NoData values detected, consider filling or masking.")
            
            # Check for CRS
            if src.crs is None:
                recommendations.append("Raster lacks coordinate reference system, consider setting one.")
            elif str(src.crs) != 'EPSG:4674':
                recommendations.append("Convert coordinate system to SIRGAS 2000 (EPSG:4674) for consistency with Brazilian standards.")
            
            # Check for very high resolution that may slow processing
            if src.res[0] < 5 and src.res[1] < 5 and src.width * src.height > 10000000:
                recommendations.append("High resolution raster, consider resampling for faster processing.")
            
            report['preprocessing_recommendations'] = recommendations
            
            return report
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {'error': str(e), 'error_details': error_details}

# Function to convert numpy data types to native Python types
def convert_numpy_to_native(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(i) for i in obj]
    else:
        return obj

# Function to save the analysis report as a JSON file
def save_report_as_json(report, output_path):
    # Convert numpy data types to native Python types
    report = convert_numpy_to_native(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

# Main function to iterate over files and perform analysis
def main():
    # Base directory for the project
    base_dir = os.path.abspath(os.getcwd())
    data_dir = os.path.join(base_dir, 'data')
    raw_data_dir = os.path.join(data_dir, 'raw')
    output_dir = os.path.join(base_dir, 'analysis_reports')
    os.makedirs(output_dir, exist_ok=True)
    
    # Define an area of interest for clipping
    area_of_interest = None
    try:
        sorocaba_file = os.path.join(data_dir, 'raw', 'sorocaba.gpkg')
        if os.path.exists(sorocaba_file):
            area_of_interest = gpd.read_file(sorocaba_file)
            print(f"Using area of interest from sorocaba.gpkg with {len(area_of_interest)} features")
    except Exception as aoi_error:
        print(f"Warning: Could not load area of interest: {str(aoi_error)}")
    
    # Track processed and skipped files
    processed_files = []
    skipped_files = []
    error_files = []
    
    # List of specific files to ensure they're included in the analysis
    specific_files = [
        os.path.join(raw_data_dir, 'sorocaba_setores_censitarios.gpkg'),
        # Add other specific files here if needed
    ]
    
    # Function to process a single file
    def process_file(file_path):
        file_name = os.path.basename(file_path)
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"\nAnalyzing {file_name} ({file_size_mb:.2f} MB)...")
            
            # Skip hidden files
            if file_name.startswith('.'):
                print(f"Skipping hidden file: {file_name}")
                skipped_files.append(file_name)
                return
            
            if file_name.lower().endswith(('.gpkg', '.shp')):
                print(f"Processing as geospatial file...")
                reports = analyze_geospatial_file(file_path, area_of_interest)
                has_error = False
                
                for report in reports:
                    layer_name = report.get('layer', 'unknown')
                    output_path = os.path.join(output_dir, f"{file_name}_{layer_name}_analysis.json")
                    save_report_as_json(report, output_path)
                    print(f"Report saved: {output_path}")
                    
                    if 'error' in report:
                        error_files.append(f"{file_name} (layer: {layer_name})")
                        has_error = True
                
                if not has_error:
                    processed_files.append(file_name)
                
            elif file_name.lower().endswith('.csv'):
                print(f"Processing as CSV file...")
                report = analyze_csv_file(file_path)
                output_path = os.path.join(output_dir, f"{file_name}_analysis.json")
                save_report_as_json(report, output_path)
                print(f"Report saved: {output_path}")
                
                if 'error' in report:
                    error_files.append(file_name)
                else:
                    processed_files.append(file_name)
                    
            elif file_name.lower().endswith(('.tif', '.tiff')):
                print(f"Processing as TIFF file...")
                report = analyze_tiff_file(file_path)
                output_path = os.path.join(output_dir, f"{file_name}_analysis.json")
                save_report_as_json(report, output_path)
                print(f"Report saved: {output_path}")
                
                if 'error' in report:
                    error_files.append(file_name)
                else:
                    processed_files.append(file_name)
            else:
                print(f"Skipping unsupported file type: {file_name}")
                skipped_files.append(file_name)
                
        except Exception as e:
            import traceback
            print(f"Critical error processing {file_name}: {str(e)}")
            print(traceback.format_exc())
            error_files.append(file_name)
    
    # Process specific files first
    for file_path in specific_files:
        if os.path.exists(file_path):
            print(f"Processing specific file: {file_path}")
            process_file(file_path)
        else:
            alt_path = file_path.replace('\\', '/')  # Try alternative path format
            if os.path.exists(alt_path):
                print(f"Processing specific file (alt path): {alt_path}")
                process_file(alt_path)
            else:
                print(f"Warning: Specific file not found: {file_path}")
    
    # Process files in the data directory
    for dir_to_scan in [data_dir, raw_data_dir]:
        if os.path.exists(dir_to_scan):
            print(f"\nScanning directory: {dir_to_scan}")
            for file_name in os.listdir(dir_to_scan):
                file_path = os.path.join(dir_to_scan, file_name)
                
                # Skip directories
                if os.path.isdir(file_path):
                    print(f"Skipping directory: {file_name}")
                    continue
                
                # Skip files that were already processed
                if os.path.basename(file_path) in processed_files:
                    print(f"Skipping already processed file: {file_name}")
                    continue
                
                process_file(file_path)
    
    # Check for absolute path (for Windows compatibility)
    setores_path = r"F:\TESE_MESTRADO\geoprocessing\data\raw\sorocaba_setores_censitarios.gpkg"
    if os.path.exists(setores_path) and os.path.basename(setores_path) not in processed_files:
        print(f"\nProcessing sorocaba_setores_censitarios.gpkg from absolute path...")
        process_file(setores_path)
    
    # Print summary
    print("\n=== Analysis Complete ===")
    print(f"Total files processed successfully: {len(processed_files)}")
    print(f"Total files skipped (unsupported): {len(skipped_files)}")
    print(f"Total files with errors: {len(error_files)}")
    
    if error_files:
        print("\nFiles with errors:")
        for file in error_files:
            print(f"  - {file}")
            
    # Write summary to file
    summary = {
        'processed_files': processed_files,
        'skipped_files': skipped_files,
        'error_files': error_files,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(os.path.join(output_dir, 'analysis_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main() 