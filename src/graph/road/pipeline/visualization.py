# -*- coding: utf-8 -*-
"""
Visualization Functions for Road Network Analysis

This module contains functions for visualizing road networks,
graph structures, and analysis results.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import networkx as nx
import geopandas as gpd
import pandas as pd
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import folium
import contextily as cx

def plot_road_network(gdf, node_gdf=None, fig_size=(15, 15), title='Road Network', save_path=None):
    """
    Plot the road network from GeoDataFrames.
    
    Args:
        gdf: GeoDataFrame with road geometries
        node_gdf: Optional GeoDataFrame with node geometries
        fig_size: Size of the figure
        title: Title for the plot
        save_path: Path to save the figure, or None to display
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot roads
    gdf.plot(ax=ax, linewidth=1, alpha=0.7)
    
    # Plot nodes if provided
    if node_gdf is not None:
        node_gdf.plot(ax=ax, markersize=5, color='red')
    
    # Add basemap
    try:
        cx.add_basemap(ax, crs=gdf.crs)
    except Exception as e:
        print(f"Could not add basemap: {e}")
    
    ax.set_title(title, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_graph(G, pos=None, node_color='skyblue', node_size=100, edge_color='gray', 
               fig_size=(10, 10), title='Graph Visualization', save_path=None):
    """
    Plot a NetworkX graph.
    
    Args:
        G: NetworkX graph
        pos: Dictionary with node positions
        node_color: Color for nodes
        node_size: Size for nodes
        edge_color: Color for edges
        fig_size: Size of the figure
        title: Title for the plot
        save_path: Path to save the figure, or None to display
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    if pos is None:
        pos = nx.spring_layout(G)
    
    nx.draw_networkx(
        G, 
        pos=pos,
        with_labels=False,
        node_color=node_color,
        node_size=node_size,
        edge_color=edge_color,
        alpha=0.7,
        ax=ax
    )
    
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_node_classes(G, pos=None, node_class_attr='class', cmap='viridis', 
                      fig_size=(12, 12), title='Node Classes', save_path=None):
    """
    Visualize node classes in a graph.
    
    Args:
        G: NetworkX graph
        pos: Dictionary with node positions
        node_class_attr: Attribute containing node classes
        cmap: Colormap to use
        fig_size: Size of the figure
        title: Title for the plot
        save_path: Path to save the figure, or None to display
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    if pos is None:
        pos = nx.spring_layout(G)
    
    # Extract classes
    classes = [G.nodes[n].get(node_class_attr, 0) for n in G.nodes()]
    
    # Draw the graph
    nodes = nx.draw_networkx_nodes(
        G,
        pos=pos,
        node_size=100,
        node_color=classes,
        cmap=plt.get_cmap(cmap),
        alpha=0.8,
        ax=ax
    )
    
    edges = nx.draw_networkx_edges(
        G,
        pos=pos,
        edge_color='gray',
        alpha=0.5,
        ax=ax
    )
    
    # Add a colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(nodes, cax=cax)
    
    ax.set_title(title, fontsize=16)
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_confusion_matrix(conf_matrix, class_names=None, fig_size=(10, 8), 
                         title='Confusion Matrix', save_path=None):
    """
    Visualize a confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix as array
        class_names: List of class names
        fig_size: Size of the figure
        title: Title for the plot
        save_path: Path to save the figure, or None to display
        
    Returns:
        Figure and axes objects
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Normalize the confusion matrix
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(
        conf_matrix_norm, 
        annot=True, 
        fmt='.2f', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def create_interactive_map(gdf, node_gdf=None, class_column=None, 
                          tooltip_columns=None, output_path=None):
    """
    Create an interactive Folium map.
    
    Args:
        gdf: GeoDataFrame with geometries
        node_gdf: Optional GeoDataFrame with node geometries
        class_column: Column to use for coloring
        tooltip_columns: Columns to show in tooltips
        output_path: Path to save the HTML map
        
    Returns:
        Folium map object
    """
    # Create a copy to avoid modifying the original
    gdf_map = gdf.copy()
    
    # Ensure we're working with lat/lon
    if gdf_map.crs and not gdf_map.crs.to_string() == 'EPSG:4326':
        gdf_map = gdf_map.to_crs(epsg=4326)
    
    # Get center of the map
    center = [gdf_map.unary_union.centroid.y, gdf_map.unary_union.centroid.x]
    
    # Create map
    m = folium.Map(location=center, zoom_start=13, tiles='cartodbpositron')
    
    # Function to determine color based on class
    def get_color(feature):
        if class_column and class_column in gdf_map.columns:
            unique_classes = gdf_map[class_column].unique()
            colors = plt.cm.tab10.colors
            color_dict = {cls: '#%02x%02x%02x' % tuple([int(255*c) for c in colors[i % len(colors)]])
                          for i, cls in enumerate(unique_classes)}
            return color_dict.get(feature[class_column], '#3388ff')
        return '#3388ff'
    
    # Add roads
    if 'LineString' in gdf_map.geometry.type.unique() or 'MultiLineString' in gdf_map.geometry.type.unique():
        style_function = lambda x: {
            'color': get_color(x['properties']),
            'weight': 3,
            'opacity': 0.7
        }
        
        tooltip_function = lambda x: folium.Tooltip(
            "<br>".join([f"{col}: {x['properties'][col]}" for col in tooltip_columns])
        ) if tooltip_columns else None
        
        folium.GeoJson(
            gdf_map.__geo_interface__,
            style_function=style_function,
            tooltip=tooltip_function
        ).add_to(m)
    
    # Add nodes if provided
    if node_gdf is not None:
        # Ensure we're working with lat/lon
        if node_gdf.crs and not node_gdf.crs.to_string() == 'EPSG:4326':
            node_gdf = node_gdf.to_crs(epsg=4326)
            
        for idx, row in node_gdf.iterrows():
            tooltip_text = "<br>".join([f"{col}: {row[col]}" for col in tooltip_columns]) if tooltip_columns else ""
            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.7,
                tooltip=tooltip_text
            ).add_to(m)
    
    # Save map if output path is provided
    if output_path:
        m.save(output_path)
    
    return m 