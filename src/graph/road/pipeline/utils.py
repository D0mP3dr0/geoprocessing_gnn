# -*- coding: utf-8 -*-
"""
Utility Functions

This module contains various utility functions used across the road network
analysis pipeline.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import torch
from datetime import datetime

def setup_logging(log_file=None, level="INFO"):
    """
    Set up logging configuration.
    
    Args:
        log_file: Path to log file, or None for console only
        level: Logging level
        
    Returns:
        Logger object
    """
    # Create logger
    logger = logging.getLogger('road_network_analysis')
    
    # Set level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logger.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_json(data, file_path):
    """
    Save data as JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save JSON file
        
    Returns:
        Path to saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Convert numpy arrays and other non-serializable objects to lists
    def serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    # Save JSON file
    with open(file_path, 'w') as f:
        json.dump(data, f, default=serialize, indent=2)
    
    return file_path

def load_json(file_path):
    """
    Load data from JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data

def calculate_network_metrics(G):
    """
    Calculate various network metrics for a graph.
    
    Args:
        G: NetworkX graph
        
    Returns:
        Dictionary of metrics
    """
    import networkx as nx
    
    metrics = {}
    
    # Basic metrics
    metrics['num_nodes'] = G.number_of_nodes()
    metrics['num_edges'] = G.number_of_edges()
    metrics['density'] = nx.density(G)
    
    # Connected components
    metrics['num_components'] = nx.number_connected_components(G)
    components = list(nx.connected_components(G))
    metrics['largest_component_size'] = len(max(components, key=len))
    
    # Only calculate these metrics for connected graphs
    if nx.is_connected(G):
        # Path-based metrics
        metrics['diameter'] = nx.diameter(G)
        metrics['radius'] = nx.radius(G)
        metrics['avg_shortest_path'] = nx.average_shortest_path_length(G)
        
        # Centrality metrics (for a subset of nodes to improve performance)
        if G.number_of_nodes() > 1000:
            # For large graphs, sample nodes
            sample_nodes = np.random.choice(list(G.nodes()), size=1000, replace=False)
            sample_g = G.subgraph(sample_nodes)
            betweenness = nx.betweenness_centrality(sample_g)
            closeness = nx.closeness_centrality(sample_g)
        else:
            betweenness = nx.betweenness_centrality(G)
            closeness = nx.closeness_centrality(G)
        
        metrics['avg_betweenness'] = np.mean(list(betweenness.values()))
        metrics['avg_closeness'] = np.mean(list(closeness.values()))
    
    # Clustering metrics
    metrics['avg_clustering'] = nx.average_clustering(G)
    
    return metrics

def create_timestamp_directory(base_dir, prefix="run"):
    """
    Create a directory with timestamp for storing results.
    
    Args:
        base_dir: Base directory
        prefix: Prefix for directory name
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dir_name = f"{prefix}_{timestamp}"
    dir_path = os.path.join(base_dir, dir_name)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path

def check_gpu():
    """
    Check GPU availability and return device information.
    
    Returns:
        Device to use and GPU info
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_info = {
            'name': torch.cuda.get_device_name(0),
            'count': torch.cuda.device_count(),
            'memory_allocated': torch.cuda.memory_allocated(0) / 1024**2,
            'memory_reserved': torch.cuda.memory_reserved(0) / 1024**2,
        }
        print(f"GPU available: {gpu_info['name']}")
    else:
        device = torch.device('cpu')
        gpu_info = None
        print("GPU not available, using CPU.")
    
    return device, gpu_info

def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        labels: Array of labels
        
    Returns:
        Array of class weights
    """
    # Count number of examples in each class
    class_counts = np.bincount(labels)
    
    # Calculate weights inversely proportional to class frequencies
    weights = 1.0 / class_counts
    
    # Normalize weights so they sum to the number of classes
    weights = weights * len(class_counts) / np.sum(weights)
    
    return weights

def save_results_summary(results_dict, output_path):
    """
    Save a summary of results to text and markdown files.
    
    Args:
        results_dict: Dictionary containing results
        output_path: Base path for output files (without extension)
        
    Returns:
        Dictionary with paths to saved files
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as text file
    txt_path = f"{output_path}.txt"
    with open(txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ROAD NETWORK ANALYSIS RESULTS SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for section, data in results_dict.items():
            f.write(f"{section.upper()}\n")
            f.write("-"*80 + "\n")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"{key}: {value}\n")
            else:
                f.write(f"{data}\n")
            
            f.write("\n")
    
    # Save as markdown file
    md_path = f"{output_path}.md"
    with open(md_path, 'w') as f:
        f.write("# Road Network Analysis Results Summary\n\n")
        f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        for section, data in results_dict.items():
            f.write(f"## {section.title()}\n\n")
            
            if isinstance(data, dict):
                for key, value in data.items():
                    f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
            else:
                f.write(f"{data}\n")
            
            f.write("\n")
    
    return {
        'text': txt_path,
        'markdown': md_path
    }

def convert_time_format(seconds):
    """
    Convert seconds to a human-readable time format.
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s" 