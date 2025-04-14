# -*- coding: utf-8 -*-
"""
Reporting and Quality Assessment

This module contains functions for generating reports and assessing
the quality of road network analysis results.
"""

import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import networkx as nx

def generate_quality_report(G, predictions, true_labels, idx_to_class=None, output_dir='reports'):
    """
    Generate a quality report for node classification.
    
    Args:
        G: NetworkX graph of road network
        predictions: Node class predictions
        true_labels: True node classes
        idx_to_class: Dictionary mapping class indices to names
        output_dir: Directory to save reports
        
    Returns:
        Dictionary with report paths
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # If idx_to_class not provided, create default mapping
    if idx_to_class is None:
        unique_classes = sorted(list(set(true_labels)))
        idx_to_class = {i: f"Class_{i}" for i in unique_classes}
    
    # Generate classification report
    class_names = [idx_to_class.get(i, f"Class_{i}") for i in sorted(idx_to_class.keys())]
    class_report = classification_report(true_labels, predictions, output_dict=True)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Calculate accuracy
    acc = accuracy_score(true_labels, predictions)
    
    # Generate network statistics
    network_stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'graph_density': nx.density(G),
        'avg_clustering': nx.average_clustering(G),
        'connected_components': nx.number_connected_components(G)
    }
    
    # Calculate edge correlation matrix (correlation between connected nodes)
    edge_correlation = {}
    for u, v in G.edges():
        if 'class' in G.nodes[u] and 'class' in G.nodes[v]:
            u_class = G.nodes[u]['class']
            v_class = G.nodes[v]['class']
            key = f"{idx_to_class.get(u_class, u_class)}-{idx_to_class.get(v_class, v_class)}"
            edge_correlation[key] = edge_correlation.get(key, 0) + 1
    
    # Save node coordinates for visualization
    node_coords = {}
    for node, data in G.nodes(data=True):
        if 'x' in data and 'y' in data:
            node_coords[node] = {
                'x': data['x'],
                'y': data['y'],
                'class': idx_to_class.get(data.get('class', -1), "Unknown")
            }
    
    # Save reports
    report_files = {}
    
    # Classification report
    class_report_path = os.path.join(output_dir, f"classification_report_{timestamp}.json")
    with open(class_report_path, 'w') as f:
        json.dump(class_report, f, indent=2)
    report_files['classification_report'] = class_report_path
    
    # Edge correlation
    edge_corr_path = os.path.join(output_dir, f"edge_correlation_matrix_{timestamp}.json")
    with open(edge_corr_path, 'w') as f:
        json.dump(edge_correlation, f, indent=2)
    report_files['edge_correlation'] = edge_corr_path
    
    # Node coordinates
    node_coords_path = os.path.join(output_dir, f"node_coords_{timestamp}.json")
    with open(node_coords_path, 'w') as f:
        json.dump(node_coords, f, indent=2)
    report_files['node_coords'] = node_coords_path
    
    # Training results
    training_results = {
        'accuracy': acc,
        'num_samples': len(true_labels),
        'num_classes': len(class_names),
        'confusion_matrix': conf_matrix.tolist(),
        'class_mapping': idx_to_class
    }
    
    training_results_path = os.path.join(output_dir, f"training_results_{timestamp}.json")
    with open(training_results_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    report_files['training_results'] = training_results_path
    
    # Create a comprehensive quality report
    quality_report = {
        'timestamp': timestamp,
        'accuracy': acc,
        'classification_metrics': class_report,
        'network_statistics': network_stats,
        'class_distribution': {
            idx_to_class.get(i, f"Class_{i}"): int((np.array(true_labels) == i).sum()) 
            for i in sorted(set(true_labels))
        }
    }
    
    # Save as JSON
    quality_json_path = os.path.join(output_dir, f"quality_report_{timestamp}.json")
    with open(quality_json_path, 'w') as f:
        json.dump(quality_report, f, indent=2)
    report_files['quality_json'] = quality_json_path
    
    # Save as formatted text
    quality_txt_path = os.path.join(output_dir, f"quality_report_{timestamp}.txt")
    with open(quality_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"ROAD NETWORK CLASSIFICATION QUALITY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("CLASSIFICATION ACCURACY\n")
        f.write("-"*80 + "\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        
        f.write("CLASS DISTRIBUTION\n")
        f.write("-"*80 + "\n")
        for cls, count in quality_report['class_distribution'].items():
            f.write(f"{cls}: {count} nodes\n")
        f.write("\n")
        
        f.write("NETWORK STATISTICS\n")
        f.write("-"*80 + "\n")
        for stat, value in network_stats.items():
            f.write(f"{stat.replace('_', ' ').title()}: {value}\n")
        f.write("\n")
        
        f.write("CLASSIFICATION REPORT\n")
        f.write("-"*80 + "\n")
        f.write(classification_report(true_labels, predictions, target_names=class_names))
        f.write("\n")
    
    report_files['quality_txt'] = quality_txt_path
    
    # Generate project summary
    summary = {
        'project_name': 'Road Network Analysis',
        'timestamp': timestamp,
        'num_nodes': network_stats['num_nodes'],
        'num_edges': network_stats['num_edges'],
        'accuracy': acc,
        'report_files': {k: os.path.basename(v) for k, v in report_files.items()}
    }
    
    summary_path = os.path.join(output_dir, f"project_summary_{timestamp}.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    report_files['summary'] = summary_path
    
    # Generate markdown report
    md_report_path = os.path.join(output_dir, f"final_report_{timestamp}.md")
    with open(md_report_path, 'w') as f:
        f.write("# Road Network Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Network Overview\n\n")
        f.write(f"- **Nodes:** {network_stats['num_nodes']}\n")
        f.write(f"- **Edges:** {network_stats['num_edges']}\n")
        f.write(f"- **Density:** {network_stats['graph_density']:.6f}\n")
        f.write(f"- **Avg. Clustering:** {network_stats['avg_clustering']:.6f}\n")
        f.write(f"- **Connected Components:** {network_stats['connected_components']}\n\n")
        
        f.write("## Classification Results\n\n")
        f.write(f"- **Overall Accuracy:** {acc:.4f}\n\n")
        
        f.write("### Class Distribution\n\n")
        f.write("| Class | Count | Precision | Recall | F1-Score |\n")
        f.write("|-------|-------|-----------|--------|----------|\n")
        for cls_idx, cls_name in idx_to_class.items():
            cls_str = str(cls_idx)
            if cls_str in class_report:
                data = class_report[cls_str]
                count = int((np.array(true_labels) == cls_idx).sum())
                f.write(f"| {cls_name} | {count} | {data['precision']:.4f} | {data['recall']:.4f} | {data['f1-score']:.4f} |\n")
        f.write("\n")
        
        f.write("## Summary\n\n")
        f.write("This report provides an overview of road network analysis results. ")
        f.write("The model classified road network nodes with various accuracy metrics as shown above. ")
        f.write("Additional details can be found in the JSON reports.\n\n")
        
        f.write("### Generated Files\n\n")
        for report_type, path in report_files.items():
            f.write(f"- **{report_type.replace('_', ' ').title()}:** {os.path.basename(path)}\n")
    
    report_files['markdown'] = md_report_path
    
    return report_files

def load_reports(report_dir, timestamp=None):
    """
    Load reports from the specified directory.
    
    Args:
        report_dir: Directory containing reports
        timestamp: Specific timestamp to load, or None for latest
        
    Returns:
        Dictionary with loaded report data
    """
    # List all json files in the directory
    json_files = [f for f in os.listdir(report_dir) if f.endswith('.json')]
    
    if not json_files:
        raise FileNotFoundError(f"No JSON report files found in {report_dir}")
    
    # If timestamp not specified, find the latest one
    if timestamp is None:
        timestamps = []
        for f in json_files:
            parts = f.split('_')
            if len(parts) >= 2:
                try:
                    ts = parts[-1].replace('.json', '')
                    if len(ts) == 15:  # YYYYmmdd_HHMMSS format
                        timestamps.append(ts)
                except:
                    continue
        
        if not timestamps:
            raise ValueError("Could not identify timestamps in filenames")
        
        timestamp = max(timestamps)
    
    # Load reports with the specified timestamp
    reports = {}
    for f in json_files:
        if timestamp in f:
            key = '_'.join(f.split('_')[:-1])  # Remove timestamp
            file_path = os.path.join(report_dir, f)
            with open(file_path, 'r') as file:
                reports[key] = json.load(file)
    
    return reports 