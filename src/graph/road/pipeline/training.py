# -*- coding: utf-8 -*-
"""
Training and Evaluation Functions

This module contains functions for training and evaluating GNN models
on road network data.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

def train(model, optimizer, data, epochs=200, patience=20):
    """
    Train the GNN model.
    
    Args:
        model: GNN model to train
        optimizer: Optimizer for parameter updates
        data: PyTorch Geometric Data object
        epochs: Maximum number of training epochs
        patience: Early stopping patience
        
    Returns:
        Trained model, training history
    """
    # Placeholder for the train function from the original code
    model.train()
    history = {'loss': [], 'val_loss': [], 'val_acc': []}
    
    # Early stopping setup
    best_val_loss = float('inf')
    no_improve = 0
    
    for epoch in range(epochs):
        # Training step
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = torch.nn.functional.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item()
            pred = out.argmax(dim=1)
            val_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum().item() / data.val_mask.sum().item()
        
        # Record history
        history['loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
                
        model.train()
    
    return model, history

def evaluate(model, data, mask):
    """
    Evaluate the model on a specific data mask.
    
    Args:
        model: Trained GNN model
        data: PyTorch Geometric Data object
        mask: Mask to select nodes for evaluation
        
    Returns:
        Evaluation metrics
    """
    # Placeholder for the evaluate function from the original code
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        loss = torch.nn.functional.cross_entropy(out[mask], data.y[mask]).item()
        pred = out.argmax(dim=1)
        acc = (pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
        
        # Generate more detailed metrics
        y_true = data.y[mask].cpu().numpy()
        y_pred = pred[mask].cpu().numpy()
        
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
    return {
        'loss': loss,
        'accuracy': acc,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'predictions': pred[mask].cpu().numpy(),
        'true_labels': data.y[mask].cpu().numpy()
    }

def save_results(results, model, history, output_dir='results', prefix=''):
    """
    Save training results, model, and evaluation metrics.
    
    Args:
        results: Evaluation results dictionary
        model: Trained model
        history: Training history
        output_dir: Directory to save results
        prefix: Prefix for output filenames
        
    Returns:
        Dictionary with paths to saved files
    """
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if prefix:
        prefix = f"{prefix}_"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, f"{prefix}model_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save training history
    history_path = os.path.join(output_dir, f"{prefix}training_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump({k: [float(val) for val in v] for k, v in history.items()}, f, indent=2)
    
    # Save evaluation results
    results_path = os.path.join(output_dir, f"{prefix}evaluation_results_{timestamp}.json")
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for k, v in results.items():
        if isinstance(v, np.ndarray):
            serializable_results[k] = v.tolist()
        elif isinstance(v, dict):
            serializable_results[k] = {kk: vv if not isinstance(vv, np.ndarray) else vv.tolist() 
                                      for kk, vv in v.items()}
        else:
            serializable_results[k] = v
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return {
        'model': model_path,
        'history': history_path,
        'results': results_path
    }

def plot_training_history(history, output_path=None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training metrics
        output_path: Path to save the plot, or None to display
        
    Returns:
        Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['loss'], label='Training Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot accuracy
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path)
    
    return fig 