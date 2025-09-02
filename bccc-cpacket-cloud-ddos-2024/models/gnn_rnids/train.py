import pickle
import argparse
import torch
import torch.nn as nn
from model import GNN_NIDS
import time
import os
import pandas as pd
import logging
import datetime
import json
import torchmetrics
from dataset import create_dataloader

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-dir", type=str, required=True,
                   help="Directory with graph_fold_*.pickle")
    p.add_argument("--hyperparams-path", type=str, required=True,
                   help="Path to hyperparameters JSON file")
    p.add_argument("--log-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--k-folds", type=int, default=10)
    p.add_argument("--patience", type=int, default=100)
    return p.parse_args()

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = os.path.join(log_dir, f"training_{dt}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(fh),
            logging.StreamHandler()
        ]
    )
    return fh

def create_metrics():
    return {
        'train_accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=3),
        'train_precision': torchmetrics.Precision(task='multiclass', num_classes=3, average='macro'),
        'train_recall': torchmetrics.Recall(task='multiclass', num_classes=3, average='macro'),
        'train_f1': torchmetrics.F1Score(task='multiclass', num_classes=3, average='macro'),
        'val_accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=3),
        'val_precision': torchmetrics.Precision(task='multiclass', num_classes=3, average='macro'),
        'val_recall': torchmetrics.Recall(task='multiclass', num_classes=3, average='macro'),
        'val_f1': torchmetrics.F1Score(task='multiclass', num_classes=3, average='macro')
    }

def reset_metrics(metrics):
    for metric in metrics.values():
        metric.reset()

def train_one_epoch(model, dataloader, optimizer, criterion, device, metrics):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (flow_node_batches, subgraph) in enumerate(dataloader):
        subgraph = subgraph.to(device)
        
        for flow_nodes in flow_node_batches:
            flow_nodes = flow_nodes.to(device)
            optimizer.zero_grad()
            
            host_features = subgraph.nodes['host'].data['h']
            flow_features = subgraph.nodes['flow'].data['h']
            
            logits = model(subgraph, host_features, flow_features)
            predictions = logits[flow_nodes]
            labels = subgraph.nodes['flow'].data['label'][flow_nodes]
            
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update metrics
            pred_classes = torch.argmax(predictions, dim=1)
            metrics['train_accuracy'].update(pred_classes, labels)
            metrics['train_precision'].update(pred_classes, labels)
            metrics['train_recall'].update(pred_classes, labels)
            metrics['train_f1'].update(pred_classes, labels)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Compute metrics
    train_metrics = {
        'loss': avg_loss,
        'accuracy': metrics['train_accuracy'].compute().item(),
        'precision': metrics['train_precision'].compute().item(),
        'recall': metrics['train_recall'].compute().item(),
        'f1': metrics['train_f1'].compute().item()
    }
    
    return train_metrics

def evaluate_model(model, dataloader, criterion, device, metrics):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (flow_node_batches, subgraph) in enumerate(dataloader):
            subgraph = subgraph.to(device)
            
            for flow_nodes in flow_node_batches:
                flow_nodes = flow_nodes.to(device)
                
                host_features = subgraph.nodes['host'].data['h']
                flow_features = subgraph.nodes['flow'].data['h']
                
                logits = model(subgraph, host_features, flow_features)
                predictions = logits[flow_nodes]
                labels = subgraph.nodes['flow'].data['label'][flow_nodes]
                
                loss = criterion(predictions, labels)
                total_loss += loss.item()
                num_batches += 1
                
                # Update metrics
                pred_classes = torch.argmax(predictions, dim=1)
                metrics['val_accuracy'].update(pred_classes, labels)
                metrics['val_precision'].update(pred_classes, labels)
                metrics['val_recall'].update(pred_classes, labels)
                metrics['val_f1'].update(pred_classes, labels)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    # Compute metrics
    val_metrics = {
        'loss': avg_loss,
        'accuracy': metrics['val_accuracy'].compute().item(),
        'precision': metrics['val_precision'].compute().item(),
        'recall': metrics['val_recall'].compute().item(),
        'f1': metrics['val_f1'].compute().item()
    }
    
    return val_metrics

def train_fold(fold, train_graph, val_graph, hyperparams, device, args):
    logging.info(f"Training fold {fold + 1}")
    
    # Create model
    model = GNN_NIDS(
        in_feats_host=hyperparams['in_feats_host'],
        in_feats_flow=hyperparams['in_feats_flow'],
        hidden_size=hyperparams['hidden_size'],
        num_classes=hyperparams['num_classes'],
        num_iterations=hyperparams['num_iterations']
    ).to(device)
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_graph, 
        batch_size=hyperparams['batch_size_train'],
        shuffle=True
    )
    val_dataloader = create_dataloader(
        val_graph,
        batch_size=hyperparams['batch_size_test'],
        shuffle=False
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    # Create metrics
    metrics = create_metrics()
    for metric in metrics.values():
        metric.to(device)
    
    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    training_history = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Reset metrics
        reset_metrics(metrics)
        
        # Train
        train_metrics = train_one_epoch(model, train_dataloader, optimizer, criterion, device, metrics)
        
        # Reset validation metrics
        for key in metrics:
            if key.startswith('val_'):
                metrics[key].reset()
        
        # Validate
        val_metrics = evaluate_model(model, val_dataloader, criterion, device, metrics)
        
        epoch_time = time.time() - start_time
        
        # Log progress
        if epoch % 10 == 0:
            logging.info(f"Fold {fold + 1}, Epoch {epoch + 1}/{args.epochs}")
            logging.info(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
            logging.info(f"Val - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
            logging.info(f"Time: {epoch_time:.2f}s")
        
        # Save history
        history_entry = {
            'fold': fold + 1,
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'train_precision': train_metrics['precision'],
            'train_recall': train_metrics['recall'],
            'train_f1': train_metrics['f1'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'epoch_time': epoch_time
        }
        training_history.append(history_entry)
        
        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            # Save best model
            model_path = os.path.join(args.log_dir, f'best_model_fold_{fold + 1}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'hyperparams': hyperparams,
                'val_f1': best_val_f1,
                'epoch': epoch + 1
            }, model_path)
        else:
            patience_counter += 1
            
        if patience_counter >= args.patience:
            logging.info(f"Early stopping at epoch {epoch + 1}")
            break
    
    logging.info(f"Fold {fold + 1} completed. Best val F1: {best_val_f1:.4f}")
    
    return training_history, best_val_f1

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Setup logging
    log_file = setup_logging(args.log_dir)
    logging.info(f"Starting training with device: {device}")
    
    # Load hyperparameters
    with open(args.hyperparams_path, 'r') as f:
        hyperparams = json.load(f)
    
    logging.info(f"Loaded hyperparameters: {hyperparams}")
    
    # K-fold cross validation
    all_histories = []
    fold_scores = []
    
    for fold in range(args.k_folds):
        # Load fold data
        train_graph_path = os.path.join(args.graph_dir, f"graph_fold_{fold}_train.pickle")
        val_graph_path = os.path.join(args.graph_dir, f"graph_fold_{fold}_val.pickle")
        
        if not os.path.exists(train_graph_path) or not os.path.exists(val_graph_path):
            logging.warning(f"Fold {fold + 1} data not found, skipping...")
            continue
        
        with open(train_graph_path, 'rb') as f:
            train_graph = pickle.load(f)
        
        with open(val_graph_path, 'rb') as f:
            val_graph = pickle.load(f)
        
        # Train fold
        fold_history, best_f1 = train_fold(fold, train_graph, val_graph, hyperparams, device, args)
        
        all_histories.extend(fold_history)
        fold_scores.append(best_f1)
    
    # Save complete training history
    history_df = pd.DataFrame(all_histories)
    history_path = os.path.join(args.log_dir, 'training_history.csv')
    history_df.to_csv(history_path, index=False)
    
    # Calculate and log final results
    if fold_scores:
        mean_f1 = sum(fold_scores) / len(fold_scores)
        std_f1 = (sum((x - mean_f1) ** 2 for x in fold_scores) / len(fold_scores)) ** 0.5
        
        logging.info(f"Training completed!")
        logging.info(f"Mean F1 across {len(fold_scores)} folds: {mean_f1:.4f} ± {std_f1:.4f}")
        logging.info(f"Individual fold scores: {fold_scores}")
        logging.info(f"Training history saved to: {history_path}")
        logging.info(f"Logs saved to: {log_file}")
        
        # Save final summary
        summary = {
            'mean_f1': mean_f1,
            'std_f1': std_f1,
            'fold_scores': fold_scores,
            'hyperparams': hyperparams,
            'args': vars(args)
        }
        
        summary_path = os.path.join(args.log_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Training summary saved to: {summary_path}")
    else:
        logging.error("No folds were successfully processed!")

if __name__ == "__main__":
    main()
