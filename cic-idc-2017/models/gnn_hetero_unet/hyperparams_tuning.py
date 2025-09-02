import os
import argparse
import json
import logging
import pickle
import datetime
import torch
import torch.nn as nn
import torchmetrics
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import time
from model import HeteroGraphUNet
from dataset import create_datalaoder

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--optuna-dir",        type=str, required=True,
                   help="Directory to store Optuna logs & DB")
    p.add_argument("--graph-path",        type=str, required=True,
                   help="Path to one fold’s graph .pickle")
    p.add_argument("--n-trials",          type=int, default=200)
    p.add_argument("--study-name",        type=str, default="hyperopt")
    return p.parse_args()

def load_graph(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

def create_train_dataloader(g, config):
    # Get flow nodes with train mask
    train_flow_nodes = torch.arange(g.number_of_nodes('flow'))[g.nodes['flow'].data['train_mask']]
    # Create batches
    train_batches = [train_flow_nodes[i:i + config['flow_batch_size']] for i in range(0, len(train_flow_nodes), config['flow_batch_size'])]
    # Filter out small batches
    train_batches = [batch for batch in train_batches if len(batch) == config['flow_batch_size']]
    return create_datalaoder(g, train_batches, config['batch_size_train'], True, max_connected_flows=config['max_connected_flows'])

def create_validation_dataloader(g, config):
    # Get flow nodes with validation mask
    val_flow_nodes = torch.arange(g.number_of_nodes('flow'))[g.nodes['flow'].data['val_mask']]
    # Create batches
    val_batches = [val_flow_nodes[i:i + config['flow_batch_size']] for i in range(0, len(val_flow_nodes), config['flow_batch_size'])]
    # Filter out small batches
    val_batches = [batch for batch in val_batches if len(batch) == config['flow_batch_size']]
    return create_datalaoder(g, val_batches, config['batch_size_validation'], False, max_connected_flows=config['max_connected_flows'])



# Cache for dataloaders to avoid recreating them
train_data_loaders_dict = {}
validation_data_loaders_dict = {}


def objective(trial, loaded_graph, in_feats_dict, out_feats_dict, rel_names, canonical_etypes):
    config = {
        'hidden_feats': trial.suggest_categorical('hidden_size', [64, 128, 256]),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-2),
        'flow_batch_size': trial.suggest_categorical('flow_batch_size', [100, 200, 300]),
        'batch_size_train': trial.suggest_categorical('batch_size_train', [16, 32, 64]),
        'batch_size_validation': trial.suggest_categorical('batch_size_validation', [16, 32, 64]),
        'max_connected_flows': trial.suggest_categorical('max_connected_flows', [10, 20, 30, 40, 50]),
        'depth': trial.suggest_categorical('depth', [2, 3, 4, 5]),
        'pool_ratio': trial.suggest_categorical('pool_ratio', [0.4, 0.5, 0.6, 0.7]),
        'epochs': 50,
        'early_stopping_patience': 20,
    }
    
    # Setup logger
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get or create dataloaders
    dataloader_key = (config['flow_batch_size'], config['batch_size_train'], config['max_connected_flows'])
    if dataloader_key in train_data_loaders_dict:
        training_dataloader = train_data_loaders_dict[dataloader_key]
    else:
        training_dataloader = create_train_dataloader(loaded_graph, config)
        train_data_loaders_dict[dataloader_key] = training_dataloader

    val_dataloader_key = (config['flow_batch_size'], config['batch_size_validation'], config['max_connected_flows'])
    if val_dataloader_key in validation_data_loaders_dict:
        validation_dataloader = validation_data_loaders_dict[val_dataloader_key]
    else:
        validation_dataloader = create_validation_dataloader(loaded_graph, config)
        validation_data_loaders_dict[val_dataloader_key] = validation_dataloader
    
    # Initialize model
    model = HeteroGraphUNet(
        in_feats_dict=in_feats_dict,
        hidden_feats=config['hidden_feats'],
        out_feats_dict=out_feats_dict,
        rel_names=rel_names,
        canonical_etypes=canonical_etypes,
        depth=config['depth'],
        pool_ratio=config['pool_ratio']
    ).to(device)
    
    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.BCEWithLogitsLoss()
    
    # Set up metrics
    metrics = {
        'accuracy': torchmetrics.Accuracy(task="binary", threshold=0.5),
        'precision': torchmetrics.Precision(task="binary", threshold=0.5),
        'recall': torchmetrics.Recall(task="binary", threshold=0.5),
        'f1': torchmetrics.F1Score(task="binary", threshold=0.5)
    }
    metrics = {k: v.to(device) for k, v in metrics.items()}
    
    best_f1 = 0
    epochs_no_improve = 0
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_flow_nodes, subgraph in training_dataloader:
            subgraph = subgraph.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(subgraph, subgraph.ndata['feat'])
            labels = subgraph.nodes['flow'].data['label'].float().to(device)
            
            # Calculate loss
            loss = criterion(logits.squeeze(), labels)
            train_loss += loss.item()
            num_batches += 1
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            for metric in metrics.values():
                metric.reset()
            
            val_loss = 0.0
            val_batches = 0
            
            for batch_flow_nodes, subgraph in validation_dataloader:
                subgraph = subgraph.to(device)
                
                # Forward pass
                logits = model(subgraph, subgraph.ndata['feat'])
                labels = subgraph.nodes['flow'].data['label'].float().to(device)
                
                # Calculate loss
                loss = criterion(logits.squeeze(), labels)
                val_loss += loss.item()
                val_batches += 1
                
                # Get predictions
                preds = torch.sigmoid(logits).round()
                
                # Update metrics
                for metric in metrics.values():
                    metric.update(preds.squeeze(), labels)
            
            # Compute metrics
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            results = {k: v.compute().item() for k, v in metrics.items()}
            
        
        # Report to Optuna
        trial.report(results['f1'], epoch)
        
        # Check for pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping check
        if results['f1'] > best_f1:
            best_f1 = results['f1']
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config['early_stopping_patience']:
                break
    
    return best_f1

def main():
    args = parse_args()
    os.makedirs(args.optuna_dir, exist_ok=True)

    # Load the graph once
    loaded_graph = load_graph(args.graph_path)

    # Extract feature dimensions
    in_feats_dict = {}
    for ntype in loaded_graph.ntypes:
        in_feats = loaded_graph.nodes[ntype].data['feat'].shape[1]
        in_feats_dict[ntype] = in_feats

    out_feats_dict = {
        'flow': 1  # Binary classification for 'flow' nodes
    }

    # Get relation names and canonical edge types
    rel_names = []
    for src, etype, dst in loaded_graph.canonical_etypes:
        if etype not in rel_names:
            rel_names.append(etype)
    canonical_etypes = loaded_graph.canonical_etypes

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        storage=f"sqlite:///{args.optuna_dir}/optuna_study.db",
        study_name=f"{args.study_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
        load_if_exists=True
    )
    
    study.optimize(lambda t: objective(
                       t, loaded_graph, in_feats_dict, out_feats_dict, rel_names, canonical_etypes
                   ), n_trials=args.n_trials)
    
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    best_params_path = os.path.join(args.optuna_dir, "best_hyperparams.json")
    with open(best_params_path, 'w') as f:
        json.dump(best_trial.params, f, indent=4)

if __name__ == "__main__":
    main()