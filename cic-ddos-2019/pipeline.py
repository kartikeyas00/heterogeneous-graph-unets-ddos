#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser(description="Run complete ML pipeline for CIC-DDoS-2019 detection")
    p.add_argument("--base-dir", type=str, required=True,
                   help="Base directory for the project")
    p.add_argument("--input-features", type=str, required=True,
                   help="Path to input features CSV file")
    p.add_argument("--output-base", type=str, required=True,
                   help="Base output directory for all results")
    p.add_argument("--n-trials", type=int, default=200,
                   help="Number of Optuna trials for hyperparameter tuning")
    p.add_argument("--epochs", type=int, default=1000,
                   help="Number of training epochs")
    p.add_argument("--k-folds", type=int, default=10,
                   help="Number of K-folds")
    p.add_argument("--patience", type=int, default=50,
                   help="Early stopping patience")
    p.add_argument("--skip-preprocessing", action="store_true",
                   help="Skip preprocessing steps")
    p.add_argument("--skip-hyperopt", action="store_true",
                   help="Skip hyperparameter optimization")
    p.add_argument("--only-model", choices=["gnn_rnids", "gnn_egraph_sage", "gnn_hetero_unet"],
                   help="Run pipeline for only specified model")
    return p.parse_args()


def run_command(cmd, cwd=None, description=""):
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {cwd or os.getcwd()}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        print("SUCCESS")
        if result.stdout:
            print("STDOUT:", result.stdout[-500:])
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED")
        print("STDERR:", e.stderr)
        if e.stdout:
            print("STDOUT:", e.stdout)
        return False


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def fix_hyperparams_compatibility(hyperparams_path, model_name):
    """Fix parameter naming inconsistencies between tuning and training."""
    if not os.path.exists(hyperparams_path):
        return
    
    with open(hyperparams_path, 'r') as f:
        params = json.load(f)
    
    modified = False
    
    if model_name == "gnn_rnids":
        if 'batch_size_train' in params:
            params['batch_size_training'] = params.pop('batch_size_train')
            modified = True
        if 'batch_size_test' in params:
            params['batch_size_validation'] = params.pop('batch_size_test')
            modified = True
    elif model_name == "gnn_hetero_unet":
        if 'batch_size_test' in params:
            params['batch_size_validation'] = params.pop('batch_size_test')
            modified = True
    
    if modified:
        with open(hyperparams_path, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"Fixed hyperparameter compatibility for {model_name}")


def main():
    args = parse_args()
    
    base_dir = Path(args.base_dir)
    output_base = Path(args.output_base)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    graphs_dir = output_base / "graphs"
    logs_dir = output_base / "logs" / timestamp
    hyperparams_dir = output_base / "hyperparams"
    
    ensure_dir(graphs_dir)
    ensure_dir(logs_dir)
    ensure_dir(hyperparams_dir)
    
    host_graphs_dir = graphs_dir / "host_connection_graph"
    network_graphs_dir = graphs_dir / "network_flow_graph"
    
    print(f"Starting ML Pipeline for CIC-DDoS-2019")
    print(f"Base directory: {base_dir}")
    print(f"Input features: {args.input_features}")
    print(f"Output base: {output_base}")
    print(f"Timestamp: {timestamp}")
    
    success_count = 0
    total_steps = 0
    
    if not args.skip_preprocessing:
        total_steps += 1
        print(f"\nStep 1: Preprocessing Host Connection Graph")
        cmd = [
            sys.executable, "main.py",
            "--input-path", args.input_features,
            "--output-dir", str(host_graphs_dir),
            "--n-splits", str(args.k_folds)
        ]
        if run_command(cmd, cwd=base_dir / "preprocess" / "host_connection_graph", 
                      description="Host Connection Graph Preprocessing"):
            success_count += 1
    
    if not args.skip_preprocessing:
        total_steps += 1
        print(f"\nStep 2: Preprocessing Network Flow Graph")
        cmd = [
            sys.executable, "main.py",
            "--input-path", args.input_features,
            "--output-dir", str(network_graphs_dir),
            "--n-splits", str(args.k_folds)
        ]
        if run_command(cmd, cwd=base_dir / "preprocess" / "network_flow_graph",
                      description="Network Flow Graph Preprocessing"):
            success_count += 1
    
    models_to_run = []
    if args.only_model:
        models_to_run = [args.only_model]
    else:
        models_to_run = ["gnn_rnids", "gnn_egraph_sage", "gnn_hetero_unet"]
    
    for model_name in models_to_run:
        if model_name in ["gnn_rnids", "gnn_hetero_unet"]:
            graph_dir = host_graphs_dir
            graph_path = host_graphs_dir / "graph_fold_1.pickle"
        else:
            graph_dir = network_graphs_dir
            graph_path = network_graphs_dir / "graph_fold_1.pickle"
        
        model_logs_dir = logs_dir / model_name
        model_hyperparams_dir = hyperparams_dir / model_name
        ensure_dir(model_logs_dir)
        ensure_dir(model_hyperparams_dir)
        
        if not args.skip_hyperopt:
            total_steps += 1
            print(f"\nHyperparameter Tuning for {model_name.upper()}")
            cmd = [
                sys.executable, "hyperparams_tuning.py",
                "--optuna-dir", str(model_hyperparams_dir),
                "--graph-path", str(graph_path),
                "--n-trials", str(args.n_trials),
                "--study-name", f"{model_name}_study"
            ]
            if run_command(cmd, cwd=base_dir / "models" / model_name,
                          description=f"Hyperparameter tuning for {model_name}"):
                success_count += 1
                
                hyperparams_path = model_hyperparams_dir / "best_hyperparams.json"
                fix_hyperparams_compatibility(hyperparams_path, model_name)
        
        total_steps += 1
        print(f"\nTraining {model_name.upper()}")
        
        hyperparams_path = model_hyperparams_dir / "best_hyperparams.json"
        if not hyperparams_path.exists() and not args.skip_hyperopt:
            print(f"Hyperparameters not found for {model_name}, skipping training")
            continue
        elif not hyperparams_path.exists():
            if model_name == "gnn_rnids":
                default_params = {
                    "in_feats_flow": 77,
                    "in_feats_host": 77,
                    "hidden_size": 128,
                    "num_classes": 1,
                    "num_iterations": 4,
                    "learning_rate": 0.001,
                    "flow_nodes_batch_size": 300,
                    "batch_size_training": 16,
                    "batch_size_validation": 64
                }
            elif model_name == "gnn_egraph_sage":
                default_params = {
                    "node_in_dim": 128,
                    "edge_in_dim": 77,
                    "hidden_dim": 128,
                    "num_classes": 1,
                    "num_layers": 6,
                    "learning_rate": 0.001,
                    "edge_batch_size": 300,
                    "batch_size_train": 16,
                    "batch_size_test": 64
                }
            else:
                default_params = {
                    "hidden_size": 128,
                    "learning_rate": 0.001,
                    "flow_batch_size": 300,
                    "batch_size_train": 16,
                    "batch_size_validation": 64,
                    "max_connected_flows": 30,
                    "depth": 3,
                    "pool_ratio": 0.5
                }
            
            with open(hyperparams_path, 'w') as f:
                json.dump(default_params, f, indent=4)
            print(f"Created default hyperparameters for {model_name}")
        
        cmd = [
            sys.executable, "train.py",
            "--graph-dir", str(graph_dir),
            "--hyperparams-path", str(hyperparams_path),
            "--log-dir", str(model_logs_dir),
            "--epochs", str(args.epochs),
            "--k-folds", str(args.k_folds),
            "--patience", str(args.patience)
        ]
        
        if run_command(cmd, cwd=base_dir / "models" / model_name,
                      description=f"Training {model_name}"):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Pipeline Completed!")
    print(f"Success: {success_count}/{total_steps} steps")
    print(f"Results saved to: {output_base}")
    print(f"Logs: {logs_dir}")
    print(f"Hyperparameters: {hyperparams_dir}")
    
    if success_count == total_steps:
        print("All steps completed successfully!")
        return 0
    else:
        print(f"{total_steps - success_count} steps failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
