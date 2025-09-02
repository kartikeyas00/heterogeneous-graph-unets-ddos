from utils import (
    balance_data, normalize_features,
    calculate_feature_stats, build_fold_graph_with_ip_source, 
    create_kfold_splits
)
from pathlib import Path
import pickle
import os
import pandas as pd
import argparse
import logging
import datetime

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess data and create host connection graphs")
    p.add_argument("--input-path", type=str, required=True,
                   help="Path to the input CSV file with features")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Directory to save the processed graph files")
    p.add_argument("--class-ratio", type=float, default=1.0,
                   help="Class ratio for balancing (default: 1.0)")
    p.add_argument("--k-folds", type=int, default=10,
                   help="Number of k-fold splits (default: 10)")
    p.add_argument("--benign-label", type=str, default="Benign",
                   help="Label name for benign samples (default: 'Benign')")
    return p.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    args = parse_args()
    setup_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info(f"Starting preprocessing with parameters:")
    logging.info(f"  Input path: {args.input_path}")
    logging.info(f"  Output directory: {args.output_dir}")
    logging.info(f"  Class ratio: {args.class_ratio}")
    logging.info(f"  K-folds: {args.k_folds}")
    logging.info(f"  Benign label: {args.benign_label}")
    
    # Load and process data
    logging.info("Loading and processing data...")
    df_raw = pd.read_csv(args.input_path)
    
    # Convert labels to binary (0 for benign, 1 for malicious)
    df_raw['label'] = df_raw['label'].apply(lambda x: 0 if x == args.benign_label else 1)
    
    logging.info(f"Dataset shape: {df_raw.shape}")
    logging.info(f"Label distribution:")
    logging.info(f"  Benign (0): {sum(df_raw['label'] == 0)}")
    logging.info(f"  Malicious (1): {sum(df_raw['label'] == 1)}")
    
    # Create k-fold splits
    logging.info(f"Creating {args.k_folds}-fold splits...")
    fold_data = create_kfold_splits(df_raw, args.k_folds)
    
    # Process each fold
    for fold_idx, fold in enumerate(fold_data):
        logging.info(f"Processing fold {fold_idx + 1}/{args.k_folds}")
        
        # Calculate feature statistics (no data leakage)
        feature_stats = calculate_feature_stats(fold['train'])
        
        # Normalize datasets using training statistics
        normalized_train = normalize_features(fold['train'], feature_stats)
        normalized_val = normalize_features(fold['val'], feature_stats)
        normalized_test = normalize_features(fold['test'], feature_stats)
        
        # Balance the training data
        balanced_train = balance_data(normalized_train, args.class_ratio)
        
        logging.info(f"  Balanced train size: {len(balanced_train)}")
        logging.info(f"  Validation size: {len(normalized_val)}")
        logging.info(f"  Test size: {len(normalized_test)}")
        
        # Build and save graph
        graph = build_fold_graph_with_ip_source(
            balanced_train,
            normalized_val,
            normalized_test,
            feature_stats
        )
        
        # Save graph
        graph_fold_path = Path(args.output_dir) / f'graph_fold_{fold_idx}.pickle'
        with open(graph_fold_path, 'wb') as fp:
            pickle.dump(graph, fp)
        
        logging.info(f"  Saved graph to: {graph_fold_path}")
    
    logging.info("All folds processed successfully!")
    logging.info(f"Graphs saved to: {args.output_dir}")

if __name__ == "__main__":
    main()