import argparse
import pickle
import os
import pandas as pd
from pathlib import Path
from utils import (
    balance_data, normalize_features,
    calculate_feature_stats, build_fold_graph_with_ip_source, 
    create_kfold_splits
)


def parse_args():
    p = argparse.ArgumentParser(description="Build K-fold graph datasets for network flow analysis")
    p.add_argument("--input-path",       type=str, required=True,
                   help="Path to input CSV file with features")
    p.add_argument("--output-dir",       type=str, required=True,
                   help="Directory to save graph pickle files")
    p.add_argument("--class-ratio",      type=float, default=1.0,
                   help="Class balance ratio for training data")
    p.add_argument("--n-splits",         type=int, default=10,
                   help="Number of K-fold splits")
    return p.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading and processing data...")
    df_raw = pd.read_csv(args.input_path)

    print(f"Creating {args.n_splits}-fold splits...")
    fold_data = create_kfold_splits(df_raw, args.n_splits)

    for fold_idx, fold in enumerate(fold_data):
        print(f"\nProcessing fold {fold_idx + 1}/{args.n_splits}")
        feature_stats = calculate_feature_stats(fold['train'])

        # Normalize datasets using raw training statistics
        normalized_train = normalize_features(fold['train'], feature_stats)
        normalized_val = normalize_features(fold['val'], feature_stats)
        normalized_test = normalize_features(fold['test'], feature_stats)
        
        # Balance the training data
        balanced_train = balance_data(normalized_train, args.class_ratio)
            
        # Build and save graph
        graph = build_fold_graph_with_ip_source(
            balanced_train,
            normalized_val,
            normalized_test,
            feature_stats
        )
        
        # Save graph
        graph_fold_path = Path(args.output_dir) / f'graph_fold_{fold_idx + 1}.pickle'
        with open(graph_fold_path, 'wb') as fp:
            pickle.dump(graph, fp)
        
        print(f"Saved graph to: {graph_fold_path}")

    print("\nAll folds processed successfully!")


if __name__ == "__main__":
    main()