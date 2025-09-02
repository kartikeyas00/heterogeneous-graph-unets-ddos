import argparse
import os
import pickle
from pathlib import Path
import gc
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import psutil
from utils import (
    balance_data, balance_data_fixed, normalize_features, calculate_feature_stats,
    build_fold_graph_with_ip_source, create_kfold_indices
)
import time

def print_memory():
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.used/1e9:.1f}GB used / {mem.total/1e9:.1f}GB total")

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess and build graph folds")
    p.add_argument("--input-path",            type=str, required=True,
                   help="CSV of features to load")
    p.add_argument("--output-dir",            type=str, required=True,
                   help="Directory to write graph pickles and encoder")
    p.add_argument("--class-ratio",           type=float, default=1.0,
                   help="Positive/negative ratio for balancing")
    p.add_argument("--n-splits",              type=int,   default=10,
                   help="Number of folds")
    return p.parse_args()


def process_single_fold(df_raw, fold_indices, fold_num, args):
    """Process one fold using indices to slice original dataframe"""
    print(f"Processing fold {fold_num}...")
    
    train_idx = fold_indices['train_idx']
    val_idx = fold_indices['val_idx'] 
    test_idx = fold_indices['test_idx']
    
    train_data = df_raw.iloc[train_idx].copy()
    
    stats = calculate_feature_stats(train_data)
    
    train_data = normalize_features(train_data, stats)
    # train_data = balance_data(train_data, args.class_ratio)
    train_data, class_weights = balance_data_fixed(train_data, args.class_ratio)
    
    val_data = normalize_features(df_raw.iloc[val_idx].copy(), stats)
    test_data = normalize_features(df_raw.iloc[test_idx].copy(), stats)
    
    g = build_fold_graph_with_ip_source(train_data, val_data, test_data, stats)
    
    out_path = Path(args.output_dir)/f"graph_fold_{fold_num}.pickle"
    with open(out_path, "wb") as fp:
        pickle.dump(g, fp)

    # Save class weights
    weights_path = Path(args.output_dir)/f"class_weights_fold_{fold_num}.pickle"
    with open(weights_path, "wb") as fp:
        pickle.dump(class_weights, fp)
    print(f"Saved graph and weights for fold {fold_num}")

    del train_data, val_data, test_data, g
    gc.collect()
    
    print(f"Fold {fold_num} complete and cleaned up")

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading and processing data...")
    start_reading_time = time.time()
    df_raw = pd.read_csv(args.input_path)
    end_reading_time = time.time()
    print(f"Time to read file: {end_reading_time-start_reading_time}")
    df_raw["Label"] = df_raw["Label"].map({"BENIGN": 0}).fillna(1).astype(int)
    le = LabelEncoder()
    df_raw["MultiLabelEncoded"] = le.fit_transform(df_raw["Label"])
    with open(Path(args.output_dir)/"label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print(f"Creating {args.n_splits}-fold splits...")
    fold_indices_list = create_kfold_indices(df_raw['Label'].values, args.n_splits)

    for i, fold_indices in enumerate(fold_indices_list, start=1):
        print_memory()
        start_process_single_fold_time = time.time()
        process_single_fold(df_raw, fold_indices, i, args)
        end_process_single_fold_time = time.time()
        print(f"Time to process fold: {end_process_single_fold_time-start_process_single_fold_time}")
        print_memory()

    print("Done.")

if __name__ == "__main__":
    main()
