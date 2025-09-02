import pandas as pd
import numpy as np
from pathlib import Path
from imblearn.under_sampling import RandomUnderSampler
import dgl
import torch
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import random
from itertools import cycle
from collections import defaultdict
import re
from datetime import datetime


def create_kfold_splits(df, n_splits, test_size=0.2):
    """Create k-fold splits with a held-out test set."""
    # First split into train+val and test
    
    df_ = df.copy()
    # Create k-fold splits on train+val data
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_data = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(df_.index)):
        train_fold = df_.iloc[train_idx].reset_index(drop=True)
        test_data = df_.iloc[test_idx].reset_index(drop=True)
        train_data, val_data = train_test_split(
            train_fold, test_size=test_size, stratify=train_fold['Label'], random_state=42
        )
       
        fold_data.append({
            'train': train_data,
            'val': val_data,
            'test': test_data
        })
    
    return fold_data


def create_kfold_indices(labels, n_splits, test_size=0.2, random_state=42):
    """Create stratified k-fold indices instead of actual data copies"""
    
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_indices = []
    for train_val_idx, test_idx in skf.split(range(len(labels)), labels):
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=test_size, 
            stratify=labels[train_val_idx], random_state=random_state
        )
        
        fold_indices.append({
            'train_idx': train_idx,
            'val_idx': val_idx, 
            'test_idx': test_idx
        })
    
    return fold_indices


def balance_data_fixed(df, class_ratio=5.0):
    """Fixed balance function that handles actual data distribution"""
    from imblearn.under_sampling import RandomUnderSampler
    
    feature_columns = df.columns.difference([
         'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
        'Destination Port', 'Protocol', 'Timestamp', 'Label', 'MultiLabelEncoded', 'SimillarHTTP',
        'Inbound'
    ])
    
    df = df.copy().reset_index(drop=True)
    df['original_index'] = df.index
    
    benign_count = len(df[df['Label'] == 0])
    ddos_count = len(df[df['Label'] == 1])
    
    print(f"Original counts - BENIGN: {benign_count:,}, DDoS: {ddos_count:,}")
    
    # Handle the actual distribution: few BENIGN, many DDoS
    if benign_count < ddos_count:
        # Keep all BENIGN, downsample DDoS
        target_ddos = int(benign_count * class_ratio)
        sampling_strategy = {
            0: benign_count,     # Keep all BENIGN
            1: min(target_ddos, ddos_count)  # Downsample DDoS
        }
    else:
        # If somehow BENIGN > DDoS, keep all DDoS, downsample BENIGN  
        target_benign = int(ddos_count * class_ratio)
        sampling_strategy = {
            0: min(target_benign, benign_count),
            1: ddos_count
        }
    
    print(f"Target counts - BENIGN: {sampling_strategy[0]:,}, DDoS: {sampling_strategy[1]:,}")
    
    # Calculate class weights for remaining imbalance
    total_samples = sampling_strategy[0] + sampling_strategy[1]
    class_weights = {
        0: total_samples / (2 * sampling_strategy[0]),  # Weight for BENIGN
        1: total_samples / (2 * sampling_strategy[1])   # Weight for DDoS
    }
    
    print(f"Class weights - BENIGN: {class_weights[0]:.3f}, DDoS: {class_weights[1]:.3f}")
    
    # Apply undersampling
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X = df[list(feature_columns) + ['original_index']]
    y = df['Label']
    
    X_resampled, y_resampled = rus.fit_resample(X, y)
    balanced_df = df.loc[X_resampled['original_index']].drop('original_index', axis=1)
    
    return balanced_df, class_weights


def balance_data(df, class_ratio):
    """Balance the dataset according to the specified class ratio.
    class ratio = number of normal / number of malicious
    """
    feature_columns = df.columns.difference([
         'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
        'Destination Port', 'Protocol', 'Timestamp', 'Label', 'MultiLabelEncoded', 'SimillarHTTP',
        'Inbound'
    ])
    
    df = df.copy().reset_index(drop=True)
    df['original_index'] = df.index
    
    sampling_strategy = {
        0: int(len(df[df['Label'] == 1]) * class_ratio),
        1: len(df[df['Label'] == 1])
    }
    
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
    X = df[list(feature_columns) + ['original_index']]
    y = df['Label']
    
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return df.loc[X_resampled['original_index']].drop('original_index', axis=1)


def calculate_feature_stats(df):
    """Calculate mean and standard deviation for each feature."""
    feature_columns = df.columns.difference([
         'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
        'Destination Port', 'Protocol', 'Timestamp', 'Label', 'MultiLabelEncoded', 
        'SimillarHTTP', 'Inbound'
    ])
    
    df_features = df[feature_columns].replace([np.inf, -np.inf], np.nan).dropna()
    means = df_features.mean()
    std_devs = df_features.std()
    
    return {col: (means[col], std_devs[col]) for col in feature_columns}


def normalize_features(df, feature_stats):
    """Normalize features using pre-computed statistics."""
    normalized_df = df.copy()
    for feature, (mean, std) in feature_stats.items():
        if std != 0:
            normalized_df[feature] = (df[feature] - mean) / std
        else:
            normalized_df[feature] = 0
    return normalized_df

def  build_fold_graph_with_ip_port_nodes(train_df, val_df, test_df, feature_stats, number_of_node_features=128):
    """
    Create a directed DGL graph from network flow data.
    The graph is directed from source (IP:Port) to destination (IP:Port).
    
    Args:
        df: DataFrame containing network flow data
        node_feats: Number of node features to initialize
        edge_feature_cols: List of column names to use as edge features
        
    Returns:
        graph: Directed DGL graph
        node_mapping: Dictionary mapping node IDs to their indices
    """
    # Create unique identifiers for source and destination nodes
    df_combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    df_combined['src_id'] = df_combined['Source IP'] + ':' + df_combined['Source Port'].astype(str)
    df_combined['dst_id'] = df_combined['Destination IP'] + ':' + df_combined['Destination Port'].astype(str)
    
    # Get unique nodes and create mapping
    unique_nodes = sorted(set(df_combined['src_id'].unique()) | set(df_combined['dst_id'].unique()))
    node_mapping = {node: idx for idx, node in enumerate(unique_nodes)}
    
    # Create graph edges (source -> destination)
    src_indices = [node_mapping[src] for src in df_combined['src_id']]
    dst_indices = [node_mapping[dst] for dst in df_combined['dst_id']]
    
    # Create directed DGL graph
    graph = dgl.graph((src_indices, dst_indices))

    
    # Initialize node features with ones (as per paper)
    graph.ndata['feat'] = torch.ones(len(node_mapping), number_of_node_features)
    feature_df = df_combined.drop(columns=[
        'Unnamed: 0', 'Flow ID', 'Source IP', 'Source Port', 'Destination IP','src_id', 'dst_id',
        'Destination Port', 'Protocol', 'Timestamp', 'Label', 'MultiLabelEncoded', 'SimillarHTTP',
        'Inbound'
    ], errors="ignore")

    print("=== CHECKING DATA TYPES ===")
    print(feature_df.dtypes)
    print("\n=== OBJECT COLUMNS ===")
    object_cols = feature_df.select_dtypes(include=['object']).columns.tolist()
    print(f"Object columns: {object_cols}")

    print("\n=== CHECKING FOR NON-NUMERIC VALUES ===")
    for col in feature_df.columns:
        try:
            pd.to_numeric(feature_df[col], errors='raise')
        except:
            print(f"Column '{col}' has non-numeric values")
            print(f"  Data type: {feature_df[col].dtype}")
            print(f"  Unique values sample: {feature_df[col].unique()[:10]}")
            print(f"  Contains NaN: {feature_df[col].isna().any()}")
            print(f"  Contains inf: {np.isinf(feature_df[col].replace([np.inf, -np.inf], np.nan).dropna()).any()}")
            print()

    print("\n=== SUMMARY ===")
    print(f"Total columns: {len(feature_df.columns)}")
    print(f"Numeric columns: {len(feature_df.select_dtypes(include=[np.number]).columns)}")
    print(f"Non-numeric columns: {len(object_cols)}")

    print("\n=== RECOMMENDED FIX ===")
    if object_cols:
        print(f"Drop these object columns: {object_cols}")
    else:
        print("Check for inf values and handle them before tensor conversion")
    
    
    edge_features = torch.FloatTensor(feature_df.values)
    graph.edata['feat'] = torch.FloatTensor(edge_features)
    graph.edata['label'] = torch.tensor(df_combined["Label"].values)
    graph.edata['multi_label_encoded'] = torch.tensor(df_combined["MultiLabelEncoded"].values)
    train_mask = torch.zeros(len(df_combined), dtype=torch.bool)
    val_mask = torch.zeros(len(df_combined), dtype=torch.bool)
    test_mask = torch.zeros(len(df_combined), dtype=torch.bool)
    
    train_mask[:len(train_df)] = True
    val_mask[len(train_df):len(train_df)+len(val_df)] = True
    test_mask[len(train_df)+len(val_df):] = True
    graph.edata['train_mask'] = train_mask
    graph.edata['val_mask'] = val_mask
    graph.edata['test_mask'] = test_mask
    
    
    return graph