import pandas as pd
import numpy as np
from pathlib import Path
from imblearn.under_sampling import RandomUnderSampler
import dgl
import torch
from sklearn.model_selection import train_test_split, KFold


def load_and_preprocess_data(directory_path):
    """Load and preprocess all data files from the directory."""
    dfs = []
    for file_path in Path(directory_path).rglob('*'):
        if file_path.is_file():
            df = pd.read_csv(file_path, encoding="latin1").dropna(how="all")
            df.columns = df.columns.str.strip()
            df = df[df.Label.isin(["BENIGN", "DDoS"])]
            df["Timestamp"] = pd.to_datetime(df.Timestamp, format="%m/%d/%Y %H:%M", errors="coerce").astype(np.int64)
            df["Label"] = (df["Label"] != "BENIGN").astype(int)
            dfs.append(df)
    
    df_combined = pd.concat(dfs, ignore_index=True)
    return df_combined.replace([np.inf, -np.inf], np.nan).dropna()


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


def balance_data(df, class_ratio):
    """Balance the dataset according to the specified class ratio.
    class ratio = number of normal / number of malicious
    """
    feature_columns = df.columns.difference([
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
        'Destination Port', 'Protocol', 'Timestamp', 'Label'
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
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
        'Destination Port', 'Protocol', 'Timestamp', 'Label'
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


def build_fold_graph_with_ip_source(train_df, val_df, test_df, feature_stats):
    """Build a DGL graph for a single fold using already normalized data."""
    df_combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    unique_ips = (
        set(df_combined['Destination IP'].astype(str)) |
        set(df_combined['Source IP'].astype(str))
    )
    ip_to_id = {ip: i for i, ip in enumerate(unique_ips)}
    
    flow_ids = range(len(df_combined))
    src_flows = list(flow_ids)
    dst_flows = list(flow_ids)
    
    src_hosts = [
        ip_to_id[x] 
        for x in df_combined['Source IP']
    ]
    dst_hosts = [
        ip_to_id[x] 
        for x in df_combined['Destination IP']
    ]
    
    g = dgl.heterograph({
        ('host', 'from_', 'flow'): (src_hosts, src_flows),
        ('flow', 'to_', 'host'): (dst_flows, dst_hosts)
    })
    
    feature_df = df_combined.drop(columns=[
        'Flow ID', 'Source IP', 'Source Port', 'Destination IP',
        'Destination Port', 'Protocol', 'Timestamp', 'Label'
    ])
    
    g.nodes['flow'].data['feat'] = torch.FloatTensor(feature_df.values)
    g.nodes['host'].data['feat'] = torch.ones((len(unique_ips), feature_df.shape[1]))
    g.nodes['flow'].data['label'] = torch.tensor(df_combined["Label"].values)
    
    # Create masks
    train_mask = torch.zeros(len(df_combined), dtype=torch.bool)
    val_mask = torch.zeros(len(df_combined), dtype=torch.bool)
    test_mask = torch.zeros(len(df_combined), dtype=torch.bool)
    
    train_mask[:len(train_df)] = True
    val_mask[len(train_df):len(train_df)+len(val_df)] = True
    test_mask[len(train_df)+len(val_df):] = True
    
    g.nodes['flow'].data['train_mask'] = train_mask
    g.nodes['flow'].data['val_mask'] = val_mask
    g.nodes['flow'].data['test_mask'] = test_mask
    
    return g