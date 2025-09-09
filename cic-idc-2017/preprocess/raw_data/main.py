from pathlib import Path
import pandas as pd
import numpy as np
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Load and preprocess CIC-IDS 2017 data files")
    p.add_argument("--input-dir", type=str, required=True,
                   help="Directory containing raw data files")
    p.add_argument("--output-file", type=str, default="cic_ids_2017_preprocessed.csv",
                   help="Output CSV filename")
    return p.parse_args()

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

def main():
    args = parse_args()
    
    print(f"Loading data from: {args.input_dir}")
    df = load_and_preprocess_data(args.input_dir)
    df.reset_index(drop=True, inplace=True)
    
    print(f"Saving preprocessed data to: {args.output_file}")
    df.to_csv(args.output_file, index=False)
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()