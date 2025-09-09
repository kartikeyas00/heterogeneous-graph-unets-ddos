import pandas as pd
import glob
import os
import numpy as np
import argparse

def parse_args():
    p = argparse.ArgumentParser(description="Load and preprocess CIC-DDoS 2019 raw data files")
    p.add_argument("--input-dirs", type=str, nargs='+', required=True,
                   help="List of directories containing raw data files")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Output directory for processed data")
    p.add_argument("--output-filename", type=str, default="features.csv",
                   help="Output CSV filename")
    return p.parse_args()

def list_files_in_directory(directory_path_list):
    raw_file_paths = []
    for i in directory_path_list:
        raw_file_paths.extend(glob.glob(os.path.join(i, "**", "*"), recursive=True))
    return raw_file_paths

def load_and_preprocess_data(raw_file_paths):
    df_raw_all = pd.DataFrame()
    
    for file_path in raw_file_paths:
        df_raw = pd.read_csv(file_path, encoding="latin1").dropna(how="all").replace([np.inf, -np.inf], np.nan).dropna()
        df_raw_all = pd.concat([df_raw_all, df_raw], ignore_index=True)

    df_raw_all = df_raw_all.dropna()
    return df_raw_all.replace([np.inf, -np.inf], np.nan).dropna()

def main():
    args = parse_args()
    
    print(f"Input directories: {args.input_dirs}")
    print(f"Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    raw_file_paths = list_files_in_directory(args.input_dirs)
    print(f"Found {len(raw_file_paths)} files to process")
    
    df = load_and_preprocess_data(raw_file_paths)
    df.columns = df.columns.str.strip()
    
    output_path = os.path.join(args.output_dir, args.output_filename)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

if __name__ == "__main__":
    main()