#!/usr/bin/env python3
"""
Comprehensive Pipeline for CIC-IDC-2017 Dataset
==============================================

This pipeline automates the complete workflow for training and evaluating 
graph neural network models on the CIC-IDC-2017 dataset.

Workflow:
1. Data preprocessing (host connection graph and/or network flow graph)
2. Hyperparameter tuning (using Optuna)
3. Model training with best hyperparameters
4. Results collection and analysis

Usage:
    python pipeline.py --config config.json

Config file should specify:
- Input data path
- Output directories
- Models to run
- Preprocessing options
- Hyperparameter tuning settings
- Training settings
"""

import os
import json
import argparse
import logging
import datetime
import subprocess
import sys
from pathlib import Path
import pandas as pd

def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{dt}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

def run_command(cmd, description):
    """Run a shell command and log the output"""
    logging.info(f"Running: {description}")
    logging.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            logging.info(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Command failed with return code {e.returncode}")
        logging.error(f"Error output: {e.stderr}")
        return False

def preprocess_data(config):
    """Run data preprocessing for graph construction"""
    logging.info("=" * 60)
    logging.info("STEP 1: DATA PREPROCESSING")
    logging.info("=" * 60)
    
    preprocessing_configs = config.get('preprocessing', {})
    base_dir = Path(__file__).parent
    
    for graph_type, graph_config in preprocessing_configs.items():
        if not graph_config.get('enabled', False):
            logging.info(f"Skipping {graph_type} preprocessing (disabled)")
            continue
            
        logging.info(f"Running {graph_type} preprocessing...")
        
        preprocess_dir = base_dir / "preprocess" / graph_type
        output_dir = graph_config['output_dir']
        
        cmd = [
            sys.executable, 
            str(preprocess_dir / "main.py"),
            "--input-path", graph_config['input_path'],
            "--output-dir", output_dir,
            "--class-ratio", str(graph_config.get('class_ratio', 1.0)),
            "--n-splits", str(graph_config.get('k_folds', 10)),
        ]
        
        success = run_command(cmd, f"{graph_type} preprocessing")
        if not success:
            logging.error(f"Preprocessing failed for {graph_type}")
            return False
            
        logging.info(f"Completed {graph_type} preprocessing")
    
    return True

def run_hyperparameter_tuning(config):
    """Run hyperparameter tuning for all enabled models"""
    logging.info("=" * 60)
    logging.info("STEP 2: HYPERPARAMETER TUNING")
    logging.info("=" * 60)
    
    models_config = config.get('models', {})
    base_dir = Path(__file__).parent
    
    for model_name, model_config in models_config.items():
        if not model_config.get('enabled', False):
            logging.info(f"Skipping {model_name} hyperparameter tuning (disabled)")
            continue
            
        logging.info(f"Running hyperparameter tuning for {model_name}...")
        
        model_dir = base_dir / "models" / model_name
        tuning_config = model_config['hyperparameter_tuning']
        
        cmd = [
            sys.executable,
            str(model_dir / "hyperparams_tuning.py"),
            "--optuna-dir", tuning_config['optuna_dir'],
            "--graph-path", tuning_config['graph_path'],
            "--n-trials", str(tuning_config.get('n_trials', 200)),
            "--study-name", tuning_config.get('study_name', f'{model_name}_hyperopt')
        ]
        
        # Add optional class weight path if specified
        if tuning_config.get('class_weight_path'):
            cmd.extend(["--class-weight-path", tuning_config['class_weight_path']])
        
        success = run_command(cmd, f"{model_name} hyperparameter tuning")
        if not success:
            logging.error(f"Hyperparameter tuning failed for {model_name}")
            return False
            
        logging.info(f"Completed hyperparameter tuning for {model_name}")
    
    return True

def run_training(config):
    """Run training for all enabled models with best hyperparameters"""
    logging.info("=" * 60)
    logging.info("STEP 3: MODEL TRAINING")
    logging.info("=" * 60)
    
    models_config = config.get('models', {})
    base_dir = Path(__file__).parent
    
    for model_name, model_config in models_config.items():
        if not model_config.get('enabled', False):
            logging.info(f"Skipping {model_name} training (disabled)")
            continue
            
        logging.info(f"Running training for {model_name}...")
        
        model_dir = base_dir / "models" / model_name
        training_config = model_config['training']
        
        # Path to best hyperparameters from tuning
        hyperparams_path = os.path.join(
            model_config['hyperparameter_tuning']['optuna_dir'],
            'best_hyperparams.json'
        )
        
        cmd = [
            sys.executable,
            str(model_dir / "train.py"),
            "--graph-dir", training_config['graph_dir'],
            "--hyperparams-path", hyperparams_path,
            "--log-dir", training_config['log_dir'],
            "--epochs", str(training_config.get('epochs', 1000)),
            "--k-folds", str(training_config.get('k_folds', 10)),
            "--patience", str(training_config.get('patience', 100))
        ]
        
        success = run_command(cmd, f"{model_name} training")
        if not success:
            logging.error(f"Training failed for {model_name}")
            return False
            
        logging.info(f"Completed training for {model_name}")
    
    return True


def get_aggregated_results(model_name, results_json):

    df_results = pd.read_json(results_json)
    mean_results = df_results.T.mean()
    mean_results_dict = {f'{k} Mean':v for k,v in  mean_results.to_dict().items()}
    std_results = df_results.T.std()
    std_results_dict = {f'{k} Std':v for k,v in  std_results.to_dict().items()}
    df = pd.DataFrame({**mean_results_dict, **std_results_dict}, index=[model_name])
    df['Model'] = model_name
    df = df[['Model', 'Final Accuracy Mean', 'Final Accuracy Std', 'Final F1 Score Mean', 'Final F1 Score Std',
             'Final Precision Mean', 'Final Precision Std', 'Final Recall Mean', 'Final Recall Std']]

    return df

def collect_results(config):
    """Collect and summarize results from all experiments"""
    logging.info("=" * 60)
    logging.info("STEP 4: RESULTS COLLECTION")
    logging.info("=" * 60)
    
    results_dir = config.get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    models_config = config.get('models', {})
    
    # Collect hyperparameter tuning results
    hyperopt_results = {}
    for model_name, model_config in models_config.items():
        if not model_config.get('enabled', False):
            continue
            
        optuna_dir = model_config['hyperparameter_tuning']['optuna_dir']
        hyperparams_file = os.path.join(optuna_dir, 'best_hyperparams.json')
        
        if os.path.exists(hyperparams_file):
            with open(hyperparams_file, 'r') as f:
                hyperopt_results[model_name] = json.load(f)
    
    # Save hyperparameter results
    hyperopt_summary_path = os.path.join(results_dir, 'hyperparameter_results.json')
    with open(hyperopt_summary_path, 'w') as f:
        json.dump(hyperopt_results, f, indent=2)
    
    logging.info(f"Hyperparameter results saved to: {hyperopt_summary_path}")
    
    # Collect training results
    df_final_resuls = pd.DataFrame()
    for model_name, model_config in models_config.items():
        if not model_config.get('enabled', False):
            continue
            
        log_dir = model_config['training']['log_dir']
        summary_file = os.path.join(log_dir, 'final_metrics.json')
        
        if os.path.exists(summary_file):
            df_final_resuls = pd.concat([df_final_resuls, get_aggregated_results(model_name, summary_file)], ignore_index=True)

    # Save training results
    training_summary_path = os.path.join(results_dir, 'training_results.csv')
    df_final_resuls.to_csv(training_summary_path, index=False)

    logging.info(f"Training results saved to: {training_summary_path}")
    
    return True

def validate_config(config):
    """Validate the configuration file"""
    required_sections = ['preprocessing', 'models']
    for section in required_sections:
        if section not in config:
            logging.error(f"Missing required section in config: {section}")
            return False
    
    # Validate that at least one preprocessing or model is enabled
    preprocessing_enabled = any(
        cfg.get('enabled', False) for cfg in config.get('preprocessing', {}).values()
    )
    models_enabled = any(
        cfg.get('enabled', False) for cfg in config.get('models', {}).values()
    )
    
    if not preprocessing_enabled and not models_enabled:
        logging.error("No preprocessing steps or models are enabled in config")
        return False
    
    return True

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True,
                       help="Path to the configuration JSON file")
    parser.add_argument("--skip-preprocessing", action='store_true',
                       help="Skip preprocessing step")
    parser.add_argument("--skip-hyperparameter-tuning", action='store_true',
                       help="Skip hyperparameter tuning step")  
    parser.add_argument("--skip-training", action='store_true',
                       help="Skip training step")
    parser.add_argument("--log-dir", type=str, default="pipeline_logs",
                       help="Directory for pipeline logs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Setup logging
    log_file = setup_logging(args.log_dir)
    
    logging.info("=" * 60)
    logging.info("CIC-IDC-2017 PIPELINE")
    logging.info("=" * 60)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config file: {e}")
        return 1
    
    # Validate configuration
    if not validate_config(config):
        return 1
    
    logging.info(f"Loaded configuration from: {args.config}")
    logging.info(f"Pipeline logs saved to: {log_file}")
    
    # Run pipeline steps
    if not args.skip_preprocessing:
        if not preprocess_data(config):
            logging.error("Preprocessing failed. Stopping pipeline.")
            return 1
    
    if not args.skip_hyperparameter_tuning:
        if not run_hyperparameter_tuning(config):
            logging.error("Hyperparameter tuning failed. Stopping pipeline.")
            return 1
    
    if not args.skip_training:
        if not run_training(config):
            logging.error("Training failed. Stopping pipeline.")
            return 1
    
    # Collect results
    if not collect_results(config):
        logging.error("Results collection failed.")
        return 1
    
    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETED SUCCESSFULLY!")
    logging.info("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
