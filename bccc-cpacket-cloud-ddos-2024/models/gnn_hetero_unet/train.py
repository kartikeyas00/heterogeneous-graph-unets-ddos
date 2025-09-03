import pickle
import argparse
import torch
import torch.nn as nn
import time
import os
import pandas as pd
import logging
import datetime
import json
import torchmetrics
from model import HeteroGraphUNet
from dataset import create_dataloader

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--graph-dir", type=str, required=True,
                   help="Directory with graph_fold_*.pickle")
    p.add_argument("--hyperparams-path", type=str, required=True,
                   help="Path to hyperparameters JSON file")
    p.add_argument("--log-dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--k-folds", type=int, default=10)
    p.add_argument("--patience", type=int, default=50)
    return p.parse_args()

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = os.path.join(log_dir, f"training_{dt}.log")
    logging.basicConfig(filename=fh, level=logging.INFO)
    return logging.getLogger()

def load_hyperparams(path):
    with open(path) as f:
        return json.load(f)

def load_class_weight(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)

def train_epoch(epoch, train_dataloader, model, optimizer, criterion, device, logger):
    model.train()
    epoch_loss = 0
    start_time = time.time()
    
    for flow_nodes, sub_graph in train_dataloader:
        sub_graph = sub_graph.to(device)
        host_features = sub_graph.nodes['host'].data['feat']
        flow_features = sub_graph.nodes['flow'].data['feat']
        labels = sub_graph.ndata['label']['flow'].float().to(device)
        logits = model(sub_graph, host_features, flow_features)
        
        loss = criterion(logits.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    epoch_loss /= len(train_dataloader)
    end_time = time.time()
    epoch_time = end_time - start_time
    logger.info(f'Epoch {epoch}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.2f}s')
    
    return epoch_loss

def evaluate(epoch, test_dataloader, model, criterion, accuracy_metric, precision_metric,
             recall_metric, f1_metric, confmat_metric, device, logger):
    model.eval()
    eval_loss = 0
    accuracy_metric.reset()
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    confmat_metric.reset()
    
    with torch.no_grad():
        for flow_nodes, sub_graph in test_dataloader:
            sub_graph = sub_graph.to(device)
            host_features = sub_graph.nodes['host'].data['feat'].to(device)
            flow_features = sub_graph.nodes['flow'].data['feat'].to(device)
            labels = sub_graph.ndata['label']['flow'].float().to(device)
            logits = model(sub_graph, host_features, flow_features)
            loss = criterion(logits.squeeze(), labels)
            eval_loss += loss.item()
            
            preds = torch.sigmoid(logits).round()
            
            accuracy_metric.update(preds.squeeze(), labels)
            precision_metric.update(preds.squeeze(), labels)
            recall_metric.update(preds.squeeze(), labels)
            f1_metric.update(preds.squeeze(), labels)
            confmat_metric.update(preds.squeeze(), labels)
    
    eval_loss /= len(test_dataloader)
    acc = accuracy_metric.compute().item()
    prec = precision_metric.compute().item()
    rec = recall_metric.compute().item()
    f1 = f1_metric.compute().item()
    confmat = confmat_metric.compute()
    tn, fp, fn, tp = confmat.flatten().tolist()
    
    logger.info(f'Evaluation - Loss: {eval_loss:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}')
    logger.info(f'Confusion Matrix - TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}')
    
    return acc, prec, rec, f1, eval_loss, tp, fp, fn, tn

def main():
    args = parse_args()
    
    logger = setup_logging(args.log_dir)
    hyperparams = load_hyperparams(args.hyperparams_path)
    
    with open(os.path.join(args.log_dir, 'hyperparameters.json'), 'w') as fp:
        json.dump(hyperparams, fp, indent=4)
    
    checkpoint_dir = os.path.join(args.log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    metrics_file = os.path.join(args.log_dir, 'metrics.csv')
    metrics_columns = ['fold', 'epoch', 'loss', 'accuracy', 'precision', 'recall', 'f1']
    metrics_df = pd.DataFrame(columns=metrics_columns)
    metrics_df.to_csv(metrics_file, index=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    accuracy_metric = torchmetrics.Accuracy(task="binary", threshold=0.5).to(device)
    precision_metric = torchmetrics.Precision(task="binary", threshold=0.5).to(device)
    recall_metric = torchmetrics.Recall(task="binary", threshold=0.5).to(device)
    f1_metric = torchmetrics.F1Score(task="binary", threshold=0.5).to(device)
    confmat_metric = torchmetrics.ConfusionMatrix(task="binary", num_classes=2).to(device)
    
    final_kfolds_metrics = {}
    
    for i in range(1, args.k_folds + 1):
        
        
        accuracy_metric.reset()
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()
        
        with open(f"{args.graph_dir}/graph_fold_{i}.pickle", 'rb') as fp:
            g = pickle.load(fp)
        
        # Extract feature dimensions
        in_feats_dict = {}
        for ntype in g.ntypes:
            in_feats = g.nodes[ntype].data['feat'].shape[1]
            in_feats_dict[ntype] = in_feats

        out_feats_dict = {
            'flow': 1  # Binary classification for 'flow' nodes
        }

        # Get relation names and canonical edge types
        rel_names = []
        for src, etype, dst in g.canonical_etypes:
            if etype not in rel_names:
                rel_names.append(etype)
        canonical_etypes = g.canonical_etypes

        model = HeteroGraphUNet(
            in_feats_dict=in_feats_dict,
            hidden_feats=hyperparams['hidden_size'],
            out_feats_dict=out_feats_dict,
            rel_names=rel_names,
            canonical_etypes=canonical_etypes,
            depth=hyperparams['depth'],
            pool_ratio=hyperparams['pool_ratio']
        )
        model = model.to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        criterion = nn.BCEWithLogitsLoss()
        
        train_flow_nodes = g.nodes('flow')[g.nodes['flow'].data['train_mask']]
        test_flow_nodes = g.nodes('flow')[g.nodes['flow'].data['test_mask']]
        
        train_batches = [train_flow_nodes[j:j + hyperparams['flow_nodes_batch_size']]
                        for j in range(0, len(train_flow_nodes), hyperparams['flow_nodes_batch_size'])]
        train_batches = [batch for batch in train_batches if len(batch) == hyperparams['flow_nodes_batch_size']]
        
        test_batches = [test_flow_nodes[j:j + hyperparams['flow_nodes_batch_size']]
                       for j in range(0, len(test_flow_nodes), hyperparams['flow_nodes_batch_size'])]
        test_batches = [batch for batch in test_batches if len(batch) == hyperparams['flow_nodes_batch_size']]
        
        train_dataloader = create_dataloader(g, train_batches, hyperparams['batch_size_train'], False)
        test_dataloader = create_dataloader(g, test_batches, hyperparams['batch_size_test'], False)
        
        best_f1 = 0
        epochs_no_improve = 0
        logger.info(f'K Fold ----> {i}')
        
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            
            train_loss = train_epoch(epoch, train_dataloader, model, optimizer, criterion, device, logger)
            
            if epoch % 5 == 0:
                acc, prec, rec, f1, eval_loss, tp, fp, fn, tn = evaluate(
                    epoch, test_dataloader, model, criterion, accuracy_metric,
                    precision_metric, recall_metric, f1_metric, confmat_metric, device, logger)
                
                metrics_df = pd.concat([metrics_df, pd.DataFrame({
                    'fold': i,
                    'epoch': epoch,
                    'loss': train_loss,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'true_positives': tp,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'true_negatives': tn
                }, index=[0])], ignore_index=True)
                metrics_df.to_csv(metrics_file, index=False)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_model_path = os.path.join(checkpoint_dir, f'best_model_fold_{i}_epoch_{epoch}.pth')
                    torch.save(model.state_dict(), best_model_path)
                    logging.info(f'New best model saved with F1 score: {best_f1:.4f}')
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    logging.info(f'No improvement in F1 score for {epochs_no_improve} epoch(s)')
                
                if epochs_no_improve >= args.patience:
                    logging.info('Early stopping triggered.')
                    break
            
            epoch_end_time = time.time()
            total_epoch_time = epoch_end_time - epoch_start_time
            logging.info(f'Epoch {epoch} completed in {total_epoch_time:.2f}s')
        
        if best_f1 > 0:
            logging.info('Loading the best model for final evaluation.')
            model.load_state_dict(torch.load(best_model_path))
            model.to(device)
            
            final_acc, final_prec, final_rec, final_f1, final_eval_loss, final_tp, final_fp, final_fn, final_tn = evaluate(
                epoch, test_dataloader, model, criterion, accuracy_metric,
                precision_metric, recall_metric, f1_metric, confmat_metric, device, logger)
            logging.info(f'Final Evaluation - Loss: {final_eval_loss:.4f}, Accuracy: {final_acc:.4f}, Precision: {final_prec:.4f}, Recall: {final_rec:.4f}, F1: {final_f1:.4f}')
            logging.info(f'Confusion Matrix - TP: {final_tp}, FP: {final_fp}, FN: {final_fn}, TN: {final_tn}')
        else:
            logging.info('No improvement observed during training. Using the last epoch for final evaluation.')
            final_acc, final_prec, final_rec, final_f1, final_eval_loss, final_tp, final_fp, final_fn, final_tn = evaluate(
                epoch, test_dataloader, model, criterion, accuracy_metric,
                precision_metric, recall_metric, f1_metric, confmat_metric, device, logger)
            logging.info(f'Final Evaluation - Loss: {final_eval_loss:.4f}, Accuracy: {final_acc:.4f}, Precision: {final_prec:.4f}, Recall: {final_rec:.4f}, F1: {final_f1:.4f}')
            logging.info(f'Confusion Matrix - TP: {final_tp}, FP: {final_fp}, FN: {final_fn}, TN: {final_tn}')
        
        final_metrics = {
            'Final Loss': final_eval_loss,
            'Final Accuracy': final_acc,
            'Final Precision': final_prec,
            'Final Recall': final_rec,
            'Final F1 Score': final_f1,
            'True Positives': final_tp,
            'False Positives': final_fp,
            'False Negatives': final_fn,
            'True Negatives': final_tn
        }
        final_kfolds_metrics[i] = final_metrics
    
    with open(os.path.join(args.log_dir, 'final_metrics.json'), 'w') as fp:
        json.dump(final_kfolds_metrics, fp, indent=4)
    
    logging.info('Training and evaluation completed successfully.')

if __name__ == "__main__":
    main()
