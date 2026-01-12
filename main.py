import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops

import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '4'
os.environ['TORCH_NUM_THREADS'] = '4'

from logger import *
from dataset import load_dataset
from data_utils import eval_acc, eval_rocauc, load_fixed_splits
from eval import *
from parse import parse_method, parser_add_main_args
from knn_graph import build_knn_graph
from stability import NodeNorm, PairNorm, LabelSmoothingCrossEntropy
from metrics_tracker import MetricsTracker

def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

### Parse args ###
parser = argparse.ArgumentParser(description='Training Pipeline for Node Classification')
parser_add_main_args(parser)
args = parser.parse_args()
if not args.global_dropout:
    args.global_dropout = args.dropout
print(args)

fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
else:
    visible_device_index = int(args.device) if torch.cuda.is_available() else -1
    device = torch.device("cuda:" + str(visible_device_index)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(device)
        except Exception as e:
            print(f"CUDA device {device} unavailable, using CPU instead: {e}")
            device = torch.device("cpu")

### Load and preprocess data ###
dataset = load_dataset(args.data_dir, args.dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

split_idx_lst = load_fixed_splits(args.data_dir, dataset, name=args.dataset)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Load method ###
model = parse_method(args, n, c, d, device)


### Loss function (Single-class, Multi-class) ###
if args.label_smoothing and args.label_smoothing > 0.0:
    criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

metrics_tracker = MetricsTracker(args.dataset, args.method)

model.train()
print('MODEL:', model)

# Optional input feature normalization
in_norm = None
if getattr(args, 'input_norm', 'none') == 'nodenorm':
    in_norm = NodeNorm().to(device)
elif getattr(args, 'input_norm', 'none') == 'pairnorm':
    in_norm = PairNorm().to(device)

### Training loop ###
for run in range(args.runs):
    run_start_time = metrics_tracker.start_run()

    if args.dataset in ('coauthor-cs', 'coauthor-physics', 'amazon-computer', 'amazon-photo'):
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run % len(split_idx_lst)]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    model._global = False
    # Build optimizer
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay, lr=args.lr)

    best_val = float('-inf')
    best_test = float('-inf')
    best_epoch = 0
    best_out = None
    
    if args.save_model:
        save_model(args, model, optimizer, run)

    for epoch in range(args.local_epochs + args.global_epochs):
        if epoch == args.local_epochs:
            print("start global attention!!!!!!")
            if args.save_model and not args.no_resume:
                try:
                    model, optimizer = load_model(args, model, optimizer, run)
                except Exception as e:
                    print(f"The checkpoint could not be loaded.The training will start from scratch：{e}")
            model._global = True
        
        model.train()
        optimizer.zero_grad()


        x_in = dataset.graph['node_feat']
        if in_norm is not None:
            x_in = in_norm(x_in)
        out = model(x_in, dataset.graph['edge_index'])
        if isinstance(criterion, LabelSmoothingCrossEntropy):
            logits = out
            loss = criterion(logits[train_idx], dataset.label.squeeze(1)[train_idx])
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
        
        # Optimization step
        loss.backward()
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args)

        logger.add_result(run, result[:-1])

        if result[1] > best_val:
            best_val = result[1]
            best_test = result[2]
            best_epoch = epoch
            best_out = result[-1].clone() if isinstance(result[-1], torch.Tensor) else result[-1]  # 保存最佳输出
            if args.save_model:
                save_model(args, model, optimizer, run)

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Best Valid: {100 * best_val:.2f}%, '
                  f'Best Test: {100 * best_test:.2f}%')
    

    run_duration = metrics_tracker.end_run(run_start_time)
    if run_duration is not None:
        print(f"Run {run + 1} duration: {run_duration:.2f} seconds")
    

    if best_out is not None and metrics_tracker.enabled:
        metrics_tracker.record_run_metrics(
            labels=dataset.label,
            outputs=best_out,
            split_idx=split_idx,
            run_id=run,
            best_epoch=best_epoch
        )
        metrics_tracker.print_run_summary(run)
    
    logger.print_statistics(run)

results = logger.print_statistics()

### Save results ###
if metrics_tracker.enabled:
    print("\n Calculating and printing detailed indicators...")
    metrics_tracker.print_aggregated_summary()

save_result(args, results)

