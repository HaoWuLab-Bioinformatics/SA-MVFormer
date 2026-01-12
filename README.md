# SA-MVFormer
SA-MVFormer: A Structure-Aware Multi-View Graph Transformer for Node Classification

## The framework of SA-MVFormer
| ![SA-MVFormer.png](/figures/SA-MVFormer.png) | 
|:--:| 
| Figure1: An overview of the SA-MVFormer architecture. |
## Overview

The folder "**data**" contains the graph datasets used for training and evaluation, including Amazon, Coauthor, and HeterophilousGraph datasets, along with pre-computed data splits for train/validation/test sets.

The file "**main.py**" is the main entry point of the training pipeline for node classification tasks, which handles argument parsing, dataset loading, model initialization, training loop, and result logging.

The file "**model.py**" is the code of the network architecture, implementing the SA_MVFormer model with local-to-global attention mechanism, including GlobalAttn module and multi-view fusion capabilities.

The file "**dataset.py**" is the code for loading and preprocessing various graph datasets, including Amazon, Coauthor, and HeterophilousGraph datasets from different sources.

The file "**data_utils.py**" contains utility functions for dataset operations, including loading fixed data splits, evaluation metrics computation (accuracy, ROC-AUC, F1-score), and dataset download URLs.

The file "**eval.py**" contains the evaluation functions for model performance assessment on train/validation/test splits, supporting both GPU and CPU evaluation modes.

The file "**fusionblock.py**" is the code of the fusion block module that combines graph neural network features with attention-based features using different fusion methods (concat, weighted) and GNN types (GCN, SAGE).

The file "**knn_graph.py**" contains functions for building K-nearest neighbor graphs from node features using cosine similarity.

The file "**logger.py**" is the code for logging training results, tracking performance metrics across multiple runs, and saving/loading model checkpoints.

The file "**metrics_tracker.py**" is the code for tracking detailed evaluation metrics (accuracy, precision, recall, F1-score) across multiple runs and computing aggregated statistics.

The file "**parse.py**" contains the argument parser configuration and model initialization function, defining all command-line arguments for training configuration and model hyperparameters.

The file "**stability.py**" contains normalization techniques (NodeNorm, PairNorm) and label smoothing cross-entropy loss for improving training stability and generalization.


## Requirements
* python 3.9
* pytorch 2.0.1 (CUDA 11.7)
* torch_geometric 2.3.0
* numpy 1.21.0
* scipy 1.7.0
* scikit-learn 1.0.0

## Python environment setup with Conda (Linux)
```bash
conda create -n sa-mvformer python=3.9
conda activate sa-mvformer
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
pip install numpy scipy scikit-learn

conda clean --all
```

## Running SA-MVFormer
```bash
conda activate sa-mvformer
# running a single experiment on amazon-photo
python main.py --dataset amazon-photo --hidden_channels 64 --local_epochs 100 --global_epochs 100 --lr 0.001 --runs 1 --local_layers 5 --global_layers 2 --weight_decay 5e-5 --dropout 0.7 --in_dropout 0.2 --num_heads 8 --fusion_gnn_type gcn --fusion_gnn_layer 2  --fusion_graph_weight 0.6 --view_weights 0.8 0.2 --view_knn_k 5 --input_norm nodenorm --label_smoothing 0.1  --no_resume

# running all experiments with full batch training
bash run.sh
```