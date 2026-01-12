import torch
import torch.nn.functional as F
from stability import NodeNorm, PairNorm, LabelSmoothingCrossEntropy

@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        x_in = dataset.graph['node_feat']
        if getattr(args, 'input_norm', 'none') == 'nodenorm':
            x_in = NodeNorm()(x_in)
        elif getattr(args, 'input_norm', 'none') == 'pairnorm':
            x_in = PairNorm()(x_in)
        out = model(x_in, dataset.graph['edge_index'])

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])

    if isinstance(criterion, LabelSmoothingCrossEntropy):
        valid_loss = criterion(out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out

@torch.no_grad()
def evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args, device, result=None):
    if result is not None:
        out = result
    else:
        model.eval()

    model.to(torch.device("cpu"))
    dataset.label = dataset.label.to(torch.device("cpu"))
    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    out = model(x, edge_index)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    
    if isinstance(criterion, LabelSmoothingCrossEntropy):
        valid_loss = criterion(out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out
