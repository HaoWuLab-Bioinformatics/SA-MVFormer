import torch
import numpy as np
import time


class MetricsTracker:
    def __init__(self, dataset_name, method_name, enabled_datasets=None):
        self.dataset_name = dataset_name
        self.method_name = method_name
        
        if enabled_datasets is None:
            self.enabled_datasets = {
                'amazon-computer', 'amazon-photo', 
                'coauthor-physics', 'coauthor-cs',
                'roman-empire', 'minesweeper'
            }
        else:
            self.enabled_datasets = set(enabled_datasets)
        

        self.enabled = dataset_name in self.enabled_datasets
        

        self.run_results = []
        self.run_times = []
        
        if self.enabled:
            print(f"Metrics tracking enabled: {dataset_name}")
    
    def start_run(self):
        if not self.enabled:
            return
        return time.time()
    
    def end_run(self, start_time):
        if not self.enabled or start_time is None:
            return
        elapsed = time.time() - start_time
        self.run_times.append(elapsed)
        return elapsed
    
    @torch.no_grad()
    def compute_metrics(self, y_true, y_pred, average='macro'):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            if y_pred.dim() > 1 and y_pred.shape[1] > 1:
                y_pred = torch.argmax(y_pred, dim=1)
            y_pred = y_pred.cpu().numpy()

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        num_classes = int(max(y_true.max(), y_pred.max()) + 1)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
        
        for t, p in zip(y_true, y_pred):
            if 0 <= t < num_classes and 0 <= p < num_classes:
                confusion[int(t), int(p)] += 1

        tp = np.diag(confusion).astype(np.float64)
        fp = confusion.sum(axis=0).astype(np.float64) - tp
        fn = confusion.sum(axis=1).astype(np.float64) - tp
        support = confusion.sum(axis=1).astype(np.float64)

        precision_per_class = np.divide(tp, tp + fp, 
                                        out=np.zeros_like(tp), 
                                        where=(tp + fp) > 0)
        recall_per_class = np.divide(tp, tp + fn, 
                                     out=np.zeros_like(tp), 
                                     where=(tp + fn) > 0)
        f1_per_class = np.divide(2 * precision_per_class * recall_per_class,
                                precision_per_class + recall_per_class,
                                out=np.zeros_like(precision_per_class),
                                where=(precision_per_class + recall_per_class) > 0)
        

        accuracy = float(tp.sum() / confusion.sum()) if confusion.sum() > 0 else 0.0
        

        if average == 'micro':
            tp_sum = tp.sum()
            fp_sum = fp.sum()
            fn_sum = fn.sum()
            precision = float(tp_sum / (tp_sum + fp_sum)) if (tp_sum + fp_sum) > 0 else 0.0
            recall = float(tp_sum / (tp_sum + fn_sum)) if (tp_sum + fn_sum) > 0 else 0.0
            f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        elif average == 'weighted':
            total = support.sum() if support.sum() > 0 else 1.0
            weights = support / total
            precision = float((precision_per_class * weights).sum())
            recall = float((recall_per_class * weights).sum())
            f1 = float((f1_per_class * weights).sum())
        else:  # macro
            precision = float(precision_per_class.mean()) if len(precision_per_class) > 0 else 0.0
            recall = float(recall_per_class.mean()) if len(recall_per_class) > 0 else 0.0
            f1 = float(f1_per_class.mean()) if len(f1_per_class) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    
    def record_run_metrics(self, labels, outputs, split_idx, run_id, best_epoch):
        if not self.enabled:
            return
        
        metrics_dict = {}
        for split_name in ['train', 'valid', 'test']:
            if split_name in split_idx:
                idx = split_idx[split_name]
                y_true = labels[idx]
                y_pred = outputs[idx]
                
                metrics = self.compute_metrics(y_true, y_pred, average='macro')
                metrics_dict[split_name] = metrics

        run_time = self.run_times[run_id] if run_id < len(self.run_times) else 0.0
        self.run_results.append({
            'run_id': run_id,
            'best_epoch': best_epoch,
            'run_time': run_time,
            'metrics': metrics_dict
        })
    
    def print_run_summary(self, run_id):
        if not self.enabled or run_id >= len(self.run_results):
            return
        
        result = self.run_results[run_id]
        print("\n" + "="*60)
        print(f"Run {run_id + 1} - Detailed Metrics Summary")
        print("="*60)
        print(f"Runtime: {result['run_time']:.2f} seconds")
        print(f"Best Epoch: {result['best_epoch']}")
        
        for split_name in ['train', 'valid', 'test']:
            if split_name in result['metrics']:
                m = result['metrics'][split_name]
                print(f"\n{split_name.upper()}:")
                print(f"  Accuracy:  {m['accuracy']*100:.2f}%")
                print(f"  Precision: {m['precision']*100:.2f}%")
                print(f"  Recall:    {m['recall']*100:.2f}%")
                print(f"  F1-Score:  {m['f1_score']*100:.2f}%")
        print("="*60 + "\n")
    
    def compute_aggregated_stats(self):
        if not self.enabled or len(self.run_results) == 0:
            return None
        
        aggregated = {}
        metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for split_name in ['train', 'valid', 'test']:
            aggregated[split_name] = {}
            
            for metric_name in metric_names:
                values = []
                for result in self.run_results:
                    if split_name in result['metrics']:
                        values.append(result['metrics'][split_name][metric_name])
                
                if values:
                    values = np.array(values)
                    aggregated[split_name][metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'values': values.tolist()
                    }
        
        return aggregated
    
    def print_aggregated_summary(self):
        if not self.enabled:
            return
        
        aggregated = self.compute_aggregated_stats()
        if aggregated is None:
            return
        
        print("\n" + "="*70)
        print("Aggregated Statistics Across All Runs (Mean ± Std)")
        print("="*70)
        
        for split_name in ['train', 'valid', 'test']:
            if split_name in aggregated and aggregated[split_name]:
                print(f"\n{split_name.upper()}:")
                metrics = aggregated[split_name]
                
                for metric_name, display_name in [
                    ('accuracy', 'Accuracy'),
                    ('precision', 'Precision'),
                    ('recall', 'Recall'),
                    ('f1_score', 'F1-Score')
                ]:
                    if metric_name in metrics:
                        mean = metrics[metric_name]['mean'] * 100
                        std = metrics[metric_name]['std'] * 100
                        print(f"  {display_name:10s}: {mean:6.2f}% ± {std:5.2f}%")
        
        if self.run_times:
            avg_time = np.mean(self.run_times)
            std_time = np.std(self.run_times)
            print(f"\nRuntime: {avg_time:.2f} ± {std_time:.2f} seconds")
        
        print("="*70 + "\n")

