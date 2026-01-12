import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def build_knn_graph(x, k=10, metric='cosine', normalize=True, device='cpu'):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()

    if normalize:
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
    elif x.shape[1] > 512:
        from sklearn.decomposition import PCA
        x = PCA(256).fit_transform(x)

    import os
    max_threads = min(8, os.cpu_count() or 4)
    knn = NearestNeighbors(n_neighbors=k + 1, metric=metric, n_jobs=max_threads)
    knn.fit(x)
    distances, indices = knn.kneighbors(x)

    indices = indices[:, 1:]
    distances = distances[:, 1:]

    weights = np.exp(-distances) 
    src = np.repeat(np.arange(x.shape[0]), k)
    dst = indices.reshape(-1)
    edge_weight = weights.reshape(-1)

    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(edge_weight, dtype=torch.float, device=device)

    return edge_index, edge_weight
