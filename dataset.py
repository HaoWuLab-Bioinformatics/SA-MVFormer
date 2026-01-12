import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Amazon, Coauthor, HeterophilousGraphDataset

import numpy as np
import scipy.sparse as sp
from os import path
import gdown
import scipy

from data_utils import dataset_drive_url


class NCDataset(object):
    def __init__(self, name):
        self.name = name
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.6, valid_prop=.2, label_num_per_class=20):
        split_idx = None
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname, sub_dataname=''):
    print(dataname)
    if dataname in  ('amazon-photo', 'amazon-computer'):
        dataset = load_amazon_dataset(data_dir, dataname)
    elif dataname in  ('coauthor-cs', 'coauthor-physics'):
        dataset = load_coauthor_dataset(data_dir, dataname)
    elif dataname in ('roman-empire', 'minesweeper'):
        dataset = load_hetero_dataset(data_dir, dataname)
    else:
        raise ValueError('Invalid dataname')
    return dataset



def load_hetero_dataset(data_dir, name):
    torch_dataset = HeterophilousGraphDataset(name=name.capitalize(), root=data_dir)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)


    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_amazon_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'amazon-photo':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                                 name='Photo', transform=transform)
    elif name == 'amazon-computer':
        torch_dataset = Amazon(root=f'{data_dir}Amazon',
                                 name='Computers', transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset

def load_coauthor_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    if name == 'coauthor-cs':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='CS', transform=transform)
    elif name == 'coauthor-physics':
        torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                                 name='Physics', transform=transform)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset
