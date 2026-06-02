"""
Dataset utilities for graph-based Job Shop Scheduling experiments.

The dataset stores scheduling states as graph snapshots consisting of

- adjacency matrices A,
- node features X,
- machine assignment matrices M.

Each sample is converted into a PyTorch Geometric Data object and can
be used for GAE/VGAE training and evaluation.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

class JSPGraphDataset(Dataset):
    """
    JSP graph dataset stored as adjacency matrices, node features,
    and machine assignments.
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, mmap_mode="r")

        self.A_all = data["A"]
        self.X_all = data["X"]
        self.M_all = data["M"]

        self.n_jobs = int(data["n_jobs"][0])
        self.n_machines = int(data["n_machines"][0])

    def __len__(self) -> int:
        return self.A_all.shape[0]

    def __getitem__(self, idx: int) -> Data:
        adjacency = self.A_all[idx]
        node_features = self.X_all[idx]
        machine_one_hot = self.M_all[idx]

        num_nodes = adjacency.shape[0]

        # Adjacency matrix -> edge list
        src, dst = np.nonzero(adjacency > 0)
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

        # Job identity features
        job_ids = np.arange(num_nodes) // self.n_machines
        job_one_hot = np.eye(self.n_jobs, dtype=np.float32)[job_ids]

        node_features = np.concatenate([node_features, job_one_hot], axis=1,
        )

        # Job precedence chains
        precedence_src = []
        precedence_dst = []

        for job in range(self.n_jobs):
            start = job * self.n_machines

            for op in range(self.n_machines - 1):
                precedence_src.append(start + op)
                precedence_dst.append(start + op + 1)

        precedence_edge_index = torch.tensor([precedence_src, precedence_dst], dtype=torch.long)

        machine_ids = (torch.tensor(machine_one_hot, dtype=torch.float32).argmax(dim=1).long())

        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            machine=machine_ids,
            precedence_edge_index=precedence_edge_index,
        )


def split_dataset(dataset: Dataset, val_ratio: float, test_ratio: float, seed: int = 42):
    """
    Random train/validation/test split.
    """
    num_samples = len(dataset)
    indices = list(range(num_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    num_test = int(num_samples * test_ratio)
    num_val = int(num_samples * val_ratio)
    num_train = num_samples - num_val - num_test
    train_idx = indices[:num_train]
    val_idx = indices[num_train:num_train + num_val]
    test_idx = indices[num_train + num_val:]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, val_set, test_set


def build_loaders(npz_path: str, batch_size: int, val_ratio: float, test_ratio: float):
    """
    Creates train, validation and test dataloaders.
    """
    dataset = JSPGraphDataset(npz_path)
    train_set, val_set, test_set = split_dataset(dataset, val_ratio, test_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,)

    return (train_set, val_set, test_set, train_loader, val_loader, test_loader)