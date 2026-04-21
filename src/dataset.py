import numpy as np
import random
import torch
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

class JSPGraphDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, mmap_mode="r")
        self.A_all = data["A"]                  # (N, T, T)
        self.X_all = data["X"]                  # (N, T, F)
        self.M_all = data["M"]                  # (N, T, M) one-hot machine
        self.n_jobs = int(data["n_jobs"][0])
        self.n_machines = int(data["n_machines"][0])

    def __len__(self):
        return self.A_all.shape[0]

    def __getitem__(self, idx):
        A = self.A_all[idx]
        X = self.X_all[idx]
        M = self.M_all[idx]                     # (T, M)
        T = A.shape[0]

        # edge_index
        src, dst = np.nonzero(A > 0)
        edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)

        # add job one-hot
        job_id = np.arange(T) // self.n_machines
        job_oh = np.eye(self.n_jobs, dtype=np.float32)[job_id]
        X_aug = np.concatenate([X, job_oh], axis=1)

        # precedence edges (same job chain)
        prec_src = []
        prec_dst = []
        for j in range(self.n_jobs):
            start = j * self.n_machines
            for k in range(self.n_machines - 1):
                prec_src.append(start + k)
                prec_dst.append(start + k + 1)

        precedence_edge_index = torch.tensor([prec_src, prec_dst], dtype=torch.long)

        machine_ids = torch.tensor(M, dtype=torch.float32).argmax(dim=1).long()
        return Data(
            x=torch.tensor(X_aug, dtype=torch.float32),
            edge_index=edge_index,
            machine=machine_ids,
            precedence_edge_index=precedence_edge_index,
        )

def split_dataset(dataset: Dataset, val_ratio: float, test_ratio: float, seed: int = 42):
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)
    return train_set, val_set, test_set

def build_loaders(npz_path: str, batch_size: int, val_ratio: float, test_ratio: float):
    dataset = JSPGraphDataset(npz_path)
    train_set, val_set, test_set = split_dataset(dataset, val_ratio, test_ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_set, val_set, test_set, train_loader, val_loader, test_loader