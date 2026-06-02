"""
Utility functions for JSP graph representation learning.
"""

import os
import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import degree

def load_jsp_txt(path: str) -> np.ndarray:
    """Loads a Taillard-style JSP .txt file as [machines, durations]."""

    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    rows = np.array(
        [list(map(int, row.split())) for row in lines[1:]],
        dtype=np.int64,
    )

    machines = rows[:, 0::2]
    durations = rows[:, 1::2]

    return np.stack([machines, durations], axis=0)


def generate_jsp_instance(
    n_jobs: int,
    n_machines: int,
    min_processing_time: int = 2,
    max_processing_time: int = 100,
) -> np.ndarray:
    """Generates a random JSP instance as [machines, durations]."""

    machine_order = [
        random.sample(list(range(n_machines)), n_machines)
        for _ in range(n_jobs)
    ]

    processing_times = [
        [
            random.randint(min_processing_time, max_processing_time)
            for _ in range(n_machines)
        ]
        for _ in range(n_jobs)
    ]

    return np.array([machine_order, processing_times], dtype=np.int64)


def npz_to_data_list(npz_path: str) -> list[Data]:
    """Converts stored graph states from .npz format to PyG Data objects."""

    data = np.load(npz_path)

    A_all = data["A"]
    X_all = data["X"]
    M_all = data["M"]

    n_jobs = int(data["n_jobs"][0])
    n_machines = int(data["n_machines"][0])

    data_list = []

    for i in range(A_all.shape[0]):
        adjacency = A_all[i]
        node_features = X_all[i]
        machine_one_hot = M_all[i]

        num_nodes = adjacency.shape[0]

        # Adjacency matrix -> edge list
        src, dst = np.nonzero(adjacency > 0)
        edge_index = torch.tensor(
            np.stack([src, dst], axis=0),
            dtype=torch.long,
        )

        # Job identity features
        job_ids = np.arange(num_nodes) // n_machines
        job_one_hot = np.eye(n_jobs, dtype=np.float32)[job_ids]

        node_features = np.concatenate(
            [node_features, job_one_hot],
            axis=1,
        )

        # Job precedence chains
        precedence_src = []
        precedence_dst = []

        for job in range(n_jobs):
            start = job * n_machines

            for op in range(n_machines - 1):
                precedence_src.append(start + op)
                precedence_dst.append(start + op + 1)

        precedence_edge_index = torch.tensor(
            [precedence_src, precedence_dst],
            dtype=torch.long,
        )

        machine_ids = (
            torch.tensor(machine_one_hot, dtype=torch.float32)
            .argmax(dim=1)
            .long()
        )

        data_list.append(
            Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=edge_index,
                machine=machine_ids,
                precedence_edge_index=precedence_edge_index,
            )
        )

    return data_list


def split_for_link_pred(
    data_list: list[Data],
    val_ratio: float,
    test_ratio: float,
    seed: int = 42,
) -> tuple[list[Data], list[Data], list[Data]]:
    """Splits graph samples into train, validation, and test sets."""

    data_list = data_list.copy()

    rng = random.Random(seed)
    rng.shuffle(data_list)

    n = len(data_list)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    n_train = n - n_val - n_test

    train_list = data_list[:n_train]
    val_list = data_list[n_train:n_train + n_val]
    test_list = data_list[n_train + n_val:]

    return train_list, val_list, test_list


def edge_diff(edge_a: torch.Tensor, edge_b: torch.Tensor) -> torch.Tensor:
    """Returns edges in edge_a that are not in edge_b."""

    set_a = set(map(tuple, edge_a.t().tolist()))
    set_b = set(map(tuple, edge_b.t().tolist()))

    diff = list(set_a - set_b)

    if len(diff) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=edge_a.device)

    return torch.tensor(
        diff,
        dtype=torch.long,
        device=edge_a.device,
    ).t()


def edge_sort(edges: torch.Tensor) -> torch.Tensor:
    """Sorts edges lexicographically by source and destination."""

    if edges.numel() == 0:
        return edges

    src = edges[0]
    dst = edges[1]

    max_node = int(edges.max()) + 1
    key = src * max_node + dst

    order = torch.argsort(key)

    return edges[:, order]


def compute_pna_degree_histogram(data_list: list[Data]) -> torch.Tensor:
    """Computes the degree histogram required by PNAConv."""

    max_degree = 0

    for data in data_list:
        d = degree(
            data.edge_index[1],
            num_nodes=data.num_nodes,
            dtype=torch.long,
        )
        max_degree = max(max_degree, int(d.max()))

    deg = torch.zeros(max_degree + 1, dtype=torch.long)

    for data in data_list:
        d = degree(
            data.edge_index[1],
            num_nodes=data.num_nodes,
            dtype=torch.long,
        )
        deg += torch.bincount(d, minlength=deg.numel())

    return deg


def save_encoder(
    model,
    path: str,
    config: dict,
    size_name: str,
    epoch: int,
    val_metrics: dict,
    in_channels: int,
    deg: torch.Tensor | None = None,
) -> None:
    """Saves encoder weights together with config and validation metrics."""

    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "encoder_state_dict": model.encoder.state_dict(),
        "config": {
            "model": config["model"],
            "gnn_type": config["gnn_type"],
            "in_dim": int(in_channels),
            "hidden_dim": int(config["hidden_channels"]),
            "latent_dim": int(config["latent_channels"]),
            "batch_size": int(config["batch_size"]),
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "epochs": int(config["epochs"]),
            "size": size_name,
        },
        "metrics": {
            "epoch": int(epoch),
            "val_loss": float(val_metrics["loss"]),
            "val_auc": float(val_metrics["auc"]),
            "val_ap": float(val_metrics["ap"]),
        },
        "deg": deg.cpu() if deg is not None else None,
    }

    torch.save(ckpt, path)


def clb(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Computes normalized critical lower bound features."""

    durations = X[:, -1].astype(np.float32)
    edge_mask = A > 0
    indegree = edge_mask.sum(axis=0).astype(int)

    queue = list(np.where(indegree == 0)[0])
    values = durations.copy()

    for u in queue:
        for v in np.where(edge_mask[u])[0]:
            values[v] = max(values[v], values[u] + durations[v])
            indegree[v] -= 1

            if indegree[v] == 0:
                queue.append(v)

    if len(queue) != A.shape[0]:
        raise ValueError("A has a cycle. CLB requires a DAG.")

    values = values / values.max()

    return values.reshape(-1, 1)


def build_allowed_edge_index(
    num_nodes: int,
    machine: torch.Tensor,
    precedence_edge_index: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    """
    Builds all candidate reconstruction edges:
    same-machine directed pairs plus precedence edges.
    """

    device = machine.device

    nodes = torch.arange(num_nodes, device=device)
    uu, vv = torch.meshgrid(nodes, nodes, indexing="ij")

    same_machine = (
        (batch[uu] == batch[vv])
        & (machine[uu] == machine[vv])
        & (uu != vv)
    )

    same_machine_edges = torch.stack(
        [uu[same_machine], vv[same_machine]],
        dim=0,
    )

    allowed = torch.cat(
        [same_machine_edges, precedence_edge_index.to(device)],
        dim=1,
    )

    return torch.unique(allowed, dim=1)