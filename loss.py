import torch
import torch.nn.functional as F
from utils import edge_diff, edge_sort

def build_allowed_edge_index(num_nodes: int, machine: torch.Tensor, precedence_edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    Allowed candidates = all directed pairs within same machine (u!=v)
    plus precedence edges. Returns (2, E_allowed).
    """
    device = machine.device
    T = num_nodes

    # same-machine all directed pairs (u!=v)
    u = torch.arange(T, device=device)
    uu, vv = torch.meshgrid(u, u, indexing="ij")  # (T,T)
    same = (batch[uu] == batch[vv]) & (machine[uu] == machine[vv]) & (uu != vv)
    same_pairs = torch.stack([uu[same], vv[same]], dim=0)  # (2, Esame)

    # concat + unique
    allowed = torch.cat([same_pairs, precedence_edge_index.to(device)], dim=1)

    return allowed

def masked_recon_loss(model, z: torch.Tensor, pos_edge_index: torch.Tensor, allowed_edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:

    neg_edge_index = edge_diff(allowed_edge_index, pos_edge_index)
    neg_edge_index = edge_sort(neg_edge_index)
    edge_index = torch.cat([pos_edge_index, neg_edge_index], axis=1)

    y = torch.cat([
        torch.ones(pos_edge_index.size(1), device=z.device),
        torch.zeros(neg_edge_index.size(1), device=z.device),
    ], dim=0)

    logits = model.decoder(z, edge_index)  # logits
    return F.binary_cross_entropy_with_logits(logits, y)