"""
Evaluate a trained GAE/VGAE encoder on a JSP graph dataset.

The script reports global reconstruction metrics and grouped metrics by the
number of positive edges in each graph.
"""

import argparse
import json
import os
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE, VGAE
from tokenjsp.encoder import Encoder, VariationalEncoder
from tokenjsp.utils import build_allowed_edge_index, edge_diff, edge_sort, npz_to_data_list


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)

    parser.add_argument("--model", type=str, choices=["gae", "vgae"], default="gae")
    parser.add_argument("--gnn_type", type=str, default="gcn")
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--latent_channels", type=int, default=32)

    return parser.parse_args()


def build_model(args, in_channels, deg=None):
    if args.model == "gae":
        encoder = Encoder(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=args.latent_channels,
            gnn_type=args.gnn_type,
            deg=deg,
        )
        return GAE(encoder).to(DEVICE)

    if args.model == "vgae":
        encoder = VariationalEncoder(
            in_channels=in_channels,
            hidden_channels=args.hidden_channels,
            out_channels=args.latent_channels,
            gnn_type=args.gnn_type,
            deg=deg,
        )
        return VGAE(encoder).to(DEVICE)

    raise ValueError(f"Unknown model type: {args.model}")


def load_encoder_weights(model, weights_path):
    state = torch.load(weights_path, map_location=DEVICE)

    if isinstance(state, dict) and "encoder_state_dict" in state:
        state = state["encoder_state_dict"]

    model.encoder.load_state_dict(state)
    model.eval()

    print(f"Loaded encoder weights from: {weights_path}")


@torch.no_grad()
def evaluate_batch(model, batch):
    batch = batch.to(DEVICE)
    num_nodes = int(batch.num_nodes)

    z = model.encode(batch.x, batch.edge_index)

    allowed_edge_index = build_allowed_edge_index(
        num_nodes=num_nodes,
        machine=batch.machine,
        precedence_edge_index=batch.precedence_edge_index,
        batch=batch.batch,
    )

    pos_edge_index = edge_sort(batch.edge_index)
    neg_edge_index = edge_sort(edge_diff(allowed_edge_index, pos_edge_index))

    edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

    labels = torch.cat(
        [
            torch.ones(pos_edge_index.size(1), device=DEVICE),
            torch.zeros(neg_edge_index.size(1), device=DEVICE),
        ],
        dim=0,
    )

    probs = model.decoder(z, edge_index).view(-1)

    y_np = labels.cpu().numpy()
    p_np = probs.cpu().numpy()

    auc = roc_auc_score(y_np, p_np)
    ap = average_precision_score(y_np, p_np)

    num_pos = int(pos_edge_index.size(1))
    mean_pos = float(probs[:num_pos].mean().item())
    mean_neg = float(probs[num_pos:].mean().item()) if neg_edge_index.numel() > 0 else float("nan")

    prec_edge_index = edge_sort(batch.precedence_edge_index).to(DEVICE)

    pos_ids = pos_edge_index[0] * num_nodes + pos_edge_index[1]
    prec_ids = prec_edge_index[0] * num_nodes + prec_edge_index[1]

    prec_in_pos = prec_edge_index[:, torch.isin(prec_ids, pos_ids)]
    nonprec_pos = pos_edge_index[:, ~torch.isin(pos_ids, prec_ids)]

    if prec_in_pos.numel() > 0:
        p_prec = model.decoder(z, prec_in_pos).view(-1)
        mean_prec = float(p_prec.mean().item())
    else:
        mean_prec = float("nan")

    if nonprec_pos.numel() > 0:
        p_nonprec = model.decoder(z, nonprec_pos).view(-1)
        mean_nonprec = float(p_nonprec.mean().item())
    else:
        mean_nonprec = float("nan")

    return {
        "num_pos": num_pos,
        "auc": auc,
        "ap": ap,
        "mean_pos": mean_pos,
        "mean_neg": mean_neg,
        "mean_prec": mean_prec,
        "mean_nonprec": mean_nonprec,
    }


def update_groups(groups, metrics):
    num_pos = metrics["num_pos"]

    if num_pos not in groups:
        groups[num_pos] = {
            "auc": [],
            "ap": [],
            "mean_pos": [],
            "mean_neg": [],
            "mean_prec": [],
            "mean_nonprec": [],
            "count": 0,
        }

    group = groups[num_pos]

    group["auc"].append(metrics["auc"])
    group["ap"].append(metrics["ap"])
    group["mean_pos"].append(metrics["mean_pos"])
    group["mean_neg"].append(metrics["mean_neg"])
    group["mean_prec"].append(metrics["mean_prec"])
    group["mean_nonprec"].append(metrics["mean_nonprec"])
    group["count"] += 1


def summarize_global(metrics_list):
    return {
        "num_graphs": int(len(metrics_list)),
        "mean_auc": float(np.mean([m["auc"] for m in metrics_list])),
        "mean_ap": float(np.mean([m["ap"] for m in metrics_list])),
        "mean_p_true": float(np.nanmean([m["mean_pos"] for m in metrics_list])),
        "mean_p_neg": float(np.nanmean([m["mean_neg"] for m in metrics_list])),
        "mean_p_precedence_true": float(np.nanmean([m["mean_prec"] for m in metrics_list])),
        "mean_p_true_nonprecedence": float(np.nanmean([m["mean_nonprec"] for m in metrics_list])),
    }


def summarize_groups(groups):
    grouped_summary = {}

    for num_pos, values in sorted(groups.items(), key=lambda kv: kv[0]):
        grouped_summary[int(num_pos)] = {
            "count": int(values["count"]),
            "mean_auc": float(np.mean(values["auc"])),
            "mean_ap": float(np.mean(values["ap"])),
            "mean_p_true": float(np.nanmean(values["mean_pos"])),
            "mean_p_neg": float(np.nanmean(values["mean_neg"])),
            "mean_p_precedence_true": float(np.nanmean(values["mean_prec"])),
            "mean_p_true_nonprecedence": float(np.nanmean(values["mean_nonprec"])),
        }

    return grouped_summary


def print_results(results):
    print("\nGlobal Results")
    print(results["global"])

    print("\nGrouped by #positive_edges")

    for num_pos, values in results["by_num_positive_edges"].items():
        print(
            f"#pos={num_pos:4d} | n={values['count']:4d} | "
            f"AUC={values['mean_auc']:.4f} | AP={values['mean_ap']:.4f} | "
            f"p(true)={values['mean_p_true']:.4f} | "
            f"p(neg)={values['mean_p_neg']:.4f} | "
            f"p(prec_true)={values['mean_p_precedence_true']:.4f} | "
            f"p(nonprec_true)={values['mean_p_true_nonprecedence']:.4f}"
        )


def save_results(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved grouped metrics to: {out_path}")


if __name__ == "__main__":
    args = parse_args()

    # Load JSP graph states and create a PyG dataloader.
    data_list = npz_to_data_list(args.data)
    loader = DataLoader(data_list, batch_size=1, shuffle=False)

    # Build the selected GAE/VGAE model.
    in_channels = data_list[0].x.size(-1)
    model = build_model(args, in_channels=in_channels)

    # Load trained encoder weights.
    load_encoder_weights(model, args.weights)

    # Evaluate all graphs and group results by number of positive edges.
    metrics_list = []
    groups = {}

    for batch in loader:
        metrics = evaluate_batch(model, batch)
        metrics_list.append(metrics)
        update_groups(groups, metrics)

    # Summarize, print, and save results.
    results = {
        "global": summarize_global(metrics_list),
        "by_num_positive_edges": summarize_groups(groups),
    }

    print_results(results)
    save_results(results, args.out)