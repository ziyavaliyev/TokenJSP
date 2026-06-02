"""
Train GAE/VGAE encoders on one or more JSP graph datasets.

The script reads a YAML config, trains one independent encoder per dataset,
evaluates reconstruction quality, and optionally logs results/artifacts to W&B.
"""

import argparse
import gc
import json
import os

import numpy as np
import torch
import wandb
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.nn import GAE, VGAE

from tokenjsp.dataset import build_loaders
from tokenjsp.encoder import Encoder, VariationalEncoder
from tokenjsp.utils import (
    build_allowed_edge_index,
    compute_pna_degree_histogram,
    edge_diff,
    edge_sort,
    save_encoder,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_data_paths(config: dict) -> dict:
    """
    Reads all entries under config['data'] whose keys start with 'data_'.

    Example:
        data_10 -> size name '10'
        data_10x10 -> size name '10x10'
        data_ft06 -> size name 'ft06'
    """

    data_paths = {}

    for key, value in config["data"].items():
        if key.startswith("data_"):
            size_name = key.replace("data_", "")
            data_paths[size_name] = value

    if not data_paths:
        raise ValueError("No dataset paths found. Expected keys like data_10 or data_20x20.")

    return data_paths


def init_wandb(config: dict) -> dict:
    wandb.init(
        project=config["logging"]["project"],
        name=config["logging"]["run_name"],
        config=config,
    )

    config = dict(wandb.config)
    wandb.config.update({"device": str(DEVICE)})

    return config


def build_model(config: dict, in_channels: int, deg: torch.Tensor):
    model_type = config["model"]["model"]

    if model_type == "gae":
        return GAE(
            Encoder(
                in_channels=in_channels,
                hidden_channels=config["model"]["hidden_channels"],
                out_channels=config["model"]["latent_channels"],
                gnn_type=config["model"]["gnn_type"],
                deg=deg,
            )
        ).to(DEVICE)

    if model_type == "vgae":
        return VGAE(
            VariationalEncoder(
                in_channels=in_channels,
                hidden_channels=config["model"]["hidden_channels"],
                out_channels=config["model"]["latent_channels"],
                gnn_type=config["model"]["gnn_type"],
                deg=deg,
            )
        ).to(DEVICE)

    raise ValueError(f"Unknown model type: {model_type}")


def train_epoch(model, loader, optimizer, use_vgae: bool) -> float:
    model.train()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        z = model.encode(batch.x, batch.edge_index)

        allowed_edge_index = build_allowed_edge_index(
            num_nodes=batch.num_nodes,
            machine=batch.machine,
            precedence_edge_index=batch.precedence_edge_index,
            batch=batch.batch,
        )

        neg_edge_index = edge_sort(edge_diff(allowed_edge_index, batch.edge_index))

        loss = model.recon_loss(z, batch.edge_index, neg_edge_index)

        if use_vgae:
            loss = loss + model.kl_loss() / batch.num_nodes

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach())
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def eval_loader(model, loader, use_vgae: bool) -> dict:
    model.eval()

    aucs, aps, losses = [], [], []
    mean_pos_list, mean_neg_list = [], []
    mean_prec_list, mean_nonprec_list = [], []

    for batch in loader:
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

        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)

        if use_vgae:
            loss = loss + model.kl_loss() / batch.num_nodes

        losses.append(float(loss.detach()))

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

        labels = torch.cat(
            [
                torch.ones(pos_edge_index.size(1), device=DEVICE),
                torch.zeros(neg_edge_index.size(1), device=DEVICE),
            ],
            dim=0,
        )

        probs = model.decoder(z, edge_index).view(-1)

        aucs.append(roc_auc_score(labels.cpu().numpy(), probs.cpu().numpy()))
        aps.append(average_precision_score(labels.cpu().numpy(), probs.cpu().numpy()))

        num_pos = int(pos_edge_index.size(1))

        mean_pos_list.append(float(probs[:num_pos].mean().item()))

        if neg_edge_index.numel() > 0:
            mean_neg_list.append(float(probs[num_pos:].mean().item()))

        prec_edge_index = edge_sort(batch.precedence_edge_index).to(DEVICE)

        pos_ids = pos_edge_index[0] * num_nodes + pos_edge_index[1]
        prec_ids = prec_edge_index[0] * num_nodes + prec_edge_index[1]

        prec_in_pos = prec_edge_index[:, torch.isin(prec_ids, pos_ids)]
        nonprec_pos = pos_edge_index[:, ~torch.isin(pos_ids, prec_ids)]

        if prec_in_pos.numel() > 0:
            p_prec = model.decoder(z, prec_in_pos).view(-1)
            mean_prec_list.append(float(p_prec.mean().item()))

        if nonprec_pos.numel() > 0:
            p_nonprec = model.decoder(z, nonprec_pos).view(-1)
            mean_nonprec_list.append(float(p_nonprec.mean().item()))

    return {
        "loss": float(np.mean(losses)),
        "auc": float(np.mean(aucs)),
        "ap": float(np.mean(aps)),
        "mean_p_true": float(np.nanmean(mean_pos_list)),
        "mean_p_neg": float(np.nanmean(mean_neg_list)),
        "mean_p_precedence_true": float(np.nanmean(mean_prec_list)),
        "mean_p_true_nonprecedence": float(np.nanmean(mean_nonprec_list)),
    }


def log_encoder_artifact(path, file_name, config, size_name, epoch, val_metrics):
    artifact = wandb.Artifact(
        name=f"{config['model']['model']}-{config['model']['gnn_type']}-{size_name}",
        type="encoder",
        metadata={
            "model": config["model"]["model"],
            "gnn_type": config["model"]["gnn_type"],
            "size": size_name,
            "hidden_channels": config["model"]["hidden_channels"],
            "latent_channels": config["model"]["latent_channels"],
            "batch_size": config["training"]["batch_size"],
            "weight_decay": config["training"]["weight_decay"],
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_ap": val_metrics["ap"],
        },
    )

    artifact.add_file(path, name=file_name)

    wandb.log_artifact(
        artifact,
        aliases=[
            "best",
            f"best-{size_name}",
            f"epoch-{epoch}",
            f"ap-{val_metrics['ap']:.4f}",
        ],
    )


def save_best_encoder(model, config, out_dir, size_name, epoch, val_metrics, in_channels, deg):
    file_name = f"{config['model']['model']}_{config['model']['gnn_type']}_{size_name}.pt"
    path = os.path.join(out_dir, file_name)

    save_encoder(
        model=model,
        path=path,
        config={
            "model": config["model"]["model"],
            "gnn_type": config["model"]["gnn_type"],
            "hidden_channels": config["model"]["hidden_channels"],
            "latent_channels": config["model"]["latent_channels"],
            "batch_size": config["training"]["batch_size"],
            "learning_rate": config["training"]["learning_rate"],
            "weight_decay": config["training"]["weight_decay"],
            "epochs": config["training"]["epochs"],
        },
        size_name=size_name,
        epoch=epoch,
        val_metrics=val_metrics,
        in_channels=in_channels,
        deg=deg,
    )

    if config["logging"]["use_wandb"]:
        log_encoder_artifact(
            path=path,
            file_name=file_name,
            config=config,
            size_name=size_name,
            epoch=epoch,
            val_metrics=val_metrics,
        )


def train_single_dataset(size_name: str, data_path: str, config: dict, out_dir: str) -> dict:
    print(f"\n===== Training size {size_name} =====")

    train_set, val_set, test_set, train_loader, val_loader, test_loader = build_loaders(
        data_path,
        config["training"]["batch_size"],
        config["training"]["val_ratio"],
        config["training"]["test_ratio"],
    )

    in_channels = train_set[0].x.size(-1)
    deg = compute_pna_degree_histogram(train_set)

    use_vgae = config["model"]["model"] == "vgae"
    model = build_model(config, in_channels, deg)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    best_val_ap = -float("inf")

    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, use_vgae)

        log = {
            "size": size_name,
            "epoch": epoch,
            f"train/{size_name}/loss": train_loss,
        }

        if epoch % config["training"]["eval_every"] == 0:
            val_metrics = eval_loader(model, val_loader, use_vgae)

            log.update(
                {
                    f"val/{size_name}/loss": val_metrics["loss"],
                    f"val/{size_name}/auc": val_metrics["auc"],
                    f"val/{size_name}/ap": val_metrics["ap"],
                }
            )

            if val_metrics["ap"] > best_val_ap:
                best_val_ap = val_metrics["ap"]
                log[f"best/{size_name}/val_ap"] = best_val_ap

                save_best_encoder(
                    model=model,
                    config=config,
                    out_dir=out_dir,
                    size_name=size_name,
                    epoch=epoch,
                    val_metrics=val_metrics,
                    in_channels=in_channels,
                    deg=deg,
                )

        print(log)

        if config["logging"]["use_wandb"]:
            wandb.log(log)

    test_metrics = eval_loader(model, test_loader, use_vgae)

    test_log = {
        f"test/{size_name}/loss": test_metrics["loss"],
        f"test/{size_name}/auc": test_metrics["auc"],
        f"test/{size_name}/ap": test_metrics["ap"],
        f"test/{size_name}/mean_p_true": test_metrics["mean_p_true"],
        f"test/{size_name}/mean_p_neg": test_metrics["mean_p_neg"],
        f"test/{size_name}/mean_p_precedence_true": test_metrics["mean_p_precedence_true"],
        f"test/{size_name}/mean_p_true_nonprecedence": test_metrics["mean_p_true_nonprecedence"],
    }

    print(test_log)

    if config["logging"]["use_wandb"]:
        wandb.log({**test_log, "final/model_path_saved": 1})

    del model, optimizer, train_set, val_set, test_set, train_loader, val_loader, test_loader
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return test_metrics


def summarize_test_metrics(metrics: list[dict]) -> dict:
    return {
        "test/loss_average": float(np.mean([m["loss"] for m in metrics])),
        "test/auc_average": float(np.mean([m["auc"] for m in metrics])),
        "test/ap_average": float(np.mean([m["ap"] for m in metrics])),
        "test/mean_p_true_average": float(np.mean([m["mean_p_true"] for m in metrics])),
        "test/mean_p_neg_average": float(np.mean([m["mean_p_neg"] for m in metrics])),
        "test/mean_p_precedence_true_average": float(
            np.mean([m["mean_p_precedence_true"] for m in metrics])
        ),
        "test/mean_p_true_nonprecedence_average": float(
            np.mean([m["mean_p_true_nonprecedence"] for m in metrics])
        ),
    }


if __name__ == "__main__":
    # Read the config
    args = parse_args()
    config = load_yaml(args.config)

    # Collect all datasets from config["data"]
    data_paths = get_data_paths(config)

    # Initialize W&B
    if config["logging"]["use_wandb"]:
        config = init_wandb(config)

    # Create output directory and save the used config.
    out_dir = os.path.join("runs", config["logging"]["run_name"])
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Train one independent encoder for each dataset.
    all_test_metrics = []

    for size_name, data_path in data_paths.items():
        test_metrics = train_single_dataset(
            size_name=size_name,
            data_path=data_path,
            config=config,
            out_dir=out_dir,
        )
        all_test_metrics.append(test_metrics)

    # Report average test metrics across all datasets.
    avg_log = summarize_test_metrics(all_test_metrics)
    print(avg_log)

    if config["logging"]["use_wandb"]:
        wandb.log(avg_log)
        wandb.finish()