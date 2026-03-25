import argparse
import json
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE, VGAE

from encoder import Encoder, VariationalEncoder
from loss import build_allowed_edge_index
from utils import npz_to_data_list, split_for_link_pred, edge_sort, edge_diff, compute_pna_degree_histogram
import wandb


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_loaders(npz_path, batch_size, val_ratio, test_ratio):
    data_list = npz_to_data_list(npz_path)
    train_list, val_list, test_list = split_for_link_pred(
        data_list,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_list, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)

    return train_list, val_list, test_list, train_loader, val_loader, test_loader

def train_epoch(model, loader, opt, vgae):
    model.train()
    total_loss, n = 0.0, 0

    for d in loader:
        d = d.to(device)
        opt.zero_grad()
        
        z = model.encode(d.x, d.edge_index)

        allowed_edge_index = build_allowed_edge_index(
            num_nodes=d.num_nodes,
            machine=d.machine,
            precedence_edge_index=d.precedence_edge_index,
            batch=d.batch,
        )

        neg_edge_index = edge_diff(allowed_edge_index, d.edge_index)
        neg_edge_index = edge_sort(neg_edge_index)

        loss = model.recon_loss(z, d.edge_index, neg_edge_index)
        if vgae:
            loss = loss + model.kl_loss() / d.num_nodes
        loss.backward()
        opt.step()

        total_loss += float(loss.detach())
        n += 1

    return total_loss / max(n, 1)

@torch.no_grad()
def eval_loader(model, loader, vgae):
    model.eval()

    aucs, aps, losses = [], [], []
    mean_pos_list, mean_neg_list = [], []
    mean_prec_list, mean_nonprec_list = [], []

    for d in loader:
        d = d.to(device)
        T = int(d.num_nodes)

        z = model.encode(d.x, d.edge_index)

        allowed_edge_index = build_allowed_edge_index(
            num_nodes=T,
            machine=d.machine,
            precedence_edge_index=d.precedence_edge_index,
            batch=d.batch,
        )
        allowed_edge_index = edge_sort(allowed_edge_index).to(device)

        pos_edge_index = edge_sort(d.edge_index).to(device)
        neg_edge_index = edge_sort(edge_diff(allowed_edge_index, pos_edge_index)).to(device)

        loss = model.recon_loss(z, pos_edge_index, neg_edge_index)
        if vgae:
            loss = loss + model.kl_loss() / d.num_nodes
        losses.append(float(loss.detach()))

        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)

        y = torch.cat([
            torch.ones(pos_edge_index.size(1), device=device),
            torch.zeros(neg_edge_index.size(1), device=device),
        ], dim=0)

        probs = model.decoder(z, edge_index).view(-1)

        y_np = y.cpu().numpy()
        p_np = probs.cpu().numpy()

        aucs.append(roc_auc_score(y_np, p_np))
        aps.append(average_precision_score(y_np, p_np))

        num_pos = int(pos_edge_index.size(1))
        mean_pos = float(probs[:num_pos].mean().item()) if num_pos > 0 else float("nan")
        mean_neg = float(probs[num_pos:].mean().item()) if neg_edge_index.numel() > 0 else float("nan")

        prec_edge_index = edge_sort(d.precedence_edge_index).to(device)

        pos_ids = pos_edge_index[0] * T + pos_edge_index[1]
        prec_ids = prec_edge_index[0] * T + prec_edge_index[1]

        prec_in_pos_mask = torch.isin(prec_ids, pos_ids)
        prec_in_pos = prec_edge_index[:, prec_in_pos_mask]

        nonprec_pos_mask = ~torch.isin(pos_ids, prec_ids)
        nonprec_pos = pos_edge_index[:, nonprec_pos_mask]

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

        mean_pos_list.append(mean_pos)
        mean_neg_list.append(mean_neg)
        mean_prec_list.append(mean_prec)
        mean_nonprec_list.append(mean_nonprec)

    return {
        "loss": float(np.mean(losses)),
        "auc": float(np.mean(aucs)),
        "ap": float(np.mean(aps)),
        "mean_p_true": float(np.nanmean(mean_pos_list)),
        "mean_p_neg": float(np.nanmean(mean_neg_list)),
        "mean_p_precedence_true": float(np.nanmean(mean_prec_list)),
        "mean_p_true_nonprecedence": float(np.nanmean(mean_nonprec_list)),
    }
def save_encoder(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.encoder.state_dict(), path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    default_config = {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "hidden_channels": 64,
        "latent_channels": 32,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "eval_every": 10,
        "model": "gae",
        "gnn_type": "gcn"
    }

    if args.use_wandb:
        wandb.init(
            project="jsp-gae",
            config=default_config,
        )
        config = wandb.config
        run_name = f"{config.model}_lr{config.learning_rate}_bs{config.batch_size}"
        wandb.run.name = run_name

        wandb.config.update({"device": str(device)})
    else:
        config = default_config

    out_dir = os.path.join("runs", args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dict(config), f, indent=2)

    train_list, val_list, test_list, train_loader, val_loader, test_loader = build_loaders(
        npz_path=args.data_path,
        batch_size=config["batch_size"],
        val_ratio=config["val_ratio"],
        test_ratio=config["test_ratio"],
    )

    in_channels = train_list[0].x.size(-1)
    vgae = True if config["model"]=="vgae" else False
    deg = compute_pna_degree_histogram(train_list)
    if config["model"]=="gae":
        model = GAE(
            Encoder(
                in_channels=in_channels,
                hidden_channels=config["hidden_channels"],
                out_channels=config["latent_channels"],
                gnn_type=config["gnn_type"],
                deg=deg
            )
        ).to(device)
    
    elif config["model"] == "vgae":
        model = VGAE(
            VariationalEncoder(
                in_channels=in_channels,
                hidden_channels=config["hidden_channels"],
                out_channels=config["latent_channels"],
                gnn_type=config["gnn_type"],
                deg=deg
            )
        ).to(device)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    best_val_auc = -float("inf")
    best_val_ap = -float("inf")

    for epoch in range(1, config["epochs"] + 1):

        train_loss = train_epoch(model, train_loader, opt, vgae)
        
        log = {
            "epoch": epoch,
            "train/loss": train_loss,
        }

        if epoch % config["eval_every"] == 0:
            val_metrics = eval_loader(model, val_loader, vgae)
            log.update({
                "val/loss": val_metrics["loss"],
                "val/auc": val_metrics["auc"],
                "val/ap": val_metrics["ap"],
                "val/mean_p_true": val_metrics["mean_p_true"],
                "val/mean_p_neg": val_metrics["mean_p_neg"],
                "val/mean_p_precedence_true": val_metrics["mean_p_precedence_true"],
                "val/mean_p_true_nonprecedence": val_metrics["mean_p_true_nonprecedence"],
            })

            # save best by val_ap
            if val_metrics["ap"] > best_val_ap:
                best_val_ap = val_metrics["ap"]
                #save_encoder(model, os.path.join(out_dir, "encoder_best_ap.pt"))
                log["best/val_ap"] = best_val_ap
                path = os.path.join(out_dir, "encoder_best.pt")
                save_encoder(model, path)

                if args.use_wandb:
                    artifact = wandb.Artifact(
                        name=f"encoder-best-{args.run_name}",
                        type="model"
                    )
                    artifact.add_file(path)
                    wandb.log_artifact(artifact)

        print(log)

        if args.use_wandb:
            wandb.log(log)

    test_metrics = eval_loader(model, test_loader, vgae)
    test_log = {
        "test/loss": test_metrics["loss"],
        "test/auc": test_metrics["auc"],
        "test/ap": test_metrics["ap"],
        "test/mean_p_true": test_metrics["mean_p_true"],
        "test/mean_p_neg": test_metrics["mean_p_neg"],
        "test/mean_p_precedence_true": test_metrics["mean_p_precedence_true"],
        "test/mean_p_true_nonprecedence": test_metrics["mean_p_true_nonprecedence"],
    }

    print(test_log)

    if args.use_wandb:
        wandb.log({
            **test_log,
            "final/model_path_saved": 1,
        })
        wandb.finish()


if __name__ == "__main__":
    main()