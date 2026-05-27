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
from utils import edge_sort, edge_diff, compute_pna_degree_histogram, save_encoder
import wandb

from dataset import build_loaders
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


"""def save_encoder(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.encoder.state_dict(), path)"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_10", type=str, required=True)
    parser.add_argument("--data_15", type=str, required=True)
    parser.add_argument("--data_20", type=str, required=True)
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
        "eval_every": 5,
        "model": "gae",
        "gnn_type": "gcn"
    }

    if args.use_wandb:
        wandb.init(
            project="jsp-gae",
            config=default_config,
        )
        config = dict(wandb.config)
        run_name = f"{config['model']}_{config['gnn_type']}"
        wandb.run.name = run_name

        wandb.config.update({"device": str(device)})
    else:
        config = default_config

    out_dir = os.path.join("runs", args.run_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(dict(config), f, indent=2)

    data_paths = {
        "10x10": args.data_10,
        "15x15": args.data_15,
        "20x20": args.data_20,
    }

    vgae = True if config["model"] == "vgae" else False

    all_test_losses = []
    all_test_aucs = []
    all_test_aps = []
    all_test_mean_p_true = []
    all_test_mean_p_neg = []
    all_test_mean_p_prec = []
    all_test_mean_p_nonprec = []

    for size_name, data_path in data_paths.items():
        print(f"\n===== Training size {size_name} =====")

        train_set, val_set, test_set, train_loader, val_loader, test_loader = build_loaders(
            data_path,
            config["batch_size"],
            config["val_ratio"],
            config["test_ratio"],
        )

        in_channels = train_set[0].x.size(-1)
        deg = compute_pna_degree_histogram(train_set)

        if config["model"] == "gae":
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
                "size": size_name,
                "epoch": epoch,
                f"train/{size_name}/loss": train_loss,
            }

            if epoch % config["eval_every"] == 0:
                val_metrics = eval_loader(model, val_loader, vgae)

                log.update({
                    f"val/{size_name}/loss": val_metrics["loss"],
                    f"val/{size_name}/auc": val_metrics["auc"],
                    f"val/{size_name}/ap": val_metrics["ap"],
                })

                if val_metrics["ap"] > best_val_ap:
                    best_val_ap = val_metrics["ap"]
                    log[f"best/{size_name}/val_ap"] = best_val_ap

                    file_name = f"{config['model']}_{config['gnn_type']}_{size_name}.pt"
                    path = os.path.join(out_dir, file_name)

                    save_encoder(
                        model=model,
                        path=path,
                        config=config,
                        size_name=size_name,
                        epoch=epoch,
                        val_metrics=val_metrics,
                        in_channels=in_channels,
                        deg=deg,
                    )

                    if args.use_wandb:
                        artifact_name = f"{config['model']}-{config['gnn_type']}-{size_name}"

                        artifact = wandb.Artifact(
                            name=artifact_name,
                            type="encoder",
                            metadata={
                                "model": config["model"],
                                "gnn_type": config["gnn_type"],
                                "size": size_name,
                                "hidden_channels": config["hidden_channels"],
                                "latent_channels": config["latent_channels"],
                                "batch_size": config["batch_size"],
                                "weight_decay": config["weight_decay"],
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

            print(log)

            if args.use_wandb:
                wandb.log(log)

        test_metrics = eval_loader(model, test_loader, vgae)
        test_log = {
            f"test/{size_name}/loss": test_metrics["loss"],
            f"test/{size_name}/auc": test_metrics["auc"],
            f"test/{size_name}/ap": test_metrics["ap"],
            f"test/{size_name}/mean_p_true": test_metrics["mean_p_true"],
            f"test/{size_name}/mean_p_neg": test_metrics["mean_p_neg"],
            f"test/{size_name}/mean_p_precedence_true": test_metrics["mean_p_precedence_true"],
            f"test/{size_name}/mean_p_true_nonprecedence": test_metrics["mean_p_true_nonprecedence"],
        }
        all_test_losses.append(test_metrics["loss"])
        all_test_aucs.append(test_metrics["auc"])
        all_test_aps.append(test_metrics["ap"])
        all_test_mean_p_true.append(test_metrics["mean_p_true"])
        all_test_mean_p_neg.append(test_metrics["mean_p_neg"])
        all_test_mean_p_prec.append(test_metrics["mean_p_precedence_true"])
        all_test_mean_p_nonprec.append(test_metrics["mean_p_true_nonprecedence"])

        print(test_log)

        if args.use_wandb:
            wandb.log({
                **test_log,
                "final/model_path_saved": 1,
            })

        del model, opt, train_set, val_set, test_set, train_loader, val_loader, test_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_log = {
        "test/loss_average": float(np.mean(all_test_losses)),
        "test/auc_average": float(np.mean(all_test_aucs)),
        "test/ap_average": float(np.mean(all_test_aps)),
        "test/mean_p_true_average": float(np.mean(all_test_mean_p_true)),
        "test/mean_p_neg_average": float(np.mean(all_test_mean_p_neg)),
        "test/mean_p_precedence_true_average": float(np.mean(all_test_mean_p_prec)),
        "test/mean_p_true_nonprecedence_average": float(np.mean(all_test_mean_p_nonprec)),
    }

    print(avg_log)

    if args.use_wandb:
        wandb.log(avg_log)
        wandb.finish()


if __name__ == "__main__":
    main()
