import torch
from torch_geometric.nn import GAE
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import json

import config_gae as C #TODO: What to do with configs
from src.encoder import Encoder
from src.utils import npz_to_data_list, edge_diff, edge_sort
from src.loss import build_allowed_edge_index

PATH = "weights/encoder_10x10_baseline.pt"
OUT_PATH = "10x10_baseline.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_list = npz_to_data_list(C.NPZ_PATH)
loader = DataLoader(data_list, batch_size=1, shuffle=False)

in_channels = data_list[0].x.size(-1)
encoder = Encoder(in_channels, C.HIDDEN_CHANNELS, C.LATENT_CHANNELS)
model = GAE(encoder).to(device)

state = torch.load(PATH, map_location=device)
model.encoder.load_state_dict(state)
model.eval()
print("Encoder loaded successfully.")

all_auc, all_ap = [], []
all_mean_pos, all_mean_neg = [], []
all_mean_prec, all_mean_nonprec = [], []

groups = {} 

with torch.no_grad():
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
        allowed_edge_index = edge_sort(allowed_edge_index)
        pos_edge_index = edge_sort(d.edge_index)
        neg_edge_index = edge_sort(edge_diff(allowed_edge_index, pos_edge_index))

        # Candidate set = pos first, then neg  -> aligns y with edge_index
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
        y = torch.cat(
            [
                torch.ones(pos_edge_index.size(1), device=device),
                torch.zeros(neg_edge_index.size(1), device=device),
            ],
            dim=0,
        )

        probs = model.decoder(z, edge_index).view(-1)

        # Metrics
        y_np = y.cpu().numpy()
        p_np = probs.cpu().numpy()
        auc = roc_auc_score(y_np, p_np)
        ap = average_precision_score(y_np, p_np)

        num_pos = int(pos_edge_index.size(1))
        mean_pos = float(probs[:num_pos].mean().item()) if num_pos > 0 else float("nan")
        mean_neg = float(probs[num_pos:].mean().item()) if neg_edge_index.numel() > 0 else float("nan")

        prec_edge_index = edge_sort(d.precedence_edge_index)

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

        all_auc.append(auc)
        all_ap.append(ap)
        all_mean_pos.append(mean_pos)
        all_mean_neg.append(mean_neg)
        all_mean_prec.append(mean_prec)
        all_mean_nonprec.append(mean_nonprec)

        if num_pos not in groups:
            groups[num_pos] = {
                "auc": [], "ap": [],
                "mean_pos": [], "mean_neg": [],
                "mean_prec": [], "mean_nonprec": [],
                "count": 0
            }

        g = groups[num_pos]
        g["auc"].append(auc)
        g["ap"].append(ap)
        g["mean_pos"].append(mean_pos)
        g["mean_neg"].append(mean_neg)
        g["mean_prec"].append(mean_prec)
        g["mean_nonprec"].append(mean_nonprec)
        g["count"] += 1

global_summary = {
    "num_graphs": int(len(all_auc)),
    "mean_auc": float(np.mean(all_auc)) if all_auc else float("nan"),
    "mean_ap": float(np.mean(all_ap)) if all_ap else float("nan"),
    "mean_p_true": float(np.nanmean(all_mean_pos)) if all_mean_pos else float("nan"),
    "mean_p_neg": float(np.nanmean(all_mean_neg)) if all_mean_neg else float("nan"),
    "mean_p_precedence_true": float(np.nanmean(all_mean_prec)) if all_mean_prec else float("nan"),
    "mean_p_true_nonprecedence": float(np.nanmean(all_mean_nonprec)) if all_mean_nonprec else float("nan"),
}

grouped_summary = {}
for num_pos, vals in sorted(groups.items(), key=lambda kv: kv[0]):
    grouped_summary[num_pos] = {
        "count": int(vals["count"]),
        "mean_auc": float(np.mean(vals["auc"])) if vals["auc"] else float("nan"),
        "mean_ap": float(np.mean(vals["ap"])) if vals["ap"] else float("nan"),
        "mean_p_true": float(np.nanmean(vals["mean_pos"])) if vals["mean_pos"] else float("nan"),
        "mean_p_neg": float(np.nanmean(vals["mean_neg"])) if vals["mean_neg"] else float("nan"),
        "mean_p_precedence_true": float(np.nanmean(vals["mean_prec"])) if vals["mean_prec"] else float("nan"),
        "mean_p_true_nonprecedence": float(np.nanmean(vals["mean_nonprec"])) if vals["mean_nonprec"] else float("nan"),
    }

results = {
    "global": global_summary,
    "by_num_positive_edges": grouped_summary,
}

print("\n Global Results")
print(results["global"])

print("\n Grouped by #positive_edges")
for k, v in results["by_num_positive_edges"].items():
    print(
        f"#pos={k:4d} | n={v['count']:4d} | "
        f"AUC={v['mean_auc']:.4f} | AP={v['mean_ap']:.4f} | "
        f"p(true)={v['mean_p_true']:.4f} | p(neg)={v['mean_p_neg']:.4f} | "
        f"p(prec_true)={v['mean_p_precedence_true']:.4f} | "
        f"p(nonprec_true)={v['mean_p_true_nonprecedence']:.4f}"
    )

with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved grouped metrics to: {OUT_PATH}")