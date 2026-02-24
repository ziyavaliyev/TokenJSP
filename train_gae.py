import torch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE
import os
import config as C
from encoder import Encoder
from utils import npz_to_data_list, split_for_link_pred, edge_sort, edge_diff
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import negative_sampling
from loss import masked_recon_loss, build_allowed_edge_index

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if C.USE_WANDB:
    import wandb
    wandb.init(project=C.PROJECT, name=C.RUN_NAME, config={k: getattr(C, k) for k in dir(C) if k.isupper()})

# Data
data_list = npz_to_data_list(C.NPZ_PATH)
train_list, val_list, test_list = split_for_link_pred(
    data_list,
    val_ratio=C.VAL_RATIO,
    test_ratio=C.TEST_RATIO,
)

train_loader = DataLoader(train_list, batch_size=C.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_list, batch_size=C.BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_list, batch_size=C.BATCH_SIZE, shuffle=False)


# Model
in_channels = train_list[0].x.size(-1)
model = GAE(Encoder(in_channels, C.HIDDEN_CHANNELS, C.LATENT_CHANNELS)).to(device)#, Decoder()
opt = torch.optim.Adam(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)


def train_epoch() -> float:
    model.train()
    total_loss, n = 0.0, 0

    for d in train_loader:
        d = d.to(device)
        opt.zero_grad()

        z = model.encode(d.x, d.edge_index)
        allowed_edge_index = build_allowed_edge_index(num_nodes=d.num_nodes, machine=d.machine, precedence_edge_index=d.precedence_edge_index, batch = d.batch)
        neg_edge_index = edge_diff(allowed_edge_index, d.edge_index)
        neg_edge_index = edge_sort(neg_edge_index)
        edge_index = torch.cat([d.edge_index, neg_edge_index], axis=1)
        loss = model.recon_loss(z, d.edge_index, neg_edge_index)
        loss.backward()
        opt.step()

        total_loss += float(loss.detach())
        n += 1

    return total_loss / max(n, 1)

@torch.no_grad()
def eval_loader(loader):
    model.eval()
    aucs, aps = [], []

    for d in loader:
        d = d.to(device)
        z = model.encode(d.x, d.edge_index)
        allowed_edge_index = build_allowed_edge_index(num_nodes=d.num_nodes, machine=d.machine, precedence_edge_index=d.precedence_edge_index, batch = d.batch)
        pos_edge_index = d.edge_index
        neg_edge_index = edge_diff(allowed_edge_index, d.edge_index)
        neg_edge_index = edge_sort(neg_edge_index)
        
        edge_index = torch.cat([pos_edge_index, neg_edge_index], axis=1)
        
        y = torch.cat([
            torch.ones(pos_edge_index.size(1), device=device),
            torch.zeros(neg_edge_index.size(1), device=device),
        ])

        probs = model.decoder(z, edge_index)

        aucs.append(roc_auc_score(y.detach().cpu().numpy(), probs.detach().cpu().numpy()))
        aps.append(average_precision_score(y.detach().cpu().numpy(), probs.detach().cpu().numpy()))

    return sum(aucs) / len(aucs), sum(aps) / len(aps)

# Training
for epoch in range(1, C.EPOCHS + 1):
    loss = train_epoch()
    log = {"epoch": epoch, "loss": loss}

    if epoch % C.EVAL_EVERY == 0:
        val_auc, val_ap = eval_loader(val_loader)
        log.update({"val_auc": val_auc, "val_ap": val_ap})

    print(log)
    if C.USE_WANDB:
        wandb.log(log)


# Test
test_auc, test_ap = eval_loader(test_loader)
print({"test_auc": test_auc, "test_ap": test_ap})
if C.USE_WANDB:
    wandb.log({"test_auc": test_auc, "test_ap": test_ap})

# Save the model
os.makedirs("checkpoints", exist_ok=True)
torch.save(model.encoder.state_dict(), "checkpoints/encoder.pt")
if C.USE_WANDB:
    wandb.save("checkpoints/encoder.pt")