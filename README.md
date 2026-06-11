# TokenJSP

**TokenJSP: Graph Representation Learning for Job Shop Scheduling Problems**

TokenJSP is a research framework for learning graph representations of Job Shop Scheduling Problem (JSP) states using Graph Autoencoders (GAE) and Variational Graph Autoencoders (VGAE).

The repository provides tools for:

- Generating graph datasets from JSP instances
- Training graph encoders using GAE and VGAE
- Evaluating learned graph representations

The project was developed as part of a Master's thesis on graph-based tokenization methods for JSP state representations.

---

# Repository Structure

```text
tokenjsp/
│
├── configs/                     # YAML training and evaluation configs
├── datasets/                    # Downloaded datasets
├── trained_encoder_weights/     # Pretrained encoder checkpoints
├── evaluation/                  # Evaluation outputs
├── scripts/
│   ├── generate_dataset.py
│   ├── train.py
│   └── test.py
│
├── src/tokenjsp/
│   ├── dataset.py
│   ├── encoder.py
│   └── utils.py
│
├── pyproject.toml
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/ziyavaliyev0703/TokenJSP.git
cd TokenJSP
```

Create a Python environment:

```bash
conda create -n tokenjsp python=3.10
conda activate tokenjsp
```

Install the package:

```bash
pip install -e .
```

---

# Dataset Download

The datasets are hosted on Zenodo:

**Dataset DOI**

[TokenJSP Dataset (Zenodo)](https://zenodo.org/records/20511312)

Download all dataset files and place them inside the `datasets/` directory:

```text
datasets/
├── gae_dataset_jsp_6x6.npz
├── gae_dataset_jsp_10x10.npz
├── gae_dataset_jsp_15x15.npz
└── gae_dataset_jsp_20x20.npz
```

---

# Dataset Description

The datasets consist of graph snapshots collected during random rollouts in a Job Shop Scheduling environment.

For every scheduling state, the following information is stored:

### Adjacency Matrix (A)

Represents the current graph structure of the scheduling state.

```python
A.shape = (T, T)
```

where

```text
T = n_jobs × n_machines
```

---

### Node Features (X)

Each operation is represented by:

1. Scheduled flag
2. Critical Lower Bound (CLB)

```python
X.shape = (T, F)
```

---

### Machine Assignment Matrix (M)

One-hot machine encoding for every operation.

```python
M.shape = (T, n_machines)
```

---

### Metadata

Each dataset additionally stores:

```python
n_jobs
n_machines
```

---

# Generating New Datasets

Datasets can also be generated again.

Example:

```bash
python scripts/generate_dataset.py \
    --n_jobs 20 \
    --n_machines 20 \
    --num_instances 25
```

The generated dataset is stored as:

```text
datasets/gae_dataset_jsp_20x20.npz
```

---

# Supported Graph Encoders

The following GNN architectures are supported:

- GCN
- GIN
- EGC
- PNA
- GAT
- GATv2

Both deterministic and variational autoencoders are implemented:

- GAE
- VGAE

---

# Training

Training is controlled via YAML configuration files.

Example:

```bash
python scripts/train.py \
    --config configs/mixed_train.yaml
```

A configuration file specifies:

- dataset paths
- model type
- GNN architecture
- hidden dimension
- latent dimension
- optimizer settings
- logging configuration

---

# Hyperparameter Sweeps

The repository supports Weights & Biases sweeps.

Create a sweep:

```bash
wandb sweep sweep_token-jsp.yaml
```

Launch an agent:

```bash
wandb agent ENTITY/PROJECT/SWEEP_ID
```

---

# Evaluation

Evaluate a trained encoder (example):

```bash
python scripts/test.py \
    --weights trained_encoder_weights/10x10/vgae_gatv2.pt \
    --data datasets/gae_dataset_jsp_10x10.npz \
    --out evaluation/results.json \
    --model vgae \
    --gnn_type gatv2
```

Reported metrics include:

- Reconstruction Loss
- ROC-AUC (AUC)
- Average Precision (AP)
- Mean probability of positive edges
- Mean probability of negative edges
- Precedence-edge reconstruction statistics

---

## Results

To evaluate the quality of the learned graph representations, we trained both Graph Autoencoders (GAE) and Variational Graph Autoencoders (VGAE) using different Graph Neural Network (GNN) encoder architectures. The models were trained on mixed-size Job Shop Scheduling Problem (JSSP) datasets and evaluated on a link prediction task.

The reported metrics include Average Precision (AP), Area Under the ROC Curve (AUC), Mean Average True (MAT), Mean Precedence Average True (MPAT), Mean Non-Precedence Average True (MNPAT), and Mean Average Negative (MAN). Higher values are desirable for AP, AUC, MAT, MPAT, and MNPAT, while lower MAN values indicate better separation between existing and non-existing graph edges.

The results show that attention-based architectures (GATv2) achieve the strongest overall reconstruction performance, while VGAE generally provides a small but consistent improvement over deterministic GAE models.

| Model | GNN | AP | AUC | MAT | MPAT | MNPAT | MAN |
|---------|---------|---------|---------|---------|---------|---------|---------|
| VGAE | GATv2 | 95.4% | 99.5% | 97.0% | 100.0% | 90.3% | 3.3% |
| GAE | GATv2 | 95.4% | 99.5% | 96.8% | 100.0% | 89.8% | 3.2% |
| VGAE | GAT | 95.1% | 99.4% | 96.9% | 100.0% | 90.3% | 3.5% |
| GAE | GAT | 95.2% | 99.4% | 96.9% | 100.0% | 90.1% | 3.5% |
| VGAE | EGConv | 95.0% | 99.4% | 97.6% | 100.0% | 92.4% | 6.3% |
| GAE | EGConv | 95.2% | 99.4% | 97.3% | 100.0% | 91.5% | 5.5% |
| VGAE | GIN | 94.6% | 99.4% | 97.5% | 99.8% | 92.6% | 6.7% |
| GAE | GIN | 93.8% | 99.4% | 97.1% | 99.8% | 91.3% | 5.5% |
| VGAE | PNA | 88.1% | 98.8% | 95.9% | 99.4% | 88.2% | 8.1% |
| GAE | PNA | 70.5% | 95.5% | 89.7% | 91.6% | 85.5% | 20.7% |

---

# Pretrained Encoders

Pretrained encoder weights are included in:

```text
trained_encoder_weights/
```

Weights are organized by:

- dataset size
- model type (GAE/VGAE)
- GNN architecture

No additional download is required.

---

# Author

Ziya Valiyev

Supervisor: Alexander Nasuta M.Sc.

RWTH Aachen University
