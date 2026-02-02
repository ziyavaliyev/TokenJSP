# config.py

PROJECT = "jsp-gae"
RUN_NAME = "baseline_test"

NPZ_PATH = "gae_dataset_jsp_6x6.npz"

DEVICE = "gpu"
BATCH_SIZE = 64

VAL_RATIO = 0.05
TEST_RATIO = 0.10
IS_UNDIRECTED = False

HIDDEN_CHANNELS = 64
LATENT_CHANNELS = 32

EPOCHS = 500
LR = 1e-3
WEIGHT_DECAY = 1e-5
EVAL_EVERY = 10

USE_WANDB = True

# dataset generation
N_JOBS = 6
N_MACHINES = 6
NUM_INSTANCES = 3000
MAX_STEPS = 10_000
OUT_NPZ = "gae_dataset_jsp_6x6.npz"