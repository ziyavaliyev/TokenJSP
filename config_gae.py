# config.py
N_JOBS = 10
N_MACHINES = 10

PROJECT = "jsp-gae"
RUN_NAME = f"baseline_{N_JOBS}x{N_MACHINES}"

NPZ_PATH = f"datasets/gae_dataset_jsp_test_{N_JOBS}x{N_MACHINES}.npz"

BATCH_SIZE = 32

VAL_RATIO = 0.05
TEST_RATIO = 0.10
IS_UNDIRECTED = False

HIDDEN_CHANNELS = 64
LATENT_CHANNELS = 32

EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 1e-5
EVAL_EVERY = 10

USE_WANDB = True

# dataset generation
NUM_INSTANCES = 100
MAX_STEPS = 10_000
OUT_NPZ = f"datasets/gae_dataset_jsp_test_{N_JOBS}x{N_MACHINES}.npz"