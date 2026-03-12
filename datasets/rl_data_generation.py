from utils import generate_jsp_instance
import numpy as np

n_machines = 10
n_jobs = 10
min_processing_time = 2
max_processing_time = 100
count = 10000
name = f"rl_dataset_test2_jsp_{n_machines}x{n_jobs}"

instances = []

for i in range(count):
    instances.append(generate_jsp_instance(n_jobs, n_machines, min_processing_time, max_processing_time))

np.savez(f"datasets/{name}.npz", data=instances)