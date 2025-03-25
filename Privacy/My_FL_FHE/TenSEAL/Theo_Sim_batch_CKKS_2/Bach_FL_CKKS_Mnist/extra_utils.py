import math
import torch
import random
import numpy as np
import os

def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])



def set_seed(seed=420):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # PyTorch GPU (if available)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable benchmarking for reproducibility
    torch.use_deterministic_algorithms(True)  # Ensures all operations are deterministic
    # torch.set_deterministic(True)  # Ensure deterministic behavior
    os.environ["PYTHONHASHSEED"] = str(seed)  # Fix hashing randomness
    torch.set_num_threads(1) 


# ============= no longer need ====================================
import pickle

def save_pickle(data, filename="simulation_data"):

    filename = f"{filename}.pkl"

    # Save the data to the pickle file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {filename}")

#========================================================================

import os
import shutil

def setup_output_directory(tracker, ckks_enable, dataset, num_clients, num_rounds, degree, pow_value, seed):
    """
    Sets up the output directory based on system configuration and dataset details.
    
    Parameters:
        tracker (object): CodeCarbon tracker or similar object with CPU model info.
        dataset (str): Name of the dataset.
        num_clients (int): Number of clients in federated learning.
        num_rounds (int): Number of training rounds.
        degree (int): Degree parameter.
        pow_value (int): Power parameter.
        seed (int): Random seed value.

    Returns:
        str: The path to the output directory.
    """
    os.makedirs("outputs", exist_ok=True)

    # Determine local or remote machine
    cpu_model = tracker._conf.get("cpu_model", "")
    machine_output_dir = "local" if cpu_model == "AMD Ryzen 5 5600G with Radeon Graphics" else "Remote"

    # Create base output directory
    base_output_dir = os.path.join("outputs", machine_output_dir)

    # Format dataset name and construct output path
    dataset_name = dataset.split("/")[-1]
    if ckks_enable:
        path = f"{dataset_name}_CKKS/C{num_clients}_R{num_rounds}_D{degree}_mCff{pow_value}/s{seed}"
    else:
        path = f"{dataset_name}/C{num_clients}_R{num_rounds}_D{degree}_mCff{pow_value}/s{seed}"
    output_dir = os.path.join(base_output_dir, path)

    # Remove existing directory and create a fresh one
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}")
    return output_dir  # Return the directory path for further use
