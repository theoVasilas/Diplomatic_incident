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


import pickle

def save_pickle(data, filename="simulation_data"):

    filename = f"{filename}.pkl"

    # Save the data to the pickle file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    print(f"Data saved to {filename}")
