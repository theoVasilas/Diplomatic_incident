import matplotlib.pyplot as plt

import os
import torch
import pickle

import random
import numpy as np


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
    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # torch.set_num_threads(1) 
    # torch.set_default_dtype(torch.float32)



