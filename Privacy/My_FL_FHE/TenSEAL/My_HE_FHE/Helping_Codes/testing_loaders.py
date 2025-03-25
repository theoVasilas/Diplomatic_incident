import os
import time

import hydra
from omegaconf import DictConfig

from costum_FL import run_sim_bachfl, run_sim_fl_CKKS, preload_datasets
from models import Net_Mnist
from extra_utils import set_seed, save_pickle
from dataloader import load_datasets

from codecarbon import EmissionsTracker


@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):

    set_seed(cfg.SEED)
    
    clients_data = preload_datasets(cfg)

    start_clent = time.time()
    for i in range(cfg.num_clients):
        trainloader = clients_data[0]["train"]
    timepassed: float = time.time() - start_clent
    print(f"preloading : time to pull data, {timepassed}")

    start_clent = time.time()
    for i in range(cfg.num_clients):
        trainloader = load_datasets(cfg, 0)[0]
    timepassed: float = time.time() - start_clent
    print(f"without preloading : time to pull data, {timepassed}")


if __name__ == "__main__":
    main()