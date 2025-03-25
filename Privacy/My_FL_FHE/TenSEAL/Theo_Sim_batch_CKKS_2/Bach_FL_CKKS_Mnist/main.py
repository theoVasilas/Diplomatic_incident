from codecarbon import EmissionsTracker
from omegaconf import DictConfig, OmegaConf
import shutil
import hydra
import json
import toml # type: ignore
import os


from custom_FL import run_sim_bachfl
from models import Net_Mnist, Net_Cifar 
from extra_utils import set_seed, setup_output_directory

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):

    SEED = cfg.SEED
    NUM_CLIENTS = cfg.num_clients
    NUM_DATASET_PARTITIONS = cfg.num_dataset_partitions
    NUM_ROUNDS = cfg.ROUNDS
    DATASET = cfg.dataset
    DEGREE = cfg.degree
    COEFF_MOD_BIT = cfg.coeff_mod_bit_sizes
    POW = min(COEFF_MOD_BIT)
    CKKS_ENABLE = cfg.CKKs_enable

    
    print("\n==== Experiment Configuration ====")
    print(f"CKKS_ENABLE {CKKS_ENABLE}")
    print("\n==================================")
    print(f"DATASET: {DATASET}")
    print(f"SEED: {SEED}")
    print("\n-- CKKS Parameters --")
    print(f"DEGREE: {DEGREE}")
    print(f"COEFF_MOD_BIT_SIZE: {COEFF_MOD_BIT}")
    print(f"GLOBAL SCALE: {POW}")
    print("\n-- Federation Settings --")
    print(f"NUM_CLIENTS: {NUM_CLIENTS}")
    print(f"NUM_ROUNDS: {NUM_ROUNDS}")
    print("===================================")

    
    #==========EmissionsTracker===================
    lock_file = "/tmp/.codecarbon.lock"
    # Check if lock file exists and remove it
    if os.path.exists(lock_file):
        os.remove(lock_file)
    tracker = EmissionsTracker(save_to_file=False, log_level="error")
    #================= Create output Folder =========================
    output_dir = setup_output_directory(tracker, CKKS_ENABLE, DATASET, NUM_CLIENTS, NUM_ROUNDS, DEGREE, POW, SEED)
    tracker.start()
    #========================================================================

    set_seed(SEED)
    model = Net_Mnist() # Net_Cifar()

    history= {"Test_acc":[1]}
    timestamps = 0.0

    history, timestamps = run_sim_bachfl(cfg, model, CKKs_enable = cfg.CKKs_enable)

    # ========================================================================
    tracker.stop()
    emissions_data = tracker.final_emissions_data.__dict__
    print(f"\n duration : {emissions_data['duration']} \n") 
    # =======================================

    all_clients_repetitions= len(timestamps["client"]) * NUM_CLIENTS
    average_client_time = sum(timestamps["client"])/ all_clients_repetitions
    print("average_client_time : ",average_client_time)

    average_server_time = sum(timestamps["server"])/ len(timestamps["server"])
    print("average_server_time : ",average_server_time,)

    # ========================================

    all_results = {
            "num_clients"  : NUM_CLIENTS,
            "num_rounds"   : NUM_ROUNDS,
            "dataset"      : DATASET,
            "DEGREE"       : DEGREE,
            "COEFF_MOD_BIT": str(COEFF_MOD_BIT),
            "POW"          : POW,
            "duration"     : emissions_data['duration'],
            "emissions"    : emissions_data['emissions'],
        }


    CodeCarbon_file = os.path.join(output_dir, "CodeCarbon_data.json")
    with open(CodeCarbon_file, "w") as f:  
        json.dump(emissions_data, f)
        f.write("\n")  # Ensure each entry is on a new line
    
    output_file = os.path.join(output_dir, "Sim_results.json")
    with open(output_file, "w") as f:  
        json.dump(all_results, f)
        f.write("\n")  # Ensure each entry is on a new line
    
    output_file = os.path.join(output_dir, "history.json")
    with open(output_file, "w") as f:  
        json.dump(history, f)
        f.write("\n")  # Ensure each entry is on a new line

    
if __name__ == "__main__":
    main()