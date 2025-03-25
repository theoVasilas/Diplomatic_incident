import os
import hydra
from omegaconf import DictConfig
from codecarbon import EmissionsTracker

from costum_FL import run_sim_bachfl, run_sim_fl_CKKS, run_sim_fl
from models import Net_Mnist
from extra_utils import set_seed, save_pickle




@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):

    tracker = EmissionsTracker(log_level="error") #log_level="error"
    tracker.start()

    set_seed(cfg.SEED)
    model = Net_Mnist()

    # hystory, timestamps = run_sim_fl_CKKS(cfg, model)
    history, timestamps = run_sim_bachfl(cfg, model)
    # history, timestamps = run_sim_fl(cfg, model)
    

    emissions: float = tracker.stop()
    print(f"emissions, {emissions}")
    
    data = {'history': history, 'timestamps': timestamps, 'emissions': emissions}   
    save_pickle(data, filename="simulation_data")


    # =================================================
    print(f"SEED: {cfg.SEED} num_clients: {cfg.num_clients}")
    average_client_time = sum(timestamps["client"])/ len(timestamps["client"])
    print("average_client_time : ",average_client_time)

    average_server_time = sum(timestamps["server"])/ len(timestamps["server"])
    print("average_server_time : ",average_server_time)
    
    print(cfg, "\n\n")
    

if __name__ == "__main__":
    main()