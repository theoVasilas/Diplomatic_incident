# import os
# os.environ["RAY_DEDUP_LOGS_AGG_WINDOW_S"] = "10"
import json


import hydra
from omegaconf import DictConfig, OmegaConf
from codecarbon import EmissionsTracker

from my_run_fn import run_my_sim

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    

    print("\n\n")
    print(f"SEED: {cfg.SEED}")
    print(f"num_clients: {cfg.num_clients}")
    print(f"ROUNDS: {cfg.ROUNDS}")
    print(f"========Dp_enabel: {cfg.Dp_enabel}")
    print(f"max_grad_norm: {cfg.max_grad_norm}")
    print(f"noise_multiplier: {cfg.noise_multiplier}")
    print(f"target_delta: {cfg.target_delta}")
    print(f"========Dp_e_enabel: {cfg.Dp_e_enabel}")
    print(f"target_epsilon: {cfg.target_epsilon}")
    print("\n\n")
    

    # ========EmissionsTracker===============
    tracker = EmissionsTracker(log_level="error", save_to_file=False)
    tracker.start()
    # =======================================

    run_my_sim(cfg,)

    # =======================================
    tracker.stop()
    emissions_data = tracker.final_emissions_data.__dict__
    print(f"\n duration : {emissions_data['duration']} \n") 
    # =======================================

    results = ({
            "num_rounds"        : cfg.ROUNDS,
            "duration"          : emissions_data['duration'],
            "emissions"         : emissions_data['emissions'],
        },
        emissions_data
    )
    
    output_file = "Sim_results.json"

    with open(output_file, "a") as f:  # Append to file
        json.dump(results, f)
        f.write("\n")  # Ensure each entry is on a new line
        json.dump(OmegaConf.to_container(cfg, resolve=True), f)  # Convert DictConfig to dict
        f.write("\n")




if __name__ == "__main__":
    main()