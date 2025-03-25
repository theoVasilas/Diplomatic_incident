import hydra
from omegaconf import DictConfig
from codecarbon import EmissionsTracker

from Mnist_Adapt.Flower.my_run_fn import run_my_sim

@hydra.main(config_path="config", config_name="config", version_base="1.1")
def main(cfg: DictConfig):

    print(f"\n ======== Dp_enabel : {cfg.Dp_enabel} =========== ")
    print(f"======== Dp_e_enabel : {cfg.Dp_e_enabel} =========== ")
    

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

    results = {
            "num_rounds"        : cfg.ROUNDS,
            "duration"          : emissions_data['duration'],
            "emissions"         : emissions_data['emissions'],
        }




if __name__ == "__main__":
    main()