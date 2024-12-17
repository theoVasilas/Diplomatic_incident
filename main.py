import pickle
from pathlib import Path

import hydra
from hydra.utils import instantiate, call
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import flwr as fl

from dataset import prepare_dataset
from client import generate_client_fn
from server import get_on_fit_config, get_evalulate_fn

from codecarbon import EmissionsTracker


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):
    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))
    save_path = HydraConfig.get().runtime.output_dir

    ## 2. Prepare your dataset
   
    ## 3. Define your clients
   
    # ====EmissionsTracker===
    tracker = EmissionsTracker()
    tracker.start()
    # ========================

    ## 5. Start Simulation
    run_simulation(
            server_app=server,
            client_app=client,
            num_supernodes=num_clients,
            backend_config=backend_config,
        )

    # ========================
    emissions: float = tracker.stop()
    print(emissions)
    # ========================

    ## 6. Save your results
    # results_path = Path(save_path) / "results.pkl"
    # results = {"history": history,"emissions": emissions } #"emissions": emissions
    # print(results)
    # with open(str(results_path), "wb") as h:
    #     pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
