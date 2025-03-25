import toml # type: ignore
import os 
import subprocess
import time

# Define different sets of variables
# configs = [
#     {"num-shares": 0.1, "reconstruction-threshold": 0.3},  # Low redundancy, moderate threshold
#     {"num-shares": 0.3, "reconstruction-threshold": 0.5},  # More secure, tolerates dropout
#     {"num-shares": 0.5, "reconstruction-threshold": 0.7},  # Balanced security and failure tolerance
#     {"num-shares": 0.7, "reconstruction-threshold": 0.9},  # Highly secure, strict threshold
#     {"num-shares": 1.0, "reconstruction-threshold": 1.0},  # Max security, must have all clients active
# ]

configs = [
    {"num-shares": 0.1, "reconstruction-threshold": 0.3},  # Low redundancy, moderate threshold
    {"num-shares": 0.5, "reconstruction-threshold": 0.7},  # Balanced security and failure tolerance
    {"num-shares": 0.7, "reconstruction-threshold": 0.9},  # Highly secure, strict threshold
]

seeds =[101, 222, 3333] #[101, 222, 3333, 4044, 5555, 6967, 7777, 8888, 9090, 1337]

ROUNDS = 10

toml_path = "pyproject.toml" 

for config in configs:
    for num_clients in [10, 50, 100]:
        for seed in seeds:

            # Load current TOML config
            with open(toml_path, "r") as f:
                toml_data = toml.load(f)

            # Modify the parameters
            toml_data["tool"]["flwr"]["app"]["config"]["SEED"]                                          = seed
            toml_data["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"]   = num_clients
            toml_data["tool"]["flwr"]["app"]["config"]["num-rounds"]                                    = ROUNDS
            
            toml_data["tool"]["flwr"]["app"]["config"]["num-shares"]                   = config["num-shares"]
            toml_data["tool"]["flwr"]["app"]["config"]["reconstruction-threshold"]     = config["reconstruction-threshold"]

            # Save updated TOML file
            with open(toml_path, "w") as f:
                toml.dump(toml_data, f)

            # Load updated TOML to verify changes
            with open(toml_path, "r") as f:
                updated_config = toml.load(f)

            # **Validation Check**
            assert updated_config["tool"]["flwr"]["app"]["config"]["SEED"] == seed, "SEED not updated correctly!"
            
            assert updated_config["tool"]["flwr"]["app"]["config"]["num-rounds"] == ROUNDS, "num-rounds not updated!"
            assert updated_config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"] == num_clients, "num-supernodes not updated!"
            
            assert updated_config["tool"]["flwr"]["app"]["config"]["num-shares"] == config["num-shares"], "num-shares not updated!"
            assert updated_config["tool"]["flwr"]["app"]["config"]["reconstruction-threshold"] == config["reconstruction-threshold"], "reconstruction-threshold not updated!"

            # print(f"✅ Config verified")

            # SEED                        = updated_config["tool"]["flwr"]["app"]["config"]["SEED"]
            # NUM_ROUNDS                  = updated_config["tool"]["flwr"]["app"]["config"]["num-rounds"]
            # NUM_SHARES                  = updated_config["tool"]["flwr"]["app"]["config"]["num-shares"]
            # RECONSTRUCTION_THRESHOLD    = updated_config["tool"]["flwr"]["app"]["config"]["reconstruction-threshold"]
            # NUM_SUPERNODES              = updated_config["tool"]["flwr"]["federations"]["local-simulation"]["options"]["num-supernodes"]
            
            # time.sleep(1)

            # Run the Flower experiment
            subprocess.run(["flwr", "run", "."])  # Ensure the `flwr` command is available
