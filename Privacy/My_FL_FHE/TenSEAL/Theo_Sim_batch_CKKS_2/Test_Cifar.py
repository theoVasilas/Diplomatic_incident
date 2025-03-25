import subprocess
import time
import yaml
import os 

keys = [
    {"degree": 32768, "coeff": [40, 21, 21, 40]},
    # {"degree": 16384, "coeff": [60, 54, 54, 60]},
    # {"degree": 8192, "coeff": [40, 20, 40]},
    # {"degree": 4096, "coeff": [30, 30, 30, 19]},
    {"degree": 2048, "coeff": [20, 20, 14]}
  ]
  

seeds = [101, 222, 3333, 4044, 5555, 6967, 7777, 8888, 9090, 1337]

ROUNDS = 25 

CKKs_enable = True

base_path = "Bach_FL_CKKS_Cifar"       # "Bach_FL_CKKS_Cifar"

path = base_path + "/config/config.yaml"  # Load existing Hydra config

for key in keys:
    for num_clients in [10]: # [10, 50, 100]
        for seed in seeds:

            # Load the YAML file
            with open(path, "r") as f:
                config = yaml.safe_load(f)

            # Modify parameters
            config["SEED"] = seed

            config["CKKs_enable"] = True
            config["num_clients"] = num_clients
            config["ROUNDS"] = ROUNDS

            config["coeff_mod_bit_sizes"] = key["coeff"]
            config["degree"] = key["degree"]
            config["pow"] = min(key["coeff"])

            # Save changes back
            with open(path, "w") as f:
                yaml.safe_dump(config, f)

            print("Updated config.yaml!")

            subprocess.run(["python", base_path + "/main.py"])
