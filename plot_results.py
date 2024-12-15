import os
import yaml 
import pickle
import matplotlib
# Use TkAgg backend for plotting
matplotlib.use('TkAgg')  # Can also try 'Agg' for non-interactive plots

import matplotlib.pyplot as plt

import subprocess
subprocess.run("clear", shell=True, executable="/bin/bash")

# Define paths
main = "/home/theo_ubuntu/Diplomatic_incident/flwr_hydra/outputs/"
date_and_time = "2024-12-15/20-39-26"
default_folder = "results.pkl"

# Dynamically construct the file path
file_path = os.path.join(main, date_and_time, default_folder)

# Load the data from the pickle file
with open(file_path, "rb") as file:
    results = pickle.load(file)

history  = results["history"]
# print(type(history))
emissions = results["emissions"]
# print(emissions)

losses_centralized = history.losses_centralized
losses_distributed = history.losses_distributed  
metrics_centralized = history.metrics_centralized

# Access losses and metrics directly (no parentheses needed)
losses_centralized = history.losses_centralized  # 2D list for centralized losses
losses_distributed = history.losses_distributed  # 2D list for distributed losses
metrics_centralized = history.metrics_centralized  # 2D list for centralized metrics

# Extract x and y values from the 2D lists
rounds_centralized_loss, centralized_loss_values = zip(*losses_centralized)  # Unpack into x and y
rounds_distributed_loss, distributed_loss_values = zip(*losses_distributed)  # Unpack into x and y
rounds_centralized_metrics, accuracy_centralized = zip(*metrics_centralized["accuracy"])  # Unpack rounds and accuracy

# Plot the data
plt.figure(figsize=(12, 6))

# Plot loss values
plt.subplot(1, 2, 1)
plt.plot(rounds_distributed_loss, distributed_loss_values, label="Loss (Distributed)", marker='o', linestyle='-', color='blue')
plt.plot(rounds_centralized_loss, centralized_loss_values, label="Loss (Centralized)", marker='x', linestyle='--', color='red')
plt.title("Loss Over Rounds")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.legend()

# Plot accuracy for centralized training
plt.subplot(1, 2, 2)
plt.plot(rounds_centralized_metrics, accuracy_centralized, label="Accuracy (Centralized)", marker='o', linestyle='-', color='green')
plt.title("Accuracy Over Rounds")
plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.legend()


config_file = ".hydra/config.yaml"  # Path to your YAML configuration file

config_path = os.path.join(main, date_and_time, config_file)

# Load the configuration from the YAML file
with open(config_path, 'r') as config:
    config_data = yaml.safe_load(config)

num_rounds = config_data["num_rounds"]
num_clients = config_data["num_clients"]
strategy = config_data["strategy"]
strategy = strategy["_target_"]

text = f"Num Rounds: {num_rounds}\nNum Clients: {num_clients}\nStrategy: {strategy} \nemissions {emissions*1000:.3f}"

plt.suptitle(f"Federated Learning Training Results \n {text}")


# Show the plots
plt.tight_layout()
plt.show()

