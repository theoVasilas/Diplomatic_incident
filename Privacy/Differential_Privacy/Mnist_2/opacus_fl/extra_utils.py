import matplotlib.pyplot as plt

import os
import torch
import pickle

import random
import numpy as np

def plot_aggregation_history(aggregation_history):
    """Plot accuracy from Aggregation history."""
    accuracy_values = [d['accuracy'] for d in aggregation_history]
    rounds = list(range(len(accuracy_values)))

    plt.figure(figsize=(8, 6))
    plt.plot(rounds, accuracy_values, marker="o", label="Accuracy", color="tab:blue")

    # Annotate points with values
    for x, y in zip(rounds, accuracy_values):
        plt.text(x, y, f"{y:.2f}", fontsize=9, ha='center', va='bottom')

    plt.title("Cumulative Accuracy over Rounds")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_evaluation_history(evaluation_history):
    """Plot loss and accuracy from Evaluation history."""
    rounds = [entry["round"] for entry in evaluation_history]
    loss = [entry["loss"] for entry in evaluation_history]
    accuracy = [entry["accuracy"] for entry in evaluation_history]

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot loss on the first y-axis
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Loss", color="tab:red")
    ax1.plot(rounds, loss, marker="o", color="tab:red", label="Loss")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    # Annotate loss points
    for x, y in zip(rounds, loss):
        ax1.annotate(f"{y:.2f}", (x, y), fontsize=9, color="tab:red", ha='center', va='bottom')

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color="tab:blue")
    ax2.plot(rounds, accuracy, marker="o", color="tab:blue", label="Accuracy")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    # Annotate accuracy points
    for x, y in zip(rounds, accuracy):
        ax2.annotate(f"{y:.2f}", (x, y), fontsize=9, color="tab:blue", ha='center', va='bottom')

    # Title and legend
    fig.suptitle("Server Loss and Accuracy over Training Rounds")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.show()


def save_results(directory: str, net, aggregation_history, evaluation_history, emissions: float):
    os.makedirs(directory, exist_ok=True)

    # Save the model
    save_path = os.path.join(directory, "net_weights.pth")
    torch.save(net.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # Save the aggregation history
    agg_history_path = os.path.join(directory, "Aggregation_history.pkl")
    with open(agg_history_path, 'wb') as agg_file:
        pickle.dump(aggregation_history, agg_file)
        print(f"Aggregation history saved to {agg_history_path}")

    # Save the evaluation history
    eval_history_path = os.path.join(directory, "evaluation_history.pkl")
    with open(eval_history_path, 'wb') as eval_file:
        pickle.dump(evaluation_history, eval_file)
        print(f"Evaluation history saved to {eval_history_path}")

    # Save the emissions data
    emissions_path = os.path.join(directory, "emissions.txt")
    with open(emissions_path, 'w') as emissions_file:
        emissions_file.write(f"Total emissions: {emissions} kg CO2e\n")
        print(f"Emissions data saved to {emissions_path}")


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
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1) 
    torch.set_default_dtype(torch.float32)
