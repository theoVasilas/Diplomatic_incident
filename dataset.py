import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import Counter
from torchvision.transforms import Compose, ToTensor, Normalize
import os
from torchvision import datasets, transforms
from tqdm import tqdm

BATCH_SIZE = 64


def get_mnist(data_path: str = "./data"):
    """Download MNIST and apply a simple transform, showing progress bar during download."""
    
    # Define the transform for the dataset
    tr = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    # Define the paths for MNIST train and test datasets
    train_data_path = os.path.join(data_path, "MNIST", "raw", "train-images-idx3-ubyte")
    test_data_path = os.path.join(data_path, "MNIST", "raw", "t10k-images-idx3-ubyte")
    
    # Check if data is already downloaded
    if os.path.exists(train_data_path) and os.path.exists(test_data_path):
        print("MNIST dataset is already installed.")
        trainset = datasets.MNIST(data_path, train=True, download=False, transform=tr)
        testset = datasets.MNIST(data_path, train=False, download=False, transform=tr)
    else:
        print("MNIST dataset not found. Downloading now...")
        
        # Use tqdm to show the progress of the download
        def download_progress_bar(n):
            with tqdm(total=n, desc="Downloading MNIST", unit='B', unit_scale=True) as pbar:
                return pbar

        # Use the 'download=True' option to download the MNIST dataset and show progress bar
        trainset = datasets.MNIST(data_path, train=True, download=True, transform=tr)
        testset = datasets.MNIST(data_path, train=False, download=True, transform=tr)
        
        # Once the download is complete, print a success message
        print("MNIST dataset download completed.")

    # Return the MNIST datasets (train and test)
    return trainset, testset


def dirichlet_partition(dataset: Dataset, num_partitions: int, alpha: float = 0.5):
    """
    Partition dataset using Dirichlet distribution to create non-IID splits.
    
    Args:
    - dataset: The dataset to partition (e.g., MNIST).
    - num_partitions: The number of partitions to create.
    - alpha: The concentration parameter of the Dirichlet distribution.
    
    Returns:
    - A list of partitions, each containing indices for its data points.
    """
    # Get the labels from the dataset
    labels = [item[1] for item in dataset]
    num_classes = len(set(labels))

    # Create class-wise indices
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Sample Dirichlet distribution for partition probabilities
    dirichlet_probs = np.random.dirichlet([alpha] * num_classes, num_partitions)

    partitions = [[] for _ in range(num_partitions)]

    # Distribute data points based on the Dirichlet distribution
    for class_label, indices in class_indices.items():
        num_samples = len(indices)
        class_probs = dirichlet_probs[class_label]
        class_distribution = np.random.multinomial(num_samples, class_probs)

        start_idx = 0
        for partition_id, num_samples_in_partition in enumerate(class_distribution):
            partitions[partition_id].extend(indices[start_idx:start_idx + num_samples_in_partition])
            start_idx += num_samples_in_partition

    return partitions


def load_datasets(partition_id: int, num_partitions: int, data_path: str = "./data"):
    """Load the federated datasets using Dirichlet Partitioning and apply transformations."""
    # Get MNIST dataset
    trainset, testset = get_mnist(data_path)

    # Partition dataset using Dirichlet Partitioning
    partitions = dirichlet_partition(trainset, num_partitions)

    # Create the partition for the given partition_id
    partition_indices = partitions[partition_id]
    partition_train = torch.utils.data.Subset(trainset, partition_indices)

    # Apply transformations to the partition
    pytorch_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    # Apply transform to training data
    partition_train = [apply_transforms(trainset[idx]) for idx in partition_train]

    trainloader = DataLoader(partition_train, batch_size=BATCH_SIZE, shuffle=True)
    valloader = DataLoader(testset, batch_size=BATCH_SIZE)

    return trainloader, valloader


def plot_class_histograms(dataset, partitions, num_partitions, max_plots=6):
    """Plot class distribution histograms for multiple partitions in one figure."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Adjust the grid size for 6 plots (2x3)
    axes = axes.flatten()  # Flatten to easily iterate over the axes

    # Plot up to `max_plots` partitions in the same figure
    for partition_id in range(min(num_partitions, max_plots)):
        partition_indices = partitions[partition_id]
        partition_data = [dataset[idx] for idx in partition_indices]

        # Extract class labels from the partition
        labels = [item[1] for item in partition_data]

        # Count the occurrences of each class
        label_counts = Counter(labels)

        # Plot histogram
        axes[partition_id].bar(label_counts.keys(), label_counts.values())
        axes[partition_id].set_xlabel('Class')
        axes[partition_id].set_ylabel('Frequency')
        axes[partition_id].set_title(f'Partition {partition_id}')
            # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()