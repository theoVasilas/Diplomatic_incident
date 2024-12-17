
from dataset import dirichlet_partition, get_mnist, plot_class_histograms

if __name__ == "__main__":
    # Load MNIST dataset
    data_path = "./data"
    trainset, _ = get_mnist(data_path)

    # Specify partition_id and num_partitions
    num_partitions = 10  # Number of partitions you want to split the data into

    # Partition the dataset using Dirichlet distribution
    partitions = dirichlet_partition(trainset, num_partitions)

    # Plot the class distribution histograms for the first 6 partitions
    plot_class_histograms(trainset, partitions, num_partitions)
