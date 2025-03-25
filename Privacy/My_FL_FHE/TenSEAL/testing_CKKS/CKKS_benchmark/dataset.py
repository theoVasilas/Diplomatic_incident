import torch
import numpy as np

def generate_data(num_samples=100, num_classes=3):
    # Define means and standard deviations for each class (one per feature)
    means = torch.tensor([[2.0, -3.0, 1.0],   # Mean for class 1
                          [1.5, 2.5, 0.5],    # Mean for class 2
                          [-1.0, 0.5, -2.0]]) # Mean for class 3
    stds = torch.tensor([[0.9, 0.8, 0.3],     # Std for class 1
                         [0.9, 0.3, 0.5],     # Std for class 2
                         [0.4, 0.5, 0.7]])    # Std for class 3

    # Generate random samples from normal distribution for each class
    samples_per_class = num_samples // num_classes
    X = torch.cat([
        torch.normal(means[i].repeat(samples_per_class, 1), stds[i].repeat(samples_per_class, 1))  # Broadcasting
        for i in range(num_classes)
    ], dim=0)

    # Generate labels (class 0, 1, 2 for each sample)
    y = torch.cat([torch.full((samples_per_class,), i) for i in range(num_classes)])

    # Shuffle the dataset to randomize class distributions
    perm = torch.randperm(X.size(0))
    X = X[perm]
    y = y[perm]

    return X, y

# # Example usage
# X, y = generate_data(num_samples=300)
# print(X.shape)  # Should be (300, 3)
# print(y.shape)  # Should be (300,)

def split_data_for_clients(X_train, y_train, NUM_CLIENTS):
    # Shuffle the data
    indices = np.random.permutation(len(X_train))
    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    # Split the shuffled data into NUM_CLIENTS parts
    data_per_client = len(X_train) // NUM_CLIENTS
    
    X_clients = []
    y_clients = []
    
    for i in range(NUM_CLIENTS):
        start_idx = i * data_per_client
        end_idx = (i + 1) * data_per_client if i != NUM_CLIENTS - 1 else len(X_train)
        
        X_clients.append(X_train_shuffled[start_idx:end_idx])
        y_clients.append(y_train_shuffled[start_idx:end_idx])
    
    return X_clients, y_clients

