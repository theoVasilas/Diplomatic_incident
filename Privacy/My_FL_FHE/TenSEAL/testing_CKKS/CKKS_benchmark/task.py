import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def training(X, y, local_model, local_epochs, batch_size=32, device='cpu'):
    # Move the model to the appropriate device (GPU or CPU)
    local_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(local_model.parameters(), lr=0.01)

    # Convert data to tensor and move it to the device
    X_tensor = X.clone().detach().to(torch.float32).to(device)
    y_tensor = y.clone().detach().to(torch.long).to(device)

    # Create DataLoader for batching
    dataset = TensorDataset(X_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    for epoch in range(local_epochs):
        local_model.train()  # Set model to training mode
        running_loss = 0.0
        
        for inputs, labels in data_loader:
            optimizer.zero_grad()
            outputs = local_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Print the loss after each epoch
        print(f'Epoch [{epoch+1}/{local_epochs}], Loss: {running_loss/len(data_loader)}')


def evaluate(model, X_test, y_test, device='cpu'):
    """
    Evaluates the model on a test set.

    Args:
    - model: The model to evaluate.
    - X_test: The test input features.
    - y_test: The true labels for the test set.
    - device: The device to run the model on ('cpu' or 'cuda').

    Returns:
    - accuracy: The accuracy of the model on the test set.
    """
    model.to(device)  # Move model to the specified device
    model.eval()  # Set model to evaluation mode

    # Convert data to tensors and move to the device
    X_test_tensor = X_test.clone().detach().to(torch.float32).to(device)
    y_test_tensor = y_test.clone().detach().to(torch.long).to(device)

    with torch.no_grad():  # No need to compute gradients for evaluation
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs, 1)  # Get the class with highest score
        correct = (predicted == y_test_tensor).sum().item()  # Count correct predictions
        total = y_test_tensor.size(0)  # Total number of samples

    accuracy = correct / total  # Calculate accuracy
    # print(f'Accuracy: {accuracy * 100:.2f}%')

    return accuracy
