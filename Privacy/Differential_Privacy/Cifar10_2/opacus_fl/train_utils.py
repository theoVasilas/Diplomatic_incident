import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

   
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, train_loader, optimizer, device, epochs=1):
    """Train the network on the training set."""
    net.to(device)
    net.train()
    
    criterion = torch.nn.CrossEntropyLoss()
    

    grad_norms = []
    
    running_loss = 0.0
    for epoch in range(epochs):
            
            for batch in train_loader:
                images, labels = batch["img"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # accese gradients
                # Store individual gradient norms
                for p in net.parameters():
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm(2).item()) 

                optimizer.step()
                running_loss += loss.item()
            
    avg_trainloss = running_loss / len(train_loader)

    # Compute statistics for gradient norms
    max_norm_before_clipping = max(grad_norms) if grad_norms else 0.0
    # grad_norms_std = np.std(grad_norms) if grad_norms else 0.0
    # one_sigma_value = np.mean(grad_norms) + grad_norms_std if grad_norms else 0.0
    grand_70_pec = np.percentile(grad_norms, 70)

    # print(f"\n max_norm_before_clipping {max_norm_before_clipping} ")
    # print(f"grad_norms_std {grad_norms_std} ")
    # print(f"grand_70_pecent {grand_70_pec} ")
    # print(f"one_sigma_value {one_sigma_value} \n")

    return avg_trainloss, max_norm_before_clipping, grand_70_pec


def train_DP(net, 
             train_loader, 
             privacy_engine, 
             optimizer, 
             target_delta, 
             device, 
             epochs=1
             ):
    
    net.to(device)
    net.train()

    criterion = torch.nn.CrossEntropyLoss()

    grad_norms = []

    running_loss = 0.0
    max_norm_before_clipping = 0.0
    for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in train_loader:
                images, labels = batch["img"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

               # Store individual gradient norms
                for p in net.parameters():
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm(2).item()) 

                optimizer.step()
                running_loss += loss.item()
    
    avg_trainloss = running_loss / len(train_loader)

    #Opacus
    print("\n\t",type(privacy_engine.accountant))

    # epsilon = privacy_engine.get_epsilon(delta=target_delta)
    epsilon = privacy_engine.accountant.get_epsilon(delta=target_delta)

    #Extra information
    max_norm_before_clipping = max(grad_norms) if grad_norms else 0.0
    grand_70_pec = np.percentile(grad_norms, 70)

    return avg_trainloss, epsilon, max_norm_before_clipping, grand_70_pec


from opacus.accountants.utils import get_noise_multiplier

def train_DP_with_epsilon(  net, 
                            train_loader, 
                            privacy_engine,
                            target_epsilon,
                            target_delta,
                            sample_rate,
                            optimizer, 
                            device, 
                            epochs=1
                            ):
    net.to(device)
    net.train()

    criterion = torch.nn.CrossEntropyLoss()

    grad_norms = []

    running_loss = 0.0
    max_norm_before_clipping = 0.0
    for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in train_loader:
                images, labels = batch["img"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Store individual gradient norms
                for p in net.parameters():
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm(2).item()) 

                optimizer.step()
                running_loss += loss.item()

    avg_trainloss = running_loss / len(train_loader)
    
    #Opacus
    # print(f"\n\n sample_rate {sample_rate} \n\n")
    noise_multiplier = get_noise_multiplier(  target_epsilon = target_epsilon,
                                                    target_delta = target_delta,
                                                    sample_rate = sample_rate,
                                                    # epochs = epochs,                 
                                                    steps = 1 ) # epochs or steps ,needs to be revisioned 

    epsilon = privacy_engine.get_epsilon(delta=target_delta)
    # print(f"\n\n epsilon {epsilon}")

    #Extra information
    max_norm_before_clipping = max(grad_norms) if grad_norms else 0.0
    grand_70_pec = np.percentile(grad_norms, 70)

    return avg_trainloss ,noise_multiplier, max_norm_before_clipping, grand_70_pec


#====================================================================================

def test(net, testloader):
    """Evaluate the network on the entire test set."""

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)

            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy

from sklearn.metrics import precision_score, recall_score, f1_score

def test_2(net, testloader):
    """Evaluate the network on the entire test set with accuracy, F1-score, precision, and recall."""

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    all_preds, all_labels = [], []

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = net(images)

            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss /= len(testloader.dataset)
    accuracy = correct / total

    # Compute Precision, Recall, and F1-Score
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    return loss, accuracy, precision, recall, f1
