import torch.nn.functional as F
import torch.nn as nn
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, train_loader, optimizer, device, epochs=1):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    running_loss = 0.0
    max_norm_before_clipping = 0.0
    for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in train_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Compute per-sample gradient norms before clipping
                per_sample_norms = []
                for p in net.parameters():
                    if p.requires_grad and hasattr(p, "grad_sample") and p.grad_sample is not None:
                        per_sample_norms.append(p.grad_sample.view(p.grad_sample.shape[0], -1).norm(2, dim=1).max().item())

                if per_sample_norms:
                    batch_max_norm = max(per_sample_norms)
                    max_norm_before_clipping = max(max_norm_before_clipping, batch_max_norm)

                optimizer.step()
                running_loss += loss.item()


    avg_trainloss = running_loss / len(train_loader)
    results = {
        "train_loss"  : avg_trainloss,
        "epsilon"      : -1,
        "max_norm_before_clipping": -1,
    }

    return results


def train_DP(net, 
             train_loader, 
             privacy_engine, 
             optimizer, 
             target_delta, 
             device, 
             epochs=1
             ):
    
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    running_loss = 0.0
    max_norm_before_clipping = 0.0
    for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in train_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Compute per-sample gradient norms before clipping
                per_sample_norms = []
                for p in net.parameters():
                    if p.requires_grad and hasattr(p, "grad_sample") and p.grad_sample is not None:
                        per_sample_norms.append(p.grad_sample.view(p.grad_sample.shape[0], -1).norm(2, dim=1).max().item())

                if per_sample_norms:
                    batch_max_norm = max(per_sample_norms)
                    max_norm_before_clipping = max(max_norm_before_clipping, batch_max_norm)

                optimizer.step()
                running_loss += loss.item()

    epsilon = privacy_engine.get_epsilon(delta=target_delta)

    avg_trainloss = running_loss / len(train_loader)
    results = {
        "train_loss"  : avg_trainloss,
        "epsilon"     : epsilon,
        "max_norm_before_clipping": max_norm_before_clipping,
    }

    return results


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
    criterion = torch.nn.CrossEntropyLoss()
    net.to(device)
    net.train()
    running_loss = 0.0
    max_norm_before_clipping = 0.0
    for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for batch in train_loader:
                images, labels = batch["image"].to(device), batch["label"].to(device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()

                # Compute per-sample gradient norms before clipping
                per_sample_norms = []
                for p in net.parameters():
                    if p.requires_grad and hasattr(p, "grad_sample") and p.grad_sample is not None:
                        per_sample_norms.append(p.grad_sample.view(p.grad_sample.shape[0], -1).norm(2, dim=1).max().item())

                if per_sample_norms:
                    batch_max_norm = max(per_sample_norms)
                    max_norm_before_clipping = max(max_norm_before_clipping, batch_max_norm)

                optimizer.step()
                running_loss += loss.item()

    print(f"\n\n sample_rate {sample_rate} \n\n")

    noise_multiplier = get_noise_multiplier(  target_epsilon = target_epsilon,
                                                    target_delta = target_delta,
                                                    sample_rate = sample_rate,
                                                    # epochs = epochs,                 
                                                    steps = 1 )

    avg_trainloss = running_loss / len(train_loader)
    results = {
        "train_loss"  : avg_trainloss,
        "noise_multiplier"      : noise_multiplier,
        "max_norm_before_clipping": max_norm_before_clipping,
    }

    return results


#====================================================================================

def test(net, testloader):
    """Evaluate the network on the entire test set."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    
    all_preds, all_labels = [], []

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"].to(device), batch["label"].to(device)
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
