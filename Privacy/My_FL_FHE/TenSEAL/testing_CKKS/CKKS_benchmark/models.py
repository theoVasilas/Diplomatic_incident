import torch

class BiggerModel(torch.nn.Module):
    def __init__(self, SEED):
        super(BiggerModel, self).__init__()
        
        self.fc1 = torch.nn.Linear(3, 6)  
        self.fc2 = torch.nn.Linear(6, 3)   

        self.relu = torch.nn.ReLU()

        # Explicitly initialize weights
        torch.manual_seed(SEED)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = x.view(-1, 3)
        x = self.relu(self.fc1(x))  
        x = self.fc2(x)  
        return x
    
# ============================ SET GET PARAMS ==========================================================

def get_parames(model):
    state_dict = model.state_dict()
    
    weights = []
    biases = []
    
    # Iterate through the model's named parameters
    for name, param in model.named_parameters():
        param_data = param.detach().cpu()
        if "weight" in name:
            weights.append(param.data.numpy())  # Append weights
        elif "bias" in name:
            biases.append(param.data.numpy())  # Append biases

    # Now `weights` and `biases` contain all the extracted parameters
    # print("Weights:", weights)
    # print("Biases:", biases)

    return weights, biases


def set_parames(model, weights, biases):
    
    # Reload the weights and biases into the model
    for i, (name, param) in enumerate(model.named_parameters()):
        if "weight" in name:
            param.data = torch.tensor(weights.pop(0), dtype=torch.float32)  # Convert back to tensor
        elif "bias" in name:
            param.data = torch.tensor(biases.pop(0), dtype=torch.float32)  # Convert back to tensor