import torch
from training_utils import evaluate, train 
from model_utils import get_parameters, set_parameters
import copy


def clients_training(global_model, client_data, cfg, verbose=False):  
      
    history = {
    "val_loss": [[] for _ in range(cfg.num_clients)],
    "val_accuracy": [[] for _ in range(cfg.num_clients)],
    }

    local_models = [copy.deepcopy(global_model) for _ in range(cfg.num_clients)]

    # params_list = []
    # local_models = [None] * cfg.num_clients
    for client_id in range(cfg.num_clients):
        # local_models[client_id] = global_model 
        # set_parameters(local_models[client_id], global_params)
        
        trainloader = client_data[client_id]["train"]
        valloader = client_data[client_id]["val"]

        optim = torch.optim.SGD(local_models[client_id].parameters(), 
                                lr = cfg.learning_rate, 
                                momentum = cfg.momentum) 
                    
        train(local_models[client_id],
                trainloader, 
                optim, 
                epochs = cfg.local_epochs, 
                verbose = verbose)
                
        loss, accuracy = evaluate(local_models[client_id], valloader)
        
        if verbose and (client_id in {0, cfg.num_clients - 1}): 
            print(f"client_id: {client_id} ")   
            print(f"loss:{loss:.5f} accuracy: {accuracy:.5f} number of sumples {len(valloader)}\n")
        
        history["val_loss"][client_id].append(loss)
        history["val_accuracy"][client_id].append(accuracy)

        # params_list.append(get_parameters(local_models[client_id]))

    return local_models, history