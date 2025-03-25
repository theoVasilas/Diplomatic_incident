import time
import random
import numpy as np
import torch
import tenseal as ts
from tqdm import tqdm  # Import tqdm for progress bar

from Strategies import Fed_Average
from dataloader import load_datasets
from training_utils import evaluate, train
from model_utils import (
    get_parameters, flatten_params, reshape_params, 
    set_parameters, batch_parameters
)
from Client import clients_training
from HE_functions import Ckks_init, Encrypt_model, Dencrypt_model
from extra_utils import set_seed, convert_size

def preload_datasets(cfg):
    start_Preload = time.time()

    # 📂 Preload dataset
    client_data = {}

    for client_id in tqdm(range(cfg.num_clients), desc="Loading Datasets", unit="client"):
        train_data, val_data, _ = load_datasets(cfg, client_id)
        client_data[client_id] = {
            "train": train_data,
            "val": val_data
        }

    time_preloding_data = time.time() - start_Preload
    print(f"Data preloading time: {time_preloding_data:.2f} seconds")

    return client_data, time_preloding_data


def run_sim_bachfl(cfg, global_model, CKKs_enable=False):
    set_seed(cfg.SEED)  # Ensure reproducibility
    
    generate_contex = time.time()
    # 🔐 Generate CKKS context
    if CKKs_enable:
        context = Ckks_init(cfg.degree, cfg.coeff_mod_bit_sizes, cfg.pow)
    slot_count = cfg.degree // 2
    print(f"Slot count: {slot_count}")

    time_generate_contex = time.time() - generate_contex

    # 📂 Preload dataset
    client_data, time_preloding_data = preload_datasets(cfg)


    Global_history = {
                      "Test_loss": [],
                      "Test_acc": [],
                      "Test_precision":[],
                      "Test_recall":[],
                      "Test_f1":[],
                      "val_loss": [], 
                      "val_accuracy": [], 
                      "Server_val_loss": [], 
                      "Server_val_acc": [],
                      }
    Global_shapes = [p.shape for p in get_parameters(global_model)]
    

    timestamps = {
                "time_generate_contex"  :time_generate_contex ,
                "time_preloding_data"   : time_preloding_data,
                "client"                : [],
                "server"                : [], 
                "BACHING"               : []
                }
    print(f"\n\n ================== Simulation Start ==================")

    for iter in range(cfg.ROUNDS):
        print(f"========= Round {iter+1}/{cfg.ROUNDS} =========")

        ## =========== CLIENT TRAINING ==================== #
        start_clent = time.time()

        local_models, local_history = clients_training(global_model, client_data, cfg, verbose=True)

        Global_history["val_loss"].append(local_history["val_loss"])
        Global_history["val_accuracy"].append(local_history["val_accuracy"])

        ## =========== ENCRYPT LOCAL MODELS ============= ##
        start_baching = time.time()
        clients_encrypted_batches = []

        for model in local_models:
            flat_params = flatten_params(get_parameters(model))[0]
            batched = batch_parameters(flat_params, slot_count)
            
            if CKKs_enable:
                encrypted_batches = [ts.ckks_vector(context, batch) for batch in batched]
            else:
                encrypted_batches = [ np.array(batch) for batch in batched]

            clients_encrypted_batches.append(encrypted_batches)
        
        timestamps["BACHING"].append(time.time() - start_baching)

        timestamps["client"].append(time.time() - start_clent)  # Measure encryption time

        ## =========== FEDERATED AVERAGING (Encrypted) ============= ##
        start_server = time.time()

        sumed_encrypted_batches = [
            sum(batch) for batch in zip(*clients_encrypted_batches)
        ]

        if CKKs_enable:
            decrypted_params = np.concatenate([batch.decrypt() for batch in sumed_encrypted_batches]) / cfg.num_clients
        else :
            decrypted_params = np.concatenate([batch for batch in sumed_encrypted_batches]) / cfg.num_clients

        reshaped_params = reshape_params(decrypted_params, Global_shapes)
        set_parameters(global_model, reshaped_params)

        ## =========== SERVER EVALUATION ==================== ##
        _, valloader, _ = load_datasets(cfg, 0, 1)
        loss, accuracy, _, _, _ = evaluate(global_model, valloader)

        print(f"SERVER eval: Round {iter+1}: loss {loss:.5f}, accuracy {accuracy:.5f}")
        timestamps["server"].append(time.time() - start_server)

        Global_history["Server_val_loss"].append(loss)
        Global_history["Server_val_acc"].append(accuracy)

    ## =========== FINAL TEST EVALUATION ============= ##
    print(f"================== Simulation Finished ==================")
    _, _, testloader = load_datasets(cfg, 0, 1)
    Test_loss, Test_acc, Test_precision, Test_recall, Test_f1 = evaluate(global_model, testloader)
    print(f"Test loss: {loss:.5f}, Test accuracy: {accuracy:.5f} \n\n")

    Global_history["Test_precision"].append(Test_precision)
    Global_history["Test_recall"].append(Test_recall)
    Global_history["Test_loss"].append(Test_loss)
    Global_history["Test_acc"].append(Test_acc)
    Global_history["Test_f1"].append(Test_f1)

    print(f" num of ckks batches for the model : {len(clients_encrypted_batches[0]) }, example of weights {clients_encrypted_batches[0][0]}")

    return Global_history, timestamps
