import time
import torch
import random
import numpy as np
import tenseal as ts

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
    # 📂 Preload dataset
    start_client = time.time()

    print("Preloading dataset...")
    clients_data = {}

    for client_id in range(cfg.num_clients):
        trainloader, valloader, _ = load_datasets(cfg, client_id)
        clients_data[client_id] = {"train": trainloader, "val": valloader}

        if (client_id + 1) % 10 == 0 :
            print(f"Client {client_id + 1}/{cfg.num_clients} data loaded.")  # Log progress

    print("Done preloading dataset.")

    timepassed = time.time() - start_client
    print(f"Preloading time: {timepassed:.2f} seconds")

    return clients_data


def run_sim_fl(cfg, global_model):
    set_seed(cfg.SEED)  # Ensure reproducibility

    # 🔐 Generate CKKS context
    # context = Ckks_init(cfg.degree, cfg.coeff_mod_bit_sizes, cfg.pow)
    # slot_count = cfg.degree // 2

    # 📂 Preload dataset
    client_data = preload_datasets(cfg)

    Global_history = {"val_loss": [], 
                      "val_accuracy": [], 
                      "Server_val_loss": [], 
                      "Server_val_acc": [],
                      "Test_loss": [],
                      "Test_acc": []}
    Global_shapes = [p.shape for p in get_parameters(global_model)]
    
    timestamps = {"client": [], "server": []}

    print(f"\n\n ================== Simulation Start ==================")

    for iter in range(cfg.ROUNDS):
        print(f"\t========= Round {iter+1}/{cfg.ROUNDS} =========")

        ## =========== CLIENT TRAINING ==================== #
        start_clent = time.time()
        local_models, local_history = clients_training(global_model, client_data, cfg, verbose=False)

        Global_history["val_loss"].append(local_history["val_loss"])
        Global_history["val_accuracy"].append(local_history["val_accuracy"])

        ## =========== ENCRYPT LOCAL MODELS ============= ##
        start = time.time()
        flatten_params_list = []

        for model in local_models:
            flat_params = flatten_params(get_parameters(model))[0]
            # batched = batch_parameters(flat_params, slot_count)

            # encrypted_batches = [ts.ckks_vector(context, batch) for batch in batched]
            # encrypted_batches = [ np.array(batch) for batch in batched]
            flatten_params_list.append(flat_params)

        timestamps["client"].append(time.time() - start_clent)  # Measure encryption time

        ## =========== FEDERATED AVERAGING (Encrypted) ============= ##
        start_server = time.time()

        avg_array = Fed_Average(flatten_params_list)

        # decrypted_params = np.concatenate([batch.decrypt() for batch in sumed_encrypted_batches]) / cfg.num_clients
        # decrypted_params = np.concatenate([batch for batch in sumed_encrypted_batches]) / cfg.num_clients

        reshaped_params = reshape_params(avg_array, Global_shapes)
        set_parameters(global_model, reshaped_params)

        ## =========== SERVER EVALUATION ==================== ##
        _, valloader, _ = load_datasets(cfg, 0, 1)
        loss, accuracy = evaluate(global_model, valloader)

        print(f"SERVER eval: Round {iter+1}: loss {loss:.5f}, accuracy {accuracy:.5f}")
        timestamps["server"].append(time.time() - start_server)

        Global_history["Server_val_loss"].append(loss)
        Global_history["Server_val_acc"].append(accuracy)

    ## =========== FINAL TEST EVALUATION ============= ##
    print(f"================== Simulation Finished ==================")
    _, _, testloader = load_datasets(cfg, 0, 1)
    Test_loss, Test_acc = evaluate(global_model, testloader)
    print(f"Test loss: {Test_loss:.5f},Test accuracy: {Test_acc:.5f} \n\n")
    Global_history["Test_loss"].append(Test_loss)
    Global_history["Test_acc"].append(Test_acc)

    return Global_history, timestamps


def run_sim_bachfl(cfg, global_model):
    set_seed(cfg.SEED)  # Ensure reproducibility

    # 🔐 Generate CKKS context
    # context = Ckks_init(cfg.degree, cfg.coeff_mod_bit_sizes, cfg.pow)
    slot_count = cfg.degree // 2

    # 📂 Preload dataset
    client_data = preload_datasets(cfg)

    Global_history = {"val_loss": [], 
                      "val_accuracy": [], 
                      "Server_val_loss": [], 
                      "Server_val_acc": [],
                      "Test_loss": [],
                      "Test_acc": []}
    Global_shapes = [p.shape for p in get_parameters(global_model)]
    
    timestamps = {"client": [], "server": []}

    print(f"\n \t ================== Simulation Start ==================")

    for iter in range(cfg.ROUNDS):
        print(f"\t\t ========= Round {iter+1}/{cfg.ROUNDS} =========")

        ## =========== CLIENT TRAINING ==================== #
        start_clent = time.time()
        local_models, local_history = clients_training(global_model, client_data, cfg, verbose=False)

        Global_history["val_loss"].append(local_history["val_loss"])
        Global_history["val_accuracy"].append(local_history["val_accuracy"])

        ## =========== ENCRYPT LOCAL MODELS ============= ##
        start = time.time()
        clients_encrypted_batches = []

        for model in local_models:
            flat_params = flatten_params(get_parameters(model))[0]
            batched = batch_parameters(flat_params, slot_count)

            # encrypted_batches = [ts.ckks_vector(context, batch) for batch in batched]
            encrypted_batches = [ np.array(batch) for batch in batched]
            clients_encrypted_batches.append(encrypted_batches)

        timestamps["client"].append(time.time() - start_clent)  # Measure encryption time

        ## =========== FEDERATED AVERAGING (Encrypted) ============= ##
        start_server = time.time()

        sumed_encrypted_batches = [
            sum(batch) for batch in zip(*clients_encrypted_batches)
        ]

        # decrypted_params = np.concatenate([batch.decrypt() for batch in sumed_encrypted_batches]) / cfg.num_clients
        decrypted_params = np.concatenate([batch for batch in sumed_encrypted_batches]) / cfg.num_clients

        reshaped_params = reshape_params(decrypted_params, Global_shapes)
        set_parameters(global_model, reshaped_params)

        ## =========== SERVER EVALUATION ==================== ##
        _, valloader, _ = load_datasets(cfg, 0, 1)
        loss, accuracy = evaluate(global_model, valloader)

        print(f"SERVER eval: Round {iter+1}: loss {loss:.5f}, accuracy {accuracy:.5f}")
        timestamps["server"].append(time.time() - start_server)

        Global_history["Server_val_loss"].append(loss)
        Global_history["Server_val_acc"].append(accuracy)

    ## =========== FINAL TEST EVALUATION ============= ##
    print(f"================== Simulation Finished ==================")
    _, _, testloader = load_datasets(cfg, 0, 1)
    Test_loss, Test_acc = evaluate(global_model, testloader)
    print(f"Test loss: {Test_loss:.5f},Test accuracy: {Test_acc:.5f} \n\n")
    Global_history["Test_loss"].append(Test_loss)
    Global_history["Test_acc"].append(Test_acc)

    return Global_history, timestamps


def run_sim_fl_CKKS(cfg, global_model):
    set_seed(cfg.SEED)  # Ensure reproducibility

    # 🔐 Generate CKKS context
    context = Ckks_init(cfg.degree, cfg.coeff_mod_bit_sizes, cfg.pow)
    slot_count = cfg.degree // 2

    # 📂 Preload dataset
    client_data = preload_datasets(cfg)

    Global_history = {"val_loss": [], 
                    "val_accuracy": [], 
                    "Server_val_loss": [], 
                    "Server_val_acc": [],
                    "Test_loss": [],
                    "Test_acc": []}
    Global_shapes = [p.shape for p in get_parameters(global_model)]
    
    timestamps = {"client": [], "server": [], "ENCRYPT": []}

    print(f"\n \t ================== Simulation Start ==================")

    for iter in range(cfg.ROUNDS):
        print(f"\t ========= Round {iter+1}/{cfg.ROUNDS} =========")

        ## =========== CLIENT TRAINING ==================== #
        start_clent = time.time()

        local_models, local_history = clients_training(global_model, client_data, cfg, verbose=False)
        
        timestamps["client"].append(time.time() - start_clent)  # Measure time

        Global_history["val_loss"].append(local_history["val_loss"])
        Global_history["val_accuracy"].append(local_history["val_accuracy"])

        ## =========== ENCRYPT LOCAL MODELS ============= ##
        start_encreption = time.time()

        clients_encrypted_batches = []

        for model in local_models:
            flat_params = flatten_params(get_parameters(model))[0]
            batched = batch_parameters(flat_params, slot_count)

            encrypted_batches = [ts.ckks_vector(context, batch) for batch in batched]
            clients_encrypted_batches.append(encrypted_batches)

        timestamps["ENCRYPT"].append(time.time() - start_encreption)  # Measure encryption time

        ## =========== FEDERATED AVERAGING (Encrypted) ============= ##
        start_server = time.time()

        sumed_encrypted_batches = [
            sum(batch) for batch in zip(*clients_encrypted_batches)
        ]

        decrypted_params = np.concatenate([batch.decrypt() for batch in sumed_encrypted_batches]) / cfg.num_clients
        reshaped_params = reshape_params(decrypted_params, Global_shapes)
        set_parameters(global_model, reshaped_params)

        ## =========== SERVER EVALUATION ==================== ##
        _, valloader, _ = load_datasets(cfg, 0, 1)
        loss, accuracy = evaluate(global_model, valloader)

        print(f"SERVER eval: Round {iter+1}: loss {loss:.5f}, accuracy {accuracy:.5f}")
        timestamps["server"].append(time.time() - start_server)

        Global_history["Server_val_loss"].append(loss)
        Global_history["Server_val_acc"].append(accuracy)

    ## =========== FINAL TEST EVALUATION ============= ##
    print(f"================== Simulation Finished ==================")
    _, _, testloader = load_datasets(cfg, 0, 1)
    Test_loss, Test_acc = evaluate(global_model, testloader)
    print(f"Test loss: {Test_loss:.5f},Test accuracy: {Test_acc:.5f} \n\n")
    Global_history["Test_loss"].append(Test_loss)
    Global_history["Test_acc"].append(Test_acc)

    return Global_history, timestamps


