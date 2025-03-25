from task import *
from models import *
from dataset import *
from HE_functions import *
from sklearn.model_selection import train_test_split


def fedavg_sim(NUM_CLIENTS, iterations, model):

    data_train, data_eval = federeted_data_gen(NUM_CLIENTS)

    global_model = model
    res = evaluate(global_model, data_eval["sample"], data_eval["label"])
    print(f"stranting Accuracy: {res* 100:.2f} \n\n")

    for iter in range(iterations):

        local_models = [None] * NUM_CLIENTS
        for client_id in range(NUM_CLIENTS):
            local_models[client_id] = model 

            weights, biases = get_parames(global_model)
            set_parames(local_models[client_id], weights, biases)

            local_ecpoches = 1
            training(data_train[client_id]["sample"],
                    data_train[client_id]["label"],
                    local_models[client_id],
                    local_ecpoches)
            
            res = evaluate(local_models[client_id], data_eval["sample"], data_eval["label"])
            print(f"local Accuracy: {res* 100:.2f}")

        sum_weight = 0
        sum_bias  = 0
        for local_model in local_models:
            weights, biases = get_parames(local_model)

            orig_shapes, flat_weights, flat_biases = flatten_params(weights, biases)
            #========== Send params ============#

            sum_weight = sum_weight + flat_weights
            sum_bias  = sum_bias + flat_biases

        #========== Receive params ============#

        average_weight = sum_weight / NUM_CLIENTS
        average_bias  = sum_bias  / NUM_CLIENTS

        weights_reversed, biases_reversed = reverse_params(orig_shapes, average_weight, average_bias)

        set_parames(global_model, weights_reversed, biases_reversed)
        
        res = evaluate(global_model, data_eval["sample"], data_eval["label"])
        print(f"\n Iteration {iter} global Accuracy: {res* 100:.2f}\n")



def fedAvg_HE(NUM_CLIENTS, iterations, model):

    X, y = generate_data(num_samples = NUM_CLIENTS*100)
    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)
    X_clients, y_clients = split_data_for_clients(X_train, y_train, NUM_CLIENTS)
    # ========================================================================

    global_model = None
    global_model = model
    evaluate(global_model, X_eval, y_eval)
    print(f"stranting Accuracy: {evaluate(global_model, X_eval, y_eval)* 100:.2f} \n\n")


    context = Ckks_init(2048,[20, 20, 14])
    
    for iter in range(iterations):

        # weights, biases = get_parames(global_model)
        local_ecpoches = 1
        local_models = [None] * NUM_CLIENTS
        for client_id in range(NUM_CLIENTS):

            local_models[client_id] = model

            weights, biases = get_parames(global_model)
            set_parames(local_models[client_id], weights, biases)

            training(X_clients[client_id], y_clients[client_id], local_models[client_id], local_ecpoches)
            print(f"local Accuracy: {evaluate(local_models[client_id], X_eval, y_eval)* 100:.2f}")



        sum_encrypted_weight = 0
        sum_encrypted_bias  = 0

        sum_weight = 0
        sum_bias  = 0
        for local_model in local_models:

            weights, biases = get_parames(local_model)

            orig_shapes, flat_weights, flat_biases = flatten_params(weights, biases)
            encrypted_weight, encrypted_bias = Encrypt_model(context, flat_weights, flat_biases)

            #========== Send params ============#
            # HE
            sum_encrypted_weight = sum_encrypted_weight + encrypted_weight
            sum_encrypted_bias  = sum_encrypted_bias + encrypted_bias
            
            # not encrepted
            sum_weight  = sum_weight + flat_weights
            sum_bias    = sum_bias + flat_biases

        #========== Receive params ============#

        decrypted_sum_weight, decrypted_sum_bias = Dencrypt_model(sum_encrypted_weight, sum_encrypted_bias)

        decrypted_avg_weight = decrypted_sum_weight / NUM_CLIENTS
        decrypted_avg_bias  = decrypted_sum_bias  / NUM_CLIENTS

        avrg_weight = sum_weight / NUM_CLIENTS
        avrg_bias  = sum_bias  / NUM_CLIENTS

        weights_reversed, biases_reversed = reverse_params(orig_shapes,
                                                            decrypted_avg_weight,
                                                            decrypted_avg_bias)

        error = sum(abs(decrypted_avg_weight-avrg_weight)) + sum(abs(decrypted_avg_bias - avrg_bias))
        print(f"sum of encreprion error {error}")

        set_parames(global_model, weights_reversed, biases_reversed)
        print(f"\n Iteration {iter} global Accuracy: {evaluate(global_model, X_eval, y_eval)* 100:.2f}\n")


