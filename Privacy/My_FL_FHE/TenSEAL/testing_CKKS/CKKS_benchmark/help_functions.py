import matplotlib.pyplot as plt
import tenseal as ts
import numpy as np
import torch
import time
import math
import csv

from HE_functions import Ckks_init, Encrypt_model, Dencrypt_model
from models import get_parames
from HE_functions import flatten_params, reverse_params

# ================= OPPERATION COMPARISON  =============================
def comper_diff(old_weights, new_weights):
    # Compute the expected weights by dividing the original weights by 2
    expected_weights = [w + 2 for w in old_weights]

    # Compute the difference between the expected and actual weights
    weight_diff = [w - r for w, r in zip(expected_weights, new_weights)]

    # Initialize max weight difference as 0
    max_weight_diff = 0
    
    # Find the maximum absolute weight difference
    for w_diff in weight_diff:
        new_max_weight_diff = np.max(np.abs(w_diff))
        if new_max_weight_diff > max_weight_diff:
            max_weight_diff = new_max_weight_diff

    return max_weight_diff


def comper_mul_diff(old_weights, new_weights):
    # Compute the expected weights by dividing the original weights by 2
    expected_weights = [w * 2 for w in old_weights]

    # Compute the difference between the expected and actual weights
    weight_diff = [w - r for w, r in zip(expected_weights, new_weights)]

    # Initialize max weight difference as 0
    max_weight_diff = 0
    
    # Find the maximum absolute weight difference
    for w_diff in weight_diff:
        new_max_weight_diff = np.max(np.abs(w_diff))
        if new_max_weight_diff > max_weight_diff:
            max_weight_diff = new_max_weight_diff

    return max_weight_diff


def comper_weights_add(old_weights, weights_added):
    # Compute the expected weights by dividing the original weights by 2
    expected_weights = [w + w for w in old_weights]

    # Compute the difference between the expected and actual weights
    weight_diffs = [w - r for w, r in zip(expected_weights, weights_added)]

    # Initialize max weight difference as 0
    max_weight_diff = 0
    
    # Find the maximum absolute weight difference
    for index, w_diff in enumerate(weight_diffs):
        new_max_weight_diff = np.max(np.abs(w_diff))
        if new_max_weight_diff > max_weight_diff:
            # print(index)
            max_weight_diff = new_max_weight_diff

    return max_weight_diff, weight_diffs

# ==================== SAVE CSV ========================================
def load_csv_as_dicts(file_path):
    """
    Load data from a CSV file into a list of dictionaries.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        list: A list of dictionaries where keys are column names 
              and values are parsed based on the content.
    """
    data = []
    with open(file_path, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            parsed_row = {}
            for key, value in row.items():
                # Attempt to parse each value into int, float, or leave as string
                try:
                    parsed_row[key] = int(value)
                except ValueError:
                    try:
                        parsed_row[key] = float(value)
                    except ValueError:
                        parsed_row[key] = value
            data.append(parsed_row)
            
    return data

def save_to_csv(data, name):
    """
    Save a list of dictionaries or tuples to a dynamically named CSV file.

    Args:
        data (list): A list of dictionaries or tuples to save.
        headers (list): A list of column names for the CSV file.
        name (str): Name of the data (used to create the file name).
    """
    file_path = f"{name}.csv"  # Create a file name dynamically
    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        # writer.writerow(headers)  # Write header row
        for row in data:
            writer.writerow(row)
    print(f"Data saved to '{file_path}'")


def find_acceptable_params_for_CKKS(model, csv_configs):

    weights, biases = get_parames(model)

    acceptable_configurations = [["degree", "coeff"]]

    for config in csv_configs:

        degree = config['degree']
        coeff = list(map(int, config['coeff'].strip("[]").split(",")))
        
        print("with degree:", degree, "and coeff:", coeff)
        try: 
            # =================== Encrypt ============================= #
            context = Ckks_init(degree, coeff)
            orig_shapes, flat_weights, flat_biases = flatten_params(weights, biases)
            encrypted_weight, encrypted_bias = Encrypt_model(context, flat_weights, flat_biases)
            # # ====================================== Calculations ============================= #

            # ============ multiplication ===========================#
            encrypted_weight = encrypted_weight * 2
            encrypted_bias = encrypted_bias * 2
            print(f"Performed homomorphic multiplication by 2.")


            # ============ Addition ==================================#
            encrypted_weight = encrypted_weight + 2
            encrypted_bias = encrypted_bias + 2
            print("Performed homomorphic Addition.")


            # # =========== Addition encrypted weights ===============#
            encrypted_weight = encrypted_weight + encrypted_weight
            encrypted_bias = encrypted_bias + encrypted_bias
            print("Performed homomorphic Addition.")

            # # ============================================================================== #
            decrypted_weight, decrypted_bias = Dencrypt_model(encrypted_weight, encrypted_bias)
            weights_reversed, biases_reversed = reverse_params(orig_shapes, decrypted_weight, decrypted_bias)
            # # ============================================================================== #

        
            acceptable_configurations.append((degree, coeff))

        except Exception as e:
            print(f"Failed with error: {str(e)}")

    if len(acceptable_configurations) > 0 :
        save_to_csv(acceptable_configurations, "valid_config_new")
    return acceptable_configurations


def test_pressiosion_of_keys(model, valid_config):

    operations = [  "multiplication_by2",
                    "Addition_weight_2",
                    "Add_encrp_weights"]

    results = [["degree", "coeff", "sum_Diff", "sum_exec_time"]]
    for config in valid_config:
        
        # resete weight and biases
        weights, biases = get_parames(model)

        # =================== KEY ============================= #
        degree = config['degree']
        coeff = list(map(int, config['coeff'].strip("[]").split(",")))
        print("with degree:", degree, "and coeff:", coeff)

        context = Ckks_init(degree, coeff)

        diff = 0
        time_sum = 0
        for operation in operations:

            start_time = time.time()

            # =================== Encrypt ============================= #
            orig_shapes, flat_weights, flat_biases = flatten_params(weights, biases)
            encrypted_weight, encrypted_bias = Encrypt_model(context, flat_weights, flat_biases)
            # =================== Encrypt ============================= #

            # # ====================================== Opperations ============================= #

            if operation == "multiplication_by2":
                # ============ multiplication ===========================#
                encrypted_weight = encrypted_weight * 2

                encrypted_bias = encrypted_bias * 2
                # print(f"Performed homomorphic multiplication by 2.")
            elif operation == "Addition_weight_2":
                # ============ Addition ==================================#
                encrypted_weight = encrypted_weight + 2
                encrypted_bias = encrypted_bias + 2
                # print("Performed homomorphic Addition.")

            elif operation == "Add_encrp_weights":
                # # =========== Addition encrypted weights ===============#
                encrypted_weight = encrypted_weight + encrypted_weight
                encrypted_bias = encrypted_bias + encrypted_bias
                # print("Performed homomorphic Addition.")

            # # ====================================== Opperations ============================= #


            # # ========================Dencrypt============================================== #
            decrypted_weight, decrypted_bias = Dencrypt_model(encrypted_weight, encrypted_bias)
            weights_reversed, biases_reversed = reverse_params(orig_shapes, decrypted_weight, decrypted_bias)
            # # ========================Dencrypt============================================== #
            
            end_time = time.time()    
            execution_time = end_time - start_time
            # print(f"The function took {execution_time:.3f} seconds to execute.")

            # ====================== Test Results ===================================#
            max_diff = 0
            if operation == "multiplication_by2":
                max_diff = comper_mul_diff(weights, weights_reversed)
            elif operation == "Addition_weight_2":
                max_diff = comper_diff(weights, weights_reversed)
            elif operation == "Add_encrp_weights":
                max_diff, diffs = comper_weights_add(weights, weights_reversed)
            
            # ====================== Test Results ===================================#

            diff = diff + max_diff
            time_sum = time_sum + execution_time

            print(f"{operation} : error {max_diff} time spend {execution_time}")


        print(f"sum diff {diff} sum time {time_sum}")
        results.append((degree, coeff, diff, time_sum))

    save_to_csv(results, "config_stats")

    return results

# ================= PLOTS ===============================================
def plot_config_stats(file_nemae = "Results/config_stats.csv"):
    'Pressision vs Execution Time'

    data = load_csv_as_dicts(file_nemae)

    diffs = [data["sum_Diff"] for data in data]
    time_sums = [data["sum_exec_time"] for data in data]
    degrees = [data["degree"] for data in data]
    coeffs = [data["coeff"] for data in data]  # Extract the 'coeff' column as a list of strings
    coeffs = [list(map(int, coeff.strip("[]").split(","))) for coeff in coeffs]  # Process each string individually

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(time_sums, diffs, color='blue', alpha=0.7, label='Configurations')

    for time, diff, coeff, degree in zip(time_sums, diffs, coeffs, degrees):
        plt.text(time, diff, f'{min(coeff)},{degree}', fontsize=12, ha='left', va='bottom', alpha=0.7, rotation=45)

    plt.title('Pressision vs Execution Time', fontsize=16)
    plt.xlabel('Execution Time (seconds)', fontsize=14)
    plt.ylabel('Pressision', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.show()

# ============== test Configurations ==================================
def convert_size(size_bytes):
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


# ============== mesure size of CKKS outputs ==========================
import psutil
import gc

def mesure_size_chifertext(model, valid_config):

    process = psutil.Process()

    weights, biases = get_parames(model)
    
    size_benchmarks = [["Polynomial modulus",
                        "Coefficient modulus sizes",
                        "ALL",
                        "Public_key",
                        "Secret key",
                        "Galois keys",
                        "Relin keys",
                        "none", ]
                        ]
    
    size_of_ciphertext = [["degree", "coeffs", "context_size", "encrypted_weight_size"]]
    
    enc_type = ts.ENCRYPTION_TYPE.ASYMMETRIC

    for config in valid_config:
        context_size = 0 
        encrypted_weight_size = 0

        degree = config['degree']
        coeffs = list(map(int, config['coeff'].strip("[]").split(",")))
        
        print("benchmark with degree:", degree, "and coeff:", coeffs)
        
        # =================== CKKS KEY mesurments ============================= #
        context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=degree,
            coeff_mod_bit_sizes=coeffs,
            encryption_type=enc_type,
        )
        pow = min(coeffs)
        context.global_scale = 2**pow
        context.generate_galois_keys()
        context.generate_relin_keys()


        all = context.serialize(save_public_key=True, save_secret_key=True, save_galois_keys=True, save_relin_keys=True)
        context_size = len(all)
        all_str = convert_size(len(all))

        Public_key = context.serialize(save_public_key=True, save_secret_key=False, save_galois_keys=False, save_relin_keys=False)
        Public_key_str = convert_size(len(Public_key))

        Secret_key = context.serialize(save_public_key=False, save_secret_key=True, save_galois_keys=False, save_relin_keys=False)
        Secret_key_str = convert_size(len(Secret_key))

        galois_key = context.serialize(save_public_key=False, save_secret_key=False, save_galois_keys=True, save_relin_keys=False)
        galois_key_str = convert_size(len(galois_key))

        relin_keys = context.serialize(save_public_key=False, save_secret_key=False, save_galois_keys=False, save_relin_keys=True)
        relin_keys_str = convert_size(len(relin_keys))

        none_keys = context.serialize(save_public_key=False, save_secret_key=False, save_galois_keys=False, save_relin_keys=False)
        none_keys_str = convert_size(len(none_keys))

        size_benchmarks.append([degree,
                                coeffs,
                                all_str,
                                Public_key_str,
                                Secret_key_str,
                                galois_key_str,
                                relin_keys_str,
                                none_keys_str])

        
        # # ====================================== ciphertext ============================= #
        orig_shapes, flat_weights, flat_biases = flatten_params(weights, biases)
        encrypted_weight, encrypted_bias = Encrypt_model(context, flat_weights, flat_biases)
        encrypted_weight_size = len(encrypted_weight.serialize())
        print(f"Total size of encrypted weights: {convert_size(encrypted_weight_size)}")

        print([degree, coeffs, context_size, encrypted_weight_size], "\n")
        size_of_ciphertext.append([degree, coeffs, context_size, encrypted_weight_size])


    save_to_csv(size_of_ciphertext, "ciphertext_benchmarks")
    save_to_csv(size_benchmarks, "keys_benchmarks")

    
    return size_of_ciphertext, size_benchmarks

def plot_ckks_size(file_nemae = "Results/config_mesurments.csv"):
    'Pressision vs Execution Time'

    data = load_csv_as_dicts(file_nemae)

    encrypted_weight_size = [data["encrypted_weight_size"] for data in data]
    context_size = [data["context_size"] for data in data]

    degrees = [data["degree"] for data in data]
    coeffs = [data["coeffs"] for data in data]  # Extract the 'coeff' column as a list of strings
    coeffs = [list(map(int, coeff.strip("[]").split(","))) for coeff in coeffs]  # Process each string individually

    # Plot 1: Degree vs Size of Encrypted Weight
    plt.figure(figsize=(12, 8))
    plt.scatter(degrees, encrypted_weight_size, color='blue', alpha=0.7, label='Configurations')

    for degree, encrypted_weight_size, coeff in zip(degrees, encrypted_weight_size, coeffs):
        plt.text(degree, encrypted_weight_size, f'{min(coeff)},{degree}', fontsize=12, ha='left', va='bottom', alpha=0.7)

    plt.title('degree vs size of weight ', fontsize=16)
    plt.xlabel('size of weight (Bytes)', fontsize=14)
    plt.ylabel('degree', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.show()

    # Plot 2: Degree vs Context Size
    plt.figure(figsize=(12, 8))
    plt.scatter(degrees, context_size, color='green', alpha=0.7, label='Context Size')

    for degree, ctx_size, coeff in zip(degrees, context_size, coeffs):
        plt.text(degree, ctx_size, f'{min(coeff)},{degree}', fontsize=10, ha='left', va='bottom', alpha=0.7)

    plt.title('Degree vs Context Size', fontsize=16)
    plt.xlabel('Degree', fontsize=14)
    plt.ylabel('Context Size (Bytes)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

# ============== Extra ==================================

import psutil
process = psutil.Process()

def mesure_memory():
    # Get current memory usage in bytes
    memory_info = process.memory_info()
    print(f"Memory Usage: {memory_info.rss / (1024 ** 2)} MB")