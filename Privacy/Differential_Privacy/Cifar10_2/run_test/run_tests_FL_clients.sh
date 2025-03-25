#!/bin/bash

execute() {
    # Set environment variable for deterministic behavior in CUDA >= 10.2
    export CUBLAS_WORKSPACE_CONFIG=:4096:8

    python ~/Diplomatic_incident/Differential_Privacy/Mnist_2/opacus_fl/main.py \
        SEED="$seed" \
        num_clients="$num_clients" \
        num_dataset_partitions="$num_clients"\
        ROUNDS="$ROUNDS" \
        Dp_enabel="$Dp_enabel" \
        Dp_e_enabel="$Dp_e_enabel"\
        folder_name="$folder_name" 
}

execute_n() {
    # Set environment variable for deterministic behavior in CUDA >= 10.2
    # export CUBLAS_WORKSPACE_CONFIG=:4096:8

    python ~/Diplomatic_incident/Differential_Privacy/Mnist_2/opacus_fl/main.py \
        SEED="$seed" \
        num_clients="$num_clients" \
        ROUNDS="$ROUNDS" \
        Dp_enabel="$Dp_enabel" \
        max_grad_norm="$max_grad_norm" \
        noise_multiplier="$noise_multiplier" \
        target_delta="$target_delta" \
        Dp_e_enabel="$Dp_e_enabel"
}

execute_e() {
    # Set environment variable for deterministic behavior in CUDA >= 10.2
    # export CUBLAS_WORKSPACE_CONFIG=:4096:8

    python ~/Diplomatic_incident/Differential_Privacy/Mnist_2/opacus_fl/main.py \
        SEED="$seed" \
        num_clients="$num_clients" \
        ROUNDS="$ROUNDS" \
        Dp_enabel="$Dp_enabel" \
        max_grad_norm="$max_grad_norm" \
        noise_multiplier="$noise_multiplier" \
        target_delta="$target_delta" \
        Dp_e_enabel="$Dp_e_enabel" \
        target_epsilon="$target_epsilon" \
        folder_name="$folder_name"
}


# General
seeds=(101 222 3333 4044 5555 6967 7777 8888 9090 1337)

learning_rate_values=(0.01 0.005)

num_clients_values=(5 50 100)
# num_dataset_partitions=${num_clients_values} 

# Simulation parameters
ROUNDS=20

# Opacus
Dp_enabel="False"

# Noise multipliers for different levels of privacy
noise_multiplier=0.1  
max_grad_norm=1.0  # Clipping norms
target_delta=1e-2 

Dp_e_enabel="False"
target_epsilon=0.5 

now=$(date +"%m-%d")
for k in "${!learning_rate_values[@]}"; do  
    learning_rate="${learning_rate_values[k]}"

    for i in "${!num_clients_values[@]}"; do  
        num_clients="${num_clients_values[i]}"

        for seed in "${seeds[@]}"; do  
            if [[ "$Dp_enabel" == "True" && "$Dp_e_enabel" == "True" ]]; then
                folder_name="DP_e_${now}/C${num_clients}_R${ROUNDS}_mgn${max_grad_norm}_td${target_delta}_te${target_epsilon}/s${seed}"
                execute_e

            elif [[ "$Dp_enabel" == "True" ]]; then
                folder_name="DP_n_${now}/C${num_clients}_R${ROUNDS}_mgn${max_grad_norm}_td${target_delta}_nm${noise_multiplier}/s${seed}"
                execute_n

            else
                echo "test the grads for a naked FL (num_clients == num_dataset_partitions)"
                folder_name="FL_${now}/C${num_clients}_R${ROUNDS}_lr${learning_rate}/s${seed}"
                execute
            fi
        done
    done
done