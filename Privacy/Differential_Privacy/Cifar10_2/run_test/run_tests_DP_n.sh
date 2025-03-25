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
        max_grad_norm="$max_grad_norm" \
        noise_multiplier="$noise_multiplier" \
        target_delta="$target_delta" \
        Dp_e_enabel="$Dp_e_enabel" \
        learning_rate="$learning_rate" \
        folder_name="$folder_name"
}

# General
seeds=(101 222 3333 4044 5555 6967 7777 8888 9090 1337)

learning_rate=0.005

num_clients_values=(5 50 100)
# num_clients=2               # "${num_clients_values[2]}"
# num_dataset_partitions=100  # "${num_clients}"

# Simulation parameters
ROUNDS=20

# Opacus
Dp_enabel="True"

# Noise multipliers for different levels of privacy
noise_mult_values=(0.1 0.2 0.5 0.7 0.8 1.0 2 5 10 20 50 100) 
max_norm_values=(0.09)  # Clipping norms
delta_values=(1e-2 1e-5) 

Dp_e_enabel="False"


now=$(date +"%m-%d")
for i in "${!noise_mult_values[@]}"; do  
    noise_multiplier="${noise_mult_values[i]}"

    for j in "${!max_norm_values[@]}"; do  
        max_grad_norm="${max_norm_values[j]}"

        for k in "${!delta_values[@]}"; do
            target_delta="${delta_values[k]}"

            for seed in "${seeds[@]}"; do  

                folder_name="DP_n${now}/C${num_clients}_R${ROUNDS}_mgn${max_grad_norm}_td${target_delta}_nm${noise_multiplier}/s${seed}"
                execute
                # python ~/Diplomatic_incident/Differential_Privacy/Mnist_2/opacus_fl/test.py
                # sleep 0.01
            done
        done
    done
done

# cd ~/Diplomatic_incident/Differential_Privacy/Mnist_2/run_test
# chmod +x run_tests_DP_n.sh
# ./run_tests_DP_n.sh