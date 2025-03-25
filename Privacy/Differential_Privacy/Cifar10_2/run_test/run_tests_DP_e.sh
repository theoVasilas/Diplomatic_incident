#!/bin/bash

# General
seeds=(111 222 3333 4044 5555 6967 7777 8888 9090 1337)

learning_rate=0.01

num_clients_values=(10 50 100)
# num_clients="${num_clients_values[2]}"
# num_dataset_partitions="${num_clients}"

# Simulation parameters
ROUNDS=20

# Opacus
Dp_enabel="True"
noise_multiplier=1
max_grad_norm=0.1  # Clipping norms
delta_values=(1e-2 1e-5) 

Dp_e_enabel="True"
epsilon_values=(1.0 0.5)  # Used if Dp_e_enabel=True

now=$(date +"%m-%d")
for j in "${!num_clients_values[@]}"; do
    num_clients="${num_clients_values[j]}"

    for i in "${!epsilon_values[@]}"; do  
        target_epsilon="${epsilon_values[i]}"

        for k in "${!delta_values[@]}"; do
            target_delta="${delta_values[k]}"

            for seed in "${seeds[@]}"; do  
                folder_name="DP_e_${now}/C${num_clients}_R${ROUNDS}_mgn${max_grad_norm}_td${target_delta}_te${target_epsilon}/s${seed}"
                
                
                # export CUBLAS_WORKSPACE_CONFIG=:4096:8

                python ~/Diplomatic_incident/Differential_Privacy/Mnist_2/opacus_fl/main.py \
                    SEED="$seed" \
                    num_clients="$num_clients" \
                    ROUNDS="$ROUNDS" \
                    Dp_enabel="$Dp_enabel" \
                    target_delta="$target_delta" \
                    Dp_e_enabel="$Dp_e_enabel" \
                    learning_rate="$learning_rate" \
                    target_epsilon="$target_epsilon" \
                    folder_name="$folder_name" 


                # python ~/Diplomatic_incident/Differential_Privacy/Mnist_2/opacus_fl/test.py
                # sleep 0.01
            done
        done
    done
done

# cd ~/Diplomatic_incident/Differential_Privacy/Mnist_2/run_test
# chmod +x run_tests_DP_e.sh
# ./run_tests_DP_e.sh