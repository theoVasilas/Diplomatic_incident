#!/bin/bash

seeds=(101 222 3333 4044 5555 6967 7777 8888 9090 1337 )
num_clients_values=(2 10 50 100 )

max_norm_values=(1.0 2.0 3.0 )    # clipping_norm
noise_mult_values=(1.0 2 5 10) 
delta_values=(1e-1 1e-2 1e-3 1e-4 1e-5) #
# epsilon=(50 )        # Opacus calulates automaticaly 

ROUNDS_values=(20 )

matrix=(1)
for i in "${!num_clients_values[@]}"
do  
    ROUNDS="${ROUNDS_values[0]}"
    max_grad_norm="${max_norm_values[1]}"
    noise_multiplier="${noise_mult_values[0]}"
    
    num_clients="${num_clients_values[i]}"
    target_delta="${delta_values[i]}"

    for seed in "${seeds[@]}"
    do  
        clear
        echo " "
        echo " "
        echo " Running experiment Differcial Privacy "
        echo "-Overwriting..."
        echo " SEED=$seed "
        echo " num_clients=$num_clients "
        echo " ROUNDS=$ROUNDS "
        echo " max_grad_norm=$max_grad_norm "
        echo " noise_multiplier=$noise_multiplier "
        echo " target_delta=$target_delta "
        echo " "
        echo " "
        python Differential_Privacy/Mnist_2/opacus_fl/main.py \
            SEED=$seed num_clients=$num_clients ROUNDS=$ROUNDS \
            max_grad_norm=$max_grad_norm noise_multiplier=$noise_multiplier \
            target_delta=$target_delta   
    done
done