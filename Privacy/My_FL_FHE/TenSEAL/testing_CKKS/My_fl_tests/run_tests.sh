#!/bin/bash
# 101 222 3333 404 5555 6969 777 8888 9090 1337
# 2048 4096 8192 16384 32768
# 2 5 10 20 50 100
degrees=(32768 )
seeds=(101 222 3333 404 5555 6969)
num_clients_values=(5 20)

# Loop through each combination of SEED and num_clients
for degree in "${degrees[@]}"
do
    for num_clients in "${num_clients_values[@]}"
    do
        for seed in "${seeds[@]}"
        do
            echo "Running experiment with SEED=$seed  num_clients=$num_clients degree=$degree"
            echo ""
            python /home/theo_ubuntu/Diplomatic_incident/HE/TenSEAL/My_HE_FHE/main.py SEED=$seed num_clients=$num_clients degree=$degree
        done
    done
done