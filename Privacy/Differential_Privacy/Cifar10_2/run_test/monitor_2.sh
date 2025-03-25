#!/bin/bash

# Base directory (update this if needed)
BASE_DIR="Differential_Privacy/Mnist_2/run_test/outputs"

# Function to find the latest experiment folder
find_latest_experiment() {
    DATE=$(date +%m-%d)
    LATEST_MAIN_DIR=$(ls -td "$BASE_DIR"/dif_clients_"$DATE"* 2>/dev/null | head -1)

    if [ -z "$LATEST_MAIN_DIR" ]; then
        echo "No dif_clients_* directory found!"
        return 1
    fi

    LATEST_SUB_DIR=$(ls -td "$LATEST_MAIN_DIR"/* 2>/dev/null | head -1)
    if [ -z "$LATEST_SUB_DIR" ]; then
        echo "No subfolders found inside $LATEST_MAIN_DIR!"
        return 1
    fi

    LATEST_SEED_DIR=$(ls -td "$LATEST_SUB_DIR"/* 2>/dev/null | head -1)
    if [ -z "$LATEST_SEED_DIR" ]; then
        echo "No subfolders found inside $LATEST_SUB_DIR!"
        return 1
    fi

    echo "$LATEST_SEED_DIR"
    return 0
}

# Function to monitor the JSON file with tail -f
monitor_json() {
    local latest_experiment
    latest_experiment=$(find_latest_experiment) || return 1

    cd "$latest_experiment" || return 1
    echo "Monitoring: $latest_experiment"

    # Print the contents of .hydra/overrides.yaml if it exists
    OVERRIDES_FILE=".hydra/overrides.yaml"
    if [ -f "$OVERRIDES_FILE" ]; then
        echo "======================"
        echo "Overrides Configuration:"
        echo "======================"
        cat "$OVERRIDES_FILE"
        echo "======================"
    else
        echo "No overrides.yaml file found in .hydra!"
    fi

    # Monitor server_evaluation_history.json using tail -f
    local json_file="server_evaluation_history.json"
    if [ ! -f "$json_file" ]; then
        echo "File $json_file not found! Looking for a new experiment..."
        return 1
    fi

    echo "Tailing $json_file..."
    tail -f "$json_file" &
    TAIL_PID=$!

    # Monitor file size changes while tail is running
    local prev_size=$(stat -c %s "$json_file")
    while true; do
        sleep 20  # Adjust sleep time based on update frequency

        if [ ! -f "$json_file" ]; then
            echo "File deleted! Restarting search..."
            break
        fi

        current_size=$(stat -c %s "$json_file")
        if [ "$current_size" -eq "$prev_size" ]; then
            echo "File $json_file stopped updating! Restarting search..."
            break
        fi

        prev_size=$current_size
    done

    # Kill the tail process when restarting
    kill $TAIL_PID
    return 0
}

# Loop to restart monitoring when needed
while true; do
    monitor_json
    echo "Restarting process..."
    sleep 2
done
