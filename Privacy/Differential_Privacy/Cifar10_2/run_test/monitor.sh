# clear
# cd Differential_Privacy/Mnist_2/run_test/
# chmod +x monitor.sh
# ./monitor.sh

#!/bin/bash

# Base directory (update this if needed)
BASE_DIR="Differential_Privacy/Mnist_2/run_test/outputs"

# Find the latest "dif_clients_XX-XX" folder
DATE=$(date +%m-%d)
LATEST_MAIN_DIR=$(ls -td "$BASE_DIR"/dif_clients_"$DATE"* 2>/dev/null | head -1)
echo "Open $LATEST_MAIN_DIR"

# If no matching directory found, exit
if [ -z "$LATEST_MAIN_DIR" ]; then
    echo "No dif_clients_* directory found!"
    exit 1
fi

# Find the latest subfolder inside the latest dif_clients_* directory
LATEST_SUB_DIR=$(ls -td "$LATEST_MAIN_DIR"/* 2>/dev/null | head -1)
echo "Open $LATEST_SUB_DIR"

# If no subfolder found, exit
if [ -z "$LATEST_SUB_DIR" ]; then
    echo "No subfolders found inside $LATEST_MAIN_DIR!"
    exit 1
fi

# Find the latest subfolder inside the latest dif_clients_* directory
LATEST_SEED_DIR=$(ls -td "$LATEST_SUB_DIR"/* 2>/dev/null | head -1)
echo "Open $LATEST_SEED_DIR"

# If no subfolder found, exit
if [ -z "$LATEST_SEED_DIR" ]; then
    echo "No subfolders found inside $LATEST_SUB_DIR!"
    exit 1
fi

# Move into the latest subfolder
cd "$LATEST_SEED_DIR" || exit 1

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

# Show real-time updates to the JSON file
tail -f server_evaluation_history.json
