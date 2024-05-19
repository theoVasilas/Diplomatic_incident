#!/bin/bash

BRANCH_FILE=".last_branch"

# Check if a commit message was provided
if [ -z "$1" ]; then
    echo "Error: No commit message provided."
    echo "Usage: ./gitmit.sh \"commit message\" [branch_name]"
    exit 1
fi

# Assigning variables for better readability
COMMIT_MESSAGE="$1"

# Check if a branch name was provided, else use the last saved branch name
if [ -z "$2" ]; then
    if [ -f "$BRANCH_FILE" ]; then
        BRANCH_NAME=$(cat "$BRANCH_FILE")
        echo "Using saved branch name: $BRANCH_NAME"
    else
        echo "Error: No branch name provided and no saved branch name found."
        echo "Usage: ./gitmit.sh \"commit message\" [branch_name]"
        exit 1
    fi
else
    BRANCH_NAME="$2"
    echo "$BRANCH_NAME" > "$BRANCH_FILE"
fi

# Run git commands
git add *
git commit -m "$COMMIT_MESSAGE"
git push origin "$BRANCH_NAME"
