#!/bin/bash

# Check if a commit message was provided
if [ -z "$1" ]; then
    echo "Error: No commit message provided."
    echo "Usage: ./gitmit.sh \"commit message\" branch_name"
    exit 1
fi

# Check if a branch name was provided
if [ -z "$2" ]; then
    echo "Error: No branch name provided."
    echo "Usage: ./gitmit.sh \"commit message\" branch_name"
    exit 1
fi

# Assigning variables for better readability
COMMIT_MESSAGE="$1"
BRANCH_NAME="$2"

# Run git commands
git add *
git commit -m "$COMMIT_MESSAGE"
git push origin "$BRANCH_NAME"
