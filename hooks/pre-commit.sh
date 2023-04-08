#!/bin/bash

echo "Stashing unstaged changes..."
git stash save --keep-index --include-untracked

echo "Formatting code..."
cargo fmt

echo "Checking code..."
if ! cargo check; then
    echo "Code check failed with errors. Popping stash..."
    git stash pop --quiet
    exit 1
fi

echo "Code check successful. Popping stash..."
git stash pop --quiet
