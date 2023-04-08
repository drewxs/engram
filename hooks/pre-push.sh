#!/bin/bash

echo "Stashing unstaged changes..."
git stash save --keep-index --include-untracked

echo "Running tests..."
if ! cargo test; then
    echo "Some tests failed. Popping stash..."
    git stash pop --quiet
    exit 1
fi

echo "All tests passed. Popping stash..."
git stash pop --quiet
