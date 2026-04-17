#!/bin/bash

# Create and setup conda environment for culminate

echo "Creating conda environment 'culminate'..."
conda create -n culminate python=3.10 -y

echo "Activating environment..."
conda activate culminate

echo "Installing dependencies..."
conda install numpy pandas astropy -y

echo "Setup complete! Activate the environment with: conda activate culminate"
