#!/bin/bash
# This script launches the GemmaKagle AI video analysis pipeline
# It ensures the proper conda environment is activated before execution

# Navigate to the project directory containing the GemmaKagle pipeline
cd /home/fullzecure/dev/google/gemma3n

# Load conda environment activation script (required when not using login shells)
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the dedicated conda environment named "g1"
conda activate g1

# Run the Python script that starts the video-to-AI inference workflow
python gemmaKagle.py