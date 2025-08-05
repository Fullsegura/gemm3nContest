#!/bin/bash

# Navigate to the working directory containing the VLM alerts pipeline
cd /home/fullzecure/dev/google/gemma3n/vlm_alerts

# Source the Conda environment loader to prepare for activating the Python environment
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the conda environment 'g1' which includes required dependencies (e.g., PyTorch, Transformers, etc.)
conda activate g1

# Run the main inference script with the following parameters:
# --model_url: local API endpoint for image/video captioning or description
# --video_file: path to the video input (mining activity example)
# --api_key: not needed for local usage, placeholder
# --loop_video: enable looping of video input
# --hide_query: disable prompt display for cleaner UI or headless execution
python3 mainKagle.py --model_url http://localhost:8000/describe --video_file mining3.mp4 --api_key "No_Needed" --loop_video --hide_query