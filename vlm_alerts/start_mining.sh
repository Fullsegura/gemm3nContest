#!/bin/bash
cd /home/fullzecure/dev/google/gemma3n/vlm_alerts
source ~/miniconda3/etc/profile.d/conda.sh
conda activate g1
python3 mainKagle.py --model_url http://localhost:8000/describe --video_file mining3.mp4 --api_key "No_Needed" --loop_video --hide_query
