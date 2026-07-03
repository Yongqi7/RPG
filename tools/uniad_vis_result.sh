#!/bin/bash

python ./tools/analysis_tools/visualize/run.py \
    --predroot output/results_uniad_base.pkl \
    --out_folder ./output/vis_track_uniad_base \
    --demo_video ./output/vis_track_uniad_base/output_video_uniad_base.mp4 \
    --project_to_cam