<div align="center">   
  
# RPG 

</div>

[![Paper](https://img.shields.io/badge/arXiv-2511.18757-b31b1b.svg)](https://arxiv.org/abs/2511.18757)
[![Project](https://img.shields.io/badge/Project-RPG-green)](https://github.com/Yongqi7/RPG)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-orange)](https://drive.google.com/drive/folders/1K4gA79MLWn5xc8NzPs8l89PUvrWWFTeV)
[![License](https://img.shields.io/badge/License-TBD-lightgrey)](#license)

https://github.com/user-attachments/assets/ab74daee-a8cc-4c1c-883c-dfb19660014b

## Overview

This repository builds a cooperative end-to-end autonomous driving models RPG. It extends UniAD from single-vehicle camera perception to multi-vehicle V2V scenarios, where an ego vehicle can use sender-side information for tracking, mapping, motion forecasting, occupancy prediction, and planning.

The repository focuses on reference points guided cooperation and utilities for bandwidth-limit fusion analysis.

Main components include:

- UniAD-based end-to-end driving pipeline for camera-only cooperative perception and planning.
- Agent query fusion modules for matching, aligning, complementing, and fusing sender queries into ego tracking queries.
- Reference points fusion switches for studying compact cooperative communication.
- Configurations for cooperative tracking, tracking-map, and end-to-end experiments.
- Dataset converters and evaluation utilities for M3CAD/OpenV2V-style multi-vehicle data.

## Getting Started

This project follows the structure of UniAD and related cooperative autonomous driving codebases, with additional modules for multi-agent fusion.

- [Installation](docs/INSTALL.md)
- [Data Preparation](docs/DATA_PREP.md)
- [Training and Evaluation](docs/TRAIN_EVAL.md)

## Installation

Install the submodules and project dependencies before running training or evaluation.

```bash
cd submodules/nuscenes-devkit/setup/
pip install -e .
cd ../../..

cd submodules/OpenCOOD
pip install -e .
cd ../..
```

For the full environment setup, including PyTorch, MMCV, MMDetection3D, and pretrained UniAD weights, see [docs/INSTALL.md](docs/INSTALL.md).

## Data Preparation

Download the dataset from [Google Drive](https://drive.google.com/drive/folders/1K4gA79MLWn5xc8NzPs8l89PUvrWWFTeV), then prepare it following the layout described in [docs/DATA_PREP.md](docs/DATA_PREP.md). Large datasets, generated info files, checkpoints, logs, and visualization outputs should stay outside Git tracking.

## Training and Evaluation

Example distributed evaluation command:

```bash
./tools/uniad_dist_eval.sh \
    ./projects/configs/submodule_fusion/submodules_e2e.py \
    /PATH/TO/YOUR/CKPT.pth \
    N_GPUS
```

Example distributed training command:

```bash
./tools/uniad_dist_train.sh \
    ./projects/configs/submodule_fusion/submodules_e2e.py \
    N_GPUS
```

For more details and alternative launchers, see [docs/TRAIN_EVAL.md](docs/TRAIN_EVAL.md).

## Visualization

Use the UniAD visualization utility after producing prediction results:

```bash
python ./tools/analysis_tools/visualize/run.py \
    --predroot /PATH/TO/YOUR/RESULTS.pkl \
    --out_folder /PATH/TO/YOUR/OUTPUT \
    --demo_video demo.avi \
    --project_to_cam True
```

<!-- ## TODO

- [ ] Add public checkpoints. -->


## Contact

For questions, feedback, or collaborations, please open a GitHub issue or contact the repository maintainers.

## Citation

If you find this repository useful in your research, please cite our paper: [From Features to Reference Points: Lightweight and Adaptive Fusion for Cooperative Autonomous Driving](https://arxiv.org/abs/2511.18757).

```bibtex
@article{zhu2025features,
  title={From Features to Reference Points: Lightweight and Adaptive Fusion for Cooperative Autonomous Driving},
  author={Zhu, Yongqi and Zhu, Morui and Chen, Qi and Qu, Deyuan and Luo, Isabella and Fu, Song and Yang, Qing},
  journal={arXiv preprint arXiv:2511.18757},
  year={2025}
}
```

## Acknowledgements

This repository builds on and adapts components from the following projects:

- [UniAD](https://github.com/OpenDriveLab/UniAD)
- [M3CAD](https://github.com/zhumorui/M3CAD)
- [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD)
- [nuScenes devkit](https://github.com/nutonomy/nuscenes-devkit)
- [Bench2DriveZoo](https://github.com/Thinklab-SJTU/Bench2DriveZoo)

## License

The license for this repository is to be determined. Please check the licenses of upstream projects before using or redistributing derived components.
