# Implementing a Deep Cascade of Convolutional Neural Networks for MR Image Reconstruction on Newborn Brain Data with Domain Adaptation

This repository contains code for the project components of ENSF 619 and ENEL 645. Code components for the final projects are structured as follows:

## `/colab`

Contains the training and testing notebooks with Google Colab links for execution on GPU hardware. It is recommended to use the "premium" class GPUs (NVIDIA A100 40GB) for reasonable training and testing performance. The notebooks mount my personal google drive to access datasets, however this can be modified in the code to accept other filesystem options upon request.

## `/final_project`

Contains the Python modules required for all components of the project. Below is a breakdown of each module and subdirectory.

### `/final_project/contrast_inversion.py`

Tools for performing contrast inversion on adult MRI data. Executing the module directly will process all of the adult original images and save them to disk. This requires the adult data to be on disk and follow the directory conventions established in `paths.py`.

### `/final_project/model.py`

All PyTorch code pertaining to DANN model architecture.

### `/final_project/data.py`

Contains the PyTorch dataset class based on `torch.utils.data.TensorDataset` along with utility functions for processing images and loading data as NumPy arrays. This requires the adult original, negative, and newborn data to be on disk and follow the directory conventions established in `paths.py`.

### `/final_project/deprecated.py`

Deprecated code from exploration or saved for later use in the project. This is a general "graveyard" for code which may be useful.

### `/final_project/exploration.py`

Freeform exploration module taking advantage of VSCode Interactive Python functionality (`# %%` cell magic). This is a constantly evolving module used to expedite development locally.

### `/final_project/model.py`

All PyTorch code pertaining to DeepCascade model architecture.

### `/final_project/paths.py`

Module defining directory structure and relevant project paths for consistency. Paths are defined relative to the module. If you wish to change the directory structure, please use this module to modify paths as other modules import from here for consistency.

### `/final_project/sampling.py`

Contains functions for generating 2D sampling masks for undersampling k-space images. Code here has been adapted from [https://github.com/rmsouza01/CD-Deep-Cascade-MR-Reconstruction](https://github.com/rmsouza01/CD-Deep-Cascade-MR-Reconstruction).

### `/final_project/transform.py`

Contains the transform class applying image transforms to training data for augmentation.

### `/final_project/utils.py`

Contains general utility functions used for data processing, model training and validation, and generating figures.
