# MELTNET

## Overview
MELTNET is a deep learning tool that emulates the NEMO ocean model to calculate Antarctic ice shelf melt rates. MELTNET consists of two seperate networks: a segmentation network (MELTNET_SEG) and a DAE network (MELTNET_DAE). This repository contains the MATLAB code necessary to train both networks and make predictions using those trained networks. The research paper describing MELTNET can be found [here](https://tc.copernicus.org/preprints/tc-2021-396/).

## Requirements
MELTNET is written entirely in MATLAB and makes use of the following MATLAB toolboxes:
1. [Deep learning toolbox](https://uk.mathworks.com/products/deep-learning.html)
2. [Computer vision toolbox](https://uk.mathworks.com/products/computer-vision.html)
3. [Parallel computing toolbox](https://uk.mathworks.com/products/parallel-computing.html)

## How to
MELTNET consists of two networks: 
1. A segmentation network that converts input geometries into a single channel labelled melt rate map (referred to as MELTNET_SEG). MELTNET_SEG is trained on batches of 4-channel images (64x64x4xN) by calculating a loss function against target images. Training is done with the *trainMELTNET_SEG* function and requires two folders containing input test cases called *training_inputs* and *validation_inputs*, as well as corresponding folders containing the groundtruth segmented images in *training_targets* and *validation_targets*.

## License
