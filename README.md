# Eye Tracking Data Analysis

This repository contains code and resources that have been developed and used in the Adolphs Lab for analyzing eye tracking data.

## Table of Contents

- [Overview](#overview)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Data preprocessing](#data-preprocessing)
    - [Visualization of gaze data](#visualization-of-gaze-data)
    - [Basic feature extraction from video stimulus](#basic-feature-extraction-from-video-stimulus)
- [Acknowledgement](#acknowledgement)
- [Reference](#reference)

## Overview
Eye tracking is a powerful tool for studying visual attention and gaze behavior. By tracking eye movements, researchers can measure where people look, how long they look, and how their gaze patterns change over time, providing insight into their visual processing and attentional biases. However, eye tracking data can be complex and difficult to analyze without the right tools.

This repository provides a set of tools for analyzing eye tracking data. These tools include Python scripts and notebooks for preprocessing raw gaze data obtained from an eye tracker, visualizing gaze data using scatter plots and heat maps, performing similarity analysis between different subjects' gaze data, and conducting statistical analyses based on time spent looking at specific areas of interest (AOIs) within the visual stimulus. 

Please note that this repository is a work in progress and we are actively adding new features and improving existing ones. We appreciate your patience and understanding as we continue to work on this project. Thank you for your interest in our work!

## Getting Started
To help users understand and utilize the various functionalities of this repository, we have included detailed scripts and sample data in the [`examples`](./examples/) directory. Here we describe how to install the repository to a local machine and use the provided sample data to demonstrate some of the key features of the repository.

### Installation
We recommend using Anaconda and a new environment to install this repository on your local machine. This will ensure that the necessary dependencies are installed and managed in a separate environment, without interfering with other Python projects you may have installed. To get started, follow these steps:

- Install Anaconda or Miniconda by following the instructions on the official [website](https://www.anaconda.com/).

- Clone/download this repository to your local machine and navigate to the root directory of the repository in your terminal using the commands:
  ```bash
  git clone https://github.com/adolphslab/adolphslab_eyetracking
  cd adolphslab_eyetracking
  ```

- Create a new environment using the provided ‘make_env.yml’ file. This will create a new environment with the necessary dependencies installed. 
  ```bash
  conda env create --file make_env.yml
  ```

- Activate the environment and install this repository in editable mode using pip:
  ```bash
  conda activate eyetracking_env
  pip install -e .
  ```

Now we can use the repository and its features in this conda environment. For more information on using Anaconda and conda environments, please refer to their official [documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).


### Usage
This repository includes the following features for eye tracking data analysis.

### Data preprocessing
Eye tracking data can be recorded using a variety of eye tracking systems, each providing data in a different format. This variance in data format makes it difficult to implement a standard way of preprocessing data. We have included sample scripts in this repository (see [`examples/preprocess_rawdata/`](./examples/preprocess_rawdata/) ) that perform preprocessing of eye tracking data from some common data formats and put into a HDF file. The subsequent analyses were conducted using this standardized file format. The HDF (Hierarchical Data Format) file allows for efficient storage, retrieval, and manipulation of large amounts of data. 

In this hierarchical format, we organized the data based on subject IDs and task or stimulus names. An example of this hierarchical organization can be seen in the sample data provided in the [`examples/sample_data/sample_input/`](./examples/sample_data/sample_input/) directory. In the HDF file, groups and subgroups were created to organize the data, with each group representing a specific subject and stimulus combination. For instance, the HDF file included groups such as '/RA1001/office_sample', '/RA1002/office_sample', '/RA1003/office_sample', and so on. Each of these subjectID/stimulus combination contains a data frame where columns are recording time (in seconds), x and y coordinates of gaze data (in pixels), and pupil diameter (in arbitrary units) across recording time.   

### Visualization of gaze data
This section demonstrates how to load gaze data from a HDF file and visualize it using scatter plots and heat maps. A Jupyter Notebook is included in the [`examples/sampledata`](./examples/sample_data/visualize_etdata.ipynb) directory, which demonstrates how to utilize the related functions based on the sample data provided. We outline the main steps for visualizing the gaze data below:

```bash
# Please see ./examples/sample_data/visualize_etdata.ipynb for details

import os
import pandas as pd
from alabeye.etdata import ETdata

# Main directory for experiment data.
expdata_dir = os.path.abspath('./examples/sample_data/sample_input')

# HDF file of gaze data (see ../preprocess_rawdata about how to prepare this file)
hdf_filename = 'office_sample_etdata_compressed.hdf5'

hdf_file = os.path.join(expdata_dir, hdf_filename)

# 'ETdata' class from the 'alabeye.etdata' module handles most of the interface with the data
# an initial look into the hdf file to learn available data in the hdf file 
print(ETdata(data_file=hdf_file).available_tasks)

# load data for a specific task:
task2load = 'office_sample'

# the video file name associated with this particular experiment task
stimname_map = {'office_sample':'office_sample_vid.mp4'}

# in addition we add some optional info about the participants/subjects
subj_info_file = os.path.join(expdata_dir, 'participant_info.csv')

# use the 'ETdata' class for loading varios information about the experiment
sample_etdata = ETdata(taskname=task2load, data_file=hdf_file, 
                       subj_info_file=subj_info_file,
                       use_subj_info=True, # use the group info for the subjects 
                       stim_dir=expdata_dir, # directory where the video stimulus is
                       stimname_map=stimname_map,
                       )

# A quick visualization of gaze data
sample_etdata_re.visualize_gaze(save_viz=True,
                                show_viz=True, # if you get an openCV error, set to False
                                )
```

Those will produce the visualization:

![gaze scatters](./examples/sample_data/gaze_visualization/office_sample_vid_ETgaze.gif)


Similarly,  
```bash
# Visualize two groups separately with heatmaps.
sample_etdata_re.visualize_2groups(save_viz=True,
                                   show_viz=True, # if you get an openCV related error, set this to False
                                  )
```
will produce

![heatmaps viz](./examples/sample_data/gaze_visualization/office_sample_vid_compare_grps.gif)


### Basic feature extraction from video stimulus
We provide two sample scripts to demonstrate how to perform basic feature extraction from video stimulus using pre-trained neural networks, including face detection using the [Insightface](https://github.com/deepinsight/insightface) library and body part detection using the [detectron2/Densepose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) library.

To run these detection scripts, we suggest creating a separate conda environment so that the necessary dependencies can be installed without conflicts with other packages and versions in the current environment. See our notes [here](./examples/gazetime_aoi/detection/new_env.md) for creating a separate environment for the detection libraries. 

Once the installations are completed, we can use the provided detection scripts. 

To detect faces in the video stimulus, use [`facedetection_insightface_video.py`](./examples/gazetime_aoi/detection/facedetection_insightface_video.py). This script can be run from a terminal using the command:
```bash
python facedetection_insightface_video.py
```
This command will process the provided sample video located under [examples/sample_data/sample_input/] and generate a pickle file that contains the relevant face detection information. In addition, it will produce a video showing the visualization of detected face and face part (orange: eyes, blue: mouth) areas.

To use this script for other videos and to save outputs to a specific location, see the argument options:
```bash
# for instance 
python facedetection_insightface_video.py --video mynewvideo.mp4 --output_dir mynewfolder
```

To detect faces in the video stimulus, use [`facedetection_insightface_video.py`](./examples/gazetime_aoi/detection/facedetection_insightface_video.py). This script can be run from a terminal using the command:
```bash
python facedetection_insightface_video.py
```
This command will process the provided sample video located under [examples/sample_data/sample_input/] and generate a pickle file that contains the relevant face detection information. In addition, it will produce a video showing the visualization of detected face and face part (orange: eyes, blue: mouth) areas.

To use this script with other videos and save the outputs to a specific location, run the script as follows:
```bash
# for instance 
python facedetection_insightface_video.py --video mynewvideo.mp4 --output_dir mynewfolder
```

To detect body parts in the video stimulus, use [`bodyparts_densepose_video.py`](./examples/gazetime_aoi/detection/bodyparts_densepose_video.py). This script can be run from a terminal using the command:
```bash
python bodyparts_densepose_video.py
```
This command will process the provided sample video located under [examples/sample_data/sample_input/](./examples/sample_data/sample_input/) and generate a pickle file that contains the relevant face detection information (saved under [examples/sample_data/sample_output/](./examples/sample_data/sample_output/)). In addition, it will produce a video showing the visualization of detected face and face part (orange: eyes, green: mouth) areas (save in the same folder as the pickle file). 

![face detection](./examples/sample_data/sample_output/office_sample_vid_insightface.gif)

%%% <img src="./examples/sample_data/sample_output/office_sample_vid_insightface.gif" width="400" />


To use this script with other videos and save the outputs to a specific location, run the script as follows:
```bash
# for instance 
python bodyparts_densepose_video.py --video mynewvideo.mp4 --output_dir mynewfolder
```
This command will process the provided sample video located under [examples/sample_data/sample_input/] and generate a pickle file that contains the relevant body part detection information (saved under [examples/sample_data/sample_output/](./examples/sample_data/sample_output/)). In addition, it will produce a video showing the visualization of detected body parts (yellow: head, green:hands, purple: other body parts) areas (save in the same folder as the pickle file).

![body parts detection](./examples/sample_data/sample_output/office_sample_vid_densepose.gif)


These usage examples can serve as a starting point for your own projects and can be modified to suit your specific needs.

## Acknowledgement
This package has been developed as part of our eye tracking studies related to autism research. We thank the Simons Foundation Autism Research Initiative (SFARI) and the National Institute of Mental Health (NIMH) for supporting our research. 


## Reference
If you use this repository in your work, please cite our publication:
- Keles, U., Kliemann, D., Byrge, L. et al. Atypical gaze patterns in autistic adults are heterogeneous across but reliable within individuals. Molecular Autism 13, 39 (2022).   
DOI: [10.1186/s13229-022-00517-2](https://doi.org/10.1186/s13229-022-00517-2)



