## How to create a new environment for the detection libraries

We used the following commands to create a separate environment:

```bash
conda create -n aoidetect_env
conda activate aoidetect_env

# install pytorch following: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c conda-forge matplotlib pillow scikit-image tqdm h5py
conda install -c conda-forge av pims

# try installing opencv using conda
conda install -c conda-forge opencv
# if conda gets stuck in "Solving Environment", then opencv can be installed using pip
pip install opencv-python
pip install opencv-contrib-python

# to install the Insightface library for face detection:
# see: https://insightface.ai/
pip install onnxruntime-gpu 
pip install -U insightface 

# to install the Densepose library for body parts detection:
# see: https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/GETTING_STARTED.md
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose

# if above command gives an error, then try installing first the detectron2 library, then install the Densepose:
# see: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```
