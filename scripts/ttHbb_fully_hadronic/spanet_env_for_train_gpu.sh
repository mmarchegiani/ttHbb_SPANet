#!/bin/bash
#install necessary packages for training on GPU

#use within the active environment of the SPANet repository 

python -m pip install numpy
python -m pip install torch
python -m pip install h5py
python -m pip install PyYAML
python -m pip install opt_einsum
python -m pip install numba
python -m pip install pytorch_lightning
python -m pip install mdmm
python -m pip install scikit-learn