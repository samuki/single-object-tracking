#!/usr/bin/env bash


git clone https://github.com/foolwood/SiamMask.git && cd SiamMask
export SiamMask=$PWD
conda create -n siammask2 python=3.6
source activate siammask2
pip install -r requirements.txt

cd utils/pyvotkit
python setup.py build_ext --inplace
cd ../../

cd utils/pysot/utils/
python setup.py build_ext --inplace
cd ../../../

export PYTHONPATH=$PWD:$PYTHONPATH
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
export PYTHONPATH=$PWD:$PYTHONPATH

cd ../../../
python -m pip install --upgrade pip

pip install --use-feature=2020-resolver -r requirements.txt