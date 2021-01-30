#!/usr/bin/env bash

git clone https://github.com/foolwood/SiamMask.git
cd SiamMask
export SiamMask=$PWD

cd utils/pyvotkit
python3 setup.py build_ext --inplace
cd ../../

cd utils/pysot/utils/
python3 setup.py build_ext --inplace
cd ../../../

export PYTHONPATH=$PWD:$PYTHONPATH
cd $SiamMask/experiments/siammask_sharp
wget http://www.robots.ox.ac.uk/~qwang/SiamMask_DAVIS.pth
export PYTHONPATH=$PWD:$PYTHONPATH

cd ../../../
