# CrowdPig
#### This is a pig counting application thanks for [CrowdDet team](https://github.com/megvii-model/CrowdDetection).
## How to run
#### you better have a ubuntu 18.04 with CUDA 10.2 and Anaconda installed with Nvidia GPU.
```shell script
conda create -n CrowdPig python=3.6 anaconda
conda activate CrowdPig
cd $ROOT_OF_CrowdPig
cd det_tools_cuda
python3 setup.py install
cd ..
pip3 install -r requirements.txt
check config.py to make sure your data folder are right.
python3 train.py
python3 test.py
```
