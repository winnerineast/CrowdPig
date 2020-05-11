1. conda create -n CrowdPig python=3.6 anaconda
2. conda activate CrowdPig
3. cd $ROOT_OF_CrowdPig
4. cd det_tools_cuda
5. python3 setup.py install
6. cd ..
7. pip3 install -r requirements.txt
8. check config.py to make sure your data folder are right.
9. python3 train.py
10. python3 test.py