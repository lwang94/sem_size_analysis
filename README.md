# SAEMI

## Description:
--------------
SAEMI (Size Analysis for Electron Microscopy Images) is a tool for obtaining the size distribution of nanoparticles in electron microscopy images. 
It uses an image segmentation model trained through a neural network to detect nanoparticle sizes and locations. The neural network used is a u-net
with a resnet downsample created through the fastai library. From there, it then obtains a histogram of the sizes of each particle, providing 
important information such as mean particle size, median particle size etc. This tool can be used for characterizing new materials, analyzing 
environmental effects on chemical systems, ensure quality control for routine synthesis and much more. 


## Installation:
----------------
### Dependencies
saemi requires:
- Python == 3.6
- fastai == 1.0.6.0
- scikit-image == 0.16.2
- dash == 1.8.0
- gdown == 3.10.1

### User installation (TO DO)
To pip install saemi, run:
```
pip install git+git://github.com/lwang94/sem_size_analysis
```


## To Run:
----------------
To open the app first cd to the root directory and run in the command shell
```
python -m src.backend_api
```
The above will open the backend api server on localhost:5000. To interact via a front-end UI, run in a different command shell
```
python -m app
```
You should see an output that looks like this.
```
Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)
```
Simply copy and paste the http address into a local browser and press Enter.

### To run tests
With coverage:
```
pytest --cov-config=.coveragerc --cov=sem_size_analysis tests/
```