# SAEMI

## Description
--------------
SAEMI (Size Analysis for Electron Microscopy Images) is a tool for obtaining the size distribution of nanoparticles in electron microscopy images. 
It uses an image segmentation model trained through a neural network to detect nanoparticle sizes and locations. The neural network used is a [u-net](https://arxiv.org/pdf/1505.04597.pdf) 
with a resnet downsample created through the [fastai](https://github.com/fastai/fastai) library. From there, it then obtains a histogram of the sizes of each particle, providing important 
information such as mean particle size, median particle size etc. This tool can be used for characterizing new materials, analyzing environmental effects on chemical systems, 
ensure quality control for routine synthesis and much more. 

## Dependencies
----------------
- Python == 3.6
- dash == 1.9.1
- dash-canvas == 0.1.0
- fastai == 1.0.60
- gdown == 3.10.1

## Website
-------------
You can find the app hosted on Heroku [here](https://saemi.herokuapp.com/)

## Using the App
-------------
Please check the [user docs](https://github.com/lwang94/sem_size_analysis/blob/master/docs/user_docs.md) for more details. 
Below is a summary of the steps a user would take to obtain the size distribution of an electron microscopy image using this app. 

1. Preprocess the image: before using the app, remove any metadata such as the scale bar that may interfere with the deep learning prediction

2. Upload the image: upload the image to the app using the buttons on the homepage. The image will then be segmented by the model and a size distribution calculated from the segmentation

3. Postprocess the image: After tbe size distribution is calculated, the user can edit the prediction to give more accurate results in case the model did not satisfactorily segment the image. 
Check the [user docs](https://github.com/lwang94/sem_size_analysis/blob/master/docs/user_docs.md) for more details on post processing

4. Download the calculated particle sizes as a .csv file using the Download link provided. Keep in mind that the particle sizes are in units of pixels so it is up to the user to convert
that to a physical distance.


<!-- 
INCLUDE MEDIUM ARTICLE IN USING THE APP

## Installation
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


## To Run
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
-->