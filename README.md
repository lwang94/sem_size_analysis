# SEM Size Analysis

SAEMI (Size Analysis for Electron Microscopy Images) is a tool for obtaining the size distribution of nanoparticles in electron microscopy images. 
It uses an image segmentation model trained through a neural network to detect nanoparticle sizes and locations. The neural network used is a u-net
with a resnet downsample created through the fastai library. From there, it then obtains a histogram of the sizes of each particle, providing 
important information such as mean particle size, median particle size etc. This tool can be used for characterizing new materials, analyzing 
environmental effects on chemical systems, ensure quality control for routine synthesis and much more. 


### Requirements:
---------------



### To start a new project, run: