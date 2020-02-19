"""
Module containing functions for transforming data
to be put into deep learning model
"""


import numpy as np
from skimage import transform
from fastai.vision import Image
from torch import FloatTensor


def fastai_image(img):
    """Turns numpy array into fastai Image object"""
    img = FloatTensor(img)
    img = img.permute(2, 0, 1)
    return Image(img)


def resize(img, size, order=1):
    """
    Resizes img into appropriate size using Nth order interpolation

    Parameters
    ------------------------------------------
    img : ndarray
        Contains image data
    size : tuple
        Contains width and height for resized image
    order : int, optional
        Order of the spline interpolation. Has to be in range 0-5.
        Defaults to 1.

    Returns
    ------------------------------------
    resized_img : ndarray
    """
    return transform.resize(img, size, order=order)


def make_3channel(img):
    """Ensures input image is a 3 channel image for training/predicting"""
    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
    img = np.concatenate((img, img, img), axis=2)

    return img
