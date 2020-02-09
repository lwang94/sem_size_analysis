"""
Module containing functions for transforming data
to be put into deep learning model
"""


import numpy as np
from skimage import transform
from fastai.vision import Image
from torch import FloatTensor

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

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


def reflect_pad(img, remove_coord):
    """
    Replaces region in img defined by remove_coord with a
    'reflection' of the region above and below remove_coord
    """
    remove_x1, remove_y1, remove_x2, remove_y2 = remove_coord
    remove_height = remove_y2 - remove_y1
    br_ylim = remove_y2 + remove_height // 2
    tr_ylim = remove_y1 - (remove_height - remove_height // 2)
    if br_ylim >= img.shape[0]:
        br_ylim = img.shape[0]
        tr_ylim = remove_y1 - (remove_height - (br_ylim - remove_y2))
    elif tr_ylim <= 0:
        tr_ylim = 0
        br_ylim = remove_y2 - (remove_height - remove_y1)
    bot_reflect = img[remove_y2:br_ylim, remove_x1:remove_x2, :]
    bot_reflect = np.flipud(bot_reflect)
    top_reflect = img[tr_ylim: remove_y1, remove_x1:remove_x2, :]
    top_reflect = np.flipud(top_reflect)
    reflect_pad = np.concatenate((top_reflect, bot_reflect), axis=0)
    imgcopy = img.copy()
    imgcopy[remove_y1:remove_y2, remove_x1:remove_x2] = reflect_pad
    return imgcopy


def replace_constant(img, remove_coord, constant):
    remove_x1, remove_y1, remove_x2, remove_y2 = remove_coord
    imgcopy = img.copy()
    imgcopy[remove_y1:remove_y2, remove_x1:remove_x2] = constant
    return imgcopy


def replace_neighbor(img, remove_coord):
    remove_x1, remove_y1, remove_x2, remove_y2 = remove_coord
    remove_height = remove_y2 - remove_y1
    imgcopy = img.copy()
    bot_neighbor = remove_y2 + 1
    top_neighbor = remove_y1 - 1
    if bot_neighbor == img.shape[0]:
        for i in range(remove_y1, remove_y2):
            imgcopy[i, remove_x1:remove_x2, :] = imgcopy[
                                                     top_neighbor,
                                                     remove_x1:remove_x2,
                                                     :
                                                 ]
    elif top_neighbor == 0:
        for i in range(remove_y1, remove_y2):
            imgcopy[i, remove_x1:remove_x2, :] = imgcopy[
                                                     bot_neighbor,
                                                     remove_x1:remove_x2,
                                                     :
                                                 ]
    else:
        ylim = remove_y2 - (remove_height // 2)
        for i in range(remove_y1, remove_y2):
            if i <= ylim:
                imgcopy[i, remove_x1:remove_x2, :] = imgcopy[
                                                         top_neighbor,
                                                         remove_x1:remove_x2,
                                                         :
                                                     ]
            else:
                imgcopy[i, remove_x1:remove_x2, :] = imgcopy[
                                                         bot_neighbor,
                                                         remove_x1:remove_x2,
                                                         :
                                                     ]
    return imgcopy


def replace_whitewm_constant(img, roi, constant, threshold=225):
    x1, y1, x2, y2 = roi
    imgcopy = img.copy()
    img_roi = imgcopy[y1:y2, x1:x2, :]
    img_roi[img_roi >= threshold] = constant
    imgcopy[y1:y2, x1:x2, :] = img_roi
    return imgcopy


def replace_whitewm_avg(img, roi, threshold=225, method='mean'):
    x1, y1, x2, y2 = roi
    imgcopy = img.copy()
    img_roi = imgcopy[y1:y2, x1:x2, :]

    notwm = img_roi[img_roi < threshold]

    if method == 'mean':
        avg = np.mean(notwm)
    elif method == 'median':
        avg = np.median(notwm)
    else:
        raise ValueError('method must be "mean" or "median"')

    img_roi[img_roi >= threshold] = avg
    imgcopy[y1:y2, x1:x2, :] = img_roi
    return imgcopy


def whitewm_moving_avg(img, roi,
                       window_size=(5, 5), threshold=225, method='mean'):
    x1, y1, x2, y2 = roi
    imgcopy = img.copy()
    img_roi = imgcopy[y1:y2, x1:x2, :]

    for i in range(0, img_roi.shape[0], window_size[1]):
        for j in range(0, img_roi.shape[1], window_size[0]):
            window = img_roi[i:i+window_size[1], j:j+window_size[0], :]
            notwm = window[window < threshold]

            if method == 'mean':
                avg = np.mean(notwm)
            elif method == 'median':
                avg = np.median(notwm)
            else:
                raise ValueError('method must be "mean" or "median"')

            window[window >= threshold] = avg
            img_roi[i:i+window_size[1], j:j+window_size[0], :] = window

    imgcopy[y1:y2, x1:x2, :] = img_roi
    return imgcopy



# img = mpimg.imread('C:/Users/lawre/Documents/sem_size_analysis/data/dataset/good/train_x/L2_7d7ca0943bf232354a4182ce3b17c928.jpg')
# # img = reflect_pad(img, (0, 384, 1024, 768))

# remove_x1, remove_y1, remove_x2, remove_y2 = 630, 325, 1024, 768
# remove_width = remove_x2 - remove_x1

# xlim = remove_x1 - remove_width
# reflect = img[remove_y1:remove_y2, xlim:remove_x1, :]
# reflect = np.fliplr(reflect)
# imgcopy = img.copy()
# imgcopy[remove_y1:remove_y2, remove_x1:remove_x2, :] = reflect



# plt.imshow(imgcopy)
# plt.show()