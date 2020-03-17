"""
Module containing functions for transforming data
to be put into deep learning model
"""

import numpy as np
from skimage import transform
from fastai.vision import Image
from fastai.torch_core import to_half
from torch import HalfTensor


def fastai_image(img):
    """Turns numpy array into fastai Image object"""
    img = HalfTensor(img)
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

    Parameters
    -------------------------------------
    img: ndarray
        Contains image data
    remove_region: tuple
        Coordinates of box containing the remove_region. Order is: (
            top left x-coordinate,
            top left y-coordinate,
            bottom right x-coordinate,
            bottom right y-coordinate
        )

    Returns
    -------------------------------------
    ndarray
        transformed image
    """
    remove_x1, remove_y1, remove_x2, remove_y2 = remove_coord
    remove_height = remove_y2 - remove_y1

    # find the y-coordinates to reflect from
    br_ylim = remove_y2 + remove_height // 2
    tr_ylim = remove_y1 - (remove_height - remove_height // 2)

    # make sure reflection limits are not past the edge of image
    if br_ylim >= img.shape[0]:
        br_ylim = img.shape[0]
        tr_ylim = remove_y1 - (remove_height - (br_ylim - remove_y2))
    elif tr_ylim <= 0:
        tr_ylim = 0
        br_ylim = remove_y2 + (remove_height - remove_y1)

    # create reflection array
    bot_reflect = img[remove_y2:br_ylim, remove_x1:remove_x2, :]
    bot_reflect = np.flipud(bot_reflect)
    top_reflect = img[tr_ylim: remove_y1, remove_x1:remove_x2, :]
    top_reflect = np.flipud(top_reflect)
    reflect_pad = np.concatenate((top_reflect, bot_reflect), axis=0)

    # replace remove_coord with reflection array
    imgcopy = img.copy()
    imgcopy[remove_y1:remove_y2, remove_x1:remove_x2] = reflect_pad
    return imgcopy


def replace_constant(img, remove_coord, constant):
    """
    Replaces region in image defined by remove_coord with constant value

    Parameters
    -------------------------------------
    img: ndarray
        Contains image data
    remove_region: tuple
        Coordinates of box containing the remove_region. Order is: (
            top left x-coordinate,
            top left y-coordinate,
            bottom right x-coordinate,
            bottom right y-coordinate
        )
    constant: int
        Value to replace all colors in remove_region
    Returns
    -------------------------------------
    ndarray
        transformed image
    """
    remove_x1, remove_y1, remove_x2, remove_y2 = remove_coord
    imgcopy = img.copy()
    imgcopy[remove_y1:remove_y2, remove_x1:remove_x2] = constant
    return imgcopy


def replace_neighbor(img, remove_coord):
    """
    Replaces region in image defined by remove_coord with its
    neighbor in the y-axis

    Parameters
    -------------------------------------
    img: ndarray
        Contains image data
    remove_region: tuple
        Coordinates of box containing the remove_region. Order is: (
            top left x-coordinate,
            top left y-coordinate,
            bottom right x-coordinate,
            bottom right y-coordinate
        )

    Returns
    -------------------------------------
    ndarray
        transformed image
    """
    remove_x1, remove_y1, remove_x2, remove_y2 = remove_coord
    imgcopy = img.copy()

    # find neighboring values
    bot_neighbor = remove_y2 + 1
    top_neighbor = remove_y1 - 1

    # ensure neighbors are not past the edge of the image
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
    # replace remove_coord with neighboring values on top and bottom
    else:
        ylim = remove_y2 - ((remove_y2 - remove_y1) // 2)
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


def replace_whitewm_constant(img, remove_region, constant, threshold=225):
    """
    Replaces white watermarks with constant value

    Parameters
    -------------------------------------
    img: ndarray
        Contains image data
    remove_region: tuple
        Coordinates of box containing the remove_region. Order is: (
            top left x-coordinate,
            top left y-coordinate,
            bottom right x-coordinate,
            bottom right y-coordinate
        )
    contant: int
        Value to replace all values with in remove_region
    threshold: int, optional
        Color threshold for white value. Default is 225

    Returns
    -------------------------------------
    ndarray
        transformed image
    """
    x1, y1, x2, y2 = remove_region
    imgcopy = img.copy()
    img_rr = imgcopy[y1:y2, x1:x2, :]
    img_rr[img_rr >= threshold] = constant
    imgcopy[y1:y2, x1:x2, :] = img_rr
    return imgcopy


def replace_whitewm_avg(img, remove_region, threshold=225, method='mean'):
    """
    Replaces white watermarks with mean or median of values
    in remove_region.

    Parameters
    -------------------------------------
    img: ndarray
        Contains image data
    remove_region: tuple
        Coordinates of box containing the remove_region. Order is: (
            top left x-coordinate,
            top left y-coordinate,
            bottom right x-coordinate,
            bottom right y-coordinate
        )
    threshold: int, optional
        Color threshold for white value. Default is 225
    method: {'mean', 'median'}, optional
        Method of calculating average. Default is 'mean'

    Returns
    -------------------------------------
    ndarray
        transformed image
    """
    x1, y1, x2, y2 = remove_region
    imgcopy = img.copy()
    img_rr = imgcopy[y1:y2, x1:x2, :]

    # find all pixels in remove_region that is not the white watermark
    notwm = img_rr[img_rr < threshold]

    if method == 'mean':
        avg = np.mean(notwm)
    elif method == 'median':
        avg = np.median(notwm)
    else:
        raise ValueError('method must be "mean" or "median"')

    img_rr[img_rr >= threshold] = avg
    imgcopy[y1:y2, x1:x2, :] = img_rr
    return imgcopy


def whitewm_moving_avg(img, remove_region,
                       window_size=(5, 5), threshold=225, method='mean'):
    """
    Replaces white watermarks with with mean or median of values
    in remove_region.

    Parameters
    -------------------------------------
    img: ndarray
        Contains image data
    remove_region: tuple
        Coordinates of box containing the remove_region. Order is: (
            top left x-coordinate,
            top left y-coordinate,
            bottom right x-coordinate,
            bottom right y-coordinate
        )
    window_size: tuple, optional
        Size of kernel window to move across image. Default is (5, 5)
    threshold: int, optional
        Color threshold for white value. Default is 225
    method: {'mean', 'median'}
        Method for calculating average. Default is 'mean'

    Returns
    ---------------------------------------
    ndarray
        transformed image
    """
    x1, y1, x2, y2 = remove_region
    imgcopy = img.copy()
    img_rr = imgcopy[y1:y2, x1:x2, :]

    # move window through remove_region from left to right and top to bottom
    for i in range(0, img_rr.shape[0], window_size[1]):
        for j in range(0, img_rr.shape[1], window_size[0]):
            window = img_rr[i:i+window_size[1], j:j+window_size[0], :]
            notwm = window[window < threshold]

            if method == 'mean':
                avg = np.mean(notwm)
            elif method == 'median':
                avg = np.median(notwm)
            else:
                raise ValueError('method must be "mean" or "median"')

            window[window >= threshold] = avg
            img_rr[i:i+window_size[1], j:j+window_size[0], :] = window

    imgcopy[y1:y2, x1:x2, :] = img_rr
    return imgcopy
