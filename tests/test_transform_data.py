import matplotlib.image as mpimg

import sys
sys.path.insert(1, 'C:/Users/lawre/Documents/TEM_ML')
import src.transform_data as src

from pathlib import Path

def test_resize():
    grey_img = mpimg.imread('C:/Users/lawre/Documents/TEM_ML/tests/images/SEM_image_of_blood_cells.jpg')
    grey_img = src.resize(grey_img, (192, 256))
    assert grey_img.shape == (192, 256)

    img = mpimg.imread('C:/Users/lawre/Documents/TEM_ML/tests/images/SEM_image_of_red_blood_cell.jpg')
    img = src.resize(img, (192, 256))
    assert img.shape == (192, 256, 3)

def test_make_3channel():
    grey_img = mpimg.imread('C:/Users/lawre/Documents/TEM_ML/tests/images/SEM_image_of_blood_cells.jpg')
    colour_img = src.make_3channel(grey_img)
    assert colour_img.shape == (grey_img.shape[0], grey_img.shape[1], 3)
