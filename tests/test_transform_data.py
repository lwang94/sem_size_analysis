import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, 'C:/Users/lawre/Documents/TEM_ML')
import sem_size_analysis.transform_data as src

from skimage import measure
from pathlib import Path

def test_make_3channel():
    grey_img = mpimg.imread('C:/Users/lawre/Documents/TEM_ML/tests/images/SEM_image_of_blood_cells.jpg')
    colour_img = src.make_3channel(grey_img)
    assert colour_img.shape == (grey_img.shape[0], grey_img.shape[1], 3)
