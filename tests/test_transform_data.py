import matplotlib.image as mpimg
from fastai.vision import Image

from ..src import transform_data as td

from pathlib import Path


def test_fastai_image():
    """
    Tests fastai_image function in transform_data
    by asserting the output type
    """
    img_path = (
        Path(__file__).parent
        / 'images'
        / 'train_x'
        / 'L2_0a7efff5757e6b543ee1a0d17328c881.jpg'
    )
    img = mpimg.imread(img_path)
    img = td.fastai_image(img)
    assert isinstance(img, Image)


def test_resize():
    """
    Tests resize function in transform_data
    by asserting the output shape
    """
    img_path = (
        Path(__file__).parent
        / 'images'
        / 'train_x'
        / 'L2_0a7efff5757e6b543ee1a0d17328c881.jpg'
    )
    img = mpimg.imread(img_path)
    img = td.resize(img, (192, 256))
    assert img.shape == (192, 256, 3)


def test_make_3channel():
    """
    Tests make_3channel function in transform_data
    by asserting the output shape
    """
    img_path = (
        Path(__file__).parent
        / 'images'
        / 'SEM_image_of_blood_cells.jpg'
    )
    grey_img = mpimg.imread(img_path)
    colour_img = td.make_3channel(grey_img)
    assert colour_img.shape == (grey_img.shape[0], grey_img.shape[1], 3)
