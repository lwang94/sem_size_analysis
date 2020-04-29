"""Unit tests for the transform data functions"""

import matplotlib.image as mpimg
from fastai.vision import Image

from ..src import transform_data as td

from pathlib import Path
import pytest


@pytest.fixture
def test_image():
    img_path = (
        Path(__file__).parent
        / 'images'
        / 'train_x'
        / 'L2_0a7efff5757e6b543ee1a0d17328c881.jpg'
    )
    img = mpimg.imread(img_path)
    return img


def test_fastai_image(test_image):
    """
    Tests fastai_image function in transform_data
    by asserting the output type
    """
    img = test_image
    img = td.fastai_image(img)
    assert isinstance(img, Image)


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


def test_reflect_pad(test_image):
    """
    Tests reflect_pad function in transform_data
    by asserting the output shape
    """
    img = test_image
    new_img = td.reflect_pad(img, (0, 500, img.shape[1], 600))
    assert img.shape == new_img.shape

    img = test_image
    new_img = td.reflect_pad(
        img,
        (0, img.shape[0] - 100, img.shape[1], img.shape[0])
    )
    assert img.shape == new_img.shape

    img = test_image
    new_img = td.reflect_pad(img, (0, 0, img.shape[1], 100))
    assert img.shape == new_img.shape


def test_replace_constant(test_image):
    """
    Tests replace_constant function in transform_data
    by asserting the output shape
    """
    img = test_image
    new_img = td.replace_constant(img, (0, 500, img.shape[1], 600), 0)
    assert img.shape == new_img.shape


def test_replace_neighbor(test_image):
    """
    Tests replace_neighbor function in transform_data
    by asserting the output shape
    """
    img = test_image
    new_img = td.replace_neighbor(img, (0, 500, img.shape[1], 600))
    assert img.shape == new_img.shape

    img = test_image
    new_img = td.replace_constant(img, (0, 1, img.shape[1], 101), 0)
    assert img.shape == new_img.shape

    img = test_image
    new_img = td.replace_constant(
        img,
        (0, img.shape[0] - 101, img.shape[1], img.shape[0] - 1),
        0
    )
    assert img.shape == new_img.shape


def test_replace_whitewm_constant(test_image):
    """
    Tests replace_whitewm_constant function in transform_data
    by asserting the output shape
    """
    img = test_image
    new_img = td.replace_whitewm_constant(img, (0, 500, img.shape[1], 600), 0)
    assert img.shape == new_img.shape


def test_replace_whitewm_avg(test_image):
    """
    Tests replace_whitewm_avg function in transform_data
    by asserting the output shape
    """
    img = test_image
    new_img = td.replace_whitewm_avg(img, (0, 500, img.shape[1], 600))
    assert img.shape == new_img.shape

    img = test_image
    new_img = td.replace_whitewm_avg(
        img,
        (0, 500, img.shape[1], 600),
        method='median'
    )
    assert img.shape == new_img.shape


def test_whitewm_moving_avg(test_image):
    """
    Tests whitewm_moving_avg function in transform_data
    by asserting the output shape
    """
    img = test_image
    new_img = td.whitewm_moving_avg(img, (0, 500, img.shape[1], 600))
    assert img.shape == new_img.shape

    img = test_image
    new_img = td.replace_whitewm_avg(
        img,
        (0, 500, img.shape[1], 600),
        method='median'
    )
    assert img.shape == new_img.shape
