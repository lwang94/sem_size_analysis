import matplotlib.image as mpimg

from ..src import transform_data as td

from pathlib import Path

def test_resize():
    img_path = Path(__file__).parent / 'images' / 'SEM_image_of_red_blood_cell.jpg'
    img = mpimg.imread(img_path)
    img = td.resize(img, (192, 256))
    assert img.shape == (192, 256, 3)

def test_make_3channel():
    img_path = Path(__file__).parent / 'images' / 'SEM_image_of_blood_cells.jpg'
    grey_img = mpimg.imread(img_path)
    colour_img = td.make_3channel(grey_img)
    assert colour_img.shape == (grey_img.shape[0], grey_img.shape[1], 3)
