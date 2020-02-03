import matplotlib.image as mpimg

import sys
sys.path.insert(1, 'C:/Users/lawre/Documents/TEM_ML')
import src.predict as src
from src.transform_data import resize, fastai_image

from pathlib import Path
import numpy as np

def test_load_learner():
    data_path = 'C:/Users/lawre/Documents/good'
    learn = src.load_learner(
        'stage-2_bs16',
        Path(data_path) / 'train_x',
        Path(data_path) /'train_y_png',
        np.array(['background', 'particle'], dtype='<U17'),
        (192, 256),
        16
     )

    length_lg = [41, 49, 62]
    for i in range(len(learn.layer_groups)):
        assert len(learn.layer_groups[i]) == length_lg[i]

def test_predict_segment():
    data_path = 'C:/Users/lawre/Documents/good'
    learn = src.load_learner(
        'stage-2_bs16',
        Path(data_path) / 'train_x',
        Path(data_path) /'train_y_png',
        np.array(['background', 'particle'], dtype='<U17'),
        (192, 256),
        16
     )
    img = mpimg.imread('C:/Users/lawre/Documents/TEM_ML/tests/images/L2_5b095b8603ce97661d9a01918cf4bd53.jpg')
    img = resize(img, (192, 256))
    img = fastai_image(img)
    pred = src.predict_segment(learn, img)
    assert pred.shape == (1, 192, 256)
    assert len(np.unique(pred)) == 2


def test_get_size_distr():
    data_path = 'C:/Users/lawre/Documents/good'
    learn = src.load_learner(
        'stage-2_bs16',
        Path(data_path) / 'train_x',
        Path(data_path) /'train_y_png',
        np.array(['background', 'particle'], dtype='<U17'),
        (192, 256),
        16
     )
    img = mpimg.imread('C:/Users/lawre/Documents/TEM_ML/tests/images/L2_5b095b8603ce97661d9a01918cf4bd53.jpg')
    img = resize(img, (192, 256))
    img = fastai_image(img)
    size_distr = src.get_size_distr(learn, img)
    assert len(size_distr) == 1
