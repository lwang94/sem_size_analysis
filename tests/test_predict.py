import matplotlib.image as mpimg

# import sys
# sys.path.insert(1, 'C:/Users/lawre/Documents/TEM_ML')
from ..src import predict
from ..src.transform_data import resize, fastai_image

from pathlib import Path
import numpy as np
from fastai.basic_train import Learner

import pytest

@pytest.fixture
def load_learner_for_test():
    data_path = Path(__file__).parents[1] / 'data' / 'dataset' / 'good'
    learn = predict.load_learner(
        'stage-2_bs16',
        Path(data_path) / 'train_x',
        Path(data_path) /'train_y_png',
        np.array(['background', 'particle'], dtype='<U17'),
        (192, 256),
        16
     )
    return learn

def test_load_learner(load_learner_for_test):
    learn = load_learner_for_test
    assert isinstance(learn, Learner)
    length_lg = [41, 49, 62]
    for i in range(len(learn.layer_groups)):
        assert len(learn.layer_groups[i]) == length_lg[i]

def test_predict_segment(load_learner_for_test):
    learn = load_learner_for_test
    img_path = Path(__file__).parent / 'images' / 'L2_5b095b8603ce97661d9a01918cf4bd53.jpg'
    img = mpimg.imread(img_path)
    img = resize(img, (192, 256))
    img = fastai_image(img)
    pred = predict.predict_segment(learn, img)
    assert pred.shape == (1, 192, 256)
    assert list(np.unique(pred)) == [0, 1]


def test_get_size_distr():
    pred = np.array(
        [[255, 255, 255, 255, 255],
         [0,   0,   255, 255, 255],
         [0,   0,   255, 255, 255],
         [255, 255, 0,   0,   0],
         [255, 255, 0,   0,   0]]
    )

    size_distr = predict.get_size_distr(pred)
    assert len(size_distr) == 2
    assert size_distr.mean() == 5

