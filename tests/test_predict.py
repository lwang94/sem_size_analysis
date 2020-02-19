import matplotlib.image as mpimg
from fastai.vision import load_learner
import gdown

from ..src import predict
from ..src.transform_data import resize, fastai_image

from pathlib import Path
import numpy as np


def test_predict_segment():
    """
    Tests predict_segment function in predict
    by asserting the output shape and its unique
    values.
    """
    # load model
    try:
        model_path = Path(__file__).parents[1]
        learn = load_learner(model_path, 'stage-2_bs16.pkl')
    except IOError:
        url = (
            'https://drive.google.com/uc?id=1ClEUOwFhIOMNxvedps8WQN8Koy6NT5eI'
        )
        output = 'stage-2_bs16.pkl'
        gdown.download(url, output, quiet=False)

        model_path = Path(__file__).parents[1]
        learn = load_learner(model_path, output)

    # load test image
    img_path = (
        Path(__file__).parent
        / 'images'
        / 'train_x'
        / 'L2_0a7efff5757e6b543ee1a0d17328c881.jpg'
    )
    img = mpimg.imread(img_path)

    # transform image to use in model
    img = resize(img, (192, 256))
    img = fastai_image(img)

    # make prediction and assertions
    pred = predict.predict_segment(learn, img)
    assert pred.shape == (1, 192, 256)
    assert sorted(list(np.unique(pred))) == [0, 1]


def test_get_size_distr():
    """
    Tests get_size_distr function in predict
    by asserting its output shape, number of
    unique values, and mean of the size of the
    connected regions.
    """
    pred = np.array(
        [[255, 255, 255, 255, 255],
         [0,   0,   255, 255, 255],
         [0,   0,   255, 255, 255],
         [255, 255, 0,   0,   0],
         [255, 255, 0,   0,   0]]
    )

    labeled, unique, size_distr = predict.get_size_distr(pred)
    assert labeled.shape == (5, 5)
    assert len(unique) == 2
    assert len(size_distr) == 2
    assert np.allclose(size_distr.mean(), 5)
