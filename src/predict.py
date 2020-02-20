import numpy as np
from skimage import measure
from . import config as cf
from pathlib import Path
import gdown

from fastai.vision import load_learner


def load_learn(path=Path(__file__).parents[1], model=cf.MODEL):
    """
    Returns learner if model file exists in path. If not, download
    the model file into the root directory and return the learner
    """
    filename = path / model
    if filename.exists():
        learn = load_learner(path, model)
    else:
        url = cf.URL
        gdown.download(url, model, quiet=False)
        learn = load_learner(Path(__file__).parents[1], model)
    return learn


def predict_segment(learner, img):
    """
    Predicts a segmentation mask using a deep learning based model.

    Parameters
    -------------------------------------------------
    learner : Learner object
        The learner used to perform the prediction
    img : Image object
        The input image. Should be a fastai Image object.

    Returns
    ------------------------------------------------
    pred : PyTorch Tensor
        Contains segmentation mask data
    """
    pred = learner.predict(img)[0]
    return pred.data


def get_size_distr(pred):
    """
    Obtains the size distribution of particles in an image
    using a deep learning based model.

    Parameters
    -------------------------------------------------
    pred : ndarray
        The predicted segmentation mask for the image.
        Should only have 0s and 1s.

    Returns
    ------------------------------------------------
    counts : ndarray
        Contains number of pixels for each segment of the
        image as determined by the model. Does not include
        the background.
    """
    # labels each connected region with a unique value
    pred_labeled = measure.label(pred, background=255, connectivity=1)
    unique, counts = np.unique(pred_labeled, return_counts=True)
    return pred_labeled, unique[1:], counts[1:]
