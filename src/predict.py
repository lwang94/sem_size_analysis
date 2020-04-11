import numpy as np
from . import config as cf
from pathlib import Path

# import gdown
from fastai.vision import load_learner

import boto3

import cv2


def fetch_learner(path=Path(__file__).parents[1], model=cf.MODEL):
    """
    Returns learner if model file exists in path. If not, download
    the model file into the root directory and return the learner
    """
    filename = path / model
    if filename.exists():
        learn = load_learner(path, model)
    else:
        s3client = boto3.client(
            's3',
            aws_access_key_id='AKIAZBOHWE5IGXVOGMNN',
            aws_secret_access_key='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'
        )
        s3client.download_file(
            'saemimodel',
            'stage-2_bs24_rnet18.pkl',
            str(path / model)
        )

        # url = cf.MODEL_URL
        # gdown.download(url, 'stage-2_bs24_rnet18.pkl', quiet=False)
        learn = load_learner(path, model)
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
    unique, pred_labeled, stats, centroid = cv2.connectedComponentsWithStats(
                                                1 - pred,
                                                connectivity=4
                                            )
    return pred_labeled, np.arange(1, unique), stats[1:, -1]
