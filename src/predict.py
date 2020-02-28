import numpy as np
from skimage import measure
from . import config as cf
from pathlib import Path

# import io
# import boto3
# import os
import gdown
from fastai.vision import load_learner


def fetch_learner(path=Path(__file__).parents[1], model=cf.MODEL):
    """
    Returns learner if model file exists in path. If not, download
    the model file into the root directory and return the learner
    """
    filename = path / model
    if filename.exists():
        learn = load_learner(path, model)
    else:
        # client = boto3.client(
        #     's3',
        #     aws_access_key_id = os.environ['S3_KEY'],
        #     aws_secret_access_key= os.environ['S3_SECRET']
        # )
        # obj = client.get_object(Bucket='saemi-model', Key='stage-2_bs16.pkl')
        # model = io.BytesIO(obj["Body"].read())

        url = cf.MODEL_URL
        gdown.download(url, model, quiet=False)
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
    pred_labeled = measure.label(pred, background=255, connectivity=1)
    unique, counts = np.unique(pred_labeled, return_counts=True)
    return pred_labeled, unique[1:], counts[1:]
