import numpy as np
from skimage import measure
from fastai.vision import (
    SegmentationItemList,
    get_transforms,
    unet_learner,
    models,
    dice,
)


def load_learner(path_img, path_lbl, codes, input_size, bs,
                 pretrained = None, split_pct=0.2):
    """
    Load the learner used to make the prediction.

    Parameters
    ---------------------------------------
    path_img : Path object
        Path to the directory containing the training images.
    path_lbl : Path object
        Path to the directory containing the labels for the above images.
        The labels should have the same filename as their corresponding image
        but with a .png file extension.
    codes : ndarray
        Contains names corresponding to the segmented objects in your labels.
        The dtype of the array should be "<U17".
    input_size : tuple
        Contains the width and height of the input image that is accepted by
        your learner.
    bs : int
        Batch size
    pretrained : str, None, optional
        Filename of the learner to be loaded. Should be located in the
        same directory as the Learner. See the fastai Learner documentation
        for more details. Defaults to None.
    split_pct : float, optional
        The percentage of images that will be put into your validation set.
        Defaults to 20%.

    Returns
    ------------------------------------------
    learner : Learner object
        The loaded learner.
    """
    data = (SegmentationItemList.from_folder(path_img)
            .split_by_rand_pct(split_pct)
            .label_from_func(lambda x: path_lbl/f'{x.stem}.png', classes=codes)
            .transform(
                get_transforms(flip_vert=True, max_warp=None),
                tfm_y=True,
                size=input_size
            )
            .databunch(bs=bs)
            .normalize())
    learner = unet_learner(data, models.resnet34, metrics=dice)
    if pretrained != None:
        assert isinstance(pretrained, str), "pretrained must be string"
        learner.load(pretrained)
    return learner


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
    pred_labeled = measure.label(pred, background=255, connectivity=1)
    unique, counts = np.unique(pred_labeled, return_counts=True)
    return pred_labeled, unique[1:], counts[1:]
