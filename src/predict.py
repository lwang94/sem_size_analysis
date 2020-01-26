import numpy as np
from skimage import measure
from fastai.vision import (
    SegmentationItemList,
    get_transforms,
    unet_learner,
    models,
    dice
)
from training import create_db


def load_model(model_name, path_img, path_lbl, codes, input_size, bs, split_pct=0.2):
    data = create_db(path_img, path_lbl, codes, input_size, batch_size, split_pct=split_pct)
    learner = unet_learner(data, models.resnet34, metrics=dice)
    learner.load(model_name)
    return learner


def predict_segment(learner, img):
    pred learner.predict(img)[0]
    return pred.data


def get_size_distr(learner, img):
    pred = predict_segment(learner, img)
    pred_labeled = measure.label(pred, background=255, connectivity=2)
    unique, counts = np.unique(pred_labeled, return_counts=True)
    return counts[1:]

