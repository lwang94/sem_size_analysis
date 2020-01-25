from fastai import *
from fastai.vision import *

from pathlib import Path


def create_db(path_img, path_lbl, codes, size, bs, split_pct=0.2):
    """assumes labels are png"""
    data = (SegmentationItemList.from_folder(path_img)
            .split_by_rand_pct(split_pct)
            .label_from_func(lambda x: path_lbl/f'{x.stem}.png', classes=codes)
            .transform(get_transforms(flip_vert=True, max_warp=None), tfm_y=True, size=size)
            .databunch(bs=bs)
            .normalize())
    return data


def create_unet(data, path_data=None, model_dir='models'):
    return unet_learner(data, models.resnet34, metrics=dice, path=path_data, model_dir=model_dir)

def find_lr(learner, saved_model=None, unfreeze_till=None):
    if saved_model != None:
        learner.load(saved_model); #must be in path and model directory that was specified when creating unet_learner
    if unfreeze_till != None:
        learn.unfreeze()
        learn.freeze_to(unfreeze_till)
    lr_find(learn)
    learn.recorder.plot()

def train_model(learner, epochs, lr, wd, save_fname, show_losses=False, show_metrics=False):
    learner.fit_one_cycle(50, lr, wd)
    if show_losses:
        learn.recorder.plot_losses()
    if show_metrics:
        learn.recorder.plot_metrics()
    learn.save(save_fname)

def eval_model(path, codes, size, bs, learner, train_dirname='train', test_dirname='test', labels_dirname='labels'):
    data_test = (SegmentationItemList.from_folder(path)
                 .split_by_folder(train=train_dirname, valid=test_dirname)
                 .label_from_func(lambda x: path/f'{labels_dirname}/{x.stem}.png', classes=codes)
                 .transform(get_transforms(flip_vert=True, max_warp=None), tfm_y=True, size=size)
                 .databunch(bs=bs)
                 .normalize())
    learner.validate(data_test.valid_dl)

#test create_db, create_unet, and find_lr
path_img = Path('dataset/good/train_x')
path_lbl = Path('dataset/good/test_x')
data = create_db(path_img, path_lbl, array(['background', 'particles'], (192, 256), 16))
learner = create_unet(data)
find_lr(learner, unfreeze_till=2)
