"""Performs training and evalution on a segmentation model"""


from fastai.vision import (
    SegmentationItemList,
    get_transforms,
    unet_learner,
    models,
    dice
)

from config import (
    PATH_IMG,
    PATH_LBL,
    CODES,
    INPUT_SIZE,
    BATCH_SIZE,
    FREEZE_LAYER,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SAVE_MODEL,
    PATH_TO_TESTING
)


def create_db(path_img, path_lbl, codes, input_size, bs, split_pct=0.2):
    """assumes labels are png"""
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
    return data


def train_model(learner, freeze_layer, epochs, lr, wd, save_model,
                show_losses=False, show_metrics=False):
    learner.unfreeze()
    learner.freeze_to(freeze_layer)
    learner.fit_one_cycle(epochs, lr, wd)
    if show_losses:
        learner.recorder.plot_losses()
    if show_metrics:
        learner.recorder.plot_metrics()
    learner.save(save_model)


def eval_model(path, codes, input_size, bs, learner,
               train_dirname='train', test_dirname='test', labels_dirname='labels'):
    data_test = (SegmentationItemList.from_folder(path)
                 .split_by_folder(train=train_dirname, valid=test_dirname)
                 .label_from_func(
                     lambda x: path/f'{labels_dirname}/{x.stem}.png',
                     classes=codes
                 )
                 .transform(
                     get_transforms(flip_vert=True, max_warp=None),
                     tfm_y=True,
                     size=input_size
                 )
                 .databunch(bs=bs)
                 .normalize())
    eval = learner.validate(data_test.valid_dl)
    return eval


print("Creating databunch and learner...")
data = create_db(PATH_IMG, PATH_LBL, CODES, INPUT_SIZE, BATCH_SIZE)
learner = unet_learner(data, models.resnet34, metrics=dice)
print("Training model...")
train_model(
    learner,
    FREEZE_LAYER,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SAVE_MODEL
)
print("Evaluating model...")
eval = eval_model(
    PATH_TO_TESTING,
    CODES, INPUT_SIZE,
    BATCH_SIZE,
    learner
)
print(f'Loss = {eval[0]}, Accuracy = {eval[1]}')
print(
    "You have successfully trained and evaluated your model!"
    "Please find it in the appropriate directory."
)
