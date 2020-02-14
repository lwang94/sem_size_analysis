"""Performs training and evalution on a segmentation model"""


from fastai.vision import (
    SegmentationItemList,
    get_transforms,
    unet_learner,
    models,
    dice
)

import config as cf


def create_databunch(path_img, path_lbl, codes, input_size, bs, split_pct=0.2):
    """
    Creates fastai databunch object to be put into Learner object.

    Parameters
    ----------------------------------------
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
        Contains the width and height of the input image that is accepted
        by your learner.
    bs : int
        Batch size
    split_pct : float, optional
        The percentage of images that will be put into your validation set.
        Defaults to 20%.

    Returns
    ------------------------------------------
    data : Databunch
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
    return data


def train_model(learner, freeze_layer, epochs, lr, wd, save_model,
                show_losses=False, show_metrics=False):
    """
    Trains the model and saves it to the Learner path

    Parameters
    ---------------------------------------------------
    learner : Learner
        Fastai Learner object. Should be a u-net using the Dice metric.
    freeze_layer : int
        The last layer group to be frozen before training.
    epochs : int
        Number of epochs
    lr : slice object
        The learning rate. Should be determined using the learning rate finder.
    wd : float
        The weight decay.
    save_model : str
        The filename the model is saved as.
    show_losses : bool, optional
        Determines whether to plot the training and validation loss
        in a separate graph. Defaults to False.
    show_metrics : bool, optional
        Determines whether to plot the metrics during training
        in a separate graph. Defults to False.
    """
    learner.unfreeze()
    learner.freeze_to(freeze_layer)
    learner.fit_one_cycle(epochs, lr, wd)
    if show_losses:
        learner.recorder.plot_losses()
    if show_metrics:
        learner.recorder.plot_metrics()
    learner.save(save_model)


def eval_model(path, codes, input_size, bs, learner,
               train_dirname='train', test_dirname='test',
               labels_dirname='labels'):
    """
    Evaluates the model on a test set.

    Parameters
    ------------------------------
    path : Path object
        Path to the testing directory. Contained within the directory should be
        three subdirectories for the training images, the test images, and the
        labels for both training and test set.
    codes : ndarray
        Contains names corresponding to the segmented objects in your labels.
        The dtype of the array should be "<U17".
    input_size : tuple
        Contains the width and height of the input image that is accepted by
        your learner.
    bs : int
        Batch size
    learner : Learner
        Fastai Learner object. Should be a u-net using the Dice metric.
    train_dirname : str
        Name of subdirectory containing training images
    test_dirname : str
        Name of subdirectory containing test images
    labels_dirname : str
        Name of subdirectory containing labels for both training and test set.

    Returns
    ------------------------------------------------------
    eval : tuple
        Contains the training loss and the accuracy metric of the model on
        the test set.
    """
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


def train_and_eval():
    print("Creating databunch and learner...")
    data = create_databunch(
        cf.PATH_IMG,
        cf.PATH_LBL,
        cf.CODES,
        cf.INPUT_SIZE,
        cf.BATCH_SIZE
    )
    learner = unet_learner(data, models.resnet34, metrics=dice)
    print("Training model...")
    train_model(
        learner,
        cf.FREEZE_LAYER,
        cf.EPOCHS,
        cf.LEARNING_RATE,
        cf.WEIGHT_DECAY,
        cf.SAVE_MODEL
    )
    print("Evaluating model...")
    eval = eval_model(
        cf.PATH_TO_TESTING,
        cf.CODES,
        cf.INPUT_SIZE,
        cf.BATCH_SIZE,
        learner
    )
    print(f'Loss = {eval[0]}, Accuracy = {eval[1]}')
    print(
        "You have successfully trained and evaluated your model!"
        "Please find it in the appropriate directory."
    )


if __name__ == '__main__':
    train_and_eval()
