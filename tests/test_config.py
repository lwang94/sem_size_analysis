from ..src import config as cf
import numpy as np


def test_model():
    assert isinstance(cf.MODEL, str)


def test_model_url():
    assert isinstance(cf.MODEL_URL, str)


def test_version():
    assert isinstance(cf.VERSION, str)


def test_codes():
    assert isinstance(cf.CODES, np.ndarray)


def test_input_size():
    assert isinstance(cf.INPUT_SIZE, tuple)


def test_bath_size():
    assert isinstance(cf.BATCH_SIZE, int)


def test_freeze_layer():
    assert isinstance(cf.FREEZE_LAYER, int)


def test_epochs():
    assert isinstance(cf.EPOCHS, int)


def test_learning_rate():
    assert isinstance(cf.LEARNING_RATE, slice)


def test_weight_decay():
    assert isinstance(cf.WEIGHT_DECAY, float)


def test_save_model():
    assert isinstance(cf.SAVE_MODEL, str)
