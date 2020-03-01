import pytest

from ..src import backend_api as ba

import matplotlib.image as mpimg
import numpy as np
import json
from pathlib import Path


@pytest.fixture
def client():
    """Create client for testing"""
    ba.flask_app.config['TESTING'] = True

    with ba.flask_app.test_client() as client:
        yield client


@pytest.fixture
def test_img():
    """Initialize test image to be used in test functions"""
    img_path = (
        Path(__file__).parent
        / 'images'
        / 'train_x'
        / 'L2_0a7efff5757e6b543ee1a0d17328c881.jpg'
    )
    img = mpimg.imread(img_path)

    return 'data:image/jpg;base64,' + ba.numpy_2_b64(img)


@pytest.fixture
def test_arr():
    """Initialize test array to be used in test functions"""
    img_path = (
        Path(__file__).parent
        / 'images'
        / 'train_x'
        / 'L2_0a7efff5757e6b543ee1a0d17328c881.jpg'
    )
    img = mpimg.imread(img_path)
    return img


def test_numpy_2_b64(test_img):
    """
    Tests numpy_2_b64 funcion in flask_api by
    asserting the output is a string
    """
    assert isinstance(test_img, str)


def test_b64_2_numpy(test_img):
    """
    Tests b64_2_numpy function in flask_api by
    asserting the output is a numpy array
    """
    arr = ba.b64_2_numpy(test_img.split(',')[1])
    assert isinstance(arr, np.ndarray)


def test_predict(client, test_img):
    """
    Tests predict function in flask_api by
    asserting the output json contains a list
    and string with the correct keys
    """
    content_json = json.dumps({'contents': test_img})
    res = client.post(
        '/api/predict',
        json=content_json
    )
    pred = json.loads(res.data)
    assert isinstance(pred['yimage_list'], list)
    assert isinstance(pred['yimage_b64'], str)


def test_orig_size_distr(client, test_img, test_arr):
    """
    Tests orig_size_distr function in flask_api by
    asserting the output json contains lists and strings
    with the correct keys
    """
    res = client.post(
        '/api/orig_size_distr',
        json={
            'data_pred': json.dumps({
                'content_type': 'data:image/jpg;base64',
                'ximage_b64': test_img,
                'ximage_list': test_arr.tolist(),
                'yimage_b64': test_img,
                'yimage_list': test_arr[:, :, 0].tolist()
            })
        }
    )
    dat = json.loads(res.data)

    assert isinstance(dat['rgb_pred_b64'], str)
    assert isinstance(dat['rgb_pred_list'], list)
    assert isinstance(dat['labeled_list'], list)
    assert isinstance(dat['unique_list'], list)
    assert isinstance(dat['size_distr_list'], list)


def test_clicked_size_distr(client, test_img, test_arr):
    """
    Tests clicked_size_distr function in flask_api by
    asserting the output json contains lists and strings
    with the correct keys
    """
    test_list = test_arr.tolist()
    res = client.post(
        '/api/clicked_size_distr',
        json={
            'data_pred': json.dumps({
                'content_type': 'data:image/jpg;base64',
                'ximage_b64': test_img,
                'yimage_b64': test_img,
                'yimage_list': test_arr[:, :, 0].tolist()
            }),
            'click': {
                'points': [{
                    'curveNumber': 0,
                    'pointNumber': 9927,
                    'pointIndex': 9927,
                    'x': 199,
                    'y': 38
                }]
            },
            'size_distr_json': json.dumps({
                'content_type': 'data:image/jpg;base64',
                'rgb_pred_b64': test_img,
                'rgb_pred_list': test_list,
                'labeled_list': test_arr[:, :, 0].tolist(),
                'unique_list': [1, 2, 3],
                'size_distr_list': [1, 2, 3]
            })
        }
    )
    dat = json.loads(res.data)
    assert isinstance(dat['rgb_pred_b64'], str)
    assert isinstance(dat['rgb_pred_list'], list)
    assert isinstance(dat['labeled_list'], list)
    assert isinstance(dat['unique_list'], list)
    assert isinstance(dat['size_distr_list'], list)
