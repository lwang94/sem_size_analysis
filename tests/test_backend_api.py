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


def test_predict(client, test_arr):
    """
    Tests predict function in flask_api by
    asserting the output json contains a list
    and string with the correct keys
    """
    res = client.post(
        '/api/predict',
        json={
            'content_type': 'data:image/jpeg;base64',
            'contents': test_arr.tolist()
        }
    )
    pred = json.loads(res.data)
    assert isinstance(pred['yimage_list'], list)
    assert isinstance(pred['rf'], float)


def test_orig_size_distr(client, test_arr):
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
                'rf': 2,
                'yimage_list': test_arr[:, :, 0].tolist()
            })
        }
    )
    dat = json.loads(res.data)

    assert isinstance(dat['rgb_pred_list'], list)
    assert isinstance(dat['labeled_list'], list)
    assert isinstance(dat['unique_list'], list)
    assert isinstance(dat['size_distr_list'], list)


def test_clicked_size_distr(client, test_arr):
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
                'rf': 2,
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
                'rgb_pred_list': test_list,
                'labeled_list': test_arr[:, :, 0].tolist(),
                'unique_list': [1, 2, 3],
                'size_distr_list': [1, 2, 3]
            })
        }
    )
    dat = json.loads(res.data)
    assert isinstance(dat['rgb_pred_list'], list)
    assert isinstance(dat['labeled_list'], list)
    assert isinstance(dat['unique_list'], list)
    assert isinstance(dat['size_distr_list'], list)
