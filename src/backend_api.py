"""Generates server for backend API"""

from flask import Flask, request
import json
import numpy as np

from skimage.transform import resize

from . import predict as pred
from . import transform_data as td
from . import config as cf


flask_app = Flask(__name__)
host = cf.HOST
port = cf.BACKEND_PORT

# load model
learn = pred.fetch_learner()


@flask_app.route('/api/predict', methods=['POST'])
def predict():
    print('okay')
    """Obtains image segmentation prediction on image"""
    # get base64 image from requested json
    content = request.get_json()
    im = np.asarray(content['contents'], dtype=np.uint8)

    # perform data transformations
    if len(im.shape) == 2:
        im = td.make_3channel(im)
    img = resize(im, (192, 256), order=1)
    img = td.fastai_image(img)

    # make prediction
    prediction = pred.predict_segment(learn, img).astype(np.uint8)
    prediction = 255 * resize(prediction, (576, 768), order=0)
    prediction = prediction.astype(np.uint8)
    resizefactor = (
        im.shape[0] * im.shape[1]
        / (prediction.shape[0] * prediction.shape[1])
    )

    return json.dumps({
            'content_type': content['content_type'],
            'rf': resizefactor,
            'yimage_list': prediction.tolist()
    })


@flask_app.route('/api/get_size_distr', methods=['POST'])
def get_size_distr():
    """
    Obtains size distribution of image without user input. Also
    returns a version of the labeled image as a 3 channel rgb
    image to be shown on the dashboard.
    """
    # get requested json
    content = request.get_json()
    data_pred = json.loads(content['data_pred'])

    # obtain size distributions on prediction by labeling connected regions
    pred_data = np.asarray(data_pred['yimage_list'], dtype=np.uint8)
    labeled, unique, size_distr = pred.get_size_distr(pred_data)

    # rescale size_distr back to original image sizes
    size_distr = size_distr * data_pred['rf']

    return json.dumps({
            'content_type': data_pred['content_type'],
            'labeled_list': labeled.tolist(),
            'unique_list': unique.tolist(),
            'size_distr_list': size_distr.tolist()
    })


if __name__ == '__main__':
    flask_app.run(debug=True, host=host, port=port)
