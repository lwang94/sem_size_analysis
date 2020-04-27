import base64
from PIL import Image
import io
from pathlib import Path

from skimage import draw, morphology
from skimage.transform import resize
from scipy import ndimage

import numpy as np
import matplotlib.image as mpimg
import json


def b64_2_numpy(string):
    """Converts base64 encoded image to numpy array"""
    decoded = base64.b64decode(string)
    im = Image.open(io.BytesIO(decoded))
    return np.array(im)


def numpy_2_b64(arr, enc_format='png'):
    """Converts numpy array to base64 encoded image"""
    img_pil = Image.fromarray(arr)
    buff = io.BytesIO()
    img_pil.save(buff, format=enc_format)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


def upload_demo():
    """Returns demo img as base64 string"""
    fname = Path(__file__).parents[1] / 'demo_img.jpg'
    img = mpimg.imread(fname)
    return numpy_2_b64(img, enc_format='jpeg')


def apply_edits(data, ypred, size_distr_json):
    """Applies user applied edits from dash canvas"""
    labeled = {}
    for obj in data['objects'][1:]:
        mask = parse_obj(obj).astype(np.uint8)

        # if the stroke is white, add stroke to prediction
        if obj['stroke'] == 'white':
            ypred = np.bitwise_or(ypred, mask)

        # if the stroke is red, remove entire particle labeled
        # by stroke
        elif obj['stroke'] == 'red':
            # cache labeled array for if there are multiple red strokes
            if 'cache' not in labeled:
                size_distr = json.loads(size_distr_json)
                cache = np.asarray(size_distr['labeled_list'])
                labeled['cache'] = resize(
                    cache,
                    (576, 768),
                    order=0,
                    preserve_range=True
                ).astype(np.int32)

            # remove any particles that "touches" red stroke
            remove = np.unique(labeled['cache'][np.nonzero(mask)])
            for r in remove:
                ypred[np.where(labeled['cache'] == r)] = 0

        # otherwise, the stroke is black and stroke should be erased
        # from image
        else:
            ypred = np.bitwise_and(ypred, 1 - mask)
    return ypred


def parse_obj(obj):
    """Create (576, 768) binary mask from object data"""
    scale = 1 / obj['scaleX']
    path = obj['path']
    rr, cc = [], []

    # find indices of SVG pathusing bezier curve
    for (Q1, Q2) in zip(path[:-2], path[1:-1]):
        inds = draw.bezier_curve(int(round(Q1[-1] / scale)),
                                 int(round(Q1[-2] / scale)),
                                 int(round(Q2[2] / scale)),
                                 int(round(Q2[1] / scale)),
                                 int(round(Q2[4] / scale)),
                                 int(round(Q2[3] / scale)), 1)
        rr += list(inds[0])
        cc += list(inds[1])
    radius = round(obj['strokeWidth'] / 2. / scale)

    # create mask
    mask = np.zeros((576, 768), dtype=np.bool)
    mask[rr, cc] = 1
    mask = ndimage.binary_dilation(
        mask,
        morphology.disk(radius)
    )
    return mask
