import asyncio
import uvicorn
import sys

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
import json
import numpy as np

import predict as pred
import transform_data as td
import config as cf

import base64
import io
from PIL import Image


star_app = Starlette()
star_app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_headers=['X-Requested-With', 'Content-Type']
)


async def setup_learner():
    learn = pred.fetch_learner()
    return learn


async def b64_2_numpy(string):
    """Converts base64 encoded image to numpy array"""
    decoded = base64.b64decode(string)
    im = Image.open(io.BytesIO(decoded))
    return np.array(im)


async def numpy_2_b64(arr, enc_format='png'):
    """Converts numpy array to base64 encoded image"""
    img_pil = Image.fromarray(arr)
    buff = io.BytesIO()
    img_pil.save(buff, format=enc_format)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@star_app.route('/')
async def hello_world(request):
    return "Hello World!"


@star_app.route('/api/predict', methods=['POST'])
async def predict(request):
    content = await request.json()
    content = content['contents']

    content_type, content_string = content.split(',')
    im = await b64_2_numpy(content_string)

    if len(im.shape) == 2:
        im = td.make_3channel(im)
    img = td.resize(im, (192, 256))
    img = td.fastai_image(img)

    prediction = pred.predict_segment(learn, img).numpy()[0]

    lookup = np.asarray([[45, 0, 78], [153, 153, 0]], dtype=np.uint8)
    prediction3 = lookup[prediction]
    encoded_pred = await numpy_2_b64(prediction3)
    encoded_pred = content_type + ',' + encoded_pred

    return JSONResponse({
            'content_type': content_type,
            'ximage_b64': content,
            'ximage_list': im.tolist(),
            'yimage_b64': encoded_pred,
            'yimage_list': prediction.tolist()
    })


@star_app.route('/api/orig_size_distr', methods=['POST'])
async def orig_size_distr(request):
    content = await request.json()
    data_pred = json.loads(content['data_pred'])

    pred_data = np.asarray(data_pred['yimage_list'], dtype=np.uint8)
    labeled, unique, size_distr = pred.get_size_distr(pred_data)

    ximage = np.asarray(data_pred['ximage_list'])
    yimage = np.asarray(data_pred['yimage_list'])
    rf = (
        ximage.shape[0] * ximage.shape[1]
        / (yimage.shape[0] * yimage.shape[1])
    )
    size_distr = size_distr * rf

    flattened_color_arr = np.linspace(
        0,
        256 ** 3 - 1,
        num=len(unique) + 1,
        dtype=np.int64
    )

    colors = np.zeros((len(unique) + 1, 3), dtype=np.uint8)
    for i in range(len(colors)):
        colors[i] = np.array([
            (flattened_color_arr[i] // (256 ** 2)) % 256,
            (flattened_color_arr[i] // (256)) % 256,
            flattened_color_arr[i] % 256
        ])

    lookup = np.zeros((len(unique) + 1, 3), dtype=np.uint8)
    lookup[np.unique(labeled - 1)] = colors
    rgb = lookup[labeled - 1]

    encoded_rgb = await numpy_2_b64(rgb)
    encoded_rgb = data_pred['content_type'] + ',' + encoded_rgb
    print('success!')
    return JSONResponse({
            'content_type': data_pred['content_type'],
            'rgb_pred_b64': encoded_rgb,
            'rgb_pred_list': rgb.tolist(),
            'labeled_list': labeled.tolist(),
            'unique_list': unique.tolist(),
            'size_distr_list': size_distr.tolist()
    })


@star_app.route('/api/clicked_size_distr', methods=['POST'])
async def clicked_size_distr(request):
    content = await request.json()
    data_size_distr = json.loads(content['size_distr_json'])
    click = content['click']
    data_pred = json.loads(content['data_pred'])

    rgb = np.asarray(data_size_distr['rgb_pred_list'], dtype=np.uint8)
    labeled = data_size_distr['labeled_list']
    unique = np.asarray(data_size_distr['unique_list'])
    size_distr = np.asarray(data_size_distr['size_distr_list'])

    if click is not None:
        xclick, yclick = click['points'][0]['x'], click['points'][0]['y']

        remove = np.where(unique == labeled[191 - yclick][xclick])[0]
        unique = np.delete(unique, remove)
        size_distr = np.delete(size_distr, remove)
        click_r, click_g, click_b = rgb[191 - yclick, xclick, :]
        mask = (
            (rgb[:, :, 0] == click_r)
            & (rgb[:, :, 1] == click_g)
            & (rgb[:, :, 2] == click_b)
        )
        rgb[mask] = [0, 0, 0]

    encoded_rgb = await numpy_2_b64(rgb)
    encoded_rgb = data_pred['content_type'] + ',' + encoded_rgb

    return JSONResponse({
        'content_type': data_pred['content_type'],
        'rgb_pred_b64': encoded_rgb,
        'rgb_pred_list': rgb.tolist(),
        'labeled_list': labeled,
        'unique_list': unique.tolist(),
        'size_distr_list': size_distr.tolist()
    })


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=star_app, host=cf.HOST, port=cf.PORT, log_level="info")
