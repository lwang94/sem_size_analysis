import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

from dash_canvas.utils import array_to_data_url
from skimage import draw, morphology
from skimage.transform import resize
from scipy import ndimage

import json
import requests
import numpy as np

import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
import io
import base64

import app_layout as al
from app_callbacks import callbacks_backend
from app_callbacks import callbacks_misc
from app_callbacks import callbacks_hist
from app_callbacks import callbacks_images

import matplotlib.pyplot as plt
from dash_canvas.utils import parse_jsonstring

# from src import config as cf

dash_app = dash.Dash(__name__)
server = dash_app.server
backend_url = 'https://saemi-backend.herokuapp.com/'
# backend_url = f'http://{cf.HOST}:{cf.BACKEND_PORT}'

# dashboard layout
dash_app.layout = al.app_layout()

# dashboard callbacks
callbacks_backend.backend_callbacks(
    dash_app,
    backend_url
)
callbacks_misc.misc_callbacks(
    dash_app,
    al.open_txt_doc
)
callbacks_hist.hist_callbacks(dash_app)
callbacks_images.images_callbacks(dash_app)

# # helper functions
# def b64_2_numpy(string):
#     """Converts base64 encoded image to numpy array"""
#     decoded = base64.b64decode(string)
#     im = Image.open(io.BytesIO(decoded))
#     return np.array(im)


# def numpy_2_b64(arr, enc_format='png'):
#     """Converts numpy array to base64 encoded image"""
#     img_pil = Image.fromarray(arr)
#     buff = io.BytesIO()
#     img_pil.save(buff, format=enc_format)
#     return base64.b64encode(buff.getvalue()).decode("utf-8")


# def upload_demo():
#     """Returns demo img as base64 string"""
#     fname = Path(__file__).parent / 'demo_img.jpg'
#     img = mpimg.imread(fname)
#     return numpy_2_b64(img, enc_format='jpeg')




# @dash_app.callback(
#     Output('postprocessing', 'image_content'),
#     [Input('pred_json_copy', 'children')]
# )
# def image_postprocess(pred_json_copy):
#     data = json.loads(pred_json_copy)
#     pred = np.asarray(data['yimage_list'], dtype=np.uint8)
#     return array_to_data_url(pred * 255)


# @dash_app.callback(
#     Output('postprocessing', 'lineColor'),
#     [Input('colourpicker', 'value')]
# )
# def pick_color(pick):
#     if pick == 'draw':
#         colour = 'white'
#     elif pick == 'remove':
#         colour = 'red'
#     else:
#         colour = 'black'
#     return colour


# @dash_app.callback(
#     Output('postprocessing', 'lineWidth'),
#     [Input('brushwidth', 'value')]
# )
# def update_brushwidth(val):
#     return val


# @dash_app.callback(
#     Output('pred_json_copy', 'children'),
#     [Input('pred_json', 'children'),
#      Input('postprocessing', 'trigger')],
#     [State('pred_json_copy', 'children'),
#      State('size_distr_json', 'children'),
#      State('postprocessing', 'json_data')]
# )
# def postprocess_pred(pred_json, nclicks, pred_json_copy, size_distr_json, pp_data):
#     ctx = dash.callback_context
#     if ctx.triggered[-1]['prop_id'] == 'pred_json.children':
#         pred = json.loads(pred_json)
#         ypred = np.asarray(pred['yimage_list'], dtype=np.uint8)
#         ypred = 255 * resize(ypred, (576, 768), order=0)
#         ypred = ypred.astype(np.uint8)

#         resize_factor = pred['rf'] / 9

#         return json.dumps({
#             'content_type': pred['content_type'],
#             'rf': resize_factor,
#             'yimage_list': ypred.tolist()
#         })
#     else:
#         pred = json.loads(pred_json_copy)
#         ypred = np.asarray(pred['yimage_list'], dtype=np.uint8)
#         ypred = 255 * resize(ypred, (576, 768), order=0) # USE TRANSFORM_DATA AFTER UPDATING REQUIREMENTS
#         ypred = ypred.astype(np.uint8)
#         labeled = {}
#         data = json.loads(pp_data)
#         for obj in data['objects'][1:]:
#             mask = parse_obj(obj)
#             mask = mask.astype(np.uint8)
#             if obj['stroke'] == 'white':
#                 ypred = np.bitwise_or(ypred, mask)
#             elif obj['stroke'] == 'red':
#                 if 'cache' not in labeled:
#                     size_distr = json.loads(size_distr_json)
#                     cache = np.asarray(size_distr['labeled_list'])
#                     cache = resize(cache, (576, 768), order=0, preserve_range=True)
#                     labeled['cache'] = cache.astype(np.int32)
#                 indices = np.nonzero(mask)
#                 remove = np.unique(labeled['cache'][indices])
#                 for r in remove:
#                     ypred[np.where(labeled['cache']==r)] = 0
#             else:
#                 ypred = np.bitwise_and(ypred, 1 - mask)

#         return json.dumps({
#             'content_type': pred['content_type'],
#             'rf': pred['rf'],
#             'yimage_list': ypred.tolist()
#         })



# def parse_obj(obj):
#     scale = 1 / obj['scaleX']
#     path = obj['path']
#     rr, cc = [], []
#     for (Q1, Q2) in zip(path[:-2], path[1:-1]):
#         inds = draw.bezier_curve(int(round(Q1[-1] / scale)),
#                                 int(round(Q1[-2] / scale)),
#                                 int(round(Q2[2] / scale)),
#                                 int(round(Q2[1] / scale)),
#                                 int(round(Q2[4] / scale)),
#                                 int(round(Q2[3] / scale)), 1)
#         rr += list(inds[0])
#         cc += list(inds[1])
#     radius = round(obj['strokeWidth'] / 2. / scale)
#     mask = np.zeros((576, 768), dtype=np.bool)
#     mask[rr, cc] = 1
#     mask = ndimage.binary_dilation(
#         mask,
#         morphology.disk(radius)
#     )
#     return mask




# @dash_app.callback(
#     [Output('yimage', 'src'),
#      Output('yimage', 'style')],
#     [Input('pred_json_copy', 'children'),
#      Input('opacity_value', 'value')]
# )
# def display_yimage(pred_json_copy, op_val):
#     data = json.loads(pred_json_copy)

#     # change color from black and white to blue and gold
#     lookup = np.asarray([[153, 153, 0], [45, 0, 78]], dtype=np.uint8)
#     prediction3 = lookup[data['yimage_list']]
#     encoded_pred = data['content_type'] + ',' + numpy_2_b64(prediction3)

#     # specify style
#     style = {
#         'position': 'absolute',
#         'top': 0,
#         'left': 69,
#         'opacity': op_val,
#         'height': 566,
#         'width': 768
#     }
#     return encoded_pred, style


# @dash_app.callback(
#     Output('labeled_pred', 'src'),
#     [Input('size_distr_json', 'children')]
# )
# def show_labeled_pred(size_distr_json):
#     """Displays labeled prediction image"""
#     data = json.loads(size_distr_json)
#     unique = data['unique_list']
#     labeled = np.asarray(data['labeled_list'])
#     flattened_color_arr = np.linspace(
#         0,
#         256 ** 3 - 1,
#         num=len(unique) + 1,
#         dtype=np.int32
#     )

#     # represent values in flattened_color_arr as three digit number with
#     # base 256 to create a color array with shape
#     # (num unique values including background, 3)
#     colors = np.zeros((len(unique) + 1, 3), dtype=np.uint8)
#     for i in range(len(colors)):
#         colors[i] = np.array([
#             (flattened_color_arr[i] // (256 ** 2)) % 256,
#             (flattened_color_arr[i] // (256)) % 256,
#             flattened_color_arr[i] % 256
#         ])

#     # create a lookup table using colors array and convert 2D labeled array
#     # into 3D rgb array
#     lookup = np.zeros((len(unique) + 1, 3), dtype=np.uint8)
#     lookup[np.unique(labeled - 1)] = colors
#     rgb = lookup[labeled - 1]

#     # convert from numpy array to base64 image
#     # rgb = np.asarray(data['rgb_pred_list'], dtype=np.uint8)
#     encoded_rgb = data['content_type'] + ',' + numpy_2_b64(rgb)
#     return encoded_rgb

# @dash_app.callback(
#     Output('pred_json', 'children'),
#     [Input('upload-image', 'contents'),
#      Input('demo', 'n_clicks')]
# )
# def get_prediction(contents, n_clicks):
#     """
#     Gets image segmentation prediction of uploaded
#     image using trained model.
#     """
#     ctx = dash.callback_context

#     # if try the demo has been clicked, upload demo image
#     if ctx.triggered[-1]['prop_id'] == 'demo.n_clicks':
#         if n_clicks is not None:
#             imgb64 = (
#                 'data:image/jpeg;base64,'
#                 + upload_demo()
#             )
#     # otherwise, upload user image
#     else:
#         imgb64 = contents

#     # convert from base64 to numpy array
#     content_type, content_string = imgb64.split(',')
#     img = b64_2_numpy(content_string)

#     response = requests.post(
#         f'{backend_url}/api/predict',
#         json={
#             'contents': img.tolist(),
#             'content_type': content_type
#         }
#     )
#     return response.text


# @dash_app.callback(
#     Output('size_distr_json', 'children'),
#     [Input('pred_json', 'children'),
#      Input('scatterclick', 'clickData')],
#     [State('size_distr_json', 'children')]
# )
# def get_size_distr(pred_json, click, size_distr_json):
#     """
#     Obtains size distribution of particles in image
#     by assigning unique values to the connected
#     regions of the predicted segment mask.
#     """
#     ctx = dash.callback_context

#     # if a new image has been uploaded, call orig_size_distr route in flask_api
#     if ctx.triggered[-1]['prop_id'] == 'pred_json.children':
#         response = requests.post(
#             f'{backend_url}/api/orig_size_distr',
#             json={'data_pred': pred_json}
#             )
#         return response.text

#     # otherwise, call clicked_size_distr route in flask_api
#     else:
#         response = requests.post(
#             f'{backend_url}/api/clicked_size_distr',
#             json={
#                 'data_pred': pred_json,
#                 'click': click,
#                 'size_distr_json': size_distr_json
#                 }
#             )
#         return response.text


# @dash_app.callback(
#     Output('instruction_paragraph', 'children'),
#     [Input('instruction_button', 'n_clicks')]
# )
# def show_instructions(n_clicks):
#     """
#     Shows a paragraph containing a description and
#     instructions for how to use the app. Clicking
#     on the input button again will hide the paragraph.
#     """
#     if n_clicks is None or n_clicks % 2 == 0:
#         return ''
#     else:
#         return al.open_txt_doc('instructions.txt')


# @dash_app.callback(
#     Output('file_name', 'children'),
#     [Input('upload-image', 'filename')]
# )
# def display_filename(filename):
#     """Displays name of uploaded file"""
#     return filename


# @dash_app.callback(
#     [Output('ximage', 'src'),
#      Output('raw_image', 'src')],
#     [Input('upload-image', 'contents'),
#      Input('demo', 'n_clicks')]
# )
# def display_ximage(contents, n_clicks):
#     ctx = dash.callback_context

#     # if try the demo has been clicked, use the demo image
#     if ctx.triggered[-1]['prop_id'] == 'demo.n_clicks':
#         if n_clicks is not None:
#             imgb64 = (
#                 'data:image/jpeg;base64,'
#                 + upload_demo()
#             )

#     # otherwise, use the user uplaoded image
#     else:
#         imgb64 = contents
#     return imgb64, imgb64


# @dash_app.callback(
#     Output('size_distr_graph', 'figure'),
#     [Input('size_distr_json', 'children'),
#      Input('binsize', 'value')]
# )
# def update_hist(size_distr_json, value):
#     """Displays histogram of size distribution"""
#     data = json.loads(size_distr_json)
#     size_distr = np.asarray(data['size_distr_list'])
#     if value is None:
#         bin_size = 10
#     else:
#         bin_size = value

#     return {
#         'data': [go.Histogram(
#                     x=size_distr,
#                     xbins={'size': bin_size}
#                 )],
#         'layout': go.Layout(
#             title={
#                 'text': '<b>Size<br>Distribution</b>',
#                 'font': {'size': 28}
#             },
#             xaxis={'title': 'Size (pixels)'},
#             yaxis={
#                 'title': 'Count',
#                 'tickformat': ',d'
#             },
#             annotations=[
#                 go.layout.Annotation(
#                     text=(
#                         f'<b>Mean</b> = {size_distr.mean():.2E}<br><br>'
#                         f'<b>Median</b> = {np.median(size_distr):.2E}<br><br>'
#                         f'<b>Std</b> = {size_distr.std():.2E}'
#                     ),
#                     font={'size': 12},
#                     align='left',
#                     bordercolor='#B02405',
#                     borderwidth=2,
#                     borderpad=8,
#                     showarrow=False,
#                     xref='paper',
#                     yref='paper',
#                     x=0.02,
#                     y=1.30
#                 )
#             ]
#         )
#     }


# @dash_app.callback(
#     Output('download-link', 'href'),
#     [Input('size_distr_json', 'children')]
# )
# def update_download_link(size_distr_json):
#     data = json.loads(size_distr_json)
#     size_distr = np.asarray(data['size_distr_list'])

#     # create .csv link from numpy array
#     buff = io.StringIO()
#     csv_string = np.savetxt(buff, size_distr, encoding='utf-8')
#     csv_string = 'data:text/csv;charset=utf-8,' + buff.getvalue()
#     return csv_string


if __name__ == '__main__':
    dash_app.run_server(debug=True)
