import dash
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import json
import requests
import numpy as np

import matplotlib.image as mpimg
from pathlib import Path
from PIL import Image
import io
import base64

import app_layout as al
# from src import config as cf


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash_app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets
)
server = dash_app.server
backend_url = 'https://saemi-backend.herokuapp.com/'
# backend_url = f'http://{cf.HOST}:{cf.PORT}'


# helper functions
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
    fname = Path(__file__).parent / 'demo_img.jpg'
    img = mpimg.imread(fname)
    return numpy_2_b64(img, enc_format='jpeg')


# dashboard layout
dash_app.layout = al.app_layout()


@dash_app.callback(
    Output('pred_json', 'children'),
    [Input('upload-image', 'contents'),
     Input('demo', 'n_clicks')]
)
def get_prediction(contents, n_clicks):
    """
    Gets image segmentation prediction of uploaded
    image using trained model.
    """
    ctx = dash.callback_context

    # if try the demo has been clicked, upload demo image
    if ctx.triggered[-1]['prop_id'] == 'demo.n_clicks':
        if n_clicks is not None:
            imgb64 = (
                'data:image/jpeg;base64,'
                + upload_demo()
            )
    # otherwise, upload user image
    else:
        imgb64 = contents

    # convert from base64 to numpy array
    content_type, content_string = imgb64.split(',')
    img = b64_2_numpy(content_string)

    response = requests.post(
        f'{backend_url}/api/predict',
        json={
            'contents': img.tolist(),
            'content_type': content_type
        }
    )
    return response.text


@dash_app.callback(
    Output('size_distr_json', 'children'),
    [Input('pred_json', 'children'),
     Input('scatterclick', 'clickData')],
    [State('size_distr_json', 'children')]
)
def get_size_distr(pred_json, click, size_distr_json):
    """
    Obtains size distribution of particles in image
    by assigning unique values to the connected
    regions of the predicted segment mask.
    """
    ctx = dash.callback_context

    # if a new image has been uploaded, call orig_size_distr route in flask_api
    if ctx.triggered[-1]['prop_id'] == 'pred_json.children':
        response = requests.post(
            f'{backend_url}/api/orig_size_distr',
            json={'data_pred': pred_json}
            )
        return response.text

    # otherwise, call clicked_size_distr route in flask_api
    else:
        response = requests.post(
            f'{backend_url}/api/clicked_size_distr',
            json={
                'data_pred': pred_json,
                'click': click,
                'size_distr_json': size_distr_json
                }
            )
        return response.text


@dash_app.callback(
    Output('instruction_paragraph', 'children'),
    [Input('instruction_button', 'n_clicks')]
)
def show_instructions(n_clicks):
    """
    Shows a paragraph containing a description and
    instructions for how to use the app. Clicking
    on the input button again will hide the paragraph.
    """
    if n_clicks is None or n_clicks % 2 == 0:
        return ''
    else:
        return al.open_txt_doc('instructions.txt')


@dash_app.callback(
    Output('file_name', 'children'),
    [Input('upload-image', 'filename')]
)
def display_filename(filename):
    """Displays name of uploaded file"""
    return filename


@dash_app.callback(
    Output('ximage', 'src'),
    [Input('upload-image', 'contents'),
     Input('demo', 'n_clicks')]
)
def display_ximage(contents, n_clicks):
    ctx = dash.callback_context

    # if try the demo has been clicked, use the demo image
    if ctx.triggered[-1]['prop_id'] == 'demo.n_clicks':
        if n_clicks is not None:
            imgb64 = (
                'data:image/jpeg;base64,'
                + upload_demo()
            )

    # otherwise, use the user uplaoded image
    else:
        imgb64 = contents
    return imgb64


@dash_app.callback(
    [Output('yimage', 'src'),
     Output('yimage', 'style')],
    [Input('pred_json', 'children'),
     Input('opacity_value', 'value')]
)
def display_yimage(pred_json, op_val):
    data = json.loads(pred_json)

    # change color from black and white to blue and gold
    lookup = np.asarray([[45, 0, 78], [153, 153, 0]], dtype=np.uint8)
    prediction3 = lookup[data['yimage_list']]
    encoded_pred = data['content_type'] + ',' + numpy_2_b64(prediction3)

    # specify style
    style = {
        'position': 'absolute',
        'top': 0,
        'left': 69,
        'opacity': op_val,
        'height': 566,
        'width': 768
    }
    return encoded_pred, style


@dash_app.callback(
    Output('labeled_pred', 'src'),
    [Input('size_distr_json', 'children')]
)
def show_labeled_pred(size_distr_json):
    """Displays labeled prediction image"""
    data = json.loads(size_distr_json)

    # convert from numpy array to base64 image
    rgb = np.asarray(data['rgb_pred_list'], dtype=np.uint8)
    encoded_rgb = data['content_type'] + ',' + numpy_2_b64(rgb)
    return encoded_rgb


@dash_app.callback(
    Output('size_distr_graph', 'figure'),
    [Input('size_distr_json', 'children')]
)
def update_hist(size_distr_json):
    """Displays histogram of size distribution"""
    data = json.loads(size_distr_json)
    size_distr = np.asarray(data['size_distr_list'])
    return {
        'data': [go.Histogram(
                    x=size_distr,
                    xbins={'size': 5}
                )],
        'layout': go.Layout(
            title={
                'text': '<b>Size<br>Distribution</b>',
                'font': {'size': 28}
            },
            xaxis={'title': 'Size (pixels)'},
            yaxis={
                'title': 'Count',
                'tickformat': ',d'
            },
            annotations=[
                go.layout.Annotation(
                    text=(
                        f'<b>Mean</b> = {size_distr.mean():.2E}<br><br>'
                        f'<b>Median</b> = {np.median(size_distr):.2E}<br><br>'
                        f'<b>Std</b> = {size_distr.std():.2E}'
                    ),
                    font={'size': 12},
                    align='left',
                    bordercolor='#B02405',
                    borderwidth=2,
                    borderpad=8,
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.02,
                    y=1.30
                )
            ]
        )
    }


@dash_app.callback(
    Output('download-link', 'href'),
    [Input('size_distr_json', 'children')]
)
def update_download_link(size_distr_json):
    data = json.loads(size_distr_json)
    size_distr = np.asarray(data['size_distr_list'])

    # create .csv link from numpy array
    buff = io.StringIO()
    csv_string = np.savetxt(buff, size_distr, encoding='utf-8')
    csv_string = 'data:text/csv;charset=utf-8,' + buff.getvalue()
    return csv_string


if __name__ == '__main__':
    dash_app.run_server(debug=True)
