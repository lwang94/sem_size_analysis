import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import base64
import numpy as np
import io
from PIL import Image

import transform_data as td
import predict as pred

from pathlib import Path
import json

import time

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# dash_app = dash.Dash(server=server, external_stylesheets=external_stylesheets)
dash_app.layout = html.Div(children=[
    html.H1(children='Welcome to My App', style={'textAlign':'center'}),
    dcc.Upload(
        id='upload-image',
        children=html.Button('Upload File'),
        style={'display':'flex', 'justify-content':'center'}),
    html.Hr(),
    html.Div(id='size_distr'),
    html.Hr(),
    html.Div(id='output-images'),

    html.Div(id='intermediate-value', style={'display':'none'})
])

def b64_2_numpy(string):
    decoded = base64.b64decode(string)
    im = Image.open(io.BytesIO(decoded))
    return np.array(im)


def numpy_2_b64(arr, enc_format='png'):
    img_pil = Image.fromarray(arr)
    buff = io.BytesIO()
    img_pil.save(buff, format=enc_format)
    return base64.b64encode(buff.getvalue()).decode("utf-8")


@dash_app.callback(
    Output('intermediate-value', 'children'),
    [Input('upload-image', 'contents')]
)
def get_prediction(contents):
    # convert b64 encoded string to numpy array
    content_type, content_string = contents.split(',')
    im = b64_2_numpy(content_string)

    print('Making Prediction')
    # perform data transformation
    if len(im.shape) == 2:
        im = td.make_3channel(im)
    img = td.resize(im, (192, 256))
    img = td.fastai_image(img)

    # load model
    data_path = Path(__file__).parents[1] / 'data' / 'dataset' / 'good'
    learn = pred.load_learner(
        'stage-2_bs16',
        Path(data_path) / 'train_x',
        Path(data_path) /'train_y_png',
        np.array(['background', 'particle'], dtype='<U17'),
        (192, 256),
        16
     )

    # make prediction
    prediction = pred.predict_segment(learn, img).numpy()
    prediction = prediction[0]
    print('Finished Prediction')

    # convert numpy array to b64 encoded string
    pred_convert = np.uint8(255 * prediction)
    encoded = numpy_2_b64(pred_convert)
    encoded_pred = content_type + ',' + encoded
    return json.dumps({
        'content_type': content_type,
        'ximage_list': im.tolist(),
        'ximage_b64': contents,
        'yimage_list': prediction.tolist(),
        'yimage_b64': encoded_pred
    })


@dash_app.callback(
    Output('output-images', 'children'),
    [Input('intermediate-value', 'children')]
)
def upload_images(inter_json):
    data = json.loads(inter_json)
    return html.Div([
        html.Img(src=data['ximage_b64'], className='six columns'),
        html.Img(src=data['yimage_b64'], className='six columns')
    ])

@dash_app.callback(
    Output('size_distr', 'children'),
    [Input('intermediate-value', 'children')]
)
def update_graph(inter_json):
    data = json.loads(inter_json)
    pred_data = np.asarray(data['yimage_list'])

    labeled, unique, size_distr = pred.get_size_distr(pred_data)
    flattened_colors = np.linspace(0, 256 ** 3 - 1, num=len(unique) + 1, dtype=np.int64)
    colors = np.zeros((len(unique) + 1, 3), dtype=np.uint8)

    for i in range(len(colors)):
        colors[i] = np.array([
            (flattened_colors[i] // (256 ** 2)) % 256,
            (flattened_colors[i] // (256)) % 256,
            flattened_colors[i] % 256
        ])
    lookup = np.zeros((255, 3), dtype=np.uint8)
    lookup[np.unique(labeled)] = colors
    rgb = lookup[labeled]
    encoded_rgb = data['content_type'] + ',' + numpy_2_b64(rgb)


    x_scatter = [i for j in range(192) for i in range(256)]
    y_scatter = [j for j in range(192) for i in range(256)]

    return html.Div([
        dcc.Graph(
            figure={
                'data': [go.Scattergl(
                    x=np.array(x_scatter),
                    y=np.array(y_scatter),
                    mode='markers',
                    marker_opacity=0
                )],
                'layout': {
                    'xaxis': {
                        'range': [0, 256]
                    },
                    'yaxis': {
                        'range': [0, 192],
                        'scaleanchor': 'x',
                        'scaleratio': 1
                    },
                    'images': [{
                        'xref': 'x',
                        'yref': 'y',
                        'x': 0,
                        'y': 192,
                        'sizex': 256,
                        'sizey': 192,
                        'sizing': 'stretch',
                        'layer': 'below',
                        'source': encoded_rgb
                    }]
                }
            },
            className='four columns'
        ),
        dcc.Graph(
            figure={
                'data': [go.Histogram(
                            x=size_distr,
                            xbins={'size':1}
                        )],
                'layout': go.Layout(
                    title ='Size Distribution',
                    xaxis = {'title': 'Size', 'showgrid': False},
                    yaxis = {'title': 'Count', 'showgrid': False}
                )
            },
            className='eight columns'
        )
    ])



if __name__ == '__main__':
    dash_app.run_server(debug=True)

#Create new intermediate callback that accepts inter_json and clickData as inputs and have it return a json that labeled_pred and size_distr uses