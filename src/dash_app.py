import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

import base64
import numpy as np
import io
from PIL import Image
from matplotlib.pyplot import Figure

import transform_data as td
import predict as pred

from pathlib import Path
import json


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash_app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets
)
dash_app.layout = html.Div(children=[
    html.H1(children='Welcome to My App', style={'textAlign': 'center'}),
    dcc.Upload(
        id='upload-image',
        children=html.Button('Upload File'),
        style={'display': 'flex', 'justify-content': 'center'}
    ),
    html.Hr(),
    html.Div([
        dcc.Graph(
            id='labeled_pred',
            className='five columns'
        ),
        dcc.Graph(
            id='size_distr_graph',
            className='seven columns'
        )
    ]),
    html.Hr(),
    html.Div(id='output-images'),

    html.Div(id='pred_json', style={'display': 'none'}),
    html.Div(id='size_distr_json', style={'display': 'none'})
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
    Output('pred_json', 'children'),
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
        Path(data_path) / 'train_x',
        Path(data_path) / 'train_y_png',
        np.array(['background', 'particle'], dtype='<U17'),
        (192, 256),
        16,
        pretrained='stage-2_bs16'
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
    [Input('pred_json', 'children')]
)
def upload_images(pred_json):
    data = json.loads(pred_json)

    x_array = np.asarray(data['ximage_list'], dtype=np.uint8)
    x_array = td.resize(x_array, (192, 256))
    y_array = np.asarray(data['yimage_list'], dtype=np.uint8)
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(x_array)
    ax.imshow(y_array, alpha=0.5)
    ax.axis('off')
    buff = io.BytesIO()
    fig.savefig(buff, format='png')
    encoded_overlay = (
        data['content_type']
        + ','
        + base64.b64encode(buff.getvalue()).decode('utf-8')
    )

    return html.Div([
        html.Img(
            src=data['ximage_b64'],
            style={
                'height': '25%',
                'width': '25%'
            },
            className='four columns'
        ),
        html.Img(
            src=data['yimage_b64'],
            style={
                'height': '25%',
                'width': '25%'
            },
            className='four columns'
        ),
        html.Img(
            src=encoded_overlay,
            style={
                'height': '35%',
                'width': '35%'
            },
            className='four columns'
        )
    ])


@dash_app.callback(
    Output('size_distr_json', 'children'),
    [Input('pred_json', 'children'),
     Input('labeled_pred', 'clickData')]
)
def get_size_distr(pred_json, click):
    # load predictions
    data = json.loads(pred_json)
    pred_data = np.asarray(data['yimage_list'], dtype=np.uint8)

    # get size distributions
    labeled, unique, size_distr = pred.get_size_distr(pred_data)

    # color predictions
    flattened_colors = np.linspace(
        0,
        256 ** 3 - 1,
        num=len(unique) + 1,
        dtype=np.int64
    )
    colors = np.zeros((len(unique) + 1, 3), dtype=np.uint8)

    for i in range(len(colors)):
        colors[i] = np.array([
            (flattened_colors[i] // (256 ** 2)) % 256,
            (flattened_colors[i] // (256)) % 256,
            flattened_colors[i] % 256
        ])

    lookup = np.zeros((255, 3), dtype=np.uint8)
    lookup[np.unique(labeled)] = colors
    print(lookup)
    rgb = lookup[labeled]

    # remove clicked segments
    if click is not None:
        xclick, yclick = click['points'][0]['x'], click['points'][0]['y']
        for i in range(len(xclick)):
            remove = np.where(unique == labeled[191 - yclick[i], xclick[i]])[0]
            unique = np.delete(unique, remove)
            size_distr = np.delete(size_distr, remove)
            click_r, click_g, click_b = rgb[191 - yclick[i], xclick[i], :]
            mask = ((rgb[:, :, 0] == click_r)
                    & (rgb[:, :, 1] == click_g)
                    & (rgb[:, :, 2] == click_b))
            rgb[mask] = [0, 0, 0]

    encoded_rgb = data['content_type'] + ',' + numpy_2_b64(rgb)
    return json.dumps({
        'rgb_pred_list': rgb.tolist(),
        'rgb_pred_b64': encoded_rgb,
        'unique_list': unique.tolist(),
        'size_distr_list': size_distr.tolist()
    })


@dash_app.callback(
    Output('labeled_pred', 'figure'),
    [Input('size_distr_json', 'children')]
)
def show_labeled_pred(size_distr_json):
    data = json.loads(size_distr_json)
    return {
        'data': [go.Scattergl(
            x=np.array([i for j in range(192) for i in range(256)]),
            y=np.array([j for j in range(192) for i in range(256)]),
            mode='markers',
            marker_opacity=0
        )],
        'layout': {
            'xaxis': {
                'range': [0, 256],
                'showgrid': False,
                'showticklabels': False,
                'zeroline': False
            },
            'yaxis': {
                'range': [0, 192],
                'showgrid': False,
                'showticklabels': False,
                'zeroline': False,
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
                'source': data['rgb_pred_b64']
            }],
            'clickmode': 'event',
            'hovermode': 'closest',
        }
    }


@dash_app.callback(
    Output('size_distr_graph', 'figure'),
    [Input('size_distr_json', 'children')]
)
def update_hist(size_distr_json):
    data = json.loads(size_distr_json)
    size_distr = np.asarray(data['size_distr_list'])
    return {
        'data': [go.Histogram(
                    x=size_distr,
                    xbins={'size': 1}
                )],
        'layout': go.Layout(
            title='Size Distribution',
            xaxis={'title': 'Size', 'showgrid': False},
            yaxis={'title': 'Count', 'showgrid': False},
            annotations=[
                go.layout.Annotation(
                    text=(
                        f'Mean = {size_distr.mean()}<br>'
                        f'Median = {np.median(size_distr)}<br>'
                        f'Std = {size_distr.std()}'
                    ),
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper'
                )
            ]
        )
    }


if __name__ == '__main__':
    dash_app.run_server(debug=True)
