import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
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

# global variables
data_path = Path.cwd() / '..' / 'data' / 'dataset' / 'good'
learn = pred.load_learner(
    Path(data_path) / 'train_x',
    Path(data_path) / 'train_y_png',
    np.array(['background', 'particle'], dtype='<U17'),
    (192, 256),
    16,
    pretrained='stage-2_bs16'
)

dash_app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets
)
dash_app.layout = html.Div([
    html.H1(children='SAEMI', style={'textAlign': 'center'}),
    html.Div(
        children=[
            html.Button(
                'Instructions',
                id='instruction_button'
            ),
            html.P(
                id='instruction_paragraph',
                style={'margin-left': 150, 'margin-right': 150}
            )
        ],
        style={'textAlign': 'center'}
    ),
    dcc.Upload(
        id='upload-image',
        children=html.Button('Upload File'),
        style={'display': 'flex', 'justify-content': 'center'}
    ),
    html.H1(
        id='file_name',
        style={
            'marginTop': 25,
            'textAlign': 'center',
            'fontSize': 12
        }
    ),
    html.Hr(),
    dcc.Graph(id='size_distr_graph'),
    html.Hr(),
    html.Div([
        html.Div(
            children=[
                html.H2(children='Segments', style={'textAlign': 'center'}),
                dcc.Graph(
                    id='labeled_pred',
                    style={'justify-content': 'center'}
                ),
                html.P(
                    children="""
                    If the above image does not provide a satisfactory
                    segmentation of the uploaded image, you can click
                    on different segments to remove them from the image
                    and size distribution histogram above.
                    """,
                    style={'margin-left': 150, 'margin-right': 150}
                )
            ],
            className='six columns'
        ),
        html.Div(
            id='output-images',
            className='six columns'
        )
    ]),

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
    Output('instruction_paragraph', 'children'),
    [Input('instruction_button', 'n_clicks')]
)
def show_instructions(n_clicks):
    if n_clicks is None or n_clicks % 2 == 0:
        return ''
    else:
        return """
        SAEMI is a tool for obtaining a Size Analysis of particles
        in Electron Microscopy Images. To obtain a size analysis,
        first upload an Electron Microscopy image using the Upload
        button below. The tool will then predict how to segment the
        particles from the background using a model trained through
        deep learning. After segmentation, the tool will label each
        separate segment and count each segment's number of pixels.
        This size distribution is then displayed in a histogram which
        can be used to determine properties such as the mean size,
        median size or standard deviation of sizes in the image.
        Thank you for considering using this tool and any additional
        feedback is always welcome. Good luck!
        """


@dash_app.callback(
    Output('file_name', 'children'),
    [Input('upload-image', 'filename')]
)
def display_filename(filename):
    return filename


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

    # make prediction
    prediction = pred.predict_segment(learn, img).numpy()
    prediction = prediction[0]
    print('Finished Prediction')

    encoded_pred = np.uint8(255*prediction)
    encoded_pred = content_type + ',' + numpy_2_b64(encoded_pred)
    start = time.time()
    pred_json = json.dumps({
        'content_type': content_type,
        'ximage_b64': contents,
        'ximage_list': im.tolist(),
        'yimage_b64': encoded_pred,
        'yimage_list': prediction.tolist()
    })
    print('pred_json = ', time.time()-start)
    return pred_json


@dash_app.callback(
    Output('size_distr_json', 'children'),
    [Input('pred_json', 'children'),
     Input('labeled_pred', 'clickData')],
    [State('size_distr_json', 'children')]
)
def get_size_distr(pred_json, click, size_distr_json):
    data_pred = json.loads(pred_json)
    start = time.time()
    if (
        (size_distr_json is None)
        or (
            data_pred['ximage_list']
            != json.loads(size_distr_json)['ximage_list']
        )
    ):
        # load predictions
        pred_data = np.asarray(data_pred['yimage_list'], dtype=np.uint8)

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

        lookup = np.zeros((len(unique) + 1, 3), dtype=np.uint8)
        lookup[np.unique(labeled - 1)] = colors
        rgb = lookup[labeled - 1]

        encoded_rgb = data_pred['content_type'] + ',' + numpy_2_b64(rgb)

        sd_json = json.dumps({
            'content_type': data_pred['content_type'],
            'ximage_list': data_pred['ximage_list'],
            'rgb_pred_b64': encoded_rgb,
            'rgb_pred_list': rgb.tolist(),
            'labeled_list': labeled.tolist(),
            'unique_list': unique.tolist(),
            'size_distr_list': size_distr.tolist()
        })
        print('size_distr_json = ', time.time()-start)
        return sd_json

    else:
        data_size_distr = json.loads(size_distr_json)
        rgb = np.asarray(data_size_distr['rgb_pred_list'], dtype=np.uint8)
        labeled = data_size_distr['labeled_list']
        unique = np.asarray(data_size_distr['unique_list'])
        size_distr = np.asarray(data_size_distr['size_distr_list'])

        # remove clicked segments
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

        encoded_rgb = data_pred['content_type'] + ',' + numpy_2_b64(rgb)

        sd_json = json.dumps({
            'content_type': data_pred['content_type'],
            'ximage_list': data_pred['ximage_list'],
            'rgb_pred_b64': encoded_rgb,
            'rgb_pred_list': rgb.tolist(),
            'labeled_list': labeled,
            'unique_list': unique.tolist(),
            'size_distr_list': size_distr.tolist()
        })
        print('size_distr_json = ', time.time()-start)
        return sd_json


@dash_app.callback(
    Output('output-images', 'children'),
    [Input('pred_json', 'children')]
)
def upload_images(pred_json):
    data = json.loads(pred_json)

    return html.Div([
        html.Div(
            children=[
                html.H2(
                    children='Original Image',
                    style={'textAlign': 'left'}
                ),
                html.Img(
                    src=data['ximage_b64'],
                    style={
                        'margin-top': 0,
                        'margin-bot': 0,
                        'height': '50%',
                        'width': '50%'
                    }
                )
            ],
            style={'textAlign': 'left'}
        ),
        html.Div(
            children=[
                html.H2(
                    children='Overlay of Segments with Original',
                    style={'textAlign': 'left'}
                ),
                html.Div(
                    children=[
                        html.Img(
                            src=data['ximage_b64'],
                            style={
                                'height': '50%',
                                'width': '50%'
                            }
                        ),
                        html.Img(
                            src=data['yimage_b64'],
                            style={
                                'position': 'absolute',
                                'top': 0,
                                'left': 0,
                                'opacity': 0.5,
                                'height': '98%',
                                'width': '50%'
                            }
                        )
                    ],
                    style={'position': 'relative'}
                )
            ],
            style={'textAlign': 'left'}
        ),
    ])


@dash_app.callback(
    Output('labeled_pred', 'figure'),
    [Input('size_distr_json', 'children')]
)
def show_labeled_pred(size_distr_json):
    data = json.loads(size_distr_json)
    return {
        'data': [go.Scattergl(
            x=[i for j in range(192) for i in range(256)],
            y=[j for j in range(192) for i in range(256)],
            mode='markers',
            marker_opacity=0
        )],
        'layout': {
            'margin': {
                'r': 0,
                't': 0,
                'b': 10
            },
            'xaxis': {
                'range': [0, 255],
                'showgrid': False,
                'showticklabels': False,
                'zeroline': False
            },
            'yaxis': {
                'showgrid': False,
                'showticklabels': False,
                'zeroline': False
            },
            'images': [{
                'range': [0, 191],
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
            'height': 566,
            'width': 768
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
            title={
                'text': '<b>Size<br>Distribution</b>',
                'font': {
                    'size': 28
                }
            },
            xaxis={'title': 'Size (pixels)'},
            yaxis={'title': 'Count'},
            annotations=[
                go.layout.Annotation(
                    text=(
                        f'<b>Mean</b> = {size_distr.mean():.2E}<br><br>'
                        f'<b>Median</b> = {np.median(size_distr):.2E}<br><br>'
                        f'<b>Std</b> = {size_distr.std():.2E}'
                    ),
                    font={
                        'size': 12
                    },
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


if __name__ == '__main__':
    dash_app.run_server(debug=True)
