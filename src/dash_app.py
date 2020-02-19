import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import json
import requests
import numpy as np


# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash_app = dash.Dash(
    __name__,
    # external_stylesheets=external_stylesheets
)


def background_scatter_plot_for_clickdata():
    """
    Creates a dense scatter plot behind the labeled
    prediction image so that users can click on points
    in the image and the app can recieve the coordinates
    """
    return dcc.Graph(
        id='scatterclick',
        figure={
            'data': [
                go.Scattergl(
                    x=[i for j in range(192) for i in range(256)],
                    y=[j for j in range(192) for i in range(256)],
                    mode='markers',
                    marker_opacity=0
                )
            ],
            'layout': {
                'margin': {
                    'r': 0,
                    'l': 0,
                    't': 0,
                    'b': 0
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
                'clickmode': 'event',
                'hovermode': 'closest',
                'height': 632,
                'width': 768
            }
        },
        style={
            'position': 'absolute',
            'top': 0,
            'left': 0,
            'opacity': 0.2
        }
    )


# dashboard layout
dash_app.layout = html.Div([
    # Title
    html.H1(children='SAEMI', style={'textAlign': 'center'}),

    # Instructions
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

    # Upload file button
    dcc.Upload(
        id='upload-image',
        children=html.Button('Upload File'),
        style={'display': 'flex', 'justify-content': 'center'}
    ),

    # Display filename after uploading
    html.H1(
        id='file_name',
        style={
            'marginTop': 25,
            'textAlign': 'center',
            'fontSize': 12
        }
    ),
    html.Hr(),

    # Histogram
    dcc.Graph(id='size_distr_graph'),
    html.Hr(),

    # Images
    html.Div([
        # Labeled Prediction Image
        html.Div(
            children=[
                html.H2(children='Segments', style={'textAlign': 'center'}),
                html.P(
                    children="""
                    If the below image does not provide a satisfactory
                    segmentation of the uploaded image, you can click
                    on different segments to remove them from the image
                    and size distribution histogram below.
                    """,
                    style={'margin-left': 150, 'margin-right': 150}
                ),
                html.Div(
                    children=[
                        html.Img(
                            id='labeled_pred',
                            style={
                                'position': 'absolute',
                                'top': 32,
                                'left': 0,
                                'height': 566,
                                'width': 768
                            }
                        ),
                        background_scatter_plot_for_clickdata()
                    ],
                    style={'position': 'relative'}
                )
            ],
            className='six columns'
        ),
        # Original Image and Overlay
        html.Div(
            id='output-images',
            className='six columns'
        )
    ]),

    # Hidden Divs containing json data
    html.Div(id='pred_json', style={'display': 'none'}),
    html.Div(id='size_distr_json', style={'display': 'none'})

])


@dash_app.callback(
    Output('pred_json', 'children'),
    [Input('upload-image', 'contents')]
)
def get_prediction(contents):
    """
    Gets image segmentation prediction of uploaded
    image using trained model.
    """
    response = requests.post(
        'http://localhost:5000/api/predict',
        json={'contents': contents}
    )
    response = response.text
    return response


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
    # determine most recent callback
    ctx = dash.callback_context

    # if a new image has been uploaded, call orig_size_distr route in flask_api
    if ctx.triggered[-1]['prop_id'] == 'pred_json.children':
        response = requests.post(
            'http://localhost:5000/api/orig_size_distr',
            json={
                'data_pred': pred_json
                }
            )
        response = response.text
        return response

    # otherwise the callback should only be triggered by clicking on the
    # image/background scatter plot. Then call clicked_size_distr route
    # in flask api
    else:
        response = requests.post(
            'http://localhost:5000/api/clicked_size_distr',
            json={
                'data_pred': pred_json,
                'click': click,
                'size_distr_json': size_distr_json
                }
            )
        response = response.text
        return response


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
    """Displays name of uploaded file"""
    return filename


@dash_app.callback(
    Output('output-images', 'children'),
    [Input('pred_json', 'children')]
)
def upload_images(pred_json):
    """Displays uploaded image and an overlay with the prediction"""
    data = json.loads(pred_json)

    return html.Div([
        # Original Image
        html.Div(
            children=[
                html.H2(
                    children='Original Image',
                    style={'textAlign': 'left'}
                ),
                html.Img(
                    src=data['ximage_b64'],
                    style={
                        'height': '50%',
                        'width': '50%'
                    }
                )
            ],
            style={'textAlign': 'left'}
        ),

        # Overlay with prediction
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
    Output('labeled_pred', 'src'),
    [Input('size_distr_json', 'children')]
)
def show_labeled_pred(size_distr_json):
    """Displays labeled prediction image"""
    data = json.loads(size_distr_json)
    return data['rgb_pred_b64']


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
