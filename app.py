import dash
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import json
import requests
import numpy as np

import app_layout as al
# from src import config as cf


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash_app = dash.Dash(
    __name__,
    external_stylesheets=external_stylesheets
)
server = dash_app.server
backend_url = 'https://saemibackend.onrender.com'


# dashboard layout
dash_app.layout = al.app_layout()


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
        f'{backend_url}/api/predict',
        json={'contents': contents}
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
    Output('output-images', 'children'),
    [Input('pred_json', 'children')]
)
def upload_images(pred_json):
    """Displays uploaded image and an overlay with the prediction"""
    data = json.loads(pred_json)
    ximage = np.asarray(data['ximage_list'])
    yimage = np.asarray(data['yimage_list'])
    rf = (
        ximage.shape[0] * ximage.shape[1]
        / (yimage.shape[0] * yimage.shape[1])
    )

    return html.Div([
        # Overlay with prediction
        html.Div(
            children=[
                html.H2(children='Overlay of Original with Segments'),
                html.Div([
                    html.Div(
                        html.Div(
                            children=[
                                html.Img(
                                    src=data['ximage_b64'],
                                    style={
                                        'height': '50%',
                                        'width': '100%'
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
                                        'width': '100%'
                                    }
                                )
                            ],
                            style={'position': 'relative'}
                        ),
                        style={
                            'width': '50%',
                            'display': 'inline-block'}
                    ),
                    html.Div(
                        html.Pre(
                            f"""Size: {yimage.shape[0]}x{yimage.shape[1]} pixels
                            \n{al.open_txt_doc('resize_note.txt')}
                            """,
                            style={
                                'fontSize': 14,
                                'margin-left': 5,
                                'margin-top': 0,
                                'color': 'red'
                            }
                        ),
                        style={
                            'width': '50%',
                            'display': 'inline-block',
                            'vertical-align': 'top'}
                    )
                ])
            ],
            style={'textAlign': 'left'}
        ),
        # Original Image
        html.Div(
            children=[
                html.H2(children='Original Image'),
                html.Div([
                    html.Div(
                        html.Img(
                            src=data['ximage_b64'],
                            style={
                                'height': '50%',
                                'width': '100%'
                            }
                        ),
                        style={
                            'width': '50%',
                            'display': 'inline-block'}
                    ),
                    html.Div(
                        html.Pre(
                            f"""Size: {ximage.shape[0]}x{ximage.shape[1]} pixels
                            \nRescaling Factor: {rf:.2E}
                            \nMean pixel value: {ximage.mean():.2E}
                            """,
                            style={
                                'fontSize': 14,
                                'margin-left': 5,
                                'margin-top': 0,
                                'color': 'red'
                            }
                        ),
                        style={
                            'width': '50%',
                            'display': 'inline-block',
                            'vertical-align': 'top'}
                    )
                ])
            ],
            style={'textAlign': 'left'}
        )
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


if __name__ == '__main__':
    dash_app.run_server(debug=True)
