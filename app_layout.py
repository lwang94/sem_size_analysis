import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from pathlib import Path


def open_txt_doc(filename):
    """Opens texts in docs folder that go in front end of app"""
    path = (
        Path(__file__).parent
        / 'docs'
        / filename
    )
    with open(path, 'r') as txtfile:
        text = txtfile.read()
    return text


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


def app_layout():
    """Returns layout for the front-end of the app"""
    return html.Div([
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

        # Upload demo or file
        html.Div(
            children=[
                html.Div(
                    html.Button(
                        id='demo',
                        children='Try the Demo'
                    ),
                    style={'display': 'inline-block'}
                ),
                html.Div(
                    dcc.Upload(
                        id='upload-image',
                        children=html.Button('Upload Image'),
                    ),
                    style={'display': 'inline-block'}
                )
            ],
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
        html.A(
            'Download Data',
            id='download-link',
            download='size_distribution.csv',
            href='',
            target='_blank',
            style={'display': 'flex', 'justify-content': 'center'}
        ),
        html.Hr(),

        # Images
        html.Div([
            # Labeled Prediction Image
            html.Div(
                children=[
                    html.H2(children='Segments'),
                    html.Pre(
                        children=open_txt_doc('remove_note.txt')
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
                style={'textAlign': 'center'},
                className='six columns'
            ),
            # Original Image and Overlay
            html.Div(
                children=[
                    html.H2(children='Overlay of Original with Segments'),
                    html.Pre(
                        children=open_txt_doc('resize_warning.txt'),
                        style={'margin-bottom': '32px'}
                    ),
                    html.Div(
                        children=[
                            html.Img(
                                id='ximage',
                                style={
                                    'top': 32,
                                    'height': 566,
                                    'width': 768
                                }
                            ),
                            html.Img(
                                id='yimage'
                            )
                        ],
                        style={'position': 'relative'}
                    ),
                    dcc.Slider(
                        id='opacity_value',
                        min=0,
                        max=1,
                        step=0.1,
                        value=0.5,
                    ),
                ],
                style={'textAlign': 'center'},
                className='six columns'
            ),
        ]),
        html.Hr(),

        #Feedback
        html.Div(
            children=[
                html.H2('Your Feedback is Welcome!'),
                html.P(
                    """
                    For any questions, concerns, or suggestions,
                    please email me at:
                    """
                ),
                html.B('lawrence.fy.wang@gmail.com'),
                html.P('Also check out some of my other projects at:'),
                html.A(
                    'https://github.com/lwang94',
                    href='https://github.com/lwang94'
                )
            ],
            style={'textAlign': 'center'}
        ),

        # Hidden Divs containing json data
        html.Div(id='pred_json', style={'display': 'none'}),
        html.Div(id='size_distr_json', style={'display': 'none'})
    ])
