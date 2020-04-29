"""Generates front end layout"""

import dash_core_components as dcc
import dash_html_components as html
from dash_canvas import DashCanvas

from pathlib import Path


def open_txt_doc(filename):
    """Opens texts in docs folder that go in front end of app"""
    path = Path(__file__).parent / 'docs' / 'app' / filename
    with open(path, 'r') as txtfile:
        text = txtfile.read()
    return text


def app_layout():
    """Returns layout for the front-end of the app"""
    return html.Div([
        html.Div(
            children=[
                # Title
                html.H1(
                    children="""
                        SAEMI: Size Analysis of Electron Microscopy Images
                    """,
                    style={'textAlign': 'center'}
                ),

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

                # Display image and filename after uploading
                html.H1(
                    id='file_name',
                    style={
                        'marginTop': 25,
                        'textAlign': 'center',
                        'fontSize': 12
                    }
                ),
                html.Img(
                    id='raw_image',
                    style={
                        'height': 192,
                        'width': 256,
                        'display': 'block',
                        'margin-right': 'auto',
                        'margin-left': 'auto'
                    }
                ),

                # Loading progress
                dcc.Loading(
                    children=[
                        html.Div(id='pred_json', style={'display': 'none'}),
                        html.Div(
                            id='pred_json_copy',
                            style={'display': 'none'}
                        ),
                        html.Div(id='images_data', style={'display': 'none'}),
                        html.Div(
                            id='size_distr_json',
                            style={'display': 'none'}
                        )
                    ],
                    type='default'
                ),

                html.Hr(),
            ],
            className='one row'
        ),
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.H2(
                            'Enter Bin Size',
                            style={'fontSize': 16}
                        ),
                        dcc.Input(
                            id='binsize',
                            value=25,
                            type='number',
                            debounce=True,
                            min=1
                        )
                    ],
                    style={'display': 'inline-block'}
                ),
                # Histogram
                dcc.Graph(
                    id='size_distr_graph'
                ),
                html.A(
                    'Download Data',
                    id='download-link',
                    href='',
                    target='_blank',
                    style={'display': 'inline-block'}
                ),
                html.Hr()
            ],
            className='one row',
            style={'textAlign': 'center'}
        ),
        html.Div(
            children=[
                # Images
                html.Div([
                    # Uniquely Labeled Prediction Image
                    html.Div(
                        children=[
                            html.H2(children='Process Image'),
                            html.Pre(
                                children=open_txt_doc('postprocessing.txt'),
                                style={'margin-bottom': '32px'}
                            ),
                            html.Div(
                                children=[
                                    DashCanvas(
                                        id='postprocessing',
                                        width=768,
                                        height=576,
                                        hide_buttons=[
                                            'zoom',
                                            'pan',
                                            'line',
                                            'pencil',
                                            'rectangle',
                                            'select'
                                        ]
                                    ),
                                    dcc.RadioItems(
                                        id='showitem',
                                        options=[
                                            {'label': 'B/W', 'value': 'bw'},
                                            {
                                                'label': 'Colour',
                                                'value': 'colour'
                                            }
                                        ],
                                        value='bw',
                                        labelStyle={'display': 'inline-block'}
                                    ),
                                    dcc.Slider(
                                        id='brushwidth',
                                        min=2,
                                        max=40,
                                        step=1,
                                        value=5,
                                        tooltip={
                                            'always_visible': False,
                                            'placement': 'bottom'
                                        }
                                    ),
                                    dcc.Dropdown(
                                        id='colourpicker',
                                        options=[
                                            {
                                                'label': 'Draw (white)',
                                                'value': 'draw'
                                            },
                                            {
                                                'label': 'Remove (red)',
                                                'value': 'remove'
                                            },
                                            {
                                                'label': 'Erase (black)',
                                                'value': 'erase'
                                            }
                                        ],
                                        value='draw',
                                        style={
                                            'width': '300px',
                                            'margin': 'auto'
                                        }
                                    )
                                ],
                                style={'display': 'inline-block'}
                            ),
                        ],
                        style={'textAlign': 'center'},
                        className='six columns'
                    ),
                    # Original Image and Overlay
                    html.Div(
                        children=[
                            html.H2(
                                children='Overlay of Original with Segments'
                            ),
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
                                            'width': 768,
                                            'height': 566
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
                html.Hr()
            ],
            className='one row'
        ),
        html.Div(
            children=[
                # Feedback
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
                        html.P(
                            'Check out the user docs for more information:'
                        ),
                        html.A(
                            (
                                "https://github.com/lwang94"
                                "/sem_size_analysis/blob/master"
                                "/docs/user_docs.md"
                            ),
                            href=(
                                "https://github.com/lwang94"
                                "/sem_size_analysis/blob/master"
                                "/docs/user_docs.md"
                            )
                        )
                    ],
                    style={'textAlign': 'center'}
                )
            ],
            className='one row'
        )
    ])
