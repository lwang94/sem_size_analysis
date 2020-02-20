import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go


def instructions():
    """Instructions for using app"""
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