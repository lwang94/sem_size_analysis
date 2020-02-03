import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import base64
import numpy as np
import io
from PIL import Image

from flask import jsonify

def Add_Dash(server):
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    # app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
    dash_app = dash.Dash(server=server, external_stylesheets=external_stylesheets)
    dash_app.layout = html.Div(children=[
        html.H1(children='Welcome to My App', style={'textAlign':'center'}),
        dcc.Upload(
            id='upload-image',
            children=html.Button('Upload File'),
            style={'display':'flex', 'justify-content':'center'}),
        html.Div(id='output-image'),
        html.Div(id='hidden_json', style={'display':'none'})
    ])

    init_callbacks(dash_app)

    return dash_app.server

def init_callbacks(dash_app):
    @dash_app.callback(
        Output('output-image', 'children'),
        [Input('upload-image', 'contents')]
    )
    def upload_image(contents):
        return html.Div([
            html.Hr(),
            html.Img(src=contents, style={'display':'block', 'margin-left':'auto', 'margin-right':'auto'})
        ])

    @dash_app.callback(
        Output('hidden_json', 'children'),
        [Input('upload-image', 'contents')]
    )
    def get_img_array(contents):
        content_type, content_string = contents.split(',')
        # decode = io.BytesIO(base64.b64decode(content_string))
        # img = Image.open(decode)
        # arr = np.array(img)

        return jsonify([
            {'name':'data', 'value':content_string}
        ])

# if __name__ == '__main__':
#     app.run_server(debug=True)