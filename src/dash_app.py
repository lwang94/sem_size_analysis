import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import base64
import numpy as np
import io
from PIL import Image

import transform_data as td
import predict as pred
from pathlib import Path

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

dash_app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
# dash_app = dash.Dash(server=server, external_stylesheets=external_stylesheets)
dash_app.layout = html.Div(children=[
    html.H1(children='Welcome to My App', style={'textAlign':'center'}),
    dcc.Upload(
        id='upload-image',
        children=html.Button('Upload File'),
        style={'display':'flex', 'justify-content':'center'}),
    html.Div(id='output-image')
])


@dash_app.callback(
    Output('output-image', 'children'),
    [Input('upload-image', 'contents')]
)
def upload_image(contents):
    # convert b64 encoded string to numpy array
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded))
    img = np.array(img)

    print('Making Prediction')
    # perform data transformation
    if len(img.shape) == 2:
        img = td.make_3channel(img)
    img = td.resize(img, (192, 256))
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
    prediction = np.uint8(255 * prediction[0])
    print('Finished Prediction')

    # convert numpy array to b64 encoded string
    pred_pil = Image.fromarray(prediction)
    buff = io.BytesIO()
    pred_pil.save(buff, format='png')
    encoded_pred = content_type + ',' + base64.b64encode(buff.getvalue()).decode("utf-8")
    print(encoded_pred)

    return html.Div([
        html.Hr(),
        html.Img(src=contents, style={'display':'block', 'margin-left':'auto', 'margin-right':'auto'}, className='six columns'),
        html.Img(src=encoded_pred, style={'display':'block', 'margin-left':'auto', 'margin-right':'auto'}, className='six columns')
    ])


if __name__ == '__main__':
    dash_app.run_server(debug=True)