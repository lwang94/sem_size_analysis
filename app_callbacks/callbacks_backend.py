import dash
from dash.dependencies import Input, Output, State

import requests

from . import callbacks_util as cu


def backend_callbacks(app, backend_url):

    @app.callback(
        Output('pred_json', 'children'),
        [Input('upload-image', 'contents'),
         Input('demo', 'n_clicks')]
    )
    def get_prediction(contents, n_clicks):
        """
        Gets image segmentation prediction of uploaded
        image using trained model.
        """
        ctx = dash.callback_context

        # if try the demo has been clicked, upload demo image
        if ctx.triggered[-1]['prop_id'] == 'demo.n_clicks':
            if n_clicks is not None:
                imgb64 = (
                    'data:image/jpeg;base64,'
                    + cu.upload_demo()
                )
        # otherwise, upload user image
        else:
            imgb64 = contents

        # convert from base64 to numpy array
        content_type, content_string = imgb64.split(',')
        img = cu.b64_2_numpy(content_string)

        response = requests.post(
            f'{backend_url}/api/predict',
            json={
                'contents': img.tolist(),
                'content_type': content_type
            }
        )
        return response.text

    @app.callback(
        Output('size_distr_json', 'children'),
        [Input('pred_json_copy', 'children')]
    )
    def get_size_distr(pred_json_copy):
        """
        Obtains size distribution of particles in image
        by assigning unique values to the connected
        regions of the predicted segment mask.
        """
        response = requests.post(
            f'{backend_url}/api/get_size_distr',
            json={'data_pred': pred_json_copy}
            )
        return response.text
