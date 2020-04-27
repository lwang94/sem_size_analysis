import dash
from dash.dependencies import Input, Output, State
from dash_canvas.utils import array_to_data_url

from skimage.transform import resize

import numpy as np
import json

from . import callbacks_util as cu


def images_callbacks(app):

    @app.callback(
        Output('pred_json_copy', 'children'),
        [Input('pred_json', 'children'),
         Input('postprocessing', 'trigger')],
        [State('pred_json_copy', 'children'),
         State('size_distr_json', 'children'),
         State('postprocessing', 'json_data')]
    )
    def postprocess_pred(
        pred_json, nclicks, pred_json_copy, size_distr_json, pp_data
    ):
        """
        Creates copy of pred_json that can be
        modified from the dash canvas
        """
        ctx = dash.callback_context

        # if creating pred_json_copy for the first time
        if ctx.triggered[-1]['prop_id'] == 'pred_json.children':
            pred = json.loads(pred_json)

            # resize the prediction to fit (576, 768) dash canvas
            ypred = np.asarray(pred['yimage_list'], dtype=np.uint8)
            ypred = 255 * resize(ypred, (576, 768), order=0)
            ypred = ypred.astype(np.uint8)
            resize_factor = pred['rf'] / 9

            return json.dumps({
                'content_type': pred['content_type'],
                'rf': resize_factor,
                'yimage_list': ypred.tolist()
            })

        # otherwise, update pred_json_copy based on dash canvas edits
        else:
            pred = json.loads(pred_json_copy)
            ypred = np.asarray(pred['yimage_list'], dtype=np.uint8)
            data = json.loads(pp_data)

            # apply the edits for each user applied stroke
            ypred = cu.apply_edits(data, ypred, size_distr_json)
            return json.dumps({
                'content_type': pred['content_type'],
                'rf': pred['rf'],
                'yimage_list': ypred.tolist()
            })

    @app.callback(
        Output('images_data', 'children'),
        [Input('size_distr_json', 'children')],
        [State('pred_json_copy', 'children')]
    )
    def create_images(size_distr_json, pred_json_copy):
        size_distr = json.loads(size_distr_json)
        pred_copy = json.loads(pred_json_copy)

        # create black and white image
        pred = 255 * np.asarray(pred_copy['yimage_list'], dtype=np.uint8)
        encoded_pred = array_to_data_url(pred)

        # create blue and gold overlay
        lookup = np.asarray([[153, 153, 0], [45, 0, 78]], dtype=np.uint8)
        colour = lookup[pred_copy['yimage_list']]
        encoded_colour = (
            pred_copy['content_type']
            + ','
            + cu.numpy_2_b64(colour)
        )

        # create labeled image
        unique = size_distr['unique_list']
        labeled = np.asarray(size_distr['labeled_list'])
        flattened_colour_arr = np.linspace(
            0,
            256 ** 3 - 1,
            num=len(unique) + 1,
            dtype=np.int32
        )

        # represent values in flattened_color_arr as three digit number with
        # base 256 to create a color array with shape
        # (num unique values including background, 3)
        colours_labeled = np.zeros((len(unique) + 1, 3), dtype=np.uint8)
        for i in range(len(colours_labeled)):
            colours_labeled[i] = np.array([
                (flattened_colour_arr[i] // (256 ** 2)) % 256,
                (flattened_colour_arr[i] // (256)) % 256,
                flattened_colour_arr[i] % 256
            ])

        # create a lookup table using colors array and convert 2D labeled array
        # into 3D rgb array
        lookup_labeled = np.zeros((len(unique) + 1, 3), dtype=np.uint8)
        lookup_labeled[np.unique(labeled - 1)] = colours_labeled
        rgb_labeled = lookup_labeled[labeled - 1]

        # convert from numpy array to base64 image
        # rgb = np.asarray(data['rgb_pred_list'], dtype=np.uint8)
        encoded_rgb = (
            size_distr['content_type']
            + ','
            + cu.numpy_2_b64(rgb_labeled)
        )
        return [encoded_pred, encoded_colour, encoded_rgb]

    @app.callback(
        [Output('yimage', 'src'),
         Output('yimage', 'style')],
        [Input('images_data', 'children'),
         Input('opacity_value', 'value')]
    )
    def display_yimage(images, op_val):
        # specify style
        style = {
            'position': 'absolute',
            'top': 0,
            'left': 69,
            'opacity': op_val,
            'height': 566,
            'width': 768
        }
        return images[1], style

    @app.callback(
        Output('postprocessing', 'image_content'),
        [Input('showitem', 'value'),
         Input('images_data', 'children')]
    )
    def image_postprocess(value, images):
        if value == 'bw':
            return images[0]
        else:
            return images[2]

    @app.callback(
        Output('postprocessing', 'lineColor'),
        [Input('colourpicker', 'value')]
    )
    def pick_color(pick):
        if pick == 'draw':
            colour = 'white'
        elif pick == 'remove':
            colour = 'red'
        else:
            colour = 'black'
        return colour

    @app.callback(
        Output('postprocessing', 'lineWidth'),
        [Input('brushwidth', 'value')]
    )
    def update_brushwidth(val):
        return val
