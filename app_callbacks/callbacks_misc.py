import dash
from dash.dependencies import Input, Output

from . import callbacks_util as cu


def misc_callbacks(app, open_txt_doc):

    @app.callback(
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
            return open_txt_doc('instructions.txt')

    @app.callback(
        Output('file_name', 'children'),
        [Input('upload-image', 'filename')]
    )
    def display_filename(filename):
        """Displays name of uploaded file"""
        return filename

    @app.callback(
        [Output('ximage', 'src'),
         Output('raw_image', 'src')],
        [Input('upload-image', 'contents'),
         Input('demo', 'n_clicks')]
    )
    def display_ximage(contents, n_clicks):
        ctx = dash.callback_context

        # if try the demo has been clicked, use the demo image
        if ctx.triggered[-1]['prop_id'] == 'demo.n_clicks':
            if n_clicks is not None:
                imgb64 = (
                    'data:image/jpeg;base64,'
                    + cu.upload_demo()
                )

        # otherwise, use the user uplaoded image
        else:
            imgb64 = contents
        return imgb64, imgb64
