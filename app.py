"""Generates front end server"""

import dash

import app_layout as al
from app_callbacks import callbacks_backend
from app_callbacks import callbacks_misc
from app_callbacks import callbacks_hist
from app_callbacks import callbacks_images


dash_app = dash.Dash(__name__)
server = dash_app.server
backend_url = 'https://saemi-backend.herokuapp.com'

# set title
dash_app.title = 'SAEMI'

# dashboard layout
dash_app.layout = al.app_layout()

# dashboard callbacks
callbacks_backend.backend_callbacks(
    dash_app,
    backend_url
)
callbacks_misc.misc_callbacks(
    dash_app,
    al.open_txt_doc
)
callbacks_hist.hist_callbacks(dash_app)
callbacks_images.images_callbacks(dash_app)

if __name__ == '__main__':
    dash_app.run_server(debug=True)
