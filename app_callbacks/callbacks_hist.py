from dash.dependencies import Input, Output, State

import json
import numpy as np
import plotly.graph_objs as go
import io


def hist_callbacks(app):

    @app.callback(
        Output('size_distr_graph', 'figure'),
        [Input('size_distr_json', 'children'),
         Input('binsize', 'value')]
    )
    def update_hist(size_distr_json, value):
        """Displays histogram of size distribution"""
        data = json.loads(size_distr_json)
        size_distr = np.asarray(data['size_distr_list'])
        if value is None:
            bin_size = 10
        else:
            bin_size = value

        return {
            'data': [go.Histogram(
                        x=size_distr,
                        xbins={'size': bin_size}
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
                            f'<b>Med</b> = {np.median(size_distr):.2E}<br><br>'
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

    @app.callback(
        [Output('download-link', 'href'),
         Output('download-link', 'download')],
        [Input('size_distr_json', 'children')],
        [State('upload-image', 'filename')]
    )
    def update_download_link(size_distr_json, fname):
        data = json.loads(size_distr_json)
        size_distr = np.asarray(data['size_distr_list'])

        # create .csv link from numpy array
        buff = io.StringIO()
        csv_string = np.savetxt(buff, size_distr, encoding='utf-8')
        csv_string = 'data:text/csv;charset=utf-8,' + buff.getvalue()
        return csv_string, f'{fname}_distr.csv'
