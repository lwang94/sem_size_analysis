import transform_data as td
import predict as pred

from flask import Flask, jsonify, request
from pathlib import Path

app = Flask(__name__)
host = 'localhost'
port = '5000'

with app.app_context():
    import dash_app as da
    app = da.Add_Dash(app)

@app.route('/api/predict', methods = ['POST'])
def predict():
    img = request.get_json()
    img = img['data']  # will be a numpy array

    # perform data transformation
    if len(img.shape) == 2:
        td.make_3channel(img)
    img = td.resize(img, (192, 256))
    img = td.fastai_image(img)

    # load model
    data_path = 'C:/Users/lawre/Documents/good'
    learn = pred.load_model(
        'stage-2_bs16',
        Path(data_path) / 'train_x',
        Path(data_path) /'train_y_png',
        np.array(['background', 'particle'], dtype='<U17'),
        (192, 256),
        16
     )

    # make prediction
    prediction = pred.predict_segment(learn, img).numpy()
    prediction = prediction[0]

    # get size distribution
    size_distr = pred.get_size_distr(learn, img)

    return jsonify([
        {'name': 'Mask', 'value': prediction},
        {'name': 'Size_Distr', 'value': size_distr}
    ])

if __name__ == '__main__':
    app.run(debug=True, host=host, port=port)





