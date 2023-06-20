import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
#from skimage import io
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import pickle
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()

# You can also use pretrained model from Keras
# Check https://keras.io/applications/

model =tf.keras.models.load_model('models/ad_model3++.h5',compile=False)
#model = pickle.load(open('cnn_model.pkl','rb'))

def model_predict(img_path, model):
    img = image.load_img(img_path, grayscale=False, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        result = preds[0,0]

        if result >0.5:
            return 'AD'
        else:
            return 'NORMAL'
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    app.run()

