import re
import base64

import numpy as np

from PIL import Image
from io import BytesIO

import os
import sys

# Flask
from flask import Flask, request, render_template, Response, jsonify

# TensorFlow and tf.keras
#import tensorflow as tf
#from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def base64_to_pil(img_base64):

    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    return pil_image


def np_to_base64(img_np):

    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")


server = Flask(__name__)

MODEL_PATH = 'models/ad_model.h5'
model = load_model(MODEL_PATH)

def model_predict(img, model):
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='tf')
    
    preds = model.predict(x)
    
    return preds


@server.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@server.route('/pred', methods=['GET', 'POST'])
def pred():
    if request.method == 'POST':
        
        img = base64_to_pil(request.json)
                
        img.save("uploads\image.jpg")
        
        img_path = os.path.join(os.path.dirname(__file__),'uploads\image.jpg')
        
        os.path.isfile(img_path)
        
        img = image.load_img(img_path, target_size=(64,64))

        preds = model_predict(img, model)
        
        result = preds[0,0]
        
        print(result)
        
        if result >0.5:
            return jsonify(result="Хвороба Альцгеймера")
        else:
            return jsonify(result="Норма")

    return None


if __name__ == '__main__':
    server.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))