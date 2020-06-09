import base64
import numpy as np
import io
from PIL import Image
import keras
from keras import backend as K
from keras.models import Sequential,model_from_json
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from tensorflow.python.keras.backend import set_session

from flask import jsonify
import os
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request
import tensorflow as tf

app = Flask(__name__)
keras.backend.clear_session()



labels = ['Tüberküloz', 'Normal' , 'Zatürre']
labels = np.asarray(labels)
print(labels.shape)
print(labels)
num_classes = len(labels)



def get_model():
		global model
		json_file = open('model_4.json','r')
		loaded_model_json = json_file.read()
		json_file.close()
		model =  model_from_json(loaded_model_json)
		model.load_weights('model_4.h5')
		print("Model yüklendi!")
		model.summary()

			
def preprocess_image(image,target_size):
		if image.mode != "RGB":
			image = image.convert("RGB")
		image = image.resize(target_size)
		image= img_to_array(image)
		image = np.expand_dims(image,axis=0)

		print(image.shape)
		
		return image


def get_commonname(idx):
		sciname = labels[idx]
		print(sciname)
		return(sciname)

global graph


sess = tf.Session()

graph = tf.get_default_graph()

@app.route("/predict", methods = ["POST"])
def predict():
		message = request.get_json(force=True)
		encoded = message['image']
		decoded = base64.b64decode(encoded)
		image= Image.open(io.BytesIO(decoded))
		processed_image = preprocess_image(image,target_size=(100,100))
		prediction = session(processed_image)
		print(prediction.shape)
		sonuc = get_commonname(prediction)
		str1 = ''.join(sonuc)

		response = {
			'prediction' : str1
		}
		print(response);
		return jsonify(response)
def session(processed_image):
	with graph.as_default():
		get_model()	

		y= model.predict_classes(processed_image)
	return y