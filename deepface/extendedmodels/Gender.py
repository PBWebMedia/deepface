import numpy as np

from deepface.basemodels import VGGFace

from deepface.commons import functions

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	from keras.models import Model, Sequential
	from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
	from tensorflow.keras.models import Model, Sequential
	from tensorflow.keras.layers import Convolution2D, Flatten, Activation

#url = 'https://drive.google.com/uc?id=1wUXRVlbsni2FN9-jkS_f4UTUrm1bRLyk'


class Gender:
	def __init__(self, url='https://github.com/serengil/deepface_models/releases/download/v1.0/gender_model_weights.h5'):
		model = VGGFace.baseModel()

		#--------------------------

		classes = 2
		base_model_output = Sequential()
		base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
		base_model_output = Flatten()(base_model_output)
		base_model_output = Activation('softmax')(base_model_output)

		#--------------------------

		gender_model = Model(inputs=model.input, outputs=base_model_output)

		#--------------------------

		#load weights

		weights_path = functions.download(url, 'gender_model_weights.h5')
		gender_model.load_weights(weights_path)

		self.model = gender_model

	def predict(self, img):
		img = functions.reshape_face(img=img, target_size=(224, 224), grayscale=False)

		gender_prediction = self.model.predict(img)[0, :]

		gender = 'Unknown'
		if np.argmax(gender_prediction) == 0:
			gender = 'Woman'
		elif np.argmax(gender_prediction) == 1:
			gender = 'Man'

		return {'gender': gender}
