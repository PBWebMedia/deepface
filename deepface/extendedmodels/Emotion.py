import numpy as np

from deepface.commons import functions

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
	import keras
	from keras.models import Model, Sequential
	from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout
elif tf_version == 2:
	from tensorflow import keras
	from tensorflow.keras.models import Model, Sequential
	from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout

#url = 'https://drive.google.com/uc?id=13iUHHP3SlNg53qSuQZDdHDSDNdBP9nwy'
"""
#google drive source downloads zip
output = home+'/.deepface/weights/facial_expression_model_weights.zip'
gdown.download(url, output, quiet=False)

#unzip facial_expression_model_weights.zip
with zipfile.ZipFile(output, 'r') as zip_ref:
	zip_ref.extractall(home+'/.deepface/weights/')
"""

class Emotion:
	emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

	def __init__(self, url='https://github.com/serengil/deepface_models/releases/download/v1.0/facial_expression_model_weights.h5'):
		num_classes = 7

		model = Sequential()

		#1st convolution layer
		model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
		model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

		#2nd convolution layer
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(Conv2D(64, (3, 3), activation='relu'))
		model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

		#3rd convolution layer
		model.add(Conv2D(128, (3, 3), activation='relu'))
		model.add(Conv2D(128, (3, 3), activation='relu'))
		model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

		model.add(Flatten())

		#fully connected neural networks
		model.add(Dense(1024, activation='relu'))
		model.add(Dropout(0.2))
		model.add(Dense(1024, activation='relu'))
		model.add(Dropout(0.2))

		model.add(Dense(num_classes, activation='softmax'))

		#----------------------------

		weights_path = functions.download(url, 'facial_expression_model_weights.h5')
		model.load_weights(weights_path)

		self.model = model

	def predict(self, img):
		img = functions.reshape_face(img=img, target_size=(48, 48), grayscale=True)

		emotion_predictions = self.model.predict(img)[0, :]

		sum_of_predictions = emotion_predictions.sum()

		response = {'emotion': {}}

		for i in range(0, len(Emotion.emotion_labels)):
			emotion_label = Emotion.emotion_labels[i]
			emotion_prediction = 100 * emotion_predictions[i] / sum_of_predictions
			response['emotion'][emotion_label] = emotion_prediction

		response['dominant_emotion'] = Emotion.emotion_labels[np.argmax(emotion_predictions)]

		return response