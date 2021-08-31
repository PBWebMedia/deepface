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

#url = 'https://drive.google.com/uc?id=1nz-WDhghGQBC4biwShQ9kYjvQMpO6smj'
"""
#google drive source downloads zip
output = home+'/.deepface/weights/race_model_single_batch.zip'
gdown.download(url, output, quiet=False)

#unzip race_model_single_batch.zip
with zipfile.ZipFile(output, 'r') as zip_ref:
	zip_ref.extractall(home+'/.deepface/weights/')
"""


def loadModel(url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/race_model_single_batch.h5'):

	model = VGGFace.baseModel()

	#--------------------------

	classes = 6
	base_model_output = Sequential()
	base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
	base_model_output = Flatten()(base_model_output)
	base_model_output = Activation('softmax')(base_model_output)

	#--------------------------

	race_model = Model(inputs=model.input, outputs=base_model_output)

	#--------------------------

	#load weights

	weights_path = functions.download(url, 'race_model_single_batch.h5')
	race_model.load_weights(weights_path)

	return race_model

	#--------------------------
