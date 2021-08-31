from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Input, Add, MaxPooling2D, Flatten, Dense, Dropout

from deepface.commons import functions

#-------------------------------------

#url = 'https://drive.google.com/uc?id=1uRLtBCTQQAvHJ_KVrdbRJiCKxU8m5q2J'

def loadModel(url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/deepid_keras_weights.h5'):

	myInput = Input(shape=(55, 47, 3))

	x = Conv2D(20, (4, 4), name='Conv1', activation='relu', input_shape=(55, 47, 3))(myInput)
	x = MaxPooling2D(pool_size=2, strides=2, name='Pool1')(x)
	x = Dropout(rate=0.99, name='D1')(x)

	x = Conv2D(40, (3, 3), name='Conv2', activation='relu')(x)
	x = MaxPooling2D(pool_size=2, strides=2, name='Pool2')(x)
	x = Dropout(rate=0.99, name='D2')(x)

	x = Conv2D(60, (3, 3), name='Conv3', activation='relu')(x)
	x = MaxPooling2D(pool_size=2, strides=2, name='Pool3')(x)
	x = Dropout(rate=0.99, name='D3')(x)

	x1 = Flatten()(x)
	fc11 = Dense(160, name = 'fc11')(x1)

	x2 = Conv2D(80, (2, 2), name='Conv4', activation='relu')(x)
	x2 = Flatten()(x2)
	fc12 = Dense(160, name = 'fc12')(x2)

	y = Add()([fc11, fc12])
	y = Activation('relu', name = 'deepid')(y)

	model = Model(inputs=[myInput], outputs=y)

	#---------------------------------

	weights_path = functions.download(url, 'deepid_keras_weights.h5')
	model.load_weights(weights_path)

	return model
