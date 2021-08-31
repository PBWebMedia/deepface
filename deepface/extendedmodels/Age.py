from deepface.basemodels import VGGFace
import numpy as np

from deepface.commons import functions

import tensorflow as tf

tf_version = int(tf.__version__.split(".")[0])

if tf_version == 1:
    import keras
    from keras.models import Model, Sequential
    from keras.layers import Convolution2D, Flatten, Activation
elif tf_version == 2:
    from tensorflow import keras
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Convolution2D, Flatten, Activation


# url = 'https://drive.google.com/uc?id=1YCox_4kJ-BYeXq27uUbasu--yz28zUMV'

class Age:
    def __init__(self, url='https://github.com/serengil/deepface_models/releases/download/v1.0/age_model_weights.h5'):
        model = VGGFace.baseModel()

        # --------------------------

        classes = 101
        base_model_output = Sequential()
        base_model_output = Convolution2D(classes, (1, 1), name='predictions')(model.layers[-4].output)
        base_model_output = Flatten()(base_model_output)
        base_model_output = Activation('softmax')(base_model_output)

        # --------------------------

        age_model = Model(inputs=model.input, outputs=base_model_output)

        # --------------------------

        # load weights

        weights_path = functions.download(url, 'age_model_weights.h5')
        age_model.load_weights(weights_path)

        self.model = age_model

    def predict(self, img):
        img = functions.reshape_face(img=img, target_size=(224, 224), grayscale=False)

        age_predictions = self.model.predict(img)[0, :]
        output_indexes = np.array([i for i in range(0, 101)])
        apparent_age = np.sum(age_predictions * output_indexes)
        return int(apparent_age)  #int cast is for the exception - object of type 'float32' is not JSON serializable
