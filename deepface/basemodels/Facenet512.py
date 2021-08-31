from deepface.basemodels import Facenet
from deepface.commons import functions

def loadModel(url = 'https://github.com/serengil/deepface_models/releases/download/v1.0/facenet512_weights.h5'):

    model = Facenet.InceptionResNetV2(dimension = 512)

    #-------------------------

    weights_path = functions.download(url, 'facenet512_weights.h5')
    model.load_weights(weights_path)

    #-------------------------

    return model
