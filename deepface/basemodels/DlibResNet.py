import numpy as np

from deepface.commons import functions

class DlibResNet:

	def __init__(self):

		#this is not a must dependency
		import dlib #19.20.0

		self.layers = [DlibMetaData()]

		#---------------------

		weight_path = functions.download(
			"http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
			"dlib_face_recognition_resnet_model_v1.dat.bz2",
			unzip=True,
		)

		model = dlib.face_recognition_model_v1(weight_path)
		self.__model = model

		#---------------------

		return None #classes must return None

	def predict(self, img_aligned):

		#functions.detectFace returns 4 dimensional images
		if len(img_aligned.shape) == 4:
			img_aligned = img_aligned[0]

		#functions.detectFace returns bgr images
		img_aligned = img_aligned[:,:,::-1] #bgr to rgb

		#deepface.detectFace returns an array in scale of [0, 1] but dlib expects in scale of [0, 255]
		if img_aligned.max() <= 1:
			img_aligned = img_aligned * 255

		img_aligned = img_aligned.astype(np.uint8)

		model = self.__model

		img_representation = model.compute_face_descriptor(img_aligned)

		img_representation = np.array(img_representation)
		img_representation = np.expand_dims(img_representation, axis = 0)

		return img_representation

class DlibMetaData:
	def __init__(self):
		self.input_shape = [[1, 150, 150, 3]]