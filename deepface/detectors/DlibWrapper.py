from pathlib import Path

from deepface.commons import functions

def build_model():

	import dlib #this requirement is not a must that's why imported here

	landmarks_path = functions.download(
		"http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2",
		"shape_predictor_5_face_landmarks.dat.bz2",
		unzip=True
	)

	face_detector = dlib.get_frontal_face_detector()
	sp = dlib.shape_predictor(landmarks_path)

	detector = {}
	detector["face_detector"] = face_detector
	detector["sp"] = sp
	return detector

def detect_face(detector, img, align = True):

	import dlib #this requirement is not a must that's why imported here

	resp = []

	home = str(Path.home())

	sp = detector["sp"]

	detected_face = None
	img_region = [0, 0, img.shape[0], img.shape[1]]

	face_detector = detector["face_detector"]
	detections = face_detector(img, 1)

	if len(detections) > 0:

		for idx, d in enumerate(detections):
			left = d.left(); right = d.right()
			top = d.top(); bottom = d.bottom()
			detected_face = img[top:bottom, left:right]
			img_region = [left, top, right - left, bottom - top]

			if align:
				img_shape = sp(img, detections[idx])
				detected_face = dlib.get_face_chip(img, img_shape, size = detected_face.shape[0])

			resp.append((detected_face, img_region))


	return resp
