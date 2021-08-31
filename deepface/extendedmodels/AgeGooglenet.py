import cv2
import numpy as np

import onnxruntime as ort

from deepface.commons import functions


class AgeGooglenet:
    AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    def __init__(self, url='https://github.com/onnx/models/raw/master/vision/body_analysis/age_gender/models/age_googlenet.onnx'):
        model_path = functions.download(url, 'age_googlenet.onnx')
        self.model = ort.InferenceSession(model_path)

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img_mean = np.array([104, 117, 123])
        img = img - img_mean
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        input_name = self.model.get_inputs()[0].name
        ages = self.model.run(None, {input_name: img})
        age = AgeGooglenet.AGE_BUCKETS[ages[0].argmax()]

        return {'age_googlenet': age}
