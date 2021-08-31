import cv2
import numpy as np

import onnxruntime as ort

from deepface.commons import functions


class AgeILSVRC16:
    def __init__(self, url='https://github.com/onnx/models/raw/master/vision/body_analysis/age_gender/models/vgg_ilsvrc_16_age_chalearn_iccv2015.onnx'):
        model_path = functions.download(url, 'age_ilsvrc_16.onnx')
        self.model = ort.InferenceSession(model_path)

    def predict(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.transpose(img, [2, 0, 1])
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32)

        input_name = self.model.get_inputs()[0].name
        ages = self.model.run(None, {input_name: img})
        age = round(sum(ages[0][0] * list(range(0, 101))), 1)

        return {'age_ilsvrc16': age}
