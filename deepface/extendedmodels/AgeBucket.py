import cv2
from matplotlib import pyplot as plt

from deepface.commons import functions


class AgeBucket:
    """
    Loads and uses the bucketed age model from [https://github.com/GilLevi/AgeGenderDeepLearning] to predict age bins.
    """

    AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

    def __init__(
            self,
            prototxt_url="https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/age_net_definitions/deploy.prototxt",
            caffemodel_url="https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel"
    ):
        prototxt_path = functions.download(prototxt_url, 'age_bucket.prototxt')
        caffemodel_path = functions.download(caffemodel_url, 'age_bucket.caffemodel')

        self.model = cv2.dnn.readNetFromCaffe(
            prototxt_path,
            caffemodel_path,
        )

    def predict(self, img):
        img = img

        print(img.shape)
        plt.imshow(img[:, :, ::-1])
        plt.show()

        imageBlob = cv2.dnn.blobFromImage(
            img,
            1.0,
            (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
        )

        self.model.setInput(imageBlob)
        predictions = self.model.forward()

        print(predictions)

        i = predictions[0].argmax()
        age = AgeBucket.AGE_BUCKETS[i]
        age_confidence = predictions[0][i]
        print(f'{age}: {age_confidence}')

        return age
