import os
import cv2
import numpy as np
from tensorflow import keras

"""THIS MODEL WORKS WITH keras==2.4.3 tensorflow==2.3.1"""

model = keras.models.load_model('tumor_model3')


def detect_tumor(img_to_process: np.array) -> str:
    classes = {
        1: "Tumor",
        0: "NoTumor",
    }

    img = cv2.resize(img_to_process, (224, 224))
    res = model.predict(np.array([img]))

    class_num = np.argmax(res)
    return classes[class_num]


img = cv2.imread(r"images\Y1.jpg")
det = detect_tumor(img)

print(det)
img = cv2.resize(img, (480, 480))
if det == "Tumor":
    cv2.putText(img, "Tumor detected", (20, 40), 2, cv2.FONT_HERSHEY_PLAIN, (0, 0, 200), 2)
else:
    cv2.putText(img, "No tumor detected", (20, 40), 2, cv2.FONT_HERSHEY_PLAIN, (0, 200, 0), 2)
cv2.imshow("Res", img)
cv2.waitKey(0)