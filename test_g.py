import numpy as np
import cv2
import keras

Generator = keras.models.load_model("Generator.h5")
bruit = np.random.normal(0, 1, (1, 100))
image = Generator.predict(bruit)
image = image.reshape(16, 16, 3)
image = image * 127.5 + 127.5
image = image.astype("uint8")
nomImage = "Image Cycle N°1.jpg"
cv2.imwrite("doss/" + nomImage, image)

