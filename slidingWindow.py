from helper import pyramid
from helper import sliding_window
import argparse
import time
import cv2
from keras.models import model_from_json
import numpy as np
import math

# load the image and define the window width and height
image = cv2.imread('catTest1.jpg')
(winW, winH) = (64, 128)

# load json and create model
json_file = open('cats.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("cats.h5")

# loop over the image pyramid
for i,resized in enumerate(pyramid(image, scale=1.2)):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        clone = cv2.resize(window, (150,150))
        im = []
        im.append(clone)
        im=np.array(im)
        predictions = model.predict(im)
        print(predictions)
        clone = resized.copy()
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(0)

        '''mult = int(i * math.sqrt(1.2))
        x_new = x*mult
        y_new = x*mult
        winW_new = winW*mult
        winH_new=winH*mult
        if predictions[0][0] >= 0.7:
            cv2.rectangle(image, (x_new, y_new), (x_new + winW_new, y_new + winH_new), (0, 255, 0), 2)

        elif predictions[0][1] >= 0.7:
            cv2.rectangle(image, (x_new, y_new), (x_new + winW_new, y_new + winH_new), (255, 0, 0), 2)

        #elif predictions[0][2] >= 0.7:
         #   cv2.rectangle(image, (x_new, y_new), (x_new + winW_new, y_new + winH_new), (0, 0, 255), 2)

        clone = resized.copy()
        if predictions[0][0] >= 0.5:
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(0)

        elif predictions[0][1] >= 0.5:
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(0)

        elif predictions[0][2] >= 0.5:
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 255), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(0)

cv2.imshow("Window", image)
cv2.waitKey(0)'''