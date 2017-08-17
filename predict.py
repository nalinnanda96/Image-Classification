from keras.models import model_from_json
import cv2
import numpy as np

json_file = open('new_bicycle.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("new_bicycle.h5")

im = cv2.imread('s0.jpg')
im= cv2.resize(im, (150,150))
y =[]
y.append(im)
y=np.array(y)
predictions = model.predict(y)
print (predictions)
