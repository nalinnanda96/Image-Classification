from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import model_from_json


# dimensions of our images.
img_width, img_height = 150, 150

# load data
x_data = np.load('x_data.npy')
#x_data = np.reshape(x_data, (125, 3, 150, 150))
y_data = np.load('y_data.npy')

#spliting data
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.05, random_state=42)
batch_size = 16
input_shape = (img_width, img_height, 3)
nb_train_samples = 118
nb_validation_samples = 800
epochs = 30

#cnn model
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='same',
                          input_shape=input_shape,
                           activation='relu'))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('sigmoid'))

lr = 0.0003
sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=epochs,
          shuffle=True
         )

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

train_datagen.fit(X_train)
model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=epochs
                    )

scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

model.save_weights('new_bicycle.h5')
model_json = model.to_json()
with open("new_bicycle.json", "w") as json_file:
    json_file.write(model_json)