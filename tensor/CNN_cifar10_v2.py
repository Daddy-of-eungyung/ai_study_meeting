import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import time

start = time.time()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

gen = ImageDataGenerator(rotation_range=20, shear_range=0.2, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)

augment_ratio = 1.5
augment_size = int(augment_ratio * x_train.shape[0])

randidx = np.random.randint(x_train.shape[0], size = augment_size)
#print(randidx)

x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

x_augmented, y_augmented = gen.flow(x_augmented, y_augmented,
                                    batch_size=augment_size, shuffle=False).next()
x_train = np.concatenate( (x_train, x_augmented))
y_train = np.concatenate( (y_train, y_augmented))

s= np.arange(x_train.shape[0])
np.random.shuffle(s)
x_train = x_train[s]
y_train = y_train[s]

cnn=Sequential()

cnn.add(Conv2D(32, (3,3), activation ='relu', padding='same', input_shape=(32, 32, 3)))
cnn.add(Conv2D(32, (3,3), activation ='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(64, (3,3), activation ='relu', padding='same'))
cnn.add(Conv2D(64, (3,3), activation ='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

cnn.add(Conv2D(128, (3,3), activation ='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(128, (3,3), activation ='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Conv2D(256, (3,3), activation ='relu', padding='same'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())

cnn.add(Dense(128, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation='softmax'))

cnn.compile(loss = 'sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
hist = cnn.fit(x_train, y_train, batch_size = 256, epochs = 30, validation_data = (x_test, y_test))

cnn.evaluate(x_test, y_test)

import matplotlib.pyplot as plt

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Accuracy Trend')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.grid()
plt.show()

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss Trend')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')

plt.grid()
plt.show()

print("\ntime :", time.time() - start)
