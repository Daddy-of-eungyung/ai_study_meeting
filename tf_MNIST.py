import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import time

start = time.time()

(x_train, t_train), (x_test, t_test) = mnist.load_data()
x_train = ( x_train - 0.0) / (255.0 - 0.0)
x_test = ( x_test - 0.0) /(255.0 - 0.0)

t_train = tf.keras.utils.to_categorical(t_train, num_classes=10)
t_test = tf.keras.utils.to_categorical(t_test, num_classes=10)

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape =(28, 28)))
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

hist = model.fit(x_train, t_train, epochs = 30, validation_split= 0.3)

model.evaluate(x_test, t_test)

plt.title('Loss')
plt.grid()
plt.plot(hist.history['loss'], label='train loss')
plt.plot(hist.history['val_loss'], label='validation loss')
plt.legend(loc='best')
plt.show()


from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(6, 6))
predicted_value  = model.predict(x_test)

cm = confusion_matrix(np.argmax(t_test, axis = -1), np.argmax(predicted_value, axis = -1))
sns.heatmap(cm, annot= True, fmt ='d')
plt.show()

print(cm)
print('\n')

for i in range(10):
  print(('label = %d\t(%d/%d)\taccuracy = %.3f')%
        (i, np.max(cm[i]), np.sum(cm[i]),
         np.max(cm[i])/np.sum(cm[i])))
  
print("time :", time.time() - start)
