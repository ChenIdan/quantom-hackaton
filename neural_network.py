#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 08:41:54 2019

@author: oroxenberg
"""

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reuters = keras.datasets.



x_train = np.random.rand(10000,5)




row_sums = x_train.sum(axis=1)
x_train = x_train / row_sums[:, np.newaxis]


y_train= np.random.randint(4, size = 10000)

x_train = x_train * y_train.reshape(-1, 1)

x_train = 0.3*(x_train)




x_test = np.random.rand(100,5)




row_sums = x_test.sum(axis=1)
x_test = x_test / row_sums[:, np.newaxis]


y_test= np.random.randint(4, size = 100)

x_test = x_test * y_test.reshape(-1, 1)

x_test =  0.3*(x_test)


#train_images = train_images / 255.0

#test_images = test_images / 255.0




model = keras.Sequential()
#model.add(keras.layers.Flatten(input_shape=(28, 28)))
#model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.InputLayer(input_shape=(5,)))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation=tf.nn.softmax))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation=tf.nn.softmax))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])



model.fit(x_train, y_train, epochs=40)

test_loss, test_acc = model.evaluate(x_test,y_test)

print('Test accuracy:', test_acc)

#save the model
model.save('my_model.h5')

#new_model = tf.contrib.saved_model.load_keras_model('/Users/oroxenberg/Desktop/universtiy/lab/includ_weight_project/my_model.h5'))