#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 18:08:09 2018

@author: ndanneman
"""

### Model efforts
### Begin to learn ConvNets for one-sided games


import numpy as np

import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout
import keras


batch_size=2000
epochs=130


# reference model (very simple):
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

# TODO: is this rows, cols, channels?  Or channels, rows, cols?
input_shape = (6, 7, 1)  # (1, 6, 7)  

model = Sequential()
# TODO: how many filters?  And what size kernel?
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))  # defaults to stride=(1,1)


model.add(Flatten())
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.5))
# TODO: this is a stupid linear output function for a truncated DV!
# TODO: alter the backmapping to (0,1) and use a sigmoid output function!
model.add(Dense(1))  

model.summary()

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.rmsprop(),
              metrics=['mae'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# WITHIN SAMPLE (better be good!)
y_hat = model.predict(x_train)
to_samp = np.random.choice(len(y_train), size=200, replace=False)
plt.plot(y_train[to_samp],y_hat[to_samp], 'bo')

# OUT OF SAMPLE
y_hat = model.predict(x_test)
# sample and plot some predictions of particular game states:
to_samp = np.random.choice(len(y_test), size=200, replace=False)
plt.plot(y_test[to_samp],y_hat[to_samp], 'bo')
#plt.plot(y_test,y_hat, 'bo')


#def generator_model():
#    model = Sequential()
#    model.add(Dense(input_dim=100, output_dim=1024))
#    model.add(Activation('tanh'))
#    model.add(Dense(128*7*7))
#    model.add(BatchNormalization())
#    model.add(Activation('tanh'))
#    model.add(Reshape((7, 7, 128), input_shape=(128*7*7,)))
#    model.add(UpSampling2D(size=(2, 2)))
#    model.add(Conv2D(64, (2, 14), padding='same'))
#    model.add(Activation('tanh'))
#    model.add(UpSampling2D(size=(2, 2)))
#    model.add(Conv2D(1, (2, 28), padding='same'))
#    model.add(Activation('tanh'))
#    model.add(Reshape((28,28)))
#    model.add(TimeDistributed(Dense(28, activation = 'softmax') ) )
#    #model.add(Reshape((28,28)))
#    return model
#