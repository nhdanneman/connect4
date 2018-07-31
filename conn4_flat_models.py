#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  31 18:08:09 2018

@author: ndanneman
"""

### Model efforts
### Begin to learn FLAT NETS for one-sided games


import numpy as np

import matplotlib.pyplot as plt


from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout
import keras

#%%

batch_size=1000
epochs=350


# reference model (very simple):
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

# TODO: is this rows, cols, channels?  Or channels, rows, cols?
input_shape = (42,)  # (1, 6, 7)  

#%%

model = Sequential()
# TODO: how many filters?  And what size kernel?
model.add(Dense(35, activation='relu', input_shape=input_shape))
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))
# TODO: this is a stupid linear output function for a truncated DV!
# TODO: alter the backmapping to (0,1) and use a sigmoid output function!
model.add(Dense(1))  

model.summary()

#%%

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.rmsprop(),
              metrics=['mae'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

#%%

# WITHIN SAMPLE (better be good!)
y_hat = model.predict(x_train)
to_samp = np.random.choice(len(y_train), size=200, replace=False)
plt.plot(y_train[to_samp],y_hat[to_samp], 'bo')
plt.hlines(y=0, xmin=-1, xmax=1)

#%%
# OUT OF SAMPLE
y_hat = model.predict(x_test)
# sample and plot some predictions of particular game states:
to_samp = np.random.choice(len(y_test), size=200, replace=False)
plt.plot(y_test[to_samp],y_hat[to_samp], 'bo')
plt.hlines(y=0, xmin=-1, xmax=1)

#plt.plot(y_test,y_hat, 'bo')

#%%

