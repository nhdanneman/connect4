#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 18:08:09 2018

@author: ndanneman
"""

### Model efforts
### Begin to learn ConvNets for one-sided games


import numpy as np


from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, Dropout





# reference model (very simple):
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py

# TODO: is this rows, cols, channels?  Or channels, rows, cols?
input_shape = (1, 6, 7)  # (1, 6, 7)  

model = Sequential()
# TODO: how many filters?  And what size kernel?
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
# TODO: remember what max pooling is, and think about how it should be applied here
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.5))
# TODO: this is a stupid linear output function for a truncated DV!
model.add(Dense(1))  

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

    



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