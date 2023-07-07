#!/usr/bin/env python
# coding: utf-8

# In[75]:


import keras 
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input
from keras.regularizers import l2
from keras.models import Model


# In[89]:


def AlexNet(input_shape=(224, 224, 3), full_shape=None, n_classes=1000):
    #inputs = Input(shape=input_shape)
    inputs = Input(tensor=tf.ones(shape=full_shape, dtype='float32'))

    x = Conv2D(96, kernel_size=(11,11), strides=(4,4), activation='relu')(inputs)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2,2))(x)

    x = Conv2D(256, kernel_size=(5,5), strides=(1,1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

    x = Conv2D(384, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

    x = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)

    x = Flatten()(x)
    x = Dense(4096, input_shape=(input_shape[0]*input_shape[1]*3,), activation='relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='softmax')(x)
    
    model = Model(inputs, x, name='alexnet')
    
    return model


# In[ ]:




