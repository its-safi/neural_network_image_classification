# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 23:44:56 2021

@author: safiu
"""

import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from numpy import expand_dims

from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from glob import glob
import matplotlib.pyplot as plt

import seaborn as sns





def open_images(paths):
    '''
    Opens a batch of images, given the image path(s) as a list
    '''
    images = []
    for path in paths:
        image = load_img(path, target_size=(224,224))
        image = np.array(image)/255.0
        images.append(image)
    return np.array(images)

import os

test_dir = os.path.join(os.getcwd() , 'input\\')
test_paths = []

for label in os.listdir(test_dir):
        test_paths.append(os.path.join(test_dir,label))



# labels = os.listdir(test_dir)

img_path = os.path.join(os.getcwd() , 'input\\38795.jpg')

from tensorflow.keras.models import load_model
model1 = load_model('mobile_net.h5')

predicted_vals = []
for x in range(0,len(test_paths)-1) : 
    images = open_images([test_paths[x]])
    predicted = model1.predict(images)[0]
    predicted = np.argmax(predicted)
    print('result : ' , x, predicted)
    predicted_vals.append([x,predicted])
