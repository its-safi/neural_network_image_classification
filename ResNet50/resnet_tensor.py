# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 18:21:56 2021

@author: Hatricano
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        shear_range=0.3,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.25,
)
valid_datagen = ImageDataGenerator(
        rescale=1./255,
)

batch_size = 32

test_dir_path=os.path.join(os.getcwd(), 'input', 'testset')
train_dir_path=os.path.join(os.getcwd(), 'input', 'images')


train_data = train_datagen.flow_from_directory(
    train_dir_path,
    batch_size=batch_size,
    target_size=(224, 224),
    shuffle=True,
)


valid_data = valid_datagen.flow_from_directory(
    test_dir_path,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=False,
)


# ResNet_V2_50 = 'https://tfhub.dev/google/imagenet/resnet_v2_50/classification/5'
# #Efficientnet_b0 = "https://tfhub.dev/google/efficientnet/b0/classification/1"

# import tensorflow_hub as hub



# model_ResNet = tf.keras.Sequential([
#     hub.KerasLayer(ResNet_V2_50, trainable = False, input_shape = (224,224,3), name = 'Resnet_V2_50'),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(10, activation = 'softmax', name = 'Output_layer')
# ])

# model_ResNet.compile(
#     optimizer = tf.keras.optimizers.Adam(),
#     loss = tf.keras.losses.CategoricalCrossentropy(),
#     metrics = ['accuracy']
# )

# model_ResNet.summary()
# tf.keras.utils.plot_model(model_ResNet)


# resnet_model = model_ResNet.fit(train_data, epochs = 10, validation_data = valid_data, verbose = 1) #5

IMAGE_SIZE = [224, 224]

from tensorflow.keras.applications.resnet_v2 import ResNet50V2
ResNet_conv = ResNet50V2(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
ResNet_conv.summary()


for layer in ResNet_conv.layers[-7:]:
    layer.trainable = False
   


from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
resnet_feedforword = Flatten()(ResNet_conv.output)
prediction = Dense(101, activation='softmax')(resnet_feedforword)

from tensorflow.keras.models import Sequential,Model
model_ResNet = Model(inputs= ResNet_conv.input, outputs=prediction)
model_ResNet.summary()

model_ResNet.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['categorical_accuracy']
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
model1_es = EarlyStopping(monitor = 'loss', min_delta = 1e-11, patience = 12, verbose = 1)
model1_rlr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 6, verbose = 1)
model1_mcp = ModelCheckpoint(filepath = r'temp\model1_weights.h5', monitor = 'categorical_accuracy',save_best_only = True, verbose = 1)

model_ResNet_fit_101 = model_ResNet.fit(
  train_data,
  validation_data=valid_data,
  epochs=30,
  steps_per_epoch=75750//32,
  validation_steps=25250//32,
  callbacks=[model1_es, model1_rlr, model1_mcp]
)




model_ResNet.save('model_ResNet_fit_101.h5')
