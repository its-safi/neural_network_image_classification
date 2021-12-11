# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 15:34:22 2021

@author: Hatricano
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.models import load_model

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

batch_size = 128

train_dir_path=os.path.join(os.getcwd(), 'input', 'images')
test_dir_path=os.path.join(os.getcwd(), 'input', 'testset')


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


def visualize_images(images,labels):
    figure, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 14))
    classes = list(train_data.class_indices.keys())
    img_no = 0
    for i in range(3):
        for j in range(3):
            img = images[img_no]
            lbl = np.argmax(labels[img_no])

            ax[i,j].imshow(img)
            ax[i,j].set_title(classes[lbl])
            ax[i,j].set_axis_off()
            img_no+=1
            

images, labels = next(train_data)
visualize_images(images,labels)



images, labels = next(valid_data)
visualize_images(images,labels)



base = MobileNetV2(input_shape=(224,224,3),include_top=False,weights='imagenet')
base.trainable = False
model = Sequential()
model.add(base)
# model.add(Flatten())
model.add(GlobalAveragePooling2D())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(101, activation='softmax'))
# opt = SGD(lr=0.001, momentum=0.9)
opt = Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss = 'categorical_crossentropy',metrics=['accuracy'])

reduce_lr = ReduceLROnPlateau(monitor = 'val_accuracy',patience = 1,verbose = 1)
early_stop = EarlyStopping(monitor = 'val_accuracy',patience = 5,verbose = 1,restore_best_weights = True)
log = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard = TensorBoard(log_dir=log,write_graph=False,update_freq=100)
chkp = ModelCheckpoint('mobilenetv2_tuned.h5',monitor='val_accuracy',verbose=1,save_best_only=True)

history = model.fit(train_data, 
                    epochs=3,
                    validation_data = valid_data,
                    callbacks=[early_stop, reduce_lr, tensorboard, chkp])


import os.path

#if os.path.isfile('model_save_versions/mobile_net.h5') is False:
model1.save('model_save_versions/mobile_net.h5')

from tensorflow.keras.models import load_model
model1 = load_model('model_save_versions/mobile_net.h5')


history = model1.fit(train_data, 
                    epochs=2,
                    validation_data = valid_data,
                    callbacks=[early_stop, reduce_lr, tensorboard, chkp])