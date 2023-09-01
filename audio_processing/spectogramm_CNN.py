import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D
from keras.optimizers import SGD
from keras.regularizers import L2 
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from keras import layers
from keras import models
from IPython import display
import numpy as np
import WaveInterface
import time
import sounddevice as sd
import sys
from os import listdir

from dataprocessor import dataprocessor
from mfcc_processor import mfcc_dataprocessor
from Delta_MFCC import EvaluateDeltaMFCC

#load data and split into trainings and test data
spectrogram = np.load("audio_processing\Train_Data\set_all_spectrogram_combined.npy",allow_pickle=True) # load data
labels = np.load("audio_processing\Train_Data\set_all_label_combined.npy",allow_pickle=True) # load data

# split data into trainings and test data, 80% trainings data, 20% test data
train_spectrogram = spectrogram[:int(0.8*spectrogram.shape[0])]
train_labels = labels[:int(0.8*labels.shape[0])]
test_spectrogram = spectrogram[int(0.8*spectrogram.shape[0]):]
test_labels = labels[int(0.8*labels.shape[0]):]

# shapes of data
print(f"Data shape of train spectrogram: {train_spectrogram.shape}")
print(f"Data shape of train labels: {train_labels.shape}")
print(f"Data shape of test spectrogram: {test_spectrogram.shape}")
print(f"Data shape of test labels: {test_labels.shape}")

# CNN
model = Sequential()

model.add(Conv2D(10,(3,3),padding='same',input_shape=(252,129,1), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(10,(3,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(4,2)))   # shape = (63, 65, 10)
model.add(Dropout(0.2))

model.add(Conv2D(20,(3,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(20,(3,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(4,2)))   # shape = (15, 32, 20)
model.add(Dropout(0.2))

model.add(Conv2D(40,(3,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(40,(3,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))   # shape = (7, 16, 40)
model.add(Dropout(0.2))

model.add(Conv2D(80,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(80,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(4,2)))   # shape = (1, 8, 80)      
model.add(Dropout(0.2))

model.add(Conv2D(160,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(160,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(1,2)))   # shape = (1, 4, 160)         
model.add(Dropout(0.2))

model.add(Conv2D(320,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(230,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(1,2)))   # shape = (1, 2, 320)
model.add(Dropout(0.2))

model.add(Conv2D(640,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(640,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(1,2)))   # shape = (1, 1, 640)
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(10, activation="sigmoid", kernel_regularizer=L2(0.01), bias_regularizer=L2(0.01)))
model.add(Dense(9, activation="softmax"))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]) # optimizer = rmsprop, Adam     loss = categorical_crossentropy, CTCLoss



result = model.fit(
    train_spectrogram.reshape(-1, 252, 129, 1),
    train_labels,
    validation_data = (test_spectrogram.reshape(-1, 252, 129, 1), test_labels),
    epochs=30, # 60
    batch_size=50) # 100


model.summary()

# evaluate model
test_loss, test_acc = model.evaluate(test_spectrogram.reshape(-1, 252, 129, 1), test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")

# plot accuracy and loss
plt.figure(1)
plt.subplot(121)
plt.plot(result.history["accuracy"])
plt.plot(result.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.subplot(122)
plt.plot(result.history["loss"])
plt.plot(result.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()


model.save("audio_processing//CNN_Models//Spectrogram_CNN_combined_model.h5", include_optimizer=True)
model.save_weights("audio_processing//CNN_Models//Spectrogram_CNN_combined_weights.h5")
