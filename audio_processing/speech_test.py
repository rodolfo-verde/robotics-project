import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import L2 


# CNN
model = Sequential()

model.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(11,70,1))) 
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(10, activation="sigmoid", kernel_regularizer=L2(0.1))) 
model.add(Dropout(0.1))
model.add(Dense(9, activation="softmax"))

model.compile(optimizer=SGD(learning_rate = 0.01), loss="categorical_crossentropy", metrics=["accuracy"]) # optimizer = rmsprop, Adam     loss = categorical_crossentropy, CTCLoss

#import weights
model.load_weights("audio_processing\speech_CNN_weights.h5")


# predict
class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]
predict_mfcc = np.load(f"audio_processing\Train_Data\set_test_a1_mfcc.npy",allow_pickle=True) # load data
predict_labels = np.load(f"audio_processing\Train_Data\set_test_a1_label.npy",allow_pickle=True) # load data
index = 0
print(f"Predict shape: {predict_mfcc.shape}")
print(f"Labels shape: {predict_labels.shape}")
predict = predict_mfcc[index]
print(predict_labels[index])
#print(predict_labels[0])
prediction = model.predict(predict.reshape(-1, 11, 70, 1))
index_pred = np.argmax(prediction) #tf.argmax geht auch
index_label = np.argmax(predict_labels[index])
print(f"Prediction: {class_names[index_pred]}")
print(f"Label: {class_names[index_label]}")
