from speech_CNN import model
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

#import model
model = keras.models.load_model("speech_CNN_model.h5")
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]) # optimizer = rmsprop, Adam         loss = categorical_crossentropy, CTCLoss


# predict
class_names = ["a", "b", "c", "1", "2", "3", "rex", "stopp", "other"]
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
