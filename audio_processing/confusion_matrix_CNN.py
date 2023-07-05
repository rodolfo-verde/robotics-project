from sklearn.metrics import confusion_matrix

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

model.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(11,70,1), padding="same"))
model.add(MaxPooling2D(pool_size=(5, 5), padding="same"))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(10, activation="sigmoid", kernel_regularizer=L2(0.01), bias_regularizer=L2(0.01)))
model.add(Dense(9, activation="softmax"))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]) # optimizer = rmsprop, Adam     loss = categorical_crossentropy, CTCLoss

#import weights
model.load_weights("audio_processing/speech_CNN_weights.h5")

data_test_set_name = "set_complete_test"
# predict
class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]
predict_mfcc = np.load(f"audio_processing/Train_Data/{data_test_set_name}_mfcc.npy",allow_pickle=True) # load data
predict_labels = np.load(f"audio_processing/Train_Data/{data_test_set_name}_label.npy",allow_pickle=True) # load data
index = 0
print(f"Predict shape: {predict_mfcc.shape}")
print(f"Labels shape: {predict_labels.shape}")
predict = predict_mfcc
prediction = tf.argmax(model.predict(predict.reshape(-1, 11, 70, 1)), axis =1)

print(prediction.shape)
print(np.sum(tf.argmax(predict_labels, axis=1)==prediction))

cm = confusion_matrix(prediction, tf.argmax(predict_labels, axis=1))



fig = plt.figure()
ax = fig.add_subplot()
cax = ax.matshow(cm)
plt.title('Confusion matrix of the classifier')
fig.colorbar(cax)
ax.set_xticklabels([''] + class_names)
ax.set_yticklabels([''] + class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()