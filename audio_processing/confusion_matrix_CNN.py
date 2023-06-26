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

test_again = False

if test_again:
    data_set_name = "data_set_roman_complete"
    # load data and split into trainings and test data
    data_mfcc = np.load(f"audio_processing/Train_Data/{data_set_name}_mfcc.npy",allow_pickle=True) # load data
    data_labels = np.load(f"audio_processing/Train_Data/{data_set_name}_label.npy",allow_pickle=True) # load data

    print(f"Data shape: {data_mfcc.shape}")
    print(f"Labels shape: {data_labels.shape}")

    split_mfcc = int(len(data_mfcc[:,10,69])*0.8) # 80% trainings data, 20% test data
    split_labels = int(len(data_labels[:,8])*0.8) # 80% trainings labels, 20% test labels
    X_train = data_mfcc[:split_mfcc] # load mfccs of trainings data, 80% of data
    X_test = data_mfcc[split_mfcc:]# load test mfcc data, 20% of data
    y_train = data_labels[:split_labels] # load train labels, 80% of labels
    y_test = data_labels[split_labels:] # load test labels, 20% of labels

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

if test_again:
    model = Sequential()
    model.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(11,70,1))) 
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dropout(0.02))
    model.add(Dense(10, activation="sigmoid"))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(9, activation="softmax"))

    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]) # optimizer = rmsprop, Adam         loss = categorical_crossentropy, CTCLoss


if not test_again:
    # CNN
    model = Sequential()

    model.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(11,70,1))) 
    model.add(Dropout(0.1))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(10, activation="sigmoid", kernel_regularizer=L2(0.1)))
    model.add(Dropout(0.1))
    #model.add(BatchNormalization())
    #model.add(Flatten())
    #model.add(Dropout(0.1))
    model.add(Dense(9, activation="softmax"))

    model.compile(optimizer=SGD(learning_rate = 0.001), loss="categorical_crossentropy", metrics=["accuracy"]) # optimizer = rmsprop, Adam     loss = categorical_crossentropy, CTCLoss


if not test_again:

    #import weights
    model.load_weights("audio_processing/speech_CNN_weights.h5")

if test_again:
    result = model.fit(
        X_train.reshape(-1, 11, 70, 1), 
        y_train, 
        validation_data = (X_test.reshape(-1, 11, 70, 1), y_test),
        epochs=30, 
        batch_size=10)

    # evaluate model
    test_loss, test_acc = model.evaluate(X_test.reshape(-1, 11, 70, 1), y_test, verbose=2)
    print(f"Test accuracy: {test_acc}")


data_test_set_name = "set_big_test_0"
# predict
class_names = ["a", "b", "c", "1", "2", "3", "rex", "stopp", "other"]
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