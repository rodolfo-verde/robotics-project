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

import numpy as np
import WaveInterface
import time
import sounddevice as sd
import sys
from os import listdir

from dataprocessor import dataprocessor
from mfcc_processor import mfcc_dataprocessor

test_again = False

if test_again:
    # load data and split into trainings and test data
    data_mfcc = np.load(f"audio_processing/Train_Data/train_test_mfcc.npy",allow_pickle=True) # load data
    data_labels = np.load(f"audio_processing/Train_Data/train_test_label.npy",allow_pickle=True) # load data

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
    model.add(Dense(9, activation="softmax"))

    model.compile(optimizer=SGD(learning_rate = 0.01), loss="categorical_crossentropy", metrics=["accuracy"]) # optimizer = rmsprop, Adam     loss = categorical_crossentropy, CTCLoss

if not test_again:

    #import weights
    model.load_weights("audio_processing/speech_CNN_weights.h5")

if test_again:
    result = model.fit(
        X_train.reshape(-1, 11, 70, 1), 
        y_train, 
        validation_data = (X_test.reshape(-1, 11, 70, 1), y_test),
        epochs=30, 
        batch_size=50)

    # evaluate model
    test_loss, test_acc = model.evaluate(X_test.reshape(-1, 11, 70, 1), y_test, verbose=2)
    print(f"Test accuracy: {test_acc}")

BLOCKLENGTH = 44100
SAMPLERATE = 44100
TARGETLVL = -30
VOICETHRESHHOLD = -40
LENGTHOFVOICEACTIVITY = 10


dp = dataprocessor(BLOCKLENGTH, TARGETLVL, VOICETHRESHHOLD, LENGTHOFVOICEACTIVITY)
mp = mfcc_dataprocessor(BLOCKLENGTH)

devices = sd.query_devices()

    #safe1 stores the input from the stream to be processed later
safe1 = np.array([], dtype="float64")


    # function used from stream to get the sound input
def callback(indata, frame_count, time_info, status):
    global safe1
    safe1 = np.append(safe1, indata)
INPUTDEVICE = 1
for i in devices:
    if i['name'] == 'default':
        INPUTDEVICE = i['index']
    
#INPUTDEVICE = 7 # 1 for jonas usb mic

stream = sd.InputStream(channels=1, samplerate=SAMPLERATE, callback=callback, device=INPUTDEVICE)

class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]


with stream:
    while True:
        while(len(safe1)<BLOCKLENGTH):
            time.sleep(0.1)
            continue
        workblock = safe1[:BLOCKLENGTH]
        safe1 = safe1[BLOCKLENGTH:]
        starttime = time.time()

        words, plots = dp.processdata(workblock)

        for i in words[0]:

            mfcc = mp.mfcc_process(i)
            prediction = model.predict(mfcc[1:].reshape(-1, 11, 70, 1))
            index_pred = np.argmax(prediction) #tf.argmax geht auch
            print(f"Prediction: {class_names[index_pred]} and {prediction}")
        
        print(time.time()-starttime)

