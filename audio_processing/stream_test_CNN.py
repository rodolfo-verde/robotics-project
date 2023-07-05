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
model.load_weights("audio_processing\speech_CNN_weights.h5")

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
            print(f"Prediction: {class_names[index_pred]} and {prediction[0][index_pred]*100} %")
            print(f"Word: {class_names[0]} equals a prediction of {prediction[0][0]*100} %")
            print(f"Word: {class_names[1]} equals a prediction of {prediction[0][1]*100} %")
            print(f"Word: {class_names[2]} equals a prediction of {prediction[0][2]*100} %")
            print(f"Word: {class_names[3]} equals a prediction of {prediction[0][3]*100} %")
            print(f"Word: {class_names[4]} equals a prediction of {prediction[0][4]*100} %")
            print(f"Word: {class_names[5]} equals a prediction of {prediction[0][5]*100} %")
            print(f"Word: {class_names[6]} equals a prediction of {prediction[0][6]*100} %")
            print(f"Word: {class_names[7]} equals a prediction of {prediction[0][7]*100} %")
            print(f"Word: {class_names[8]} equals a prediction of {prediction[0][8]*100} %")
            print(f"Time: {time.time()-starttime}")
            sd.play(i)
        print(time.time()-starttime)

