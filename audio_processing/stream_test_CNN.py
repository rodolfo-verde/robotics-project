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

#import weights
model.load_weights("audio_processing/speech_CNN_weights.h5")


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
    print(i)
    if i['name'] == 'default':
        print("HIT")
        INPUTDEVICE = i['index']
    
INPUTDEVICE = 7 # 1 for jonas usb mic

stream = sd.InputStream(channels=1, samplerate=SAMPLERATE, callback=callback, device=INPUTDEVICE)

class_names = ["a", "b", "c", "1", "2", "3", "rex", "stopp", "other"]


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
            print(f"Prediction: {class_names[index_pred]}")
        
        print(time.time()-starttime)

