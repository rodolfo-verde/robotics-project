import os
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
from keras.models import load_model
import time
import sounddevice as sd

from audio_processing.dataprocessor import dataprocessor
from audio_processing.mfcc_processor import mfcc_dataprocessor

from robot_control.TickTackToe import TickTackToe

from audio_processing.word_logic import WordLogic

from scipy.io import wavfile

import threading
# This will be the main class for the Speech Recognition + TickTackToe game
# It will import the CNN model and the TickTackToe game and then run the game
# The CNN model will be used to recognize the words spoken by the user
# The TickTackToe game will be used to play the game with the user

# The main script will be the one that will run the game
# It will import the CNN model and the TickTackToe game and then run the game
# The CNN model will be used to recognize the words spoken by the user
# The TickTackToe game will be used to play the game with the user

# CNN
# CNN

model = Sequential()
a = 100

#model.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(11,70,1), padding="same"))
model.add(Conv2D(a,(3,3),padding='same',input_shape=(11,70,1), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(a,(3,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))   # shape = (5, 35, 10)
model.add(Dropout(0.2))

model.add(Conv2D(2*a,(3,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(2*a,(3,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))   # shape = (2, 17, 20)  
model.add(Dropout(0.2))

model.add(Conv2D(4*a,(3,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(4*a,(3,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))   # shape = (1, 8, 40)
model.add(Dropout(0.2))

model.add(Conv2D(8*a,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(8*a,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(1,2)))   # shape = (1, 4, 80)
model.add(Dropout(0.2))

model.add(Conv2D(16*a,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(16*a,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(1,2)))   # shape = (1, 2, 160)
model.add(Dropout(0.2))

model.add(Conv2D(32*a,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(32*a,(1,3),padding='same', activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(1,2)))   # shape = (1, 1, 320)
model.add(Dropout(0.2))

model.add(Flatten())

model.add(Dense(a, activation="sigmoid", kernel_regularizer=L2(0.01), bias_regularizer=L2(0.01)))
model.add(Dense(9, activation="softmax"))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]) # optimizer = rmsprop, Adam     loss = categorical_crossentropy, CTCLoss

# import big speech model
model.load_weights("audio_processing//CNN_Models//CNN_More_100_weights.h5")

#safe1 stores the input from the stream to be processed later
        
safe1 = np.array([], dtype="float64")


# This is the Network class
# It will be used to run the game

class Network:
    def __init__(self):
        # this is the constructor of the class
        # it will initialize the CNN model and the TickTackToe game
        self.word_logic = WordLogic()
        #self.tick_tack_toe = TickTackToe()

    
    def pre_process(self):
        # Pre Variables
        BLOCKLENGTH = 44100
        SAMPLERATE = 44100
        TARGETLVL = -30
        VOICETHRESHHOLD = -40
        LENGTHOFVOICEACTIVITY = 10


        dp = dataprocessor(BLOCKLENGTH, TARGETLVL, VOICETHRESHHOLD, LENGTHOFVOICEACTIVITY)
        mp = mfcc_dataprocessor(BLOCKLENGTH)

        devices = sd.query_devices()

        

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

        workblocklength = 32500
        mfcc = np.zeros((11, 35))
        return stream, safe1, workblocklength, mfcc, dp, mp, class_names
    def prediction(self, to_process, wordlogic, class_names, tickTackToe): 
        # prediction
        prediction = model.predict(to_process.reshape(-1, 11, 70, 1))

        index_pred = np.argmax(prediction) #tf.argmax geht auch

        print(f"Prediction             : {class_names[index_pred]} and {prediction[0][index_pred]*100} %")
                
        wordlogic.command(class_names[index_pred])
        if wordlogic.get_combination() not in ["", None]:
            print(f"Das Wort ist = {wordlogic.get_combination()}")
            if tickTackToe.command(wordlogic.get_combination()) == 1:
                print("Game Over")
                tickTackToe.reset()
                            
            wordlogic.reset_combination()



    
    def main_loop(self, stream, workblocklength, mfcc, dp, mp, class_names):
        with stream:
            wordlogic = WordLogic()
            tickTackToe = TickTackToe()
            while True:
                global safe1
                while(len(safe1)<workblocklength):
                    time.sleep(0.01)
                    continue
                workblock = safe1[:workblocklength]
                safe1 = safe1[workblocklength:]
                #ticktacktoe = TickTackToe()
                words, plots = dp.processdata(workblock)

                for i in words[0]:

                    mfcc = mp.mfcc_process(i)[1:]

                    to_process = mfcc
                    threading.Thread(target = self.prediction, args = (to_process, wordlogic, class_names, tickTackToe)).start()

                    # prediction = model.predict(to_process.reshape(-1, 11, 70, 1))

                    # index_pred = np.argmax(prediction) #tf.argmax geht auch

                    # print(f"Prediction             : {class_names[index_pred]} and {prediction[0][index_pred]*100} %")
                
                    # wordlogic.command(class_names[index_pred])
                    # if wordlogic.get_combination() not in ["", None]:
                    #     print(f"Das Wort ist = {wordlogic.get_combination()}")
                    #     if tickTackToe.command(wordlogic.get_combination()) == 1:
                    #         print("Game Over")
                    #         tickTackToe.reset()
                            
                    #     wordlogic.reset_combination()

    def run(self):
        stream, safe1, workblocklength, mfcc, dp, mp, class_names = self.pre_process()
        self.main_loop(stream, workblocklength, mfcc, dp, mp, class_names)

if __name__ == '__main__':
    Network().run()


