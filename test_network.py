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

#from robot_control.TickTackToe import TickTackToe

from audio_processing.word_logic import WordLogic

from scipy.io import wavfile
import sounddevice as sd
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
safe2 = np.array([], dtype="float64")


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

        INPUTDEVICE = None  # Default value
        INPUTDEVICE2 = None  # Default value

        def callback(indata, frame_count, time_info, status):
            global safe1
            safe1 = np.append(safe1, indata)
            global safe2
            safe2 = np.append(safe2, indata)

        # Microphone inputs

        def find_usb_audio_devices():
            devices = sd.query_devices()
            
            usb_input_devices = []

            for device in devices:
                if "USB" in device["name"] and "Input" in device["name"]:
                    usb_input_devices.append((device["name"], device["index"]))

            return usb_input_devices

        usb_input_devices = find_usb_audio_devices()

        for name, index in usb_input_devices:
            print(f"Device Name: {name}, Device Index: {index}")
            
        if len(usb_input_devices) >= 2:
            INPUTDEVICE = usb_input_devices[0]
            INPUTDEVICE2 = usb_input_devices[1]

        print(f"INPUTDEVICE: {INPUTDEVICE}, INPUTDEVICE2: {INPUTDEVICE2}")

        stream = sd.InputStream(channels=1, samplerate=SAMPLERATE, callback=callback, device=INPUTDEVICE)
        stream2 = sd.OutputStream(channels=1, samplerate=SAMPLERATE, callback=callback, device=INPUTDEVICE2)

        class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]

        workblocklength = 32500
        mfcc = np.zeros((11, 35))
        return stream, stream2, safe1, safe2, workblocklength, mfcc, dp, mp, class_names

    def prediction(self, to_process, wordlogic, class_names, tickTackToe = None): 
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



    
    def run(self):
        stream1, stream2, safe1, safe2, workblocklength, mfcc, dp, mp, class_names = self.pre_process()
        thread1 = threading.Thread(target=self.main_loop, args=(stream1, safe1, workblocklength, mfcc, dp, mp, class_names))
        thread2 = threading.Thread(target=self.main_loop, args=(stream2, safe2, workblocklength, mfcc, dp, mp, class_names))

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

    def main_loop(self, stream, safe, workblocklength, mfcc, dp, mp, class_names):
        print("main_loop started")
        with stream:
            wordlogic = WordLogic()
            while True:
                while len(safe) < workblocklength:
                    time.sleep(0.01)
                    continue

                workblock = safe[:workblocklength]
                safe = safe[workblocklength:]

                words, plots = dp.processdata(workblock)

                for i in words[0]:
                    mfcc = mp.mfcc_process(i)[1:]
                    to_process = mfcc
                    threading.Thread(target=self.prediction, args=(to_process, wordlogic, class_names)).start()


if __name__ == '__main__':
    Network().run()


