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
from keras.models import load_model

import numpy as np
import WaveInterface
import time
import sounddevice as sd
import sys
from os import listdir

from dataprocessor import dataprocessor
from mfcc_processor import mfcc_dataprocessor

from data_spectrogramm import get_spectrogram

#from TickTackToe import TickTackToe

from word_logic import WordLogic


TRAIN = False
SAVE = False
USEPROCESSOR = True


#test 
#print(f"Tesorflow version {tf.__version__}")

# load data and split into trainings and test data
data_mfcc = np.load(f"audio_processing//Train_Data//set_complete_test_mfcc.npy",allow_pickle=True) # load data
data_labels = np.load(f"audio_processing//Train_Data//set_complete_test_label.npy",allow_pickle=True) # load data

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


if TRAIN:
    result = model.fit(
        X_train.reshape(-1, 11, 70, 1),
        y_train,
        validation_data = (X_test.reshape(-1, 11, 70, 1), y_test),
        epochs=60, # 60
        batch_size=100) # 100
else:
    model.load_weights("audio_processing//CNN_Models//CNN_More_100_weights.h5")

model.summary()

# evaluate model
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 11, 70, 1), y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

if SAVE:
    model.save("audio_processing//CNN_Models//CNN_More_100_model.h5", include_optimizer=True)
    model.save_weights("audio_processing//CNN_Models//CNN_More_100_weights.h5")


if TRAIN:
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

input("Im done with the modeltraining, press enter to start testing with the stream")


modelspectro = load_model("audio_processing/CNN_Models/Spectrogram_CNN_combined_model.h5")
modelcnnlesslayers = load_model("audio_processing/CNN_Models/Final_speech_CNN_model.h5")

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
starttime = 0

workblocklength = 32500
mfcc = np.zeros((11, 35))
with stream:
    wordlogic = WordLogic()
    #tickTackToe = TickTackToe()
    while True:
        while(len(safe1)<workblocklength):
            time.sleep(0.01)
            continue
        workblock = safe1[:workblocklength]
        safe1 = safe1[workblocklength:]
        oldstarttime = starttime
        starttime = time.time()

        if USEPROCESSOR:
            #ticktacktoe = TickTackToe()
            words, plots = dp.processdata(workblock)

            for i in words[0]:

                #oldmfcc = mfcc
                mfcc = mp.mfcc_process(i)[1:]
                #print(mfcc.shape)
                #print(oldmfcc.shape)

                to_process = mfcc #np.append(oldmfcc, mfcc, axis=1).reshape(11, 70, 1)

                spectrogram = get_spectrogram(i)
                spectrogram = spectrogram.numpy()

                predictionspectro = modelspectro.predict(spectrogram.reshape(-1, 251, 129, 1))

                #print(to_process.shape)

                prediction = model.predict(to_process.reshape(-1, 11, 70, 1))

                predictioncnnlesslayers = modelcnnlesslayers.predict(to_process.reshape(-1, 11, 70, 1))
                index_pred = np.argmax(prediction) #tf.argmax geht auch
                index_pred_spectro = np.argmax(predictionspectro) #tf.argmax geht auch
                index_pred_cnnlesslayers = np.argmax(predictioncnnlesslayers)
                print(f"Prediction             : {class_names[index_pred]} and {prediction[0][index_pred]*100} %")
                print(f"Predictionspectro      : {class_names[index_pred_spectro]} and {predictionspectro[0][index_pred_spectro]*100} %")
                print(f"Predictioncnnlesslayers: {class_names[index_pred_cnnlesslayers]} and {predictioncnnlesslayers[0][index_pred_cnnlesslayers]*100} %")

                print(f"Time: {time.time()-starttime}")
                #sd.play(i)
                print(f"{starttime-oldstarttime} and {time.time()-starttime}")
            
                wordlogic.command(class_names[index_pred])
                if wordlogic.get_combination() != None:
                    print(f"This is the word rodolfo would get = {wordlogic.get_combination()}")
                    #tickTackToe.command(wordlogic.get_combination())
                    wordlogic.reset_combination()


        
        else:
            oldmfcc = mfcc
            mfcc = mp.mfcc_process(i)[1:]
            prediction = model.predict(to_process.reshape(-1, 11, 70, 1))
            index_pred = np.argmax(prediction) #tf.argmax geht auch
            print(f"Prediction: {class_names[index_pred]} and {prediction[0][index_pred]*100} %")
