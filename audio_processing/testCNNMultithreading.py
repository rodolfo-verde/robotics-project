import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D
from keras.regularizers import L2 
from keras.models import load_model

import numpy as np
import time
import sounddevice as sd

import WaveInterface

import threading

from dataprocessor import dataprocessor
from mfcc_processor import mfcc_dataprocessor

from data_spectrogramm import get_spectrogram

#from TickTackToe import TickTackToe

from word_logic import WordLogic

import asyncio

model_to_use = load_model("audio_processing/CNN_Models/Final_speech_CNN_model.h5")

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

x, r, w = WaveInterface.ReadWave("../resources/test2.wav")

words = dp.processforMT(x)


#words = dp.processdata(x, False)

#sd.play(words[8])
class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]


def predict(word: np.array):
    #print(word.shape)
    a = model_to_use.predict(word.reshape(-1,11,70,1), verbose=0)
    a = np.argmax(a)
    print(f"Prediction             : {class_names[a]}")


for i in words:
    mfcc = mp.mfcc_process(i)[1:]
    #print(mfcc.shape)
    threading.Thread(target=predict, args=([mfcc])).start()

#a = model_to_use.predict(words, batch_size=len(words), workers = len(words), use_multiprocessing=True)
#a = model_to_use.predict_on_batch(words)
#print(a)