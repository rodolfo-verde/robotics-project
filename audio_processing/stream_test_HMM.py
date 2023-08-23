import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt


import numpy as np
import WaveInterface
import time
import sounddevice as sd
import sys
from os import listdir

from dataprocessor import dataprocessor
from mfcc_processor import mfcc_dataprocessor

from tuple_hmm import DataTuple
import pickle
# load hmm model

# load model
print("Loading model...")
with open(f'audio_processing\HMM_models\hmm_model.pkl', 'rb') as f:
    hmm_model = pickle.load(f)
    print("Model loaded")

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
            mfcc_single_sequence = mfcc.reshape(1, 11, 70)
            print(mfcc_single_sequence.shape)
            preds = hmm_model.predict(mfcc_single_sequence)
            #print(preds)
            # convert preds to string
            preds = np.argmax(preds)
            print(preds)
            k = class_names[preds]
            print(k)
            sd.play(i)
        print(time.time()-starttime)
