import sounddevice as sd
import time

import numpy as np
import tensorflow as tf 
from tensorflow import keras

from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPool2D
from keras.optimizers import SGD
from keras.regularizers import L2 
from keras.models import load_model

from audio_processing.dataprocessor import dataprocessor
from audio_processing.mfcc_processor import mfcc_dataprocessor

import threading

class streamprocessor:

    _class_names: np.array
    _instream: np.array
    _streamlock: threading.Lock
    _resultlock: threading.Lock
    _to_process: np.array
    _workblocklength: int
    _dataprocessor: dataprocessor
    _mfcc_dataprocessor: mfcc_dataprocessor
    _stream: sd.InputStream
    _results: np.array
    _model: keras.models.Sequential


    # callback function for the stream, with a lock to make it safe in multithreading
    def callback(self, indata, frame_count, time_info, status):
        self._streamlock.acquire()
        self._instream = np.append(self._instream, indata)
        if len(self._instream) > self._workblocklength:
            self._to_process = self._instream[:self._workblocklength]
            self._instream = self._instream[10000:]
            self.process_next()
        self._streamlock.release()
    

    # Setting everything up
    def __init__(self):

        BLOCKLENGTH = 44100
        SAMPLERATE = 44100
        TARGETLVL = -30
        VOICETHRESHHOLD = -40
        LENGTHOFVOICEACTIVITY = 10

        self._class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]

        print("Loading model...")

        try:
            #self._model = tf.keras.models.load_model('audio_processing//CNN_Models//Final_speech_CNN_model.h5')
            self._model = tf.keras.models.load_model('audio_processing//CNN_Models//CNN_Model_Rex_simple.h5')
            #self._model.summary()
        except Exception as e:
            print(e)
            quit()

        print("Model loaded.")
        print("Setting up the processors")

        self._workblocklength = 32500
        self._dataprocessor = dataprocessor(BLOCKLENGTH, TARGETLVL, VOICETHRESHHOLD, LENGTHOFVOICEACTIVITY)
        self._mfcc_dataprocessor = mfcc_dataprocessor(BLOCKLENGTH)

        print("Processors ready")
        print("Setting up the locks")
        self._streamlock = threading.Lock()
        self._resultlock = threading.Lock()
        print(self._resultlock.locked())
        print("Locks ready")
        
        print("Setting up the stream")
        devices = sd.query_devices()

        self._results = np.array([])
        self._instream = np.array([])

        INPUTDEVICE = 1
        for i in devices:
            if i['name'] == 'default':
                INPUTDEVICE = i['index']

        self._stream = sd.InputStream(channels=1, samplerate=SAMPLERATE, callback=self.callback, device=INPUTDEVICE)
        print("Stream ready and running")

        self.run()
    

    # the function to run the stream
    def run(self):
        with self._stream:
            while True:
                self._resultlock.acquire()
                print(self._results)
                self._resultlock.release()
                time.sleep(1)


    # processing the next datablock
    def process_next(self):
        to_mfcc = self._dataprocessor.gaindata(self._to_process)
        to_predict = self._mfcc_dataprocessor.mfcc_process(to_mfcc)[1:]
        print(to_predict.shape)
        threading.Thread(target= self.predict, args= (to_predict,)).start()


    # predicting the data, hopefully in a thread and adding the result to the result array
    def predict(self, data):
        print(data.shape)
        prediction = self._model.predict(data.reshape(-1, 11, 70, 1))
        self.add_prediction(prediction)

    
    # appending the results array with a threadsafe lock
    def add_prediction(self, prediction):
        self._resultlock.acquire()
        prediction = np.argmax(prediction)

        print(self._class_names[prediction])
        self._results = np.append(self._results, [prediction], axis=0)
        self._resultlock.release()

    
    # returning the results 
    def get_results(self):
        self._resultlock.acquire()
        reti = self._results()

        self._results = np.array([])

        self._resultlock.release()

        for i in reti:
            print(i)


oink = streamprocessor()
time.sleep(10)
oink.get_results()