import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import time
import sys

from audio_processing.mfcc_processor import mfcc_dataprocessor
from audio_processing.dataprocessor import dataprocessor

import tensorflow as tf
print(tf.__version__)
from tensorflow import keras
from keras.models import load_model
import threading

import queue

# class to record audio with a stream using sounddevice

class AudioStream:
    def __init__(self, duration=5, fs=44100, channels=1):
        self.duration = duration
        self.fs = fs
        self.channels = channels
        self.audio_array = np.array([], dtype=np.float64)
        self.blocklength = 44100
        self.targetlvl = -30
        self.voiceactivitythreshold = -40
        self.lengthvoiceactivity = 10
        self.workblocklength = 32500 # test it out so that mfcc is 70
        self.mfcc = np.array([]) # variable to save mfcc data
        self.dp = dataprocessor(self.blocklength, self.targetlvl, self.voiceactivitythreshold, self.lengthvoiceactivity)
        self.mp = mfcc_dataprocessor(self.blocklength)
        self.model = None
        self.class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]
        self.command_queue = queue.Queue()
        self.stopp_command_queue = queue.Queue()

    def audio_callback(self, indata, frames, time, status):
        # save the audio data in one singular array by appending each chunk of data
        self.audio_array = np.append(self.audio_array, indata)
        pass

    def start(self):
        # Start the audio stream
        devices = sd.query_devices()
        # set the input device to the microphone to the default input device
        INPUTDEVICE = 1 # 1 for Windows, 17 or 11 for Linux
        for i in devices:
            if i['name'] == 'default':
                INPUTDEVICE = i['index']
        stream = sd.InputStream(channels=self.channels, samplerate=self.fs, callback=self.audio_callback, device=INPUTDEVICE)
        print("Recording started")

        with stream:  # Replace with your stop condition
            while True:
                #start_time = time.time()
                #while time.time() - start_time < 5:  # Listen for 5 seconds
                    #time.sleep(0.1)
                    #continue
                while len(self.audio_array) < self.workblocklength:
                    time.sleep(0.1)
                    continue
                workblock = self.audio_array[:self.workblocklength]
                self.audio_array = self.audio_array[self.workblocklength:]
                # Process the audio with your DataProcessor and MFCC classes
                words, plots = self.dp.processdata(workblock)
                for word in words[0]:
                    mfcc = self.mp.mfcc_process(word)[1:]
                    #self.mfcc = np.append(self.mfcc, mfcc)
                    self.predict(mfcc, self.class_names)
                    #sd.play(word, self.fs)

    def stop(self):
        # Stop the audio stream
        self.stream.stop()
        self.stream.close()
        print("Recording finished")

    def save_audio(self, filename):
        # save the audio data as a .wav file
        write(f"{filename}.wav", self.fs, self.audio_array)
        # Normalize audio_array to 16-bit range
        audio_array = np.int16(self.audio_array / np.max(np.abs(self.audio_array)) * 32767)
        # Write .wav file
        sd.wait()  # Wait until file is done playing
        sd.play(audio_array, self.fs)
        sd.wait()  # Wait until file is done playing
        sd.stop()
        # save array as npy file
        np.save(f"{filename}.npy", audio_array)
        # save mfcc as npy file
        np.save(f"{filename}_mfcc.npy", self.mfcc)

    def listen(self):
        # Listen for speech command
        command = input("Press enter to stop recording: ")
        return command

    def play(self):
        # play the audio data
        sd.play(self.audio_array, self.fs)
        sd.wait()  # Wait until file is done playing

    def plot(self):
        # plot the audio data
        plt.plot(self.audio_array)
        plt.show()


    def load_CNN_model(self, model_path):
        # load the CNN model
        self.model = load_model(model_path)

    def predict(self, mfcc, class_names):
        # predict the word
        mfcc = mfcc.reshape(-1, 12, 70, 1)
        prediction = self.model.predict(mfcc)
        # Save the prediction
        #self.latest_command = class_names[np.argmax(prediction)]
        predicted_command = class_names[np.argmax(prediction)]
        if predicted_command == 'stopp':
            self.stopp_command_queue.put(predicted_command)
            print('Prediction: ', predicted_command, ' with probability: ', np.max(prediction)*100,'%')
        else:
            self.command_queue.put(predicted_command)
            print('Prediction: ', predicted_command, ' with probability: ', np.max(prediction)*100,'%')
            # print the prediction
            #print('Prediction: ', self.latest_command, ' with probability: ', np.max(prediction)*100,'%')

    def main(self):
        # load the CNN model
        self.load_CNN_model('CNN_Models/CNN_Model_Rex_simple.h5')
        self.start()