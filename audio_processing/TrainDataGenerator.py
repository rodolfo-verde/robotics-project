import sounddevice as sd
import numpy as np
from os import listdir
import WaveInterface
from os import path
import random

from dataprocessor import dataprocessor
from wordprocessor import wordprocessor
from mfcc_processor import mfcc_dataprocessor

class train_data_generator:

    stored_data: np.array
    length_of_one_block: int
    data_file: str

    SAMPLERATE = 44100
    TARGETLVL = -30
    VOICETHRESHHOLD = -40
    LENGTHOFVOICEACTIVITY = 10

    dp: dataprocessor
    mp: mfcc_dataprocessor

    def __init__(self, length_of_one_block: int, data_file: str) -> None:
        self.length_of_one_block = length_of_one_block
        self.data_file = data_file

        self.stored_data = np.array([[[]]], ndmin=3)

        if path.exists(f"{self.data_file}.npy"):
            self.stored_data = np.load(f"{self.data_file}.npy", allow_pickle=True)
        
        self.dp = dataprocessor(self.SAMPLERATE, self.TARGETLVL, self.VOICETHRESHHOLD, self.LENGTHOFVOICEACTIVITY)
        self.mp = mfcc_dataprocessor(self.SAMPLERATE)
        
    
    def label_data_from_file(self, file_to_label: str):
        
        x, f, r = WaveInterface.ReadWave(file_to_label)
        
        words = self.dp.processdata(x)[0][0]
        for i in words:
            sd.play(i)
            label = input("Enter the detected word from")
            labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
            skip = False
            if label == "stop": 
                labels[6] = 1
                skip = True
            if label == "rex": 
                labels[7] = 1
                skip = True

            if label == "":
                labels[8] = 0   
                skip = True

            if not skip:
                for j in label:
                    if j == "a": labels[0] = 1
                    if j == "b": labels[1] = 1
                    if j == "c": labels[2] = 1
                    if j == "1": labels[3] = 1
                    if j == "2": labels[4] = 1
                    if j == "3": labels[5] = 1
                

            mfcc = self.mp.mfcc_process(i)
            to_store = np.array([[[mfcc], [labels]]], dtype=object)
            #print(f"{to_store.shape} and {self.stored_data.shape}")
            if self.stored_data.shape == (1, 1, 0):
                print("hit")
                self.stored_data = to_store
            else:
                self.stored_data = np.append(self.stored_data, to_store, axis=0)
            print(self.stored_data.shape)
    

    def select_file_to_label(self):
        files = listdir("audio_processing")
        selectedfiles = list()
        for i in files:
            if i[len(i)-4:] == ".wav":
                    selectedfiles.append(i)
        
        for i in range(len(selectedfiles)):
            print(f"enter {i} if you want to label {selectedfiles[i]}")
        choice = int(input())
        self.label_data_from_file(selectedfiles[choice])
    

    #def label_from_stream(self):


    
    def save_data(self):
        np.save(self.data_file, self.stored_data)


    def get_data(self):
        return self.stored_data


tg = train_data_generator(32500, "audio_processing/Train_Data/set1")
#tg.select_file_to_label()
#print(tg.get_data().shape)
tg.label_data_from_file("audio_processing/raw_commands_test.wav")
#tg.label_data_from_file("audio_processing/raw_noise_distance_commands_testt.wav")
tg.save_data()
#print(tg.get_data()[1][1])