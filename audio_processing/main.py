# This should be our main function hub for the speech recognition.
# We could use this file to call the functions from the other files and also as an import file.
# give me a list of all imports used in this folder and I will add them here
#
import numpy as np
import WaveInterface
import time
import sounddevice as sd
import sys
from os import listdir
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io.wavfile as wav
import scipy.fftpack as fft
import scipy.io as sio
import scipy

# this could be our AudioProcessing class
# it should contain all the functions we need to process the audio
# it should also contain the variables we need to store the data
# it should also contain the functions to save and load the data

from signalplotter import signalplotter
from dataprocessor import dataprocessor
from wordprocessor import wordprocessor

SAMPLERATE = 44100
TARGETLVL = -30
VOICETHRESHHOLD = -40
LENGTHOFVOICEACTIVITY = 10


PLOTDURATION = 3
VOICEBLOCKSPERPLOT = 1000//LENGTHOFVOICEACTIVITY*PLOTDURATION
VOICEBLOCKSPERSECOND = 1000//LENGTHOFVOICEACTIVITY
PLOTINFOS: np.array

choice = int(input("Enter 1 if you want to plot a .wav file \n Enter 2 if you want to plot a .wav file with pass you enter \n Enter 3 if you want to plot the audiostream\n"))

if choice == 1:
    dirlist = listdir("audio_processing/")
    choices = []
    for i in dirlist:
        if i[len(i)-4:] == ".wav":
            choices.append(i)
    print("\n-------------------\n")
    for i in range(len(choices)):
        print(f"Enter {i} for {choices[i]}")
    choice = int(input())
    x, r, w = WaveInterface.ReadWave("audio_processing/" + choices[choice])

    dp = dataprocessor(SAMPLERATE, TARGETLVL, VOICETHRESHHOLD, LENGTHOFVOICEACTIVITY)

    time.sleep(0.1)
    PLOTINFOS = dp.get_shape_info()

    fig1 = plt.figure()
    fig1.show()

    PLOTDURATION = x.shape[0]/SAMPLERATE
    VOICEBLOCKSPERPLOT = 1000//LENGTHOFVOICEACTIVITY*PLOTDURATION

    sp = signalplotter(PLOTDURATION, SAMPLERATE, VOICEBLOCKSPERPLOT, VOICEBLOCKSPERSECOND, PLOTINFOS, fig1)
    words, plots = dp.processdata(x)
    sp.update_lines(plots)

    wp = wordprocessor(SAMPLERATE)

    oink = input()
    for i in words[0]:
        doink = input("to play next word press enter")
        print(i)
        wp.playsound(np.array(i))
    
    fnok = input()