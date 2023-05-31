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
from mfcc_processor import mfcc_dataprocessor

SAMPLERATE = 44100
TARGETLVL = -30
VOICETHRESHHOLD = -40
LENGTHOFVOICEACTIVITY = 10


PLOTDURATION = 3
VOICEBLOCKSPERPLOT = 1000//LENGTHOFVOICEACTIVITY*PLOTDURATION
VOICEBLOCKSPERSECOND = 1000//LENGTHOFVOICEACTIVITY
PLOTINFOS: np.array

choice = int(input("Enter 1 if you want to plot or process a .wav file \n Enter 2 if you want to plot a .wav file with pass you enter \n Enter 3 if you want to plot the audiostream\n"))

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

    wp = wordprocessor(SAMPLERATE)

    mf = mfcc_dataprocessor(SAMPLERATE)

    sp = signalplotter(PLOTDURATION, SAMPLERATE, VOICEBLOCKSPERPLOT, VOICEBLOCKSPERSECOND, PLOTINFOS, fig1)
    words, plots = dp.processdata(x)

    #plotshitty = np.zeros(x.shape[0])

    """for i in range(len(words[0])):
        plotshitty[dp.wordindeces[i][0]:dp.wordindeces[i][1]] += np.array(wp.phasevocode_data(np.array(words[0][i])))
    
    plots = np.array([[plots[0][0], plots[0][1]], [plots[1][0], plots[1][1]], [plots[2][0], plots[2][1]], [plots[3][0], plotshitty]], dtype=object)
    print(plots.shape)"""

    mfccdata = list()

    for i in words[0]:
        mfccdata.append(mf.mfcc_process(i))
        print(mf.mfcc_process(i))
        plt.matshow(mf.mfcc_process(i), interpolation='nearest', aspect='auto')
        plt.xlabel('time [s]')
        plt.ylabel('MFCC')
        plt.show()

    sp.update_lines(plots)

    oink = input()

    plt.figure(2)
    plt.matshow(mf.mfcc_process(words[0][0]), interpolation='nearest', aspect='auto')
    plt.xlabel('time [s]')
    plt.ylabel('MFCC')
    plt.show()

    for i in words[0]:
        doink = input("to play next word press enter")
        print(i)
        wp.playsound(i)
    
    fnok = input()


if choice==3:

    BLOCKLENGTH = SAMPLERATE//int(input(f"Enter how many blocks should be processed per second"))
    PLOTDURATION = int(input(f"Enter how many seconds should ne plotted"))
    plotbool = PLOTDURATION!=0

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
    
    INPUTDEVICE = 7

    stream = sd.InputStream(channels=1, samplerate=SAMPLERATE, callback=callback, device=INPUTDEVICE)
    dp = dataprocessor(SAMPLERATE, TARGETLVL, VOICETHRESHHOLD, LENGTHOFVOICEACTIVITY)
    time.sleep(0.1)
    PLOTINFOS = dp.get_shape_info()
    PLOTINFOS = np.array([2, 2, 2, 1])

    if plotbool: fig1 = plt.figure()
    if plotbool: fig1.show()

    wp = wordprocessor(SAMPLERATE)
    mf = mfcc_dataprocessor(SAMPLERATE)

    #PLOTDURATION = PLOTDURATION*SAMPLERATE
    VOICEBLOCKSPERPLOT = 1000//LENGTHOFVOICEACTIVITY*PLOTDURATION

    if plotbool: sp = signalplotter(PLOTDURATION, SAMPLERATE, VOICEBLOCKSPERPLOT, VOICEBLOCKSPERSECOND, PLOTINFOS, fig1)

    with stream:
        while True:
            while(len(safe1)<BLOCKLENGTH):
                time.sleep(0.1)
            workblock = safe1[:BLOCKLENGTH]
            safe1 = safe1[BLOCKLENGTH:]

            words, plots = dp.processdata(workblock)
            if plotbool: sp.update_lines(plots)
            for i in words[0]:
                wp.playsound(i)
                plt.matshow(mf.mfcc_process(i), interpolation='nearest', aspect='auto')
                plt.xlabel('time [s]')
                plt.ylabel('MFCC')
                plt.show()