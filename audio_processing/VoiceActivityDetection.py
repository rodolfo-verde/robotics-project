import numpy as np
import WaveInterface
import time
import sounddevice as sd
import matplotlib.pyplot as plt 
import sys

#define parameters
Fs = 44100
bits = 16
# x = WaveInterface.ReadWave('Test.wav') --> once we have the file on wednesday after gain control
A = 1 # maximum magnitude of a wave file
BlocksizeInMs = 10
BlocksizeInSamples = int(BlocksizeInMs * Fs / 1000)
NumberOfBlocks = x.shape[0] // BlocksizeInSamples

#calculate the power of each block
L = np.zeros((NumberOfBlocks))
for n in range(NumberOfBlocks):
    idx1 = n * BlocksizeInSamples
    idx2 = idx1 + BlocksizeInSamples
    P = np.mean(x[idx1:idx2]**2)
    L[n] = 10*np.log10(2*P/(A**2))

#define threshold level after gain control
ThresholdLevel = # insert threshold level here
IsSpeech = L > ThresholdLevel


#word detection
IndexOfWord = 0
IsTriggered = False
for n in range(1, IsSpeech.shape[0]):
    CurrentSamplingPosition = int(n * BlocksizeInMs / 1000 * Fs)
    if (not IsSpeech[n-1]) and (IsSpeech[n]) and (not IsTriggered):
        IsTriggered = True
        BeginningOfWordInSamples = CurrentSamplingPosition
    if (IsSpeech[n-1]) and (not IsSpeech[n]) and (IsTriggered):
        IsTriggered = False
        EndingOfWordInSamples = CurrentSamplingPosition
        FileName = 'VAD_WordNumber_' + str(IndexOfWord) + '.wav'
        IndexOfWord += 1
        WordSamples = y[BeginningOfWordInSamples:EndingOfWordInSamples]
        WaveInterface.WriteWave(WordSamples, Fs, bits, FileName)
print(IndexOfWord, ' words detected')