# This should be our main function hub for the speech recognition.
# We could use this file to call the functions from the other files and also as an import file.
# give me a list of all imports used in this folder and I will add them here
#
import numpy as np
import WaveInterface
import time
import sounddevice as sd
import sys
import os
import pyaudio
import wave
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.io.wavfile as wav
import scipy.fftpack as fft
import scipy.io as sio
import scipy


