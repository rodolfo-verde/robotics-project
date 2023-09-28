import random
import tkinter as tk
from os import listdir
from os import path
import WaveInterface
import sounddevice as sd
import time
import numpy as np

from dataprocessor import dataprocessor
from mfcc_processor import mfcc_dataprocessor


root_tk = tk.Tk()
root_tk.geometry("600x300")
root_tk.title("mass label helper")


buttons = list()
rawlist = np.array([["a"]], ndmin=2)
mfcclist = np.array([["a"]], ndmin=2)
labellist = np.array([[]], ndmin=2)
first = True
backbut = tk.Button(master=root_tk, command= lambda: set_start_buttons(), text="<--")
backbut.place(relx=0.1, rely=0.1, anchor=tk.CENTER)


# Setting up the Buttons on the screen
def set_start_buttons():

    for i in buttons:
        i.destroy()

    button_audio_to_mfcc = tk.Button(master=root_tk, command=audio_to_mfcc, text="audio to mfcc")
    button_audio_to_mfcc.place(relx=0.3, rely=0.3, anchor=tk.CENTER)
    buttons.append(button_audio_to_mfcc)

    button_add_label_to_mfcc = tk.Button(master=root_tk, command=add_label_to_mfcc, text="add label to mfcc")
    button_add_label_to_mfcc.place(relx=0.7, rely=0.3, anchor=tk.CENTER)
    buttons.append(button_add_label_to_mfcc)


def audio_to_mfcc():

    for i in buttons:
        i.destroy()

    global dirtoselectfrom
    dirtoselectfrom = f"../resources/"

    files = listdir(dirtoselectfrom)
    selectedfiles = list()
    for i in files:
        if i[len(i)-4:] == ".wav":
                selectedfiles.append(i)
        
    for i in range(len(selectedfiles)):
        but = tk.Button(master=root_tk, command=lambda i=f"{dirtoselectfrom}{selectedfiles[i]}": button_read_wave(i), text=f"{i+1}. {selectedfiles[i]}")
        but.place(relx=0.2, rely=(i+1)/(len(selectedfiles)+1), anchor=tk.CENTER)
        buttons.append(but)


def button_read_wave(instance):
    for i in buttons:
         i.destroy()
    x, f, r = WaveInterface.ReadWave(instance)
    process_audio_to_mfcc(x, instance)


def process_audio_to_mfcc(audio, name: str):

    SAMPLERATE = 44100
    TARGETLVL = -30
    VOICETHRESHHOLD = -40
    LENGTHOFVOICEACTIVITY = 10
    dp = dataprocessor(SAMPLERATE, TARGETLVL, VOICETHRESHHOLD, LENGTHOFVOICEACTIVITY)
    mp = mfcc_dataprocessor(SAMPLERATE)

    words = dp.processdata(audio, False)

    mfccs = mp.mfcc_process_multiple(words)[:,1:,:]

    np.save(f"audio_processing/Train_Data/{name[len(dirtoselectfrom):-4]}_mfcc.npy", mfccs)

    print(f"I found {len(mfccs)} words")

    set_start_buttons()


def add_label_to_mfcc():
     print("YOIIINK not implemented yet du keggo")


set_start_buttons()
root_tk.mainloop()