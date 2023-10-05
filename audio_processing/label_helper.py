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

    for i in buttons:
        i.destroy()

    global dirtoselectfrom
    dirtoselectfrom = f"../resources/"

    files = listdir(dirtoselectfrom)
    selectedfiles = list()
    for i in files:
        if i[len(i)-4:] == ".txt":
            selectedfiles.append(i)
        
    for i in range(len(selectedfiles)):
        but = tk.Button(master=root_tk, command=lambda i=f"{dirtoselectfrom}{selectedfiles[i]}": button_read_text(i), text=f"{i+1}. {selectedfiles[i]}")
        but.place(relx=0.2, rely=(i+1)/(len(selectedfiles)+1), anchor=tk.CENTER)
        buttons.append(but)


def button_read_text(instance):
    for i in buttons:
        i.destroy()
    file = open(instance, "r")
    x = file.readlines()
    HAHAHA = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]
    labels = np.zeros((len(x), 9))
    for i in range(len(x)):
        for j in range(len(HAHAHA)):
            if x[i][:-1] == HAHAHA[j]:
                labels[i,j] = 1

    file.close()
    
    process_labels_with_mfcc(labels, instance)


def process_labels_with_mfcc(labels, instance):
    for i in buttons:
        i.destroy()

    global dirtoselectfrom
    dirtoselectfrom = f"audio_processing/Train_Data/"

    files = listdir(dirtoselectfrom)
    selectedfiles = list()
    nottoselect = list()
    for i in files:
        if i[len(i)-8:] == "mfcc.npy":
            if not nottoselect.__contains__(i[:-8]):
                selectedfiles.append(i)
                print(f"{i[:-8]} just added")
        if i[len(i)-9:] == "label.npy":
            if (selectedfiles.__contains__(f"{i[:-9]}mfcc.npy")):
                selectedfiles.remove(f"{i[:-9]}mfcc.npy")
                print(f"{i[:-9]} should be removed")
            nottoselect.append(i[:-9])
            print(f"{i[:-9]} to ignore")
        
    for i in range(len(selectedfiles)):
        but = tk.Button(master=root_tk, command=lambda i=f"{dirtoselectfrom}{selectedfiles[i]}": button_add_label(i, labels), text=f"{i+1}. {selectedfiles[i]}")
        but.place(relx=0.2, rely=(i+1)/(len(selectedfiles)+1), anchor=tk.CENTER)
        buttons.append(but)


def button_add_label(instance, labels):
    for i in buttons:
        i.destroy()
    
    mfcc = np.load(instance)

    if (mfcc.shape[0] == labels.shape[0]):
        np.save(f"{instance[:-8]}label.npy", labels)
    else:
        print("wrong amount of labels or mfccs")
        print(mfcc.shape[0])
        print(labels.shape[0])
    set_start_buttons()


set_start_buttons()
root_tk.mainloop()