import random
import tkinter as tk
from os import listdir
import WaveInterface
import sounddevice as sd
import time
import numpy as np

from dataprocessor import dataprocessor
from mfcc_processor import mfcc_dataprocessor


root_tk = tk.Tk()
root_tk.geometry("600x300")
root_tk.title("generate and laber data")


buttons = list()
rawlist = np.array([["a"]], ndmin=2)
mfcclist = np.array([[]], ndmin=2)
labellist = np.array([[]], ndmin=2)


def button_label_from_file():
    global button_label_file
    button_label_file.destroy()
    button_label_stream.destroy()
    select_file_to_label()


def select_file_to_label():
        files = listdir("audio_processing")
        selectedfiles = list()
        for i in files:
            if i[len(i)-4:] == ".wav":
                    selectedfiles.append(i)
        
        for i in range(len(selectedfiles)):
            but = tk.Button(master=root_tk, command=lambda i=f"audio_processing/{selectedfiles[i]}": button_select_file(i), text=f"{i+1}. {selectedfiles[i]}")
            but.place(relx=0.2, rely=(i+1)/(len(selectedfiles)+1), anchor=tk.CENTER)
            buttons.append(but)


def button_select_file(instance):
    for i in buttons:
         i.destroy()
    x, f, r = WaveInterface.ReadWave(instance)
    label_data(x)


def playagain(x):
    sd.play(x)


def label_data(x):
    global buttonpressed
    SAMPLERATE = 44100
    TARGETLVL = -30
    VOICETHRESHHOLD = -40
    LENGTHOFVOICEACTIVITY = 10

    dp = dataprocessor(SAMPLERATE, TARGETLVL, VOICETHRESHHOLD, LENGTHOFVOICEACTIVITY)
    words = dp.processdata(x)[0][0]
    for i in words:
        sd.play(i)
        for j in buttons:
            j.destroy()
        labellistnames = ["a", "b", "c", "1", "2", "3", "stop", "rex", "other"]
        la, lb, lc, l1, l2, l3, ls, lr, lo = tk.IntVar(), tk.IntVar(), tk.IntVar(), tk.IntVar(), tk.IntVar(), tk.IntVar(), tk.IntVar(), tk.IntVar(), tk.IntVar()
        labels = [la, lb, lc, l1, l2, l3, ls, lr, lo]
        again = tk.Button(master=root_tk, command=lambda: playagain(i), text=f"Play the sequence again")
        again.place(relx=0.5, rely=0.1, anchor=tk.CENTER)
        buttons.append(again)
        for j in range(len(labellistnames)):
            a = tk.Checkbutton(master=root_tk, variable=labels[j], text=labellistnames[j], onvalue=1, offvalue=0, bd=4)
            a.place(relx=((j%3)/(len(labellistnames)/3)+0.1), rely=(0.3+((j//3)*0.2)), anchor=tk.CENTER)
            buttons.append(a)
        buttonpressed = tk.BooleanVar()
        setlabel = tk.Button(master=root_tk, command=lambda: set_labels(i, labels), text=f"Set the labels")
        setlabel.place(relx=0.5, rely=0.9, anchor=tk.CENTER)
        buttons.append(setlabel)

        skipbut = tk.Button(master=root_tk, command=skip, text=f"Skip this")
        skipbut.place(relx=0.7, rely=0.9, anchor=tk.CENTER)
        buttons.append(skipbut)

        setlabel.wait_variable(buttonpressed)
    
    for j in buttons:
        j.destroy()
    
    print(f"{len(rawlist)} and {len(mfcclist)} and {len(labellist)}")
    save_data_set()


def set_labels(x, butlab):
    global buttonpressed

    raw = x

    mp = mfcc_dataprocessor(44100)

    mfcc = mp.mfcc_process(raw)

    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    for i in range(len(butlab)):
        if butlab[i].get()==1:
            labels[i] = 1
    
    safe(raw, mfcc, labels)

    buttonpressed.set(True)


def safe(raw, mfcc, labels):
    global rawlist
    global mfcclist
    global labellist

    if rawlist[0, 0]=="a":
        rawlist = np.array([raw], ndmin = 2)
        mfcclist = np.array([mfcc], ndmin = 2)
        labellist = np.array([labels], ndmin = 2)
    else:
        rawlist = np.append(rawlist, [raw], axis=0)
        mfcclist = np.append(mfcclist, [mfcc], axis=0)
        labellist = np.append(labellist, [labels], axis=0)


def skip():
    global buttonpressed
    buttonpressed.set(True)


def save_data_set():
    
    sd.play(rawlist[5])
    print(labellist[5])
    print("JÖÖÖÖÖLK")


def button_label_from_stream():
    print("ÖÖÖÖLK")


button_label_file = tk.Button(master=root_tk, command=button_label_from_file, text="Label data from file")
button_label_file.place(relx=0.3, rely=0.5, anchor=tk.CENTER)

button_label_stream = tk.Button(master=root_tk, command=button_label_from_stream, text="Label data from stream")
button_label_stream.place(relx=0.7, rely=0.5, anchor=tk.CENTER)

root_tk.mainloop()