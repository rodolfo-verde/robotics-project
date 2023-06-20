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
root_tk.title("generate and laber data")


buttons = list()
rawlist = np.array([["a"]], ndmin=2)
mfcclist = np.array([[]], ndmin=2)
labellist = np.array([[]], ndmin=2)


def button_label_from_file():
    for i in buttons:
        i.destroy()
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
    select_save_data_set_name()


def set_labels(x, butlab):
    global buttonpressed

    raw = x

    mp = mfcc_dataprocessor(44100)

    mfcc = mp.mfcc_process(raw)

    labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    for i in range(len(butlab)):
        if butlab[i].get()==1:
            labels[i] = 1
    
    save(raw, mfcc, labels)

    buttonpressed.set(True)


def save(raw, mfcc, labels):
    global rawlist
    global mfcclist
    global labellist

    if rawlist[0, 0]=="a":
        rawlist = np.array([raw], ndmin = 2)
        mfcclist = np.array([mfcc[1:]], ndmin = 2)
        labellist = np.array([labels], ndmin = 2)
    else:
        rawlist = np.append(rawlist, [raw], axis=0)
        mfcclist = np.append(mfcclist, [mfcc[1:]], axis=0)
        labellist = np.append(labellist, [labels], axis=0)


def skip():
    global buttonpressed
    buttonpressed.set(True)


def select_save_data_set_name():

    infile = listdir("audio_processing/Train_Data/")
    stored = list()

    for i in infile:
        if i[len(i)-9:]=="label.npy":
            stored.append(f"{i[:-10]}")
    
    nameentry = tk.Entry(master=root_tk)
    nameentry.place(relx=0.7, rely=0.4, anchor=tk.CENTER)
    buttons.append(nameentry)
    savebut = tk.Button(master=root_tk, command=lambda: save_data_set(nameentry.get()), text=f"Save to file")
    savebut.place(relx=0.7, rely=0.6, anchor=tk.CENTER)
    buttons.append(savebut)

    for i in range(len(stored)):
        but = tk.Button(master=root_tk, command=lambda i=stored[i]: set_text(i, nameentry), text=f"{i+1}. {stored[i]}")
        but.place(relx=0.2, rely=(i+1)/(len(stored)+1), anchor=tk.CENTER)
        buttons.append(but)


def set_text(text, label):
    label.delete(0, tk.END)
    label.insert(0, text)


def save_data_set(setname):
    
    for i in buttons:
        i.destroy()
    
    data_file_raw = f"audio_processing/Train_Data/{setname}_raw"
    data_file_mfcc = f"audio_processing/Train_Data/{setname}_mfcc"
    data_file_label = f"audio_processing/Train_Data/{setname}_label"

    if rawlist.shape[0]==1:
        return

    if path.exists(f"{data_file_raw}.npy"):
        stored_raw = np.load(f"{data_file_raw}.npy")
        stored_raw = np.append(stored_raw, rawlist[1:], axis=0)
    else:
        stored_raw = rawlist[1:]
    if path.exists(f"{data_file_mfcc}.npy"):
        stored_mfcc = np.load(f"{data_file_mfcc}.npy")
        stored_mfcc = np.append(stored_mfcc, mfcclist[1:], axis=0)
    else:
        stored_mfcc = mfcclist[1:]
    if path.exists(f"{data_file_label}.npy"):
        stored_label = np.load(f"{data_file_label}.npy")
        stored_label = np.append(stored_label, labellist[1:], axis=0)
    else:
        stored_label = labellist[1:]

    np.save(data_file_raw, stored_raw)    
    np.save(data_file_mfcc, stored_mfcc)    
    np.save(data_file_label, stored_label)

    data_mfcc = np.load(f"audio_processing/Train_Data/set2_mfcc.npy",allow_pickle=True) # load data
    data_labels = np.load(f"audio_processing/Train_Data/set2_label.npy",allow_pickle=True) # load data

    print(f"Data shape: {data_mfcc.shape}")
    print(f"Labels shape: {data_labels.shape}")

    set_start_buttons()


def button_label_from_stream():
    print("ÖÖÖÖLK")


def set_start_buttons():

    for i in buttons:
        i.destroy()

    button_label_file = tk.Button(master=root_tk, command=button_label_from_file, text="Label data from file")
    button_label_file.place(relx=0.3, rely=0.3, anchor=tk.CENTER)
    buttons.append(button_label_file)

    button_label_stream = tk.Button(master=root_tk, command=button_label_from_stream, text="Label data from stream")
    button_label_stream.place(relx=0.7, rely=0.3, anchor=tk.CENTER)
    buttons.append(button_label_stream)

    button_check_data = tk.Button(master=root_tk, command=check_data_button, text="check existing data")
    button_check_data.place(relx=0.3, rely=0.7, anchor=tk.CENTER)
    buttons.append(button_check_data)


def check_data_button():
    for i in buttons:
        i.destroy()
    
    infile = listdir("audio_processing/Train_Data/")
    stored = list()

    for i in infile:
        if i[len(i)-9:]=="label.npy":
            stored.append(f"{i[:-10]}")
    
    for i in range(len(stored)):
        but = tk.Button(master=root_tk, command=lambda i=stored[i]: load_data_to_check(i), text=f"{i+1}. {stored[i]}")
        but.place(relx=0.2, rely=(i+1)/(len(stored)+1), anchor=tk.CENTER)
        buttons.append(but)


def load_data_to_check(setname):
    
    data_raw = np.load(f"audio_processing/Train_Data/{setname}_raw.npy")
    data_label = np.load(f"audio_processing/Train_Data/{setname}_label.npy")

    labels = np.zeros(9)

    for i in data_label:
        labels += i

    lab = tk.Label(master=root_tk, text="")
    lab.place(relx=0.6, rely=0.6)
    buttons.append(lab)

    butplayback = tk.Button(master=root_tk, command=lambda dr=data_raw, dl = data_label, lab = lab: randdata(dr, dl, lab), text=f"{setname}")
    butplayback.place(relx=0.6, rely=0.4, anchor=tk.CENTER)
    buttons.append(butplayback)

    labelgeneral = tk.Label(master=root_tk, text=f"overall datasamples: {data_label.shape[0]}")
    labelgeneral.place(relx=0.8, rely=0.1, anchor=tk.CENTER)
    buttons.append(labelgeneral)

    labellistnames = ["a", "b", "c", "1", "2", "3", "stop", "rex", "other"]
    data_label = np.sum(data_label)

    for j in range(len(labellistnames)):
            a = tk.Label(master=root_tk, text=f"{labellistnames[j]}: {int(labels[j])}")
            a.place(relx=(0.8), rely=(0.2+j/(len(labellistnames)+2)), anchor=tk.CENTER)
            buttons.append(a)


def randdata(data_raw, data_label, lab):

    i = random.randint(0, data_raw.shape[0]-1)
    sd.play(data_raw[i])
    labels = data_label[i]
    text = ""
    labellistnames = ["a", "b", "c", "1", "2", "3", "stop", "rex", "other"]
    for i in range(len(labellistnames)):
        if labels[i] == 1:
            text += labellistnames[i]
    lab.config(text=text)


set_start_buttons()
root_tk.mainloop()