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
mfcclist = np.array([["a"]], ndmin=2)
labellist = np.array([[]], ndmin=2)
first = True
backbut = tk.Button(master=root_tk, command= lambda: set_start_buttons(), text="<--")
backbut.place(relx=0.1, rely=0.1, anchor=tk.CENTER)


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

    for i in buttons:
        i.destroy()

    global buttonpressed
    SAMPLERATE = 44100
    TARGETLVL = -30
    VOICETHRESHHOLD = -40
    LENGTHOFVOICEACTIVITY = 10

    keepvar = tk.BooleanVar()
    keepvar.set(False)

    keep = tk.Checkbutton(master=root_tk, variable=keepvar, text="safe labelboxes between data", onvalue=True, offvalue=False, bd=4)
    keep.place(relx=0.8, rely=0.1, anchor=tk.CENTER)

    dp = dataprocessor(SAMPLERATE, TARGETLVL, VOICETHRESHHOLD, LENGTHOFVOICEACTIVITY)
    words = dp.processdata(x)[0][0]
    for i in words:
        sd.play(i)

        if not keepvar.get():
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
    keep.destroy()
    
    print(f"{len(mfcclist)} and {len(labellist)}")
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
    
    save(mfcc, labels, raw)

    buttonpressed.set(True)


def save(mfcc, labels, raw):
    global mfcclist
    global labellist
    global rawlist
    global first

    if first:
        first = False
        rawlist = np.array([raw], ndmin=2)
        mfcclist = np.array([mfcc[1:]], ndmin = 2)
        labellist = np.array([labels], ndmin = 2)
    else:
        mfcclist = np.append(mfcclist, [mfcc[1:]], axis=0)
        labellist = np.append(labellist, [labels], axis=0)
        rawlist = np.append(rawlist, [raw], axis=0)
    
    print(f"{mfcclist.shape} and {labellist.shape}")


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

    if mfcclist.shape[0]==1:
        return
    
    if path.exists(f"{data_file_raw}.npy"):
        stored_raw = np.load(f"{data_file_raw}.npy")
        stored_raw = np.append(stored_raw, rawlist, axis=0)
    else:
        stored_raw = rawlist
    
    if path.exists(f"{data_file_mfcc}.npy"):
        stored_mfcc = np.load(f"{data_file_mfcc}.npy")
        stored_mfcc = np.append(stored_mfcc, mfcclist, axis=0)
    else:
        stored_mfcc = mfcclist
    if path.exists(f"{data_file_label}.npy"):
        stored_label = np.load(f"{data_file_label}.npy")
        stored_label = np.append(stored_label, labellist, axis=0)
    else:
        stored_label = labellist

    np.save(data_file_raw, stored_raw)    
    np.save(data_file_mfcc, stored_mfcc)    
    np.save(data_file_label, stored_label)

    set_start_buttons()


def callback(indata, frame_count, time_info, status):
    global save1
    save1 = np.append(save1, indata)


def button_label_from_stream():

    global save1

    for i in buttons:
        i.destroy()

    devices = sd.query_devices()

    for i in devices:
        if i['name'] == 'default':
            INPUTDEVICE = i['index']
    
    stream = sd.InputStream(channels=1, samplerate=44100, callback=callback, device=INPUTDEVICE)

    streamend = tk.BooleanVar()
    streamend.set(False)
    streamstart = tk.BooleanVar()
    streamstart.set(False)

    butstart = tk.Button(master=root_tk, command=lambda: streamstart.set(True), text=f"Start input stream")
    butstart.place(relx=0.2, rely=0.4, anchor=tk.CENTER)
    buttons.append(butstart)

    butend = tk.Button(master=root_tk, command=lambda: streamend.set(True), text=f"End input stream")
    butend.place(relx=0.2, rely=0.6, anchor=tk.CENTER)
    buttons.append(butend)

    save1 = np.array([])

    butstart.wait_variable(streamstart)
    stream.start()
    print("recording")
    butend.wait_variable(streamend)
    stream.close()

    data = save1[1:]
    print(data.shape)
    #sd.play(data)
    print("Finished")

    butlabel = tk.Button(master=root_tk, command=lambda: label_data(data), text=f"Start label the data")
    butlabel.place(relx=0.7, rely=0.4, anchor=tk.CENTER)
    buttons.append(butlabel)

    
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

    button_combine_data = tk.Button(master=root_tk, command=combine_sets, text="combine existing data")
    button_combine_data.place(relx=0.7, rely=0.7, anchor=tk.CENTER)
    buttons.append(button_combine_data)


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
    
    #data_raw = np.load(f"audio_processing/Train_Data/{setname}_raw.npy")
    data_label = np.load(f"audio_processing/Train_Data/{setname}_label.npy")
    data_mfcc = np.load(f"audio_processing/Train_Data/{setname}_mfcc.npy")

    print(f"{data_mfcc.shape} and {data_label.shape}")

    labels = np.zeros(9)

    for i in data_label:
        labels += i

    labelgeneral = tk.Label(master=root_tk, text=f"overall datasamples: {data_label.shape[0]}")
    labelgeneral.place(relx=0.8, rely=0.1, anchor=tk.CENTER)
    buttons.append(labelgeneral)

    labellistnames = ["a", "b", "c", "1", "2", "3", "stop", "rex", "other"]
    data_label = np.sum(data_label)

    for j in range(len(labellistnames)):
            a = tk.Label(master=root_tk, text=f"{labellistnames[j]}: {int(labels[j])}")
            a.place(relx=(0.8), rely=(0.2+j/(len(labellistnames)+2)), anchor=tk.CENTER)
            buttons.append(a)


def combine_sets():

    for i in buttons:
        i.destroy()
    
    infile = listdir("audio_processing/Train_Data/")
    stored = list()

    for i in infile:
        if i[len(i)-9:]=="label.npy":
            stored.append(f"{i[:-10]}")
    
    checks = list()

    for i in range(len(stored)):
        bv = tk.IntVar()
        checks.append(bv)
        but = tk.Checkbutton(master=root_tk, variable = bv, onvalue=1, offvalue=0, bd=1, text=f"{i+1}. {stored[i]}")
        but.place(relx=0.2, rely=(i+1)/(len(stored)+1), anchor=tk.CENTER)
        buttons.append(but)
    
    aug = tk.BooleanVar()
    augment = tk.Checkbutton(master=root_tk, variable=aug, onvalue=True, offvalue=False, bd=1, text="Augment data if possible")
    augment.place(relx=0.7, rely=0.3, anchor=tk.CENTER)
    
    nameentry = tk.Entry(master=root_tk)
    nameentry.place(relx=0.7, rely=0.4, anchor=tk.CENTER)
    buttons.append(nameentry)
    savebut = tk.Button(master=root_tk, command=lambda: save_combine_data_set(nameentry.get(), checks, aug), text=f"Save to file")
    savebut.place(relx=0.7, rely=0.6, anchor=tk.CENTER)
    buttons.append(savebut)


def save_combine_data_set(name, datasets, augbool):
    infile = listdir("audio_processing/Train_Data/")

    stored = list()

    toaugment = augbool.get()

    for i in infile:
        if i[len(i)-9:]=="label.npy":
            stored.append(f"{i[:-10]}")

    datatocombine = list()


    for i in range(len(stored)):
        if datasets[i].get() == 1:
            datatocombine.append(stored[i])

    with_raw = True


    if not path.exists(f"audio_processing/Train_Data/{datatocombine[0]}_raw.npy"):
        with_raw = False
        print(f"{datatocombine[0]}_raw.npy does not exist")
    

    if with_raw:
        dataraw = np.load(f"audio_processing/Train_Data/{datatocombine[0]}_raw.npy")
    datamfcc = np.load(f"audio_processing/Train_Data/{datatocombine[0]}_mfcc.npy")
    datalabel = np.load(f"audio_processing/Train_Data/{datatocombine[0]}_label.npy")

    

    for i in datatocombine[1:]:
        if not path.exists(f"audio_processing/Train_Data/{i}_raw.npy"):
            with_raw = False
        if with_raw:
            dataraw = np.append(dataraw, np.load(f"audio_processing/Train_Data/{i}_raw.npy"), axis=0)
        datamfcc = np.append(datamfcc, np.load(f"audio_processing/Train_Data/{i}_mfcc.npy"), axis=0)
        datalabel = np.append(datalabel, np.load(f"audio_processing/Train_Data/{i}_label.npy"), axis=0)

    if with_raw and toaugment:

        size = dataraw.shape[0]
        print(f"{np.max(dataraw)} and {np.max(np.abs(dataraw))}")
        noice = np.random.normal(0, 0.005, (size, 32500))
        noice2 = np.random.normal(0, 0.01, (size, 32500))
        dataraw2 = dataraw+noice
        dataraw3 = dataraw+noice2
        print(f"{dataraw2.shape} and {dataraw.shape} and {dataraw2[0]} and {dataraw[0]}")
        sd.play(dataraw2[1])

        mc = mfcc_dataprocessor(44100)

        datamfcc2 = mc.mfcc_process(dataraw2)
        datamfcc3 = mc.mfcc_process(dataraw3)
    
        dataraw = np.append(np.append(dataraw, dataraw2, axis=0), dataraw3, axis=0)
        datamfcc = np.append(np.append(datamfcc, datamfcc2, axis=0), datamfcc3, axis=0)
        datalabel = np.append(np.append(datalabel, datalabel, axis=0), datalabel, axis=0)

    if with_raw:
        rand_data_raw = np.array([np.zeros(32500)], ndmin = 2)
    rand_data_mfcc = np.array([np.zeros((11, 70))], ndmin=3)
    rand_data_label = np.array([np.zeros(9)], ndmin=2)

    for i in range(datamfcc.shape[0]):
        rand = random.randint(0, datamfcc.shape[0]-1)
        if with_raw:
            rand_data_raw = np.append(rand_data_raw, [dataraw[rand]], axis=0)
        rand_data_mfcc = np.append(rand_data_mfcc, [datamfcc[rand]], axis=0)
        rand_data_label = np.append(rand_data_label, [datalabel[rand]], axis=0)
        
        if with_raw:
            dataraw = np.delete(dataraw, rand, axis=0)
        datamfcc = np.delete(datamfcc, rand, axis=0)
        datalabel = np.delete(datalabel, rand, axis=0)
    
    if with_raw:
        rand_data_raw = rand_data_raw[1:]    
    rand_data_mfcc = rand_data_mfcc[1:]
    rand_data_label = rand_data_label[1:]


    if with_raw:
        print(f"{rand_data_raw.shape}")
    print(f"{rand_data_mfcc.shape} and {rand_data_label.shape}")



    if with_raw:
        np.save(f"audio_processing/Train_Data/{name}_raw.npy", rand_data_raw)
    np.save(f"audio_processing/Train_Data/{name}_mfcc.npy", rand_data_mfcc)
    np.save(f"audio_processing/Train_Data/{name}_label.npy", rand_data_label)

    set_start_buttons()


set_start_buttons()
root_tk.mainloop()