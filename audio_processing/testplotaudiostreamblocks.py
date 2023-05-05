import numpy as np
import WaveInterface
import time
import sounddevice as sd
import matplotlib.pyplot as plt


BLOCKLENGTH = 44100


def LowpassFilter(fc, r: int):
    assert fc < r / 2, "violation of sampling theorem"
    LengthOfFilterInSamples = 501
    n = np.arange(LengthOfFilterInSamples) - np.floor(LengthOfFilterInSamples / 2)
    t = n / r
    h_LP = np.sinc(2 * fc * t)
    w = 0.5 * (1 + np.cos(np.pi * n / LengthOfFilterInSamples))  # Hann-Window
    h = w * h_LP
    return h


def ApplyFCenter(h, f_center, r: int):
    t = np.arange(h.shape[0]) / r
    t -= np.mean(t)
    return h * np.cos(2 * np.pi * f_center * t)


def BandpassFilter(f_low, f_high, r: int):
    assert f_high > f_low, "lower frequency must be lower than higher frequency"
    assert f_high < r / 2, "violation of sampling theorem"
    h_LP = LowpassFilter((f_high - f_low) / 2, r)
    f_center = (f_low + f_high) / 2
    return ApplyFCenter(h_LP, f_center, r)


safe1 = np.array([], dtype="float64")


# function used from stream to get the sound input
def callback(indata, frame_count, time_info, status):
    global safe1
    safe1 = np.append(safe1, indata)


# setting up the stream
# device 6 for roman, and device 1 for jonas, depends on the system
# check with sounddevice.query_device() for sounddevices and use the integer
# in front of the desired device as device in the stream function
stream = sd.InputStream(channels=1, samplerate=44100, callback=callback, device=6)


x = np.arange(44100)/44100
h_BP_NB = BandpassFilter(300, 3400, 44100)


# not working yet, im on it - roman :D
with stream:

    plt.ion()
    fig1 = plt.figure(1)
    plt1 = fig1.add_subplot(211)
    plt2 = fig1.add_subplot(212)
    plt1.axis([0, 1, -1, 1])
    plt2.axis([0, 1, -1, 1])
    linenofilter, = plt1.plot(x, np.zeros(44100), 'b-')
    linewithfiler, = plt2.plot(x, np.zeros(44100), 'b-')

    while True:
        while(len(safe1)<BLOCKLENGTH):
            time.sleep(0.1)
        workblock = safe1[:BLOCKLENGTH]
        safe1 = safe1[BLOCKLENGTH:]
        filteredworkblock = np.convolve(workblock, h_BP_NB)
        #[startx, endx] = [0, 1]
        #print("bin da")
        if not np.max(np.abs(workblock))/0.99 < 0:
            workblock /= np.max(np.abs(workblock))/0.99
            filteredworkblock /= np.max(np.abs(filteredworkblock))/0.99

        print(np.max(np.abs(workblock))/0.99)
        linenofilter.set_ydata(workblock)
        linewithfiler.set_ydata(filteredworkblock[:-500])
        fig1.canvas.draw()
        fig1.canvas.flush_events()
        
        # why the f is the filter generating 500 values, independent of the length of the input
        #plt.plot(x, filteredworkblock[:-500])
        #plt.show()
        #plt.axis([startx, endx])