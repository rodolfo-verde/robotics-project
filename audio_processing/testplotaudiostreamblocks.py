import numpy as np
import WaveInterface
import time
import sounddevice as sd
import matplotlib.pyplot as plt

INPUTDEVICE = 6

devices = sd.query_devices()
print(devices)

for i in devices:
    #print(i)
    if i['name'] == 'default':
        print("HIT")
    #if i[1] == "default":
    #    INPUTDEVICE = i[0]

print(INPUTDEVICE)
#INPUTDEVICE = 5 # sets inputdevice for the stream
PLOTDURATION = 3 # plotduration in seconds
BLOCKSPERSECOND = 4 # number of blocks processed in one second, sets the blocklength for that
SAMPLERATE = 44100 # samplerate from the input
BLOCKLENGTH = SAMPLERATE//BLOCKSPERSECOND # sets the blocklength, dependend on blockspersecond
LENGTHOFVOICEACTIVITYBLOCK = 10 # sets the length in "ms" of the blocklength for the voice activity
VOICEBLOCKSPERPLOT = 1000//LENGTHOFVOICEACTIVITYBLOCK*PLOTDURATION # sets the number of voiceblocks per second, to plot activity
VOICEBLOCKSPERSECOND = 1000//LENGTHOFVOICEACTIVITYBLOCK
PLOTLENGTH = SAMPLERATE*PLOTDURATION # sets the length of the arrays to plot



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
stream = sd.InputStream(channels=1, samplerate=44100, callback=callback, device=INPUTDEVICE)


x = np.arange(PLOTLENGTH)/SAMPLERATE
x_voice = np.arange(VOICEBLOCKSPERPLOT)/VOICEBLOCKSPERSECOND
h_BP_NB = BandpassFilter(300, 3400, SAMPLERATE)
print(len(x))
print(len(x_voice))

plt.ion()
fig1 = plt.figure(1)
plt1 = fig1.add_subplot(311)
plt2 = fig1.add_subplot(312)
plt3 = fig1.add_subplot(313)
plt1v = plt1.twinx()
plt2w = plt2.twinx()

plt1.axis([0, PLOTDURATION, -2, 2])
plt2.axis([0, PLOTDURATION, -2, 2])
plt3.axis([0, PLOTDURATION, -2, 2])
plt1v.axis([0, PLOTDURATION, -90, -10])
plt2w.axis([0, PLOTDURATION, -1, 2])

printblockraw = np.zeros(PLOTLENGTH)
printblockfiltered = np.zeros(PLOTLENGTH)
printblockgained = np.zeros(PLOTLENGTH)
printvoiceactivity = np.zeros(VOICEBLOCKSPERPLOT)
printworddetection = np.zeros(VOICEBLOCKSPERPLOT)


linenofilter, = plt1.plot(x, printblockraw, 'b-')
linewithfiler, = plt2.plot(x, printblockfiltered, 'b-')
linewithgain, = plt3.plot(x, printblockgained, 'b-')
lineactivity, = plt1v.plot(x_voice, printvoiceactivity, 'r-')
linewords, = plt2w.plot(x_voice, printworddetection, 'r-')

tracker = 0
trackertime = time.time()

def printval(blockraw: np.array, blockfiltered: np.array, blockgained: np.array, voiceactivity: np.array, worddetection:np.array):
    global printblockraw
    global printblockfiltered
    global printblockgained
    global printvoiceactivity
    global printworddetection
    global tracker
    global trackertime

    printblockraw = np.append(printblockraw[len(blockraw):], blockraw)
    printblockfiltered = np.append(printblockfiltered[len(blockfiltered):], blockfiltered)
    printblockgained = np.append(printblockgained[len(blockgained):], blockgained)
    printvoiceactivity = np.append(printvoiceactivity[len(voiceactivity):], voiceactivity)
    printworddetection = np.append(printworddetection[len(worddetection):], worddetection)

    linenofilter.set_ydata(printblockraw)
    linewithfiler.set_ydata(printblockfiltered)
    linewithgain.set_ydata(printblockgained)
    lineactivity.set_ydata(printvoiceactivity)
    linewords.set_ydata(printworddetection)

    fig1.canvas.draw()
    fig1.canvas.flush_events()

def processdata(workblock: np.array) -> list[np.array, np.array, np.array]:

    filteredworkblock = np.convolve(workblock, h_BP_NB)[:-500]
    #[startx, endx] = [0, 1]
    #print("bin da")

    """
    if not np.max(np.abs(filteredworkblock))/0.99 < 0:
        workblock /= np.max(np.abs(workblock))/0.99
        filteredworkblock /= np.max(np.abs(filteredworkblock))/0.99
    """

    # maximum magnitude of a typical soundsignal after AD-Conversion
    A = 1
    targetlevel = -30


    #levelbeforegain = 10*np.log10(2*np.mean(filteredworkblock**2) / (A**2))
        
    block_mean = np.max([np.mean(filteredworkblock), 0.0001])
    block_variance = np.max([np.mean(filteredworkblock**2), 0.01])
    
    #print("------------blockmean and blockvariance-------")
    #print(block_mean)
    #print(block_variance)
    #print("----------------------------------------------")
    if block_variance>0.0001:
        gainedworkblock = np.sqrt(A*A/2*(10**(targetlevel / 10))/(block_variance - block_mean**2))*(filteredworkblock-block_mean)
    else:
        gainedworkblock = np.zeros(len(filteredworkblock))
    
    
    #print(gainedworkblock)

    #levelaftergain = 10*np.log10(2*np.mean(gainedworkblock**2) / (A**2))
    #print("------Levels befor and aftergain -------")
    #print(levelbeforegain)
    #print(levelaftergain)
    #print("----------------------------------------")

    return (workblock, filteredworkblock, gainedworkblock)


def getvoiceactivity(gainedworkblock: np.array) -> np.array:
    #print("started voice activity")
    A = 1
    blocksizeinms = LENGTHOFVOICEACTIVITYBLOCK
    blocksizeinsamples = int(blocksizeinms*SAMPLERATE/1000)
    numberofblocks = gainedworkblock.shape[0] // blocksizeinsamples
    L = np.zeros((numberofblocks))
    for n in range(numberofblocks):
        idx1 = n*blocksizeinsamples
        idx2 = idx1+blocksizeinsamples
        P = np.mean(gainedworkblock[idx1:idx2]**2)
        L[n] = 10*np.log10(2*P/(A**2))
        #print(L[n])

    #print(L.shape[0])
    #print(gainedworkblock.shape[0])

    #print("finished voice activity")

    return L


def worddetection(voiceactivity: np.array) -> list([np.array, np.array]):
    isword = False
    lenstuff = BLOCKLENGTH // (LENGTHOFVOICEACTIVITYBLOCK*SAMPLERATE//1000)
    words = np.array([])
    
    #print("-------------------------------------------")
    #print(voiceactivity.shape[0])
    #print(lenstuff)
    for i in range(lenstuff):
        if voiceactivity[i] > -45:
            #print("hit")
            if not isword:
                words = np.append(words, int(i))
                isword = True
        else:
            if isword:
                words = np.append(words, int(i))
                isword = False
    
    wordmarkers = np.zeros(lenstuff)
    #print(words)
    for i in range(len(words)//2):
        if i == len(words)-1:
            wordmarkers[i] += 1
        #print(int(a))
        #print(b)
        wordmarkers[int(words[i]):int(words[i+1])] += 1

    return [words, wordmarkers]


# kinda working, only for testint stuff - roman :D
# the automatic gain gaines to much noice, so the voice activity always detects
# this decreases with bigger blocksizes
# this only shows that i have to work on gaining properly (in the blocks) :D
with stream:

    
    """
    while(len(safe1)<SAMPLERATE):
        time.sleep(0.1)
    workblock = safe1[:BLOCKLENGTH]
    safe1 = safe1[BLOCKLENGTH:]
    [a, b, c] = processdata(workblock)
    printval(a, b, c)
    """

    while True:
        while(len(safe1)<BLOCKLENGTH):
            time.sleep(0.1)
        workblock = safe1[:BLOCKLENGTH]
        safe1 = safe1[BLOCKLENGTH:]

        [a, b, c] = processdata(workblock)
        voiceblocks = getvoiceactivity(c)
        [e, f] = worddetection(voiceblocks)
        printval(a, b, c, voiceblocks, f)

        """
        filteredworkblock = np.convolve(workblock, h_BP_NB)[:-500]
        #[startx, endx] = [0, 1]
        #print("bin da")

        if not np.max(np.abs(filteredworkblock))/0.99 < 0:
            workblock /= np.max(np.abs(workblock))/0.99
            filteredworkblock /= np.max(np.abs(filteredworkblock))/0.99

        # maximum magnitude of a typical soundsignal after AD-Conversion
        A = 2
        targetlevel = -30


        levelbeforegain = 10*np.log10(2*np.mean(filteredworkblock**2) / (A**2))
        
        block_mean = np.mean(filteredworkblock)
        block_variance = np.mean(filteredworkblock**2)

        gainedworkblock = np.sqrt(A*A/2*(10**(targetlevel / 10))/(block_variance - block_mean**2))*(filteredworkblock-block_mean)
        print(gainedworkblock)

        levelaftergain = 10*np.log10(2*np.mean(gainedworkblock**2) / (A**2))
        print("------Levels befor and aftergain -------")
        print(levelbeforegain)
        print(levelaftergain)
        print("----------------------------------------")

        print(np.max(np.abs(workblock))/0.99)
        linenofilter.set_ydata(workblock)
        linewithfiler.set_ydata(filteredworkblock)
        linewithgain.set_ydata(gainedworkblock)

        fig1.canvas.draw()
        fig1.canvas.flush_events()
        
        # why the f is the filter generating 500 values, independent of the length of the input
        #plt.plot(x, filteredworkblock[:-500])
        #plt.show()
        #plt.axis([startx, endx])
        """