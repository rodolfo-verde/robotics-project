import numpy as np
import WaveInterface
import time
import sounddevice as sd
import matplotlib.pyplot as plt

INPUTDEVICE = 6

devices = sd.query_devices()
print(devices)

for i in devices:
    print(i)
    if i['name'] == 'default':
        print("HIT")
        INPUTDEVICE = i['index']

print(INPUTDEVICE)
# if you want to set the inputdevice manually just remove the "#" infront of the next line and choose your inputdevice
# the list of all availavle inputdevices will be in the console after running once
#INPUTDEVICE = 5 # sets inputdevice for the stream
PLOTDURATION = 3 # plotduration in seconds
BLOCKSPERSECOND = 5 # number of blocks processed in one second, sets the blocklength for that
SAMPLERATE = 44100 # samplerate from the input
BLOCKLENGTH = SAMPLERATE//BLOCKSPERSECOND # sets the blocklength, dependend on blockspersecond
LENGTHOFVOICEACTIVITYBLOCK = 10 # sets the length in "ms" of the blocklength for the voice activity
VOICEBLOCKSPERPLOT = 1000//LENGTHOFVOICEACTIVITYBLOCK*PLOTDURATION # sets the number of voiceblocks per second, to plot activity
VOICEBLOCKSPERSECOND = 1000//LENGTHOFVOICEACTIVITYBLOCK
VOICEBLOCKSPERBLOCK = VOICEBLOCKSPERSECOND//BLOCKSPERSECOND
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


#safe1 stores the input from the stream to be processed later
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


# setting up all the stuff to plot the signals
x = np.arange(PLOTLENGTH)/SAMPLERATE
x_voice = np.arange(VOICEBLOCKSPERPLOT)/VOICEBLOCKSPERSECOND
h_BP_NB = BandpassFilter(300, 3400, SAMPLERATE)
print(len(x))
print(len(x_voice))

plt.ion()
fig1 = plt.figure(1)
plt1 = fig1.add_subplot(411)
plt2 = fig1.add_subplot(412)
plt3 = fig1.add_subplot(413)
plt4 = fig1.add_subplot(414)
plt1v = plt1.twinx()
plt2w = plt2.twinx()
plt3w = plt3.twinx()

plt1.set_ylabel("raw data")
plt2.set_ylabel("filtered data")
plt3.set_ylabel("automatic gained data")
plt1v.set_ylabel("voice activity")
plt2w.set_ylabel("worddetection")
plt4.set_ylabel("words")
plt3w.set_ylabel("worddetection")

plt1.axis([0, PLOTDURATION, -2, 2])
plt2.axis([0, PLOTDURATION, -2, 2])
plt3.axis([0, PLOTDURATION, -1, 1])
plt4.axis([0, PLOTDURATION, -1, 1])
plt1v.axis([0, PLOTDURATION, -90, -10])
plt2w.axis([0, PLOTDURATION, -1, 2])
plt3w.axis([0, PLOTDURATION, -1, 2])

printblockraw = np.zeros(PLOTLENGTH)
printblockfiltered = np.zeros(PLOTLENGTH)
printblockgained = np.zeros(PLOTLENGTH)
printvoiceactivity = np.zeros(VOICEBLOCKSPERPLOT)
printworddetection = np.zeros(VOICEBLOCKSPERPLOT)
printwords = np.zeros(PLOTLENGTH)
printworddetection2 = np.zeros(PLOTLENGTH)

linenofilter, = plt1.plot(x, printblockraw, 'b-')
linewithfiler, = plt2.plot(x, printblockfiltered, 'b-')
linewithgain, = plt3.plot(x, printblockgained, 'b-')
lineactivity, = plt1v.plot(x_voice, printvoiceactivity, 'r-')
lineworddetection, = plt2w.plot(x_voice, printworddetection, 'r-')
linewords, = plt4.plot(x, printwords, 'b-')
lineworddetection2, = plt3w.plot(x, printworddetection2, 'r-')


def printval(blockraw: np.array, blockfiltered: np.array, blockgained: np.array, voiceactivity: np.array, worddetection:np.array, worddetection2: np.array, words: np.array):
    global printblockraw
    global printblockfiltered
    global printblockgained
    global printvoiceactivity
    global printworddetection
    global printwords
    global printworddetection2

    printblockraw = np.append(printblockraw[len(blockraw):], blockraw)
    printblockfiltered = np.append(printblockfiltered[len(blockfiltered):], blockfiltered)
    printblockgained = np.append(printblockgained[len(blockgained):], blockgained)
    printvoiceactivity = np.append(printvoiceactivity[len(voiceactivity):], voiceactivity)
    printworddetection = np.append(printworddetection[len(worddetection):], worddetection)
    printwords = np.append(printwords[len(words):], words)
    printworddetection2 = np.append(printworddetection2[len(worddetection2):], worddetection2)

    linenofilter.set_ydata(printblockraw)
    linewithfiler.set_ydata(printblockfiltered)
    linewithgain.set_ydata(printblockgained)
    lineactivity.set_ydata(printvoiceactivity)
    lineworddetection.set_ydata(printworddetection)
    linewords.set_ydata(printwords)
    lineworddetection2.set_ydata(printworddetection2)

    fig1.canvas.draw()
    fig1.canvas.flush_events()

def processdata(workblock: np.array) -> list[np.array, np.array, np.array]:

    filteredworkblock = np.convolve(workblock, h_BP_NB)[:-500]
    #print("bin da")

    """
    if not np.max(np.abs(filteredworkblock))/0.99 < 0:
        workblock /= np.max(np.abs(workblock))/0.99
        filteredworkblock /= np.max(np.abs(filteredworkblock))/0.99
    """

    # maximum magnitude of a typical soundsignal after AD-Conversion
    A = 2
    targetlevel = -30
    """gainedworkblock = np.array([])
    c = 0.9
    x_mean = 0.0
    x_variance = 0.0
    blocksizeinms = LENGTHOFVOICEACTIVITYBLOCK
    blocksizeinsamples = int(blocksizeinms*SAMPLERATE/1000)
    numberofblocks = filteredworkblock.shape[0] // blocksizeinsamples
    L = np.zeros((numberofblocks))
    for n in range(numberofblocks):
        idx1 = n * blocksizeinsamples
        idx2 = idx1 + blocksizeinsamples
        x_Block = filteredworkblock[idx1:idx2]
        x_mean     = c * x_mean     + (1-c) * np.mean(x_Block)
        x_variance = c * x_variance + (1-c) * np.mean(x_Block**2)
        a = x_mean
        b = np.sqrt(A*A/2*(10**(targetlevel / 10))/(x_variance - x_mean**2))
        print("--------------------------")
        print(b)
        print(x_Block)
        print(a)
        gainedworkblock = np.append(gainedworkblock, b * (x_Block - a))"""

    #levelbeforegain = 10*np.log10(2*np.mean(filteredworkblock**2) / (A**2))
    
    #print("-----------------printing mean and variance-------------------")
    #print(np.mean(filteredworkblock))
    #print(np.mean(filteredworkblock**2))
    #print("--------------------------------------------------------------")

    block_mean = np.max([np.mean(filteredworkblock), 0.00000000000000001])
    block_variance = np.max([np.mean(filteredworkblock**2), 0.00000000000001])
    
    #print("------------blockmean and blockvariance-------")
    #print(block_mean)
    #print(block_variance)
    #print("----------------------------------------------")
    if block_variance>0.00001:
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


# needs an array from the voiceactivity and the the audiostream from which you want the words (raw/filtered/gained)
# returns two arrays, the first with the start and endpoint of words
# the first value is the start of the first words, the second the end the third the start of the second word an so on
# the second array contains zeros and ones, zero means that at this point there is no word, and one means that there is a word
# mostly for plotting the wordareas
def worddetection(voiceactivity: np.array, audioinput: np.array) -> list([np.array, np.array, np.array]):
    isword = False
    lenstuff = voiceactivity.shape[0] #BLOCKLENGTH // (LENGTHOFVOICEACTIVITYBLOCK*SAMPLERATE//1000)
    wordfactortodata = BLOCKLENGTH//VOICEBLOCKSPERBLOCK
    worddetection = np.array([])
    
    #print("-------------------------------------------")
    #print(voiceactivity.shape[0])
    #print(lenstuff)

    # so apperently this loop is shit and not working properly :/
    for i in range(lenstuff):
        if voiceactivity[i] > -45:
            #print("hit")
            if not isword:
                worddetection = np.append(worddetection, int(i))
                isword = True
        else:
            if isword:
                worddetection = np.append(worddetection, int(i))
                isword = False
    
    wordmarkers = np.zeros(lenstuff)
    #print(words)
    
    wordmarkers2 = np.zeros(audioinput.shape[0])

    for i in range(lenstuff):
        if voiceactivity[i]>-45:
            wordmarkers2[i*wordfactortodata:(i+1)*wordfactortodata-1] +=1

    words = np.array([])
    #print(voiceactivity.shape[0])
    #print(audioinput.shape[0])
    #print(SAMPLERATE//VOICEBLOCKSPERSECOND)
    #print(voiceactivity.shape[0]*SAMPLERATE//VOICEBLOCKSPERSECOND)
    #print("-------------------------------")
    for i in range(len(worddetection)//2):
        if i == len(worddetection)-1:
            wordmarkers[i] += 1
        #printprint(np.mean(filteredworkblock)) (int(a))
        #print(b)
        [startx, endx] = [int(worddetection[i]), int(worddetection[i+1])]
        #print(f"got a word with length {endx-startx}")
        wordmarkers[startx:endx] += 1
        words = np.append(words, audioinput[startx*wordfactortodata:(endx*wordfactortodata)])

    return [words, worddetection, wordmarkers, wordmarkers2]


# kinda working, only for testing stuff - roman :D
# the automatic gain gaines to much noice, so the voice activity always detects
# this decreases with bigger blocksizes
# this only shows that i have to work on gaining properly (in the blocks) :D
# fixed it with having a minimum mean when gaining so that if noonetalks, it doesnt push the noice up
# the printing of the words in the fourth plot is not working properly i think, will take care of that later
# and i cant test it with the microphone yet, will annoy jonas with that soonish :)
with stream:

    while True:
        while(len(safe1)<BLOCKLENGTH):
            time.sleep(0.1)
        workblock = safe1[:BLOCKLENGTH]
        safe1 = safe1[BLOCKLENGTH:]

        [raw, filtered, gained] = processdata(workblock)
        voiceblocks = getvoiceactivity(gained)
        [words, wordstartsandends, wordblocks, wordblocks2] = worddetection(voiceblocks, gained)
        printval(raw, filtered, gained, voiceblocks, wordblocks, wordblocks2, words)
