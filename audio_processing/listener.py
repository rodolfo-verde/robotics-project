import numpy as np
import WaveInterface
import time
import sounddevice as sd


def LowpassFilter(fc, r):
    assert fc < r / 2, "violation of sampling theorem"
    LengthOfFilterInSamples = 501
    n = np.arange(LengthOfFilterInSamples) - np.floor(LengthOfFilterInSamples / 2)
    t = n / r
    h_LP = np.sinc(2 * fc * t)
    w = 0.5 * (1 + np.cos(np.pi * n / LengthOfFilterInSamples))  # Hann-Window
    h = w * h_LP
    return h


def ApplyFCenter(h, f_center, r):
    t = np.arange(h.shape[0]) / r
    t -= np.mean(t)
    return h * np.cos(2 * np.pi * f_center * t)


def BandpassFilter(f_low, f_high, r):
    assert f_high > f_low, "lower frequency must be lower than higher frequency"
    assert f_high < r / 2, "violation of sampling theorem"
    h_LP = LowpassFilter((f_high - f_low) / 2, r)
    f_center = (f_low + f_high) / 2
    return ApplyFCenter(h_LP, f_center, r)


safe1 = np.array([], dtype="float64")


def callback(indata, frame_count, time_info, status):
    global safe1
    safe1 = np.append(safe1, indata)

# setting up the stream
stream = sd.InputStream(channels=1, samplerate=44100, callback=callback, device=1)


with stream:
    global safe2
    while len(safe1) < 100000:
        time.sleep(0.1)
    safe2 = safe1[:100000]


print("starting to safe")
print(len(safe2))
print(safe2.shape)
print(safe2)
h_BP_NB = BandpassFilter(300, 3400, 44100)
z_NB = np.convolve(safe2, h_BP_NB)
# /0.99 da es durch die syntax von dem /= mit dem anderen in eine Klammer kommt
z_NB /= np.amax(np.abs(z_NB)) / 0.99
safe2 /= np.amax(np.abs(safe2)) / 0.99
print(np.max(np.abs(z_NB)))
print(np.max(np.abs(safe2)))
print(len(safe2))
WaveInterface.WriteWave(safe2, 44100, 16, "TestNoFilter.wav")
WaveInterface.WriteWave(z_NB, 44100, 16, "TestWithFilter.wav")
print("safe finished")
