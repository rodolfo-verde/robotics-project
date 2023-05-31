import numpy as np
import matplotlib.pyplot as plt
import sys
import WaveInterface
import time

# Mel Frequency Cepstral Coefficients
#The standard audio feature for speech processing and word detection are the Mel Frequency Cepstral Coefficients (MFCC):
#The steps for evaluating the MFCC are:
# 1. Evaluate the spectrogram and apply step by step to each column the following algorithms:
# 2. Smooth the spectrum by applying the Mel filterbank. Typically  24 filters are used.
# 3. Apply the logarithm with an appropriate scaling:  𝑦=log(gain⋅𝑥+1).
# 4. Apply the DCT. Skip the zero-th coefficient, because it is mainly correlated to the gain of the input recording, which is negligible for audio processing. Additionally the highest coefficients from the DCT-output are also skipped. Usually,  12
#    coefficients are kept after DCT.
# 5. Optionally, the output of the DCT is multiplied with a set of weighting. This process is called Liftering. Liftering is currently not used in the MOPS.


y, Fs, bits = WaveInterface.ReadWave('audio_processing/noise_filter_commands_test.wav') # read the wave file --> we can choose the file and then create training data from it

start = time.time()
# Defining parameters
HopsizeInMilliseconds = 25 # in milliseconds, Spiertz = 10 --> wordprocessor.py uses 25 I think
hs = int(HopsizeInMilliseconds * Fs / 1000)
ws = 4*hs
w = np.hanning(ws)
FFTLen = int(2**np.ceil(np.log2(ws)))
NyquistIndex = FFTLen // 2 + 1
f = np.arange(NyquistIndex) / FFTLen * Fs

# Defining the Mel Filterbank
def Bark2KiloHertz(b):
    return 1.96 * (b + 0.53) / (26.28 - b)

CutoffFrequenciesInBark = np.arange(25)
CutoffFrequenciesInHertz = Bark2KiloHertz(CutoffFrequenciesInBark) * 1000
CenterFrequenciesInHertz = np.diff(CutoffFrequenciesInHertz) / 2
CenterFrequenciesInHertz += CutoffFrequenciesInHertz[0:CutoffFrequenciesInHertz.shape[0]-1]

T_Hertz2Bark = np.zeros((CenterFrequenciesInHertz.shape[0], NyquistIndex))
for b in range(T_Hertz2Bark.shape[0]):
    m1 = (1 - 1/np.sqrt(2)) / (CenterFrequenciesInHertz[b] - CutoffFrequenciesInHertz[b]) # first derivative of first line
    m2 = (1 - 1/np.sqrt(2)) / (CenterFrequenciesInHertz[b] - CutoffFrequenciesInHertz[b+1]) # first derivative of second line
    assert m1 > 0, 'm1 must be greater 0'
    assert m2 < 0, 'm2 must be smaller 0'
    b1 = 1 - m1 * CenterFrequenciesInHertz[b] # offset of first line
    b2 = 1 - m2 * CenterFrequenciesInHertz[b] # offset of second line
    assert b1 < 1/np.sqrt(2), 'b1 must be smaller than 1/sqrt(2)'
    assert b2 > 0, 'b2 must be greater 0'
    v1 = m1 * f + b1
    v2 = m2 * f + b2
    v3 = np.minimum(v1, v2)
    v4 = np.maximum(v3, 0.0)
    f1 = -b1 / m1 # zero crossing of the first line
    f2 = -b2 / m2 # zero crossing of the second line
    T_Hertz2Bark[b, :] = v4 / (0.5*(f2 - f1))


# Defining the DCT --> Discrete Cosine Transform
K = T_Hertz2Bark.shape[0]
T_DCT = np.zeros((K, K))
for n in range(K):
    for k in range(K):
        T_DCT[k, n] = np.cos(np.pi * (n+0.5) * k / K)


# Everything together
NumberOfColumns = int((y.shape[0] - ws) / hs + 1)
NumberOfOutputFeatures = 11 # number of output features, could vary depending on how many commands we have (Currently A1-C3 + Rex + Stopp = 11)
MFCC = np.zeros((NumberOfOutputFeatures, NumberOfColumns))
for col in range(NumberOfColumns):
    idx1 = col*hs
    idx2 = idx1 + ws
    y_block = y[idx1:idx2] * w
    Y = np.abs(np.fft.rfft(y_block, n = FFTLen))
    assert Y.shape[0] == NyquistIndex, 'wrong number of output indices of DFT'
    Y_mel = np.matmul(T_Hertz2Bark, Y)
    Y_log = np.log(1e10*Y_mel+1)
    Y_dct = np.matmul(T_DCT, Y_log)
    MFCC[:, col] = Y_dct[1:NumberOfOutputFeatures+1]
print('MFCC calculation took ' + str(time.time() - start) + ' seconds')    
# Liftering after MFCC
NormalizationGain = np.zeros((MFCC.shape[0]))
rows = 11
MFCC_normalized = np.copy(MFCC)
# jede row durchgehen energie mitteln
for r in range(rows):
    NormalizationGain[r] = np.sqrt(1/np.mean((MFCC[r,:]**2)))
    MFCC_normalized[r,:] = MFCC[r,:]*NormalizationGain[r]


# Visualization
# MFCC without normalization --> without liftering
plt.matshow(MFCC, interpolation='nearest', aspect='auto')
plt.xlabel('time [s]')
plt.ylabel('MFCC')
plt.show()

#MFCC with normalization --> with liftering
plt.matshow(MFCC_normalized, interpolation='nearest', aspect='auto')
plt.xlabel('time [s]')
plt.ylabel('MFCC (normalized)')
plt.show()

# Normalization gain
plt.plot(NormalizationGain)
plt.ylabel('Normalizaion gain')
plt.show()
print(NormalizationGain)



# MFCC_normalized = the data which is used for the neural network --> Classification
# We should save the data in a file and then use it for the neural network --> training data --> Huge array with all saved data --> then split into traint and test in CNN