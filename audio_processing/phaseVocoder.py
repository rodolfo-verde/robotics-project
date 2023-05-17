import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.linalg import solve_toeplitz
import sys
import WaveInterface


class CTransformPitchshift(object):
    
    def __init__(self):
        self.__ShiftByFactor = -1.0
        self.__XShape0 = 0
        
    def __EvaluateTransformMatrix(self, SamplingRate):
        FFTLen = 2*(self.__XShape0-1)
        Delta_f = SamplingRate / FFTLen
        self.__T = np.zeros((self.__XShape0, self.__XShape0))
        for k in range(self.__XShape0):
            TargetFrequency = Delta_f * k
            SourceFrequency = TargetFrequency / self.__ShiftByFactor
            if SourceFrequency < (SamplingRate / 2):
                # linear interpolation for neighbouring frequency bins
                n = int(SourceFrequency / Delta_f)
                self.__T[k, n + 1] = (SourceFrequency - n * Delta_f) / Delta_f
                self.__T[k, n + 0] = 1 - self.__T[k, n + 1]

    def TransformPitchshift(self, X, SamplingRate, ShiftByFactor):
        assert ShiftByFactor > 0, 'only positive shift factors for pitch are reasonable'
        SomethingChanged = np.abs(ShiftByFactor - self.__ShiftByFactor) > 1e-3
        SomethingChanged = SomethingChanged or (np.abs(self.__XShape0 - X.shape[0]) > 1e-3)
        if SomethingChanged:
            self.__ShiftByFactor = ShiftByFactor
            self.__XShape0 = X.shape[0]
            self.__EvaluateTransformMatrix(SamplingRate)
        return np.matmul(self.__T, X)
    
def Bark2KiloHertz(b):
    return 1.96 * (b + 0.53) / (26.28 - b)

def MelFilterBank(NumberOfCenterFrequenciesPerBark, NumberOfInputFrequencyBins, SamplingRate):
    FFTLen = 2 * NumberOfInputFrequencyBins - 2
    Deltaf = SamplingRate / FFTLen
    f = np.arange(NumberOfInputFrequencyBins) * Deltaf
    CutoffFrequenciesInBark = np.arange(24*NumberOfCenterFrequenciesPerBark+1) / NumberOfCenterFrequenciesPerBark
    CutoffFrequenciesInHertz = Bark2KiloHertz(CutoffFrequenciesInBark) * 1000
    CenterFrequenciesInHertz = np.diff(CutoffFrequenciesInHertz) / 2
    CenterFrequenciesInHertz += CutoffFrequenciesInHertz[0:CutoffFrequenciesInHertz.shape[0]-1]
    T_Hertz2Bark = np.zeros((CenterFrequenciesInHertz.shape[0], NumberOfInputFrequencyBins))
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
    return T_Hertz2Bark, CenterFrequenciesInHertz

class CTransformSpectralEnvelope(object):
    
    def __init__(self):
        self.__NumberOfBandsPerBark = -1.0
        self.__XShape0 = 0
    
    def __EvaluateTransformMatrix(self, SamplingRate):
        T_Hertz2Bark, CenterFrequenciesInHertz = MelFilterBank(NumberOfCenterFrequenciesPerBark = self.__NumberOfBandsPerBark, NumberOfInputFrequencyBins = self.__XShape0, SamplingRate = SamplingRate)
        D = np.eye(T_Hertz2Bark.shape[0])
        for d in range(D.shape[0]):
            D[d, d] = 1/np.sum(T_Hertz2Bark[d, :]**2)
        self.__T = np.matmul(np.transpose(T_Hertz2Bark), np.matmul(D, T_Hertz2Bark))
    
    def TransformSpectralEnvelope(self, X, SamplingRate, NumberOfBandsPerBark):
        assert NumberOfBandsPerBark > 0, 'only positive number of bands per Bark are reasonable'
        SomethingChanged = np.abs(NumberOfBandsPerBark - self.__NumberOfBandsPerBark) > 1e-3
        SomethingChanged = SomethingChanged or (np.abs(self.__XShape0 - X.shape[0]) > 1e-3)
        if SomethingChanged:
            self.__NumberOfBandsPerBark = NumberOfBandsPerBark
            self.__XShape0 = X.shape[0]
            self.__EvaluateTransformMatrix(SamplingRate)
        return np.matmul(self.__T, X)    

class CSpectralMasking(object):
    
    def __init__(self):
        self.__ZeroCounter = 0
        self.__AllCounter = 0
        #self.__ErrorCounter = 0
        
    def ProcessNewSpectrum(self, X):
        max_ind = argrelextrema(X, np.greater, axis = 0)
        Y = np.zeros(X.shape)
        Y[max_ind] = X[max_ind]
        self.__ZeroCounter += np.sum(Y < 1e-10)
        self.__AllCounter += Y.shape[0]
        #self.__ErrorCounter += np.sum(np.diff(max_ind) < 1.5) 
        return Y
    
    def GetSparsenessFactor(self):
        #print(self.__ErrorCounter)
        return self.__ZeroCounter / self.__AllCounter
class CTemporalMasking(object):
    
    def __init__(self):
        self.__X1 = 0.0
    
    def __ApplyQuantization(self, X):
        y = np.maximum(X, 0.0)
        return y

    def ProcessNewSpectrum(self, X, a = 0.5):        
        Y = self.__ApplyQuantization(X - self.__X1) + self.__X1
        self.__X1 = a * Y
        return Y
    
# Creating Vocoder structure

import RTISI

# define Parameters for the Vocoder
NumberOfBandsPerBark = 1.0 # 1-24
HopSizeInMilliseconds = 25 
OverlappingFactor = 6 # must be an integer

TemporalMaskingDamping = 0.5 # 0-1
TimeStretchFactor = 1.0 
PitchshiftFactor = 1.0
MinimumFrequencyResolutionInHertz = 20

# initializing the necessary classes
ATemporalMasking = CTemporalMasking()
ATransformSpectralEnvelope = CTransformSpectralEnvelope()
ATransformPitchshift = CTransformPitchshift()
ASpectralMasking = CSpectralMasking()

# read the input signal
x, Fs, bits = WaveInterface.ReadWave(r'C:\Users\Jonas\robotics-project\audio_processing\noise_raw_commands_test.wav') # read the input signal choose one of the test files
print('Fs = ', Fs)
print('bits = ', bits)
print('duration of input signal = ', x.shape[0] / Fs, ' s')

# derived parameters and classes
hs = int(Fs * HopSizeInMilliseconds / 1000)
ws = OverlappingFactor*hs
w_Rectangular = np.ones((ws))
w_Hann = 0.5 - 0.5 * np.cos(2*np.pi*(np.arange(ws)+0.5)/ws)
FFTLen = int(Fs / MinimumFrequencyResolutionInHertz)
if FFTLen < ws: FFTLen = ws
FFTLen = 2*int(2**np.ceil(np.log2(FFTLen)))
assert FFTLen > 2*ws, 'the algorithm assumes at least a zero padding of factor 2'

TargetHopSize = int(hs * TimeStretchFactor)

ARTISI = RTISI.CRTISI(TargetHopSize, w_Rectangular, w_Hann)

y = np.zeros((int(x.shape[0] * TimeStretchFactor * 1.0)))
for NumberOfBlocks in range(int((x.shape[0] - ws) / hs)):
    # block analysis
    idx1 = NumberOfBlocks * hs
    idx2 = idx1 + ws
    BlockAnalysis = x[idx1:idx2] * w_Hann
    SpectrumAnalysis = np.abs(np.fft.rfft(BlockAnalysis, n = FFTLen))
    
    # spectral modifications
    SpectrumSynthesis = ASpectralMasking.ProcessNewSpectrum(SpectrumAnalysis)
    SpectrumSynthesis = ATransformSpectralEnvelope.TransformSpectralEnvelope(SpectrumSynthesis, SamplingRate = Fs, NumberOfBandsPerBark = NumberOfBandsPerBark)
    SpectrumSynthesis = ATemporalMasking.ProcessNewSpectrum(SpectrumSynthesis, TemporalMaskingDamping)
    SpectrumSynthesis = ATransformPitchshift.TransformPitchshift(SpectrumSynthesis, SamplingRate = Fs, ShiftByFactor = PitchshiftFactor)
    
    # evaluate a good phase for the next audio block
    BlockSynthesis = ARTISI.ProcessNewColumnOfSpectrogram(SpectrumSynthesis)
    
    # overlap add at the target position due to timestretch
    idx1 = NumberOfBlocks * TargetHopSize
    idx2 = idx1 + ws
    y[idx1:idx2] += BlockSynthesis

# saving the results
print('Sparseness: ', ASpectralMasking.GetSparsenessFactor())
print('duration of output signal = ', y.shape[0] / Fs, ' s')
y = y *0.9 / np.amax(np.abs(y))
WaveInterface.WriteWave(y, Fs, bits, r'audio_processing\output_vocoder.wav')
