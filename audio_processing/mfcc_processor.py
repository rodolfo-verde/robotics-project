import numpy as np


class mfcc_dataprocessor:


    Fs: int
    hs: int
    ws: int
    w: np.array
    FFTLen: int
    NyquistIndex: int
    T_Hertz2Bark: np.array
    T_DCT: np.array

    def __init__(self, Fs) -> None:
        # Defining parameters
        self.Fs = Fs
        HopsizeInMilliseconds = 10 # in milliseconds, Spiertz = 10 --> wordprocessor.py uses 25 I think
        self.hs = int(HopsizeInMilliseconds * self.Fs / 1000)
        self.ws = 4*self.hs
        self.w = np.hanning(self.ws)
        self.FFTLen = int(2**np.ceil(np.log2(self.ws)))
        self.NyquistIndex = self.FFTLen // 2 + 1
        f = np.arange(self.NyquistIndex) / self.FFTLen * Fs

        # Defining the Mel Filterbank
        def Bark2KiloHertz(b):
            return 1.96 * (b + 0.53) / (26.28 - b)

        CutoffFrequenciesInBark = np.arange(25)
        CutoffFrequenciesInHertz = Bark2KiloHertz(CutoffFrequenciesInBark) * 1000
        CenterFrequenciesInHertz = np.diff(CutoffFrequenciesInHertz) / 2
        CenterFrequenciesInHertz += CutoffFrequenciesInHertz[0:CutoffFrequenciesInHertz.shape[0]-1]

        self.T_Hertz2Bark = np.zeros((CenterFrequenciesInHertz.shape[0], self.NyquistIndex))
        for b in range(self.T_Hertz2Bark.shape[0]):
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
            self.T_Hertz2Bark[b, :] = v4 / (0.5*(f2 - f1))


        # Defining the DCT --> Discrete Cosine Transform
        K = self.T_Hertz2Bark.shape[0]
        self.T_DCT = np.zeros((K, K))
        for n in range(K):
            for k in range(K):
                self.T_DCT[k, n] = np.cos(np.pi * (n+0.5) * k / K)


    def mfcc_process(self, data: np.array) -> np.array:
        NumberOfColumns = int((data.shape[0] - self.ws) / self.hs + 1)
        NumberOfOutputFeatures = 12 
        MFCC = np.zeros((NumberOfOutputFeatures, NumberOfColumns))
        for col in range(NumberOfColumns):
            idx1 = col*self.hs
            idx2 = idx1 + self.ws
            y_block = data[idx1:idx2] * self.w
            Y = np.abs(np.fft.rfft(y_block, n = self.FFTLen))
            assert Y.shape[0] == self.NyquistIndex, 'wrong number of output indices of DFT'
            Y_mel = np.matmul(self.T_Hertz2Bark, Y)
            Y_log = np.log(1e10*Y_mel+1)
            Y_dct = np.matmul(self.T_DCT, Y_log)
            MFCC[:, col] = Y_dct[1:NumberOfOutputFeatures+1]
         # Liftering after MFCC
        NormalizationGain = np.zeros((MFCC.shape[0]))
        MFCC_normalized = np.copy(MFCC)
        # jede row durchgehen energie mitteln
        for r in range(NumberOfOutputFeatures):
            NormalizationGain[r] = np.sqrt(1/np.mean((MFCC[r,:]**2)))
            MFCC_normalized[r,:] = MFCC[r,:]*NormalizationGain[r]

        return MFCC_normalized


    def mfcc_process_multiple(self, data: np.array) -> np.array:
        leni = len(data)
        reti = np.zeros((leni, 12, 70))
        for i in range(leni-1):
            print(data[i])
            reti[i] = self.mfcc_process(data[i])
            print(reti[i].size)
        return reti