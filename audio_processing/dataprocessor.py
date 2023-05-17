import numpy as np

class dataprocessor:

    raw: np.array
    filtered: np.array
    gained: np.array
    voiceactivity: np.array
    wordmarkers: np.array
    words: np.array
    wordlist: np.array

    samplerate: int
    targetlvl: int
    voicethreshhold: int
    lengthvoiceactivity: int

    _filter: int


    def LowpassFilter(self, fc, r: int):
        assert fc < r / 2, "violation of sampling theorem"
        LengthOfFilterInSamples = 501
        n = np.arange(LengthOfFilterInSamples) - np.floor(LengthOfFilterInSamples / 2)
        t = n / r
        h_LP = np.sinc(2 * fc * t)
        w = 0.5 * (1 + np.cos(np.pi * n / LengthOfFilterInSamples))  # Hann-Window
        h = w * h_LP
        return h


    def ApplyFCenter(self, h, f_center, r: int):
        t = np.arange(h.shape[0]) / r
        t -= np.mean(t)
        return h * np.cos(2 * np.pi * f_center * t)


    def BandpassFilter(self, f_low, f_high, r: int):
        assert f_high > f_low, "lower frequency must be lower than higher frequency"
        assert f_high < r / 2, "violation of sampling theorem"
        h_LP = self.LowpassFilter((f_high - f_low) / 2, r)
        f_center = (f_low + f_high) / 2
        return self.ApplyFCenter(h_LP, f_center, r)
    

    # setting up the processor
    def __init__(self, samplerate: int, targetlvl: int, voicethreshhold:int, lengthvoiceactivity: int):

        self.samplerate = samplerate
        self.targetlvl = targetlvl
        self.voicethreshhold = voicethreshhold
        self.lengthvoiceactivity = lengthvoiceactivity
        self._filter = self.BandpassFilter(300, 3400, self.samplerate)

    
    # processing data
    def processdata(self, data: np.array):

        self.raw = np.array(data)
        self.filtered = np.array(np.convolve(self.raw, self._filter)[250:-250])

        A = 1
        block_mean = np.max([np.mean(self.filtered), 0])
        block_variance = np.max([np.mean(self.filtered**2), 0.00016])
        print(f"mean {block_mean} and variance {block_variance}")

        self.gained = np.array(np.sqrt(A*A/2*(10**(self.targetlvl / 10))/(block_variance - block_mean**2))*(self.filtered-block_mean))

        lengthvoiceactivityinsamples = int(self.lengthvoiceactivity*self.samplerate/1000)

        numberofblocks = self.gained.shape[0] // lengthvoiceactivityinsamples

        self.voiceactivity = np.array(np.zeros((numberofblocks)))
        self.wordmarkers = np.array(np.zeros(self.raw.shape[0]))

        wordfactortodata = self.raw.shape[0]//numberofblocks

        for n in range(numberofblocks):
            idx1 = n*lengthvoiceactivityinsamples
            idx2 = idx1+lengthvoiceactivityinsamples
            P = np.mean(self.gained[idx1:idx2]**2)
            self.voiceactivity[n] = 10*np.log10(2*P/(A**2))
            if self.voiceactivity[n] > self.voicethreshhold:
                self.wordmarkers[n*wordfactortodata:(n+1)*wordfactortodata] += 1
        
        self.flatten_wordmarker(5000)
        self.extend_words(5000)
        self.flatten_wordmarker(5000)

        self.words = np.array(self.gained*self.wordmarkers)

        #self.wordlist = np.array([[0, 0]], ndmin=2)

        #self.wordlist = np.append(self.wordlist, np.array([[0, 0]]), axis=0)

        self.wordlist = list()

        #print(self.wordlist)

        wordbool = False
        wordtemp = np.array([])
        for i in range(self.wordmarkers.shape[0]):
            if self.wordmarkers[i]!=0:
                wordbool =  True
                wordtemp = np.append(wordtemp, self.words[i])
            else:
                if wordbool:
                    #print(f"{self.wordlist} and {wordtemp}")
                    #self.wordlist = np.append(self.wordlist, wordtemp, axis=0)
                    self.wordlist.append(wordtemp)
                    wordtemp = np.array([])
                    print(f"word detected nr{len(self.wordlist)}")
                wordbool = False
        
        return np.array([[self.wordlist], np.array([[self.raw, self.voiceactivity], [self.filtered, self.wordmarkers], [self.gained], [self.words]], dtype=object)], dtype=object)
    
    
    # gives back the shape of the returned values
    def get_shape_info(self):
        return np.array([2, 2, 1, 1])
    

    def flatten_wordmarker(self, length: int):

        counter0 = 0
        counter1 = 0

        for i in range(self.wordmarkers.shape[0]):
            if self.wordmarkers[i]==1:
                counter1 += 1
                if (counter0 < length) and (counter0 > 0):
                    self.wordmarkers[(i-counter0):i] = 1
                    print(f"cutted a pause out")
                counter0 = 0
            else:
                counter0 += 1
                if (counter1 < length) and (counter1 > 0):
                    #self.wordmarkers[(i-counter1):i] = 0
                    print(f"cutted a word out")
                counter1 = 0
    

    def extend_words(self, length: int):

        prev = 0

        for i in range(self.wordmarkers.shape[0]):
            activ = self.wordmarkers[i]
            if activ > prev:
                self.wordmarkers[i-length:i] = 1
            if activ < prev:
                self.wordmarkers[i:i+length] = 1
        prev = activ