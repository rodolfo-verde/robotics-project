import numpy as np

class dataprocessor:

    raw: np.array
    filtered: np.array
    gained: np.array
    voiceactivity: np.array
    wordmarkers: np.array
    convolved_wordmarkers: np.array
    words: np.array
    wordlist: np.array
    wordindeces: np.array

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
        block_variance = np.max([np.mean(self.filtered**2), 0.00016]) # 0.00016 is the variance of the raw_distance_commands_testt.wav
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
                self.wordmarkers[n*wordfactortodata:(n+1)*wordfactortodata] = 1

        testpattern = np.array([1, 0, -1])
        testpattern2 = np.array([1, -1])

        self.convolved_wordmarkers = np.convolve(self.wordmarkers, testpattern2, 'same')
        self.convolved_to_indeces()
        self.fix_convolved()
        self.flatten_convolved(5000)
        #self.extend_convolved(300)
        self.flatten_convolved(5000)

        print(self.convolved_wordmarkers)
        
        self.flatten_wordmarker(5000)
        #self.extend_words(2)
        self.flatten_wordmarker(5000)

        self.words = np.array(self.gained*self.wordmarkers)

        print("words are set")

        #self.wordlist = np.array([[0, 0]], ndmin=2)

        #self.wordlist = np.append(self.wordlist, np.array([[0, 0]]), axis=0)

        self.wordlist = list()

        #print(self.wordlist)

        wordstart = 0
        wordend = 0
        prev = 0
        print("entering word extraction")
        for i in range(self.wordmarkers.shape[0]):
            if self.wordmarkers[i]==1 and prev == 0:
                wordstart = i
                print(f"got a start at {i}")
            if self.convolved_wordmarkers[i] == 1:
                print(f"got wordstart via convolve at {i}")

            if self.wordmarkers[i]==0 and prev == 1:
                wordend = i
                self.wordlist.append(self.gained[wordstart:wordend])
                print(f"got a word at {i}")
            if self.convolved_wordmarkers[i] == -1:
                print(f"got wordend via convolve at {i}")
            #print(f"{self.wordmarkers[i]} and {prev}")
            prev = self.wordmarkers[i]
        print("words stored in array")
        print(len(self.wordlist))
        
        return np.array([[self.wordlist], np.array([[self.raw, self.voiceactivity], [self.filtered, self.wordmarkers], [self.gained, self.convolved_wordmarkers], [self.words]], dtype=object)], dtype=object)
    
    
    # gives back the shape of the returned values
    def get_shape_info(self):
        return np.array([2, 2, 1, 1])
    

    def convolved_to_indeces(self):

        words = np.array([[0, 0]], ndmin=2)

        start = 0
        for i in range(self.convolved_wordmarkers.shape[0]):
            active = self.convolved_wordmarkers[i]
            if active == 1:
                start = i
            if active == -1:
                words = np.append(words, [[start, i]], axis=0)
        
        print(words)

        self.wordindeces = words[1:]


    def expand_words(self, lengthof):
        for i in range(self.wordindeces.shape[0]):
            a = self.wordindeces[0]
            


    def fix_convolved(self):
        
        for i in range(self.convolved_wordmarkers.shape[0]):
            if self.convolved_wordmarkers[i] == 1:
                if self.convolved_wordmarkers[i+1] == 1:
                    self.convolved_wordmarkers[i+1] = 0
            if self.convolved_wordmarkers[i] == -1:
                if self.convolved_wordmarkers[i+1] == -1:
                    self.convolved_wordmarkers[i+1] = 0


    def flatten_convolved(self, length: int):
        print("entered flatten convolved")
        lastup = -(length+1)
        lastdown = -(length+1)
        
        for i in range(self.convolved_wordmarkers.shape[0]):
            if self.convolved_wordmarkers[i] == 1:
                lastup = i
                if i-lastdown < length:
                    self.convolved_wordmarkers[lastdown] = 0
                    self.convolved_wordmarkers[i] = 0
            
            if self.convolved_wordmarkers[i] == -1:
                lastdown = i
                if i-lastup < length:
                    self.convolved_wordmarkers[lastup] = 0
                    self.convolved_wordmarkers[i] = 0
    

    def extend_convolved(self, length: int):

        for i in range(self.wordmarkers.shape[0]):
            if self.wordmarkers[i] == 1:
                if length < i:
                    self.wordmarkers[0] = 1
                    self.wordmarkers[i] = 0
                self.wordmarkers[i-length] = 1
                self.wordmarkers[i] = 0
            
            if self.wordmarkers[self.wordmarkers.shape[0]-i-1] == -1:
                if length < i:
                    self.wordmarkers[self.wordmarkers.shape[0]-i-1] = -1
                    self.wordmarkers[self.wordmarkers.shape[0]-i-1] = 0
                self.wordmarkers[self.wordmarkers.shape[0]-i-1+length] = -1
                self.wordmarkers[self.wordmarkers.shape[0]-i-1] = 0
        


    def flatten_wordmarker(self, length: int):
        
        print("entered flatten")
        counter0 = 0
        counter1 = 0

        for i in range(self.wordmarkers.shape[0]):
            if self.wordmarkers[i]==1:
                counter1 += 1
                if (counter0 < length) and (counter0 > 0):
                    self.wordmarkers[(i-counter0):i] = 1
                    #print(f"cutted a pause out")
                counter0 = 0
            else:
                counter0 += 1
                if (counter1 < length) and (counter1 > 0):
                    self.wordmarkers[(i-counter1):i] = 0
                    #print(f"cutted a word out")
                counter1 = 0
        print("exiting flatten")
    

    def extend_words(self, length: int):

        prev = 0

        for i in range(self.wordmarkers.shape[0]):
            activ = self.wordmarkers[i]
            if activ ==1 and  prev==0:
                print("added front")
                self.wordmarkers[i-length:i] = 1
            if activ ==0 and prev==1:
                print("added end")
                self.wordmarkers[i:i+length] = 1
                i+=length
            prev = activ
        print("getting out of extend")