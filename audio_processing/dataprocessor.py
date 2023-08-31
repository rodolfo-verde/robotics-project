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
    wordfrompastblock: np.array
    wordsblocks: np.array

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
        self.wordfrompastblock = np.array([])
        self.wordsblocks = np.array([])
        self._filter = self.BandpassFilter(300, 3400, self.samplerate)

    
    # processing data
    def processdata(self, data: np.array):

        # storing raw data and then filtering it
        self.raw = np.append(self.wordfrompastblock, data)
        #print(self.raw.shape)
        self.filtered = np.array(np.convolve(self.raw, self._filter)[250:-250])

        # applying the automatic gain
        A = 1
        block_mean = np.max([np.mean(self.filtered), 0.0002])
        block_variance = np.max([np.mean(self.filtered**2), 0.003016]) # 0.00016 is the variance of the raw_distance_commands_testt.wav
        #print(f"mean {block_mean} and variance {block_variance}")

        self.gained = np.array(np.sqrt(A*A/2*(10**(self.targetlvl / 10))/(block_variance - block_mean**2))*(self.filtered-block_mean))

        # seting up values for the voice detection
        lengthvoiceactivityinsamples = int(self.lengthvoiceactivity*self.samplerate/1000)

        numberofblocks = self.gained.shape[0] // lengthvoiceactivityinsamples

        self.voiceactivity = np.array(np.zeros((numberofblocks)))
        self.wordmarkers = np.array(np.zeros(self.raw.shape[0]))

        wordfactortodata = self.raw.shape[0]//numberofblocks

        # checking the array for voice activity and marking those indeces as words
        for n in range(numberofblocks):
            idx1 = n*lengthvoiceactivityinsamples
            idx2 = idx1+lengthvoiceactivityinsamples
            P = np.mean(self.gained[idx1:idx2]**2)
            self.voiceactivity[n] = 10*np.log10(2*P/(A**2))
            if self.voiceactivity[n] > self.voicethreshhold:
                self.wordmarkers[n*wordfactortodata:(n+1)*wordfactortodata] = 1

        # testpattern = np.array([1, 0, -1])
        testpattern2 = np.array([1, -1])

        # convolvng array of the marked words with a sobel-like filter to get start and end of words
        self.convolved_wordmarkers = np.convolve(self.wordmarkers, testpattern2, 'same')
        # checking the convolved array and convert them to a matrix of word indeces
        self.convolved_to_indeces()
        # expanding words
        self.expand_words(10000, 0)

        self.wordfrompastblock = self.words_in_blocks()

        #print(self.wordindeces/44100)

        # marking the array for words to plot them
        self.words = np.array(self.gained*self.wordmarkers)

        #print("words are set")

        self.wordlist = list()

        #print("entering word extraction")

        self.convolved_wordmarkers = np.zeros(self.wordmarkers.shape[0])

        # saving all words from the list of indeces
        for i in self.wordsblocks:
            self.wordlist.append(self.gained[i[0]:i[1]])
            self.convolved_wordmarkers[i[0]:i[1]] = 1

        #print("words stored in array")
        #print(len(self.wordlist))
        
        return np.array([[self.wordlist], np.array([[self.raw, self.voiceactivity], [self.filtered, self.wordmarkers], [self.gained, self.convolved_wordmarkers], [self.words]], dtype=object)], dtype=object)
    
    
    # gives back the shape of the returned values
    def get_shape_info(self):
        return np.array([2, 2, 2, 1])
    

    # cheking the convolved array for 1 and -1
    # 1 is the start of a word and -1 the end
    # then safing those data in an matrix
    def convolved_to_indeces(self):

        words = np.array([[0, 0]], ndmin=2)

        start = 0
        for i in range(self.convolved_wordmarkers.shape[0]):
            active = self.convolved_wordmarkers[i]
            if active == 1:
                start = i
            if active == -1:
                words = np.append(words, [[start, i]], axis=0)
        
        self.wordindeces = words[1:]


    # expands the wordmarkers so that the more silent wordparts are also in the markers
    # you can increase the range of combintion of words with the second value
    # works pretty good without so far
    def expand_words(self, lengthofexpand: int, lengthofcombine: int):

        if self.wordindeces.shape[0] == 0:
            return

        # expanding the first word backwards if it is not in the range
        # if it is in the range it will set the startword of the first word to 0
        if self.wordindeces[0][0] < lengthofcombine+lengthofexpand:
            self.wordindeces[0][0] = 0
        else:
            self.wordindeces[0][0] -= lengthofexpand
        
        # expands the wordend
        self.wordindeces[0][1] += lengthofexpand

        # an array to mark the words to remove, which will be combined
        toremove = np.array([], dtype=int)

        # a loop over the rest of words
        for i in range(1,self.wordindeces.shape[0]):
            
            # expanding the wordstart further to the front and if it crosses the wordend from the word in the front
            # it will combine them in the current word and mark the ther word for deletion
            if self.wordindeces[i][0] < self.wordindeces[i-1][1]+lengthofexpand+lengthofcombine:
                self.wordindeces[i][0] = self.wordindeces[i-1][0]
                toremove = np.append(toremove, i-1)
            else:
                self.wordindeces[i][0] -= lengthofexpand
            
            # expanding wordend
            self.wordindeces[i][1] += lengthofexpand
        
        """# checkin if the end of the last word is in range of the end of the data array, if so, it expands the word till the end
        if self.wordindeces[self.wordindeces.shape[0]-1][1]+lengthofcombine+lengthofexpand > self.wordmarkers.shape[0]:
            if self.wordfrompastblock.shape[0] < 22050:
                self.wordfrompastblock = self.raw[self.wordindeces[self.wordindeces.shape[0]-1][0]:]
                toremove = np.append(toremove, self.wordindeces.shape[0]-1)
                print("word pushed back one block _______________-------------------------__________________------------___---___--___--____-____________-----------")
            else:
                self.wordindeces[self.wordindeces.shape[0]-1][1] = self.wordmarkers.shape[0]-1
                self.wordfrompastblock = np.array([])
        else:
            self.wordfrompastblock = np.array([])"""

        
        # deletes marked words which would be double by now
        self.wordindeces = np.delete(self.wordindeces, toremove, axis=0)

        #print(self.wordindeces.shape[0])

    
    def words_in_blocks(self, blocklength: int = 32500):

        words = np.array([[0, 0]], ndmin=2)

        skip = False
        for i in range(self.wordindeces.shape[0]):

            if not skip: start = self.wordindeces[i][0]

            while start<self.wordindeces[i][1]:
                if start+blocklength > self.raw.shape[0]:
                    self.wordsblocks = words[1:]
                    return self.raw[start:]
                words = np.append(words, [[start, start+blocklength]], axis=0)
                start += blocklength
                if i < self.wordindeces.shape[0]-1:
                    if start>self.wordindeces[i+1][0]:
                        skip=True
                    else:
                        skip=False
        self.wordsblocks = words[1:]
        #print(self.wordsblocks)
        return np.array([])