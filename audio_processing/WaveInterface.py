#!/usr/bin/env python

# MIT-License Copyright (c) 2018 Prof. Dr.-Ing. Martin Spiertz, FHWS Permission is hereby granted, free of charge,
# to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the
# Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions: The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY
# OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import wave
import numpy as np
import struct

import Dithering
import SignalFlowBlock
import Constants


def ReadWave(audiofilename):
    # https://stackoverflow.com/questions/2063284/what-is-the-easiest-way-to-read-wav-files-using-python-summary
    wav = wave.open(str(audiofilename), "rb")
    (nchannels, sampwidth, samplerate, nframes, comptype, compname) = wav.getparams ()
    
    frames = wav.readframes(nframes * nchannels)
    x = np.array(struct.unpack_from ("%dh" % nframes * nchannels, frames))
    if nchannels > 1:            
        #print('Number of channels = ', str(nchannels), ' , selecting first channel')
        x = x[0::nchannels]
    # scale to -1.0 -- 1.0
    nb_bits = sampwidth*8
    
    factor = float(2**(nb_bits - 1))
    samples = x / factor
    assert np.amax(np.abs(samples)) <= 1.0, 'samples out of range, clipping occured?'
    
    ADithering = Dithering.CDithering(nb_bits)
    ADithering.Initialize(samplerate)
    ADithering.Start()
    samples = ADithering.InputConnector(samples)
    ADithering.Stop()
    return samples, samplerate, nb_bits

def WriteWave(samples, samplerate, nb_bits, audiofilename):    
    ADithering = Dithering.CDithering(nb_bits)
    assert np.amax(np.abs(samples)) <= 1.0 + ADithering.GetMaximumNoiseAmplitude(), 'samples out of range, clipping occurs'
    factor = 2**(nb_bits - 1)
    buf = NumpyArrayToStruct(samples * factor)
    
    wav = wave.open(str(audiofilename), 'wb')
    sampwidth = 2
    nframes = len(samples)
    nchannels = 1
    wav.setparams((nchannels, sampwidth, samplerate, nframes, 'NONE', 'not compressed'))    
    wav.writeframes(buf)
    wav.close()

def NumpyArrayToStruct(samples):
    IntegerValues = np.round(samples)
    IntList = np.array(IntegerValues, dtype='<i2')
    return struct.pack('%sh' % len(IntList), *IntList)

class CReadWaveFileAsSignalFlowBlock(SignalFlowBlock.CSignalFlowBlock):
    
    def __init__(self, Filename, BlockSize = 1024):
        super().__init__()
        self.__filename = Filename
        self.__BlockSize = BlockSize
        
    def Initialize(self):
        self.__samples, samplerate, self.__nb_bits = ReadWave(self.__filename)
        super().Initialize(samplerate)
    
    def Start(self):
        super().Start()        
        idx1 = 0
        idx2 = self.__BlockSize
        while idx2 < self.__samples.shape[0]:
            super().InputConnector((self.__samples[idx1:idx2]))
            idx1 += self.__BlockSize
            idx2 += self.__BlockSize
        super().InputConnector((self.__samples[idx1:]))
            
class CWriteWaveFileAsSignalFlowBlock(SignalFlowBlock.CSignalFlowBlock):
    
    def __init__(self, Filename):
        super().__init__(self.__TransferFunction)
        self.__filename = Filename
        MaximumStorageUsedInBytes = 2**26
        MaximumNumberOfSamplesStoredAsFloat = int(MaximumStorageUsedInBytes/4)
        self.__Buffer = np.zeros((MaximumNumberOfSamplesStoredAsFloat))
        self.__CurrentIndex = 0
    
    def __TransferFunction(self, InputData):
        StopIndex = self.__CurrentIndex + InputData.shape[0]
        assert StopIndex < self.__Buffer.shape[0], 'too much samples stored, decrease recording duration'
        self.__Buffer[self.__CurrentIndex:StopIndex] = np.copy(InputData)
        self.__CurrentIndex = StopIndex
    
    def Stop(self):
        super().Stop
        WriteWave(self.__Buffer[0:self.__CurrentIndex], self._SamplingRate, Constants.theConstants.getAudioResolution(), self.__filename)

if __name__ == "__main__":
    import os
    
    filename1 = 'input.wav'
    Fs = 16000
    bits = 16
    x = np.sin(2*np.pi*440*np.arange(Fs*5)/Fs) * 0.999
    WriteWave(x, Fs, bits, filename1)
    y, Fs, bits = ReadWave(filename1)
    dB = 10*np.log10(np.sum(x**2) / np.sum((x-y)**2))
    for n in range(10):
        WriteWave(y, Fs, bits, filename1)
        y, Fs, bits = ReadWave(filename1)
        dB = 10*np.log10(np.sum(x**2) / np.sum((x - y)**2))
        assert (dB > 90), 'error in continuous reading/writing of wave files'
    
    filename2 = 'output.wave'
    AReadWaveFileAsSignalFlowBlock = CReadWaveFileAsSignalFlowBlock(filename1)
    AWriteWaveFileAsSignalFlowBlock = CWriteWaveFileAsSignalFlowBlock(filename2)
    AReadWaveFileAsSignalFlowBlock.RegisterOutput(AWriteWaveFileAsSignalFlowBlock)
    AReadWaveFileAsSignalFlowBlock.Initialize()
    AReadWaveFileAsSignalFlowBlock.Start()
    AReadWaveFileAsSignalFlowBlock.Stop()
    y, Fs, bits = ReadWave(filename2)
    dB = 10*np.log10(np.sum(x**2) / np.sum((x - y)**2))
    assert (dB > 90), 'error in continuous reading/writing of wave files'
    os.remove(filename1)
    os.remove(filename2)    
