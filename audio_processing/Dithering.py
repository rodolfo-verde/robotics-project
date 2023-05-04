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


import numpy as np
import Constants
import SignalFlowBlock


class CDithering(SignalFlowBlock.CSignalFlowBlock):

    def __init__(self, Bits = Constants.theConstants.getAudioResolution()):
        super().__init__(self.__TransferFunction)
        self.__NoiseFactor = 1 / float(2 ** (Bits - 1) + 1.0)

    def GetMaximumNoiseAmplitude(self):
        return self.__NoiseFactor * 0.5

    def __TransferFunction(self, input):
        DitheringNoise = (np.random.random(input.shape) - 0.5) * self.__NoiseFactor
        return input + DitheringNoise


if __name__ == '__main__':
    SamplingRate = 16000
    ADithering = CDithering()
    ADithering.GetMaximumNoiseAmplitude()
    for Bits in range(16):
        ADithering = CDithering(Bits)
        ADithering.Initialize(SamplingRate)
        ADithering.Start()
        Factor = 2 ** (Bits - 1)
        x = np.round((2 * np.random.random(10000) - 1) * Factor) / Factor
        y = ADithering.InputConnector(x)
        y = np.round(y * Factor)
        x = np.round(x * Factor)
        assert np.sum(np.absolute(x - y)) < 1e-10, 'Dithering violates more than the last bit'