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

class CSignalFlowBlock(object):

    def __init__(self, TransferFunction = None):
        self.__ListOfOutputBlocks = []
        if TransferFunction is None:
            self.__TransferFunction = self.__DefaultTransferFunction
        else:
            self.__TransferFunction = TransferFunction
        self.__LastOutput = None
        self.__OutputsActive = True
        self.__InitializationDone = False
        self.__StartDone = False
        self.__StopDone = False

    def __DefaultTransferFunction(self, NewInput):
        return NewInput

    def Initialize(self, SamplingRate):
        assert not self.__InitializationDone, 'Call Initialize() only once'
        for OutputConnector in self.__ListOfOutputBlocks:
            OutputConnector.Initialize(SamplingRate)
        self._SamplingRate = SamplingRate
        self.__InitializationDone = True       
    
    def Start(self):
        assert not self.__StartDone, 'Call Start() only once'
        assert self.__InitializationDone, 'Call Initialize() before Start()'
        for OutputConnector in self.__ListOfOutputBlocks:
            OutputConnector.Start()        
        self.__StartDone = True      
    
    def Stop(self):
        assert self.__StartDone, 'Call Start() before Stop()'
        assert not self.__StopDone, 'Call Stop() only once'
        for OutputConnector in self.__ListOfOutputBlocks:
            OutputConnector.Stop()
        self.__StopDone = True        

    def RegisterOutput(self, NextSignalFlowBlock):
        assert not self.__InitializationDone, 'register all outputs before calling Initialize()'
        self.__ListOfOutputBlocks.append(NextSignalFlowBlock)

    def InputConnector(self, NewInput):
        assert self.__StartDone, 'Call Start() before InputConnector()'
        self.__LastOutput = self.__TransferFunction(NewInput)
        if self.__OutputsActive:
            for OutputConnector in self.__ListOfOutputBlocks:
                OutputConnector.InputConnector(self.__LastOutput)
        return self.__LastOutput

    def GetLastOutput(self):
        return self.__LastOutput

    def _ActivateOutputs(self):
        self.__OutputsActive = True

    def _DeActivateOutputs(self):
        self.__OutputsActive = False


if __name__ == "__main__":
    import numpy as np


    class CEnergyEvaluator(CSignalFlowBlock):

        def __TransferFunction(self, x):
            return x ** 2

        def __init__(self):
            super().__init__(self.__TransferFunction)


    class CSumEvaluator(CSignalFlowBlock):

        def __TransferFunction(self, x):
            return np.sum(x)

        def __init__(self):
            super().__init__(self.__TransferFunction)


    class CMeanEvaluator(CSignalFlowBlock):

        def __TransferFunction(self, x):
            return np.mean(x)

        def __init__(self):
            super().__init__(self.__TransferFunction)


    class CErrorCaller(CSignalFlowBlock):

        def __TransferFunction(self, x):
            assert False, 'this function should never be called'

        def __init__(self):
            super().__init__(self.__TransferFunction)


    SamplingRate = 16000
    # testdata
    x = np.random.randn(3)
    # constructing
    AEnergyEvaluator = CEnergyEvaluator()
    ASumEvaluator = CSumEvaluator()
    AMeanEvaluator = CMeanEvaluator()
    # connecting
    AEnergyEvaluator.RegisterOutput(ASumEvaluator)
    AEnergyEvaluator.RegisterOutput(AMeanEvaluator)
    AEnergyEvaluator.Initialize(SamplingRate)
    AEnergyEvaluator.Start()
    # evaluating
    AEnergyEvaluator.InputConnector(x)
    assert np.absolute(ASumEvaluator.GetLastOutput() - np.sum(x ** 2)) < 1e-10, 'wrong sum evaluation'
    assert np.absolute(AMeanEvaluator.GetLastOutput() - np.mean(x ** 2)) < 1e-10, 'wrong mean evaluation'

    AErrorCaller = CErrorCaller()
    AEnergyEvaluator = CEnergyEvaluator()
    AEnergyEvaluator.RegisterOutput(AErrorCaller)
    AEnergyEvaluator.Initialize(SamplingRate)
    AEnergyEvaluator.Start()
    AEnergyEvaluator._DeActivateOutputs()
    y1 = AEnergyEvaluator.InputConnector(x)
    y2 = AEnergyEvaluator.InputConnector(x)
    assert np.sum((y1 - y2)**2) < 1e-10, 'SignalFlowBlocks should be stateless'
