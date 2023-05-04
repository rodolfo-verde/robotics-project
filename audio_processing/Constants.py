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

import os
from enum import Enum
import platform
from pathlib import Path

class EFactoryFeatureEvaluatorType(Enum):
    FeatureEvaluatorMFCC = 2
    FeatureEvaluatorSpectrogram = 3

class EFactoryANNModel(Enum):
    ANNModelKeras = 1
    ANNModelNumpy= 2
    ANNModelSiameseKeras = 3
    ANNModelTransferKeras = 4

class Constants:

    def __init__(self):
        self.pauseString = ''
        #self.__FeatureEvaluatorType = EFactoryFeatureEvaluatorType.FeatureEvaluatorMFCC
        self.__FeatureEvaluatorType = EFactoryFeatureEvaluatorType.FeatureEvaluatorSpectrogram

        #self.__VoiceActivityType = EFactoryVoiceActivityDetectorType.EFactoryVoiceActivityDetectorWebRtcVad
        #self.__VoiceActivityType = EFactoryVoiceActivityDetectorType.EFactoryVoiceActivityDetectorAnn
        #self.__VoiceActivityType = EFactoryVoiceActivityDetectorType.EFactoryVoiceActivityDetectorNone
        #self.__ANNModel = EFactoryANNModel.ANNModelKeras
        self.__ANNModel = EFactoryANNModel.ANNModelNumpy
        #self.__ANNModel = EFactoryANNModel.ANNModelSiameseKeras
		#self.__ANNModel = EFactoryANNModel.ANNModelTransferKeras
        #self.__useStreamTrainingsData = False
        # self.__useStreamTrainingsData = True
        self.__AudioResolution   = 16
        self.__ExtensionsVisible = False
        self.__HostPCIPAddress   = None
        self.__SocketPort        = None
        self.__ServerType        = None
        self.__SocketTimeout     = None
        self.__SocketMsgSeperatorForResults = None
        self.__HopSizeInMilliSeconds = 10
        self.__MinimumFrequencyResolutionInHertz = 30
        self.__WindowSizeInMilliSeconds = 2 * self.__HopSizeInMilliSeconds
        self.__Verbose = False
        #self.__IsLinux = platform.system() == 'Linux'
        self.__IsLinux = platform.system().find('Linux') >= 0
        #print(platform.system())
        self.__UseNonBlockingMode = False
        
        self.__WordLengthInMilliseconds = 700
        self.__RingbufferLengthInMilliseconds = 1000
        assert self.__RingbufferLengthInMilliseconds > 1.1*self.__WordLengthInMilliseconds, 'ringbufferlength too short'
        self.__MeanLevelMicrofoneIndB = -40
        self.__MinimumSoftmaxResultForClassification = 0.9
        self.__PeriodSizeMicrofoneInMilliseconds = 50
        self.__MemoryTimeKeyWordDetectorInSeconds = 0.4
        self.__MemoryTimeDenoiserInSeconds = 5*59
        self.__MemoryTimeAutomaticGainControl = 5*61
        self.__CutOffFrequencyLowInHertz = 100
        self.__CutOffFrequencyHighInHertz = 7000
        #self.pauseString = ""
        self.__NeuralNetworksDirectory = Path.joinpath(Path.cwd(), 'NeuralNetworks')
        self.__CoreNeuralNetworkDirectory = Path.joinpath(self.__NeuralNetworksDirectory, 'FeatureExtractorNetwork')
    
        pathConstantsClass = os.getcwd()
        initFile  ='Init.txt'
        #if FilePathManager.checkFile(pathConstantsClass, initFile):
        #    self.__useInitFile(initFile)
            
    def GetCutOffFrequencyLowInHertz(self):
        return self.__CutOffFrequencyLowInHertz
    
    def GetCutOffFrequencyHighInHertz(self):
        return self.__CutOffFrequencyHighInHertz

    def GetMemoryTimeAutomaticGainControl(self):
        return self.__MemoryTimeAutomaticGainControl

    def GetMemoryTimeDenoiserInSeconds(self):
        return self.__MemoryTimeDenoiserInSeconds
    
    def GetMemoryTimeKeyWordDetectorInSeconds(self):
        return self.__MemoryTimeKeyWordDetectorInSeconds
    
    def getUseNonBlockingMode(self):
        return self.__UseNonBlockingMode
    
    def getCoreNeuralNetworkDirectory(self):
        return self.__CoreNeuralNetworkDirectory
    
    def getNeuralNetWorksDirectory(self):
        return self.__NeuralNetworksDirectory
    
    def getMinimumFrequencyResolutionInHertz(self):
        return self.__MinimumFrequencyResolutionInHertz
    
    #def getShouldUseNewDatasetAugmentation(self):
    #    return self.__useStreamTrainingsData
    #    #return self.__VoiceActivityType == EFactoryVoiceActivityDetectorType.EFactoryVoiceActivityDetectorNone
    
    def getANNModel(self):
        return self.__ANNModel

    def setANNModel(self, newANNModelEnum):
        assert(isinstance(newANNModelEnum, EFactoryANNModel)), 'Enum is not from type EFactoryANNModel'
        self.__ANNModel = newANNModelEnum

    
    def getMinimumSoftmaxResultForClassification(self):
        return self.__MinimumSoftmaxResultForClassification
          
    def getMeanLevelMicrofoneIndB(self):
        return self.__MeanLevelMicrofoneIndB
    
    def getRingbufferLengthInMilliseconds(self):
        return self.__RingbufferLengthInMilliseconds
    
    def getWordLengthInMilliseconds(self):
        return self.__WordLengthInMilliseconds

    def getIsLinux(self):
        return self.__IsLinux
    
    def getOS(self):
        return platform.system()

    def getVerbose(self):
        return self.__Verbose
    
    def setVerbose(self, NewVerboseState):
        if NewVerboseState:
            self.__Verbose = True
        else:
            self.__Verbose = False
    
    def getSamplingFrequencyMicrofone(self):
        return 48000
    
    def getPeriodSizeMicrofoneInMilliseconds(self):
        return self.__PeriodSizeMicrofoneInMilliseconds

    def getAudioResolution(self):
        return self.__AudioResolution
    
    def getHopSizeInMilliSeconds(self):
        return self.__HopSizeInMilliSeconds
    
    def getWindowSizeInMilliSeconds(self):
        return self.__WindowSizeInMilliSeconds

    def __SetExtensionsVisible(self, TargetValue):
        if TargetValue=='True':
            self.__ExtensionsVisible = True
        else:
            self.__ExtensionsVisible = False
            
    def getExtensionsVisible(self):
        return self.__ExtensionsVisible
            
    def __SetHostPCIPAddress(self, TargetValue):
        self.__HostPCIPAddress=str(TargetValue)
        
    def getHostPCIPAddress(self):
        return self.__HostPCIPAddress 
        
    def __SetSocketPort(self, TargetValue):
        self.__SocketPort = int(TargetValue)
        
    def getSocketPort(self):
        return self.__SocketPort

    def setServerType(self, temp):
        self.__ServerType=temp
        
    def getServerType(self):
        return self.__ServerType

    def __SetSocketTimeout(self, TargetValue):
        self.__SocketTimeout = float(TargetValue)
        
    def getSocketTimeout(self):
        return self.__SocketTimeout

    def __SetSocketMsgSeperatorForResults(self, TargetValue):
        self.__SocketMsgSeperatorForResults = str(TargetValue)
        
    def getSocketMsgSeperatorForResults(self):
        return self.__SocketMsgSeperatorForResults
    
    def __SetFeatureEvaluatorType(self, TargetValue):
        self.__FeatureEvaluatorType = EFactoryFeatureEvaluatorType(int(TargetValue))
        
    def getFeatureEvaluatorType(self):
        return self.__FeatureEvaluatorType
    
    def getVoiceActivityDetectorType(self):
        return self.__VoiceActivityType

    def __SingleLineParser(self, line, StringToBeFind, TargetSetter):
        try:
            if line.find(StringToBeFind)!=-1:
                TargetSetter(line[line.rfind(' ')+1:-1])
        except:
            print('skipping line with unknown value: ' + str(line))

    def __useInitFile(self, initFile):
        with open(initFile,'r') as file:        
            for line in file:
                self.__SingleLineParser(line, 'ExtensionsVisible'           , self.__SetExtensionsVisible)
                self.__SingleLineParser(line, 'HostPCIPAddress'             , self.__SetHostPCIPAddress)
                self.__SingleLineParser(line, 'SocketPort'                  , self.__SetSocketPort)
                self.__SingleLineParser(line, 'ServerType'                  , self.setServerType)
                self.__SingleLineParser(line, 'SocketTimeout'               , self.__SetSocketTimeout)
                self.__SingleLineParser(line, 'SocketMsgSeperatorForResults', self.__SetSocketMsgSeperatorForResults)
                self.__SingleLineParser(line, 'FeatureEvaluatorType'        , self.__SetFeatureEvaluatorType)
    
    def printAttributes(self):
        print(self.getSamplingFrequency())
        print(self.getAudioResolution())
        print(self.getRefreshDBA())
        print(self.getExtensionsVisible())
        print(self.getHostPCIPAddress())
        print(self.getSocketPort())
        print(self.getServerType())
        print(self.getSocketTimeout())
        print(self.getSocketMsgSeperatorForResults())
        print(self.getFeatureEvaluatorType())
        print(self.getHopSizeInMilliSeconds())
        print(self.getIsLinux())
        print()

theConstants = Constants()

if __name__=='__main__':
    #import numpy as np
    
    theConstants.getAudioResolution()
    assert (not theConstants.getVerbose()), 'Verbose state should be False by default'
    #SamplingRate = theConstants.getSamplingFrequencyMicrofone()
    #assert np.absolute(SamplingRate - 16000) < 1, 'SamplingRate of microfone should be 16 kHz (for voice activity detection and linear prediction coefficients)'
    
