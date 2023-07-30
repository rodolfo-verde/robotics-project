import os
#from typing import List
import time
import pathlib

#from collections import defaultdict

#from src.gmm_hmm_asr.models import SingleGauss, HMM
#from src.gmm_hmm_asr.data import DataTuple, HMMDataTuple
import numpy as np

"""try:
    from MOPS import SignalFlowBlock
    from MOPS import Constants
    #from MOPS import WaveInterface
    from MOPS import ResultController
    from MOPS import TrainingsInterface
    from MOPS import StateMachine
    from MOPS import TrainingsDataInterface
except ImportError:
    import SignalFlowBlock
    import Constants
    #import WaveInterface
    import ResultController
    import TrainingsInterface
    import StateMachine
    import TrainingsDataInterface"""

# Code inspired by: https://github.com/desh2608/gmm-hmm-asr

PercentageOfTrainingsdata = 80
VOCABULARY = ["auf","hoch","runter","links","rechts","Stopp","zurueck",  "zu", "vor"]
NumberOfIterations = 10
NumberOfStates = 9
# insert training data here
# # load data and split into trainings and test data
data_mfcc = np.load(f"audio_processing\Train_Data\set_complete_test_mfcc.npy",allow_pickle=True) # load data
data_labels = np.load(f"audio_processing\Train_Data\set_complete_test_label.npy",allow_pickle=True) # load data

print(f"Data shape: {data_mfcc.shape}")
print(f"Labels shape: {data_labels.shape}")

split_mfcc = int(len(data_mfcc[:,10,69])*0.8) # 80% trainings data, 20% test data
split_labels = int(len(data_labels[:,8])*0.8) # 80% trainings labels, 20% test labels
X_train = data_mfcc[:split_mfcc] # load mfccs of trainings data, 80% of data
X_test = data_mfcc[split_mfcc:]# load test mfcc data, 20% of data
y_train = data_labels[:split_labels] # load train labels, 80% of labels
y_test = data_labels[split_labels:] # load test labels, 20% of labels

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")    
   

def LogWithOffset(x):
    assert np.amin(x) >= 0.0, 'negative probabilities'
    #return np.log(np.maximum(x, 10**(-80*np.log10(np.exp(1)))))
    res = -np.ones(x.shape) * (10**8)
    mask = x > 0.0
    res[mask] = np.log(x[mask])
    return res

class CTransitionMatrixUpdater(object):
    
    def __init__(self, NumberOfStates):
        self.__TransitionMatrix = np.diag(np.ones((NumberOfStates)))
        
    def ProcessNewPath(self, NewPath):
        assert np.amin(NewPath) >= 0.0, 'probabilities must be greater or equal zero'
        assert np.amax(NewPath) <= 1.0, 'probabilities must be smaller or equal zero'
        assert NewPath.shape[0] == self.__TransitionMatrix.shape[0], 'wrong size of NewPath'
        for t in range(1, NewPath.shape[1]):
            for s in range(NewPath.shape[0]-1):
                self.__TransitionMatrix[s, s+0] += NewPath[s+0, t] * NewPath[s, t-1]
                self.__TransitionMatrix[s, s+1] += NewPath[s+1, t] * NewPath[s, t-1]
    
    def GetTransitionMatrix(self):
        assert np.amin(self.__TransitionMatrix) >= 0.0, 'TransitionMatrix must be nonnegative'
        self.__TransitionMatrix[-1, -1] = 1.0
        for row in range(self.__TransitionMatrix.shape[0] - 1):        
            self.__TransitionMatrix[row, :] /= np.sum(self.__TransitionMatrix[row, :])
        #print(self.__TransitionMatrix)
        return np.copy(self.__TransitionMatrix)
        
    def Reset(self):
        self.__TransitionMatrix *= 0.0
        
class CStateModelUpdater(object):
    
    def __init__(self):
        self.Reset()
        
    def GetMean(self):
        return self.__M / np.outer(np.ones((self.__M.shape[0])), self.__c)
    
    def GetStd(self):
        return self.__S / np.outer(np.ones((self.__M.shape[0])), self.__c) - self.GetMean()**2
    
    def ProcessNewPath(self, NewPath, x):
        for t in range(x.shape[1]):
            self.__M += np.outer(x[:, t], NewPath[:, t])
            self.__S += np.outer(x[:, t]**2, NewPath[:, t])
            self.__c += NewPath[:, t]
        
    def Reset(self):
        self.__M = 0.0
        self.__S = 0.0
        self.__c = 0.0
        
def Viterbi(LoglikelihoodMatrix, TransitionMatrix):
    LogT = LogWithOffset(TransitionMatrix)
    StayInStateLikelihood = np.diag(LogT)
    ChangeStateLikelihood = np.diag(LogT, 1)
    Predecessor = np.zeros(LoglikelihoodMatrix.shape)
    LoglikelihoodMatrix[1:, 0] += LogWithOffset(np.array([0.0]))
    for t in range(1, LoglikelihoodMatrix.shape[1]):
        tmp1 = LoglikelihoodMatrix[0:LoglikelihoodMatrix.shape[0]-1, t-1] + ChangeStateLikelihood
        tmp2 = LoglikelihoodMatrix[1:, t-1] + StayInStateLikelihood[1:]
        Predecessor[1:, t] = (tmp1 >= tmp2) + 0.0
        LoglikelihoodMatrix[1:, t] += np.maximum(tmp1, tmp2)
        LoglikelihoodMatrix[0 , t] += LoglikelihoodMatrix[0, t-1] + StayInStateLikelihood[0]
    path = np.zeros(LoglikelihoodMatrix.shape)
    t = path.shape[1]
    #CurrentState = path.shape[0] - 1
    CurrentState = np.argmax(LoglikelihoodMatrix[:, path.shape[1]-1])
    while t >= 1:
        t -= 1
        path[CurrentState, t] = 1
        if (Predecessor[CurrentState, t] > 0.5) and (CurrentState > 0):
            CurrentState -= 1
    #path[:, 0] = 0
    #path[0, 0] = 1
    assert CurrentState == 0, 'wrong end state'
    assert path[0, 0] == 1, 'wrong starting state'
    assert np.amin(np.sum(np.abs(path), axis = 0)) == 1, 'wrong number of 1'
    assert np.amax(np.sum(np.abs(path), axis = 0)) == 1, 'wrong number of 1'
    assert np.amin(path) >= 0.0, 'non negative path is not allowed'
    assert np.amax(path) <= 1.0, 'path must be smaller than 1.0'
    return path

def Forward(LoglikelihoodMatrix, TransitionMatrix):
    T = LoglikelihoodMatrix.shape[1]
    LoglikelihoodMatrix[1:, 0] += LogWithOffset(np.array([0.0]))
    for t in range(1, T):
        Maximum = np.amax(LoglikelihoodMatrix[:, t-1])
        p = np.matmul(TransitionMatrix.T, np.exp(LoglikelihoodMatrix[:, t-1] - Maximum))
        LoglikelihoodMatrix[:, t] += LogWithOffset(p) + Maximum
    path = np.zeros(LoglikelihoodMatrix.shape)
    for t in range(T):
        tmp = LoglikelihoodMatrix[:, t]
        tmp = np.exp(tmp - np.amax(tmp))
        path[:, t] = tmp / np.sum(tmp)
    return path

def EvalLogLikelihood(x, M, Sinverse):
    y = x - np.outer(M, np.ones((x.shape[1])))
    det = np.sum(LogWithOffset(np.diag(Sinverse)))
    p_LL = -0.5 * np.sum(np.matmul(Sinverse, (y**2)), axis = 0) + 0.5 * det
    p_LL -= 0.5*x.shape[0]*LogWithOffset(np.array([2*np.pi]))
    return p_LL

def EvalLoglikelihoods(FeatureMatrix, M, S):
    result = np.zeros((M.shape[1], FeatureMatrix.shape[1]))
    for s in range(result.shape[0]):
        result[s, :] = EvalLogLikelihood(FeatureMatrix, M[:, s], np.diag(S[:, s]**(-1)))
    return result

class CHMMSingleCommand(object):
    
    def __init__(self, CommandString):
        self.__CommandString = CommandString
        self.__NumberOfStates = NumberOfStates
        self.__ListOfFeatures = []
        self.__M = None
        self.__S = None
        self.__T = None
        
    def __GetFileName(self):
        FolderName = 'HMMModels'
        FileName = pathlib.Path(FolderName, self.__CommandString + '.npz')
        try:
            os.mkdir(FolderName)
        except:
            pass
        return FileName          
    
    def __Initialization(self):
        if self.__M is None:
            try:
                npzfile = np.load(str(self.__GetFileName()))
                self.__M = npzfile['x']
                self.__S = npzfile['y']
                self.__T = npzfile['z']
            except:
                self.__LoadTrainingsData()
                self.__Train()
                np.savez(str(self.__GetFileName()), x = self.__M, y = self.__S, z = self.__T)


    def __LoadTrainingsData(self):
        ATrainingsDataInterface = TrainingsDataInterface.CTrainingsDataInterface()
        for CommandIndex in range(ATrainingsDataInterface.GetNumberOfCommands()):
            command = ATrainingsDataInterface.GetCommandString(CommandIndex)
            if command == self.__CommandString:
                for InstanceIndex in range(int(ATrainingsDataInterface.GetNumberOfCommandInstances(CommandIndex) * PercentageOfTrainingsdata / 100)):
                    x, Fs, bits = ATrainingsDataInterface.GetWaveOfCommandInstance(CommandIndex, InstanceIndex)
                    mat = TrainingsInterface.SamplesToFeature(x, Fs)
                    if not (mat is None):
                        self.__AppendFeature(mat)
        
    def __AppendFeature(self, NewFeature):
        self.__ListOfFeatures.append(NewFeature)
        K = NewFeature.shape[0]
        if not (self.__M is None):            
            assert K == self.__M.shape[0], 'not consistent number of features (zero-th dimension of NewFeature)'
        else:            
            self.__M = np.zeros((K, self.GetNumberOfStates()))
            self.__S = np.ones((K, self.GetNumberOfStates()))
            ATransitionMatrixUpdater = CTransitionMatrixUpdater(self.__NumberOfStates)
            self.__T = ATransitionMatrixUpdater.GetTransitionMatrix()
            
    def __Train(self):
        ATransitionMatrixUpdater = CTransitionMatrixUpdater(self.GetNumberOfStates())
        AStateModelUpdater = CStateModelUpdater()
        for iter in range(NumberOfIterations):
        #OldPath = np.ones((len(self.__ListOfFeatures)*self.GetNumberOfStates()))
        #CurrentPath = np.zeros(OldPath.shape)
        #while np.sum(OldPath - CurrentPath) > 1e-1:
            #OldPath = np.copy(CurrentPath)
            ATransitionMatrixUpdater.Reset()
            AStateModelUpdater.Reset()
            for FeatureIdx in range(len(self.__ListOfFeatures)):
                FeatureMatrix = self.__ListOfFeatures[FeatureIdx]
                if iter == 0:
                    Path = np.zeros((self.GetNumberOfStates(), FeatureMatrix.shape[1]))
                    for t in range(Path.shape[1]):
                        FractionalState = self.GetNumberOfStates() * t / Path.shape[1]
                        Path[int(FractionalState), t] = 1
                else:
                    LogLikelihoodsOfFeature = EvalLoglikelihoods(FeatureMatrix, self.__M, self.__S)
                    Path = Viterbi(LogLikelihoodsOfFeature, self.__T)
                    #Path = Forward(LogLikelihoodsOfFeature, self.__T)
                ATransitionMatrixUpdater.ProcessNewPath(Path)
                AStateModelUpdater.ProcessNewPath(Path, FeatureMatrix)
                #StartIdx = FeatureIdx * self.GetNumberOfStates()
                #CurrentPath[StartIdx:StartIdx+self.GetNumberOfStates()] = np.sum(Path, axis = 1)
            self.__T = ATransitionMatrixUpdater.GetTransitionMatrix()
            self.__M = AStateModelUpdater.GetMean()
            self.__S = AStateModelUpdater.GetStd()
            
            
    def GetNumberOfStates(self):
        return self.__NumberOfStates

    def GetCommandString(self):
        return self.__CommandString
    
    def GetNumberOfFeatures(self):
        self.__Initialization()
        return self.__M.shape[0]       
        
    def GetTransitionMatrix(self):
        self.__Initialization()
        return self.__T
    
    def GetMean(self):
        self.__Initialization()
        return self.__M
    
    def GetVariance(self):
        self.__Initialization()
        return self.__S

class CHMM(SignalFlowBlock.CSignalFlowBlock):

    def __init__(self):
        super().__init__(self.__TransferFunction)
        self.__ListOfHMMSingleCommand = []
        for command in VOCABULARY:
            self.__ListOfHMMSingleCommand.append(CHMMSingleCommand(command))
        
    def Initialize(self, SamplingRate):
        super().Initialize(SamplingRate)
        self.__ConfigureResultController()
        self.__ConfigureHMM()

    def __ConfigureResultController(self):
        ResultController.TheResultController.AddCommand('links', StateMachine.TheStateMachine.triggerMotorALeftTurn)
        ResultController.TheResultController.AddCommand('rechts', StateMachine.TheStateMachine.triggerMotorARightTurn)
        ResultController.TheResultController.AddCommand('zurueck', StateMachine.TheStateMachine.triggerMotorBRightTurn)
        ResultController.TheResultController.AddCommand('vor', StateMachine.TheStateMachine.triggerMotorBLeftTurn)
        ResultController.TheResultController.AddCommand('hoch', StateMachine.TheStateMachine.triggerMotorCRightTurn)
        ResultController.TheResultController.AddCommand('runter', StateMachine.TheStateMachine.triggerMotorCLeftTurn)
        ResultController.TheResultController.AddCommand('zu', StateMachine.TheStateMachine.triggerMotorDLeftTurn)
        ResultController.TheResultController.AddCommand('auf', StateMachine.TheStateMachine.triggerMotorDRightTurn) 
        ResultController.TheResultController.AddCommand('Stopp', StateMachine.TheStateMachine.triggerMotorStop)
        
    def __ConfigureHMM(self):
        C = len(self.__ListOfHMMSingleCommand)
        assert C > 0, 'empty command list'
        K = self.__ListOfHMMSingleCommand[0].GetNumberOfFeatures()
        S = self.__ListOfHMMSingleCommand[0].GetNumberOfStates()
        self.__Sinverse = np.zeros((K, K, S, C))
        self.__M = np.zeros((K, S, C))
        self.__TransitionMatrices = np.zeros((S, S, C))
        for c in range(C):            
            self.__TransitionMatrices[:, :, c] = self.__ListOfHMMSingleCommand[c].GetTransitionMatrix().T
            Variance = self.__ListOfHMMSingleCommand[c].GetVariance()
            Mean = self.__ListOfHMMSingleCommand[c].GetMean()
            for s in range(S):
                self.__Sinverse[:, :, s, c] = np.diag(Variance[:, s]**(-1))
                self.__M[:, s, c] = Mean[:, s]
        assert np.amin(self.__TransitionMatrices) >= 0.0, 'negative probabilities not allowed'
        
    def __TransferFunction(self, NewBlocks = None):
        t0 = time.time()
        T = NewBlocks.shape[1]
        C = self.__M.shape[2]
        S = self.__M.shape[1]
        p_LL = np.zeros((S, C, T))
        for s in range(S):
            for c in range(C):
                p_LL[s, c, :] = EvalLogLikelihood(NewBlocks, self.__M[:, s, c], self.__Sinverse[:, :, s, c])
        p_LL[1:, :, 0] += LogWithOffset(np.array([0.0]))
        p = np.zeros((S, C))
        for t in range(1, T):
            Maximum = np.amax(p_LL[:, :, t-1])
            for c in range(C):
                p[:, c] = np.matmul(self.__TransitionMatrices[:, :, c], np.exp(p_LL[:, c, t-1] - Maximum))
            p_LL[:, :, t] += LogWithOffset(p) + Maximum
        #Maximum = np.amax(p_LL[:, :, T-1])
        #p = np.exp(p_LL[:, :, T - 1] - Maximum)
        #WinnerIndex = np.argmax(LogWithOffset(np.sum(p, axis = 0)) + Maximum)
        WinnerIndex = np.argmax(p_LL[S-1, :, T-1])
        WinnerCommand = self.__ListOfHMMSingleCommand[WinnerIndex].GetCommandString()
        t1 = time.time()
        if Constants.theConstants.getVerbose(): print('HMM processing time = ', t1- t0)
        if Constants.theConstants.getVerbose(): print('command = ', WinnerCommand)
        ResultController.TheResultController.DoCommand(WinnerCommand)
        return WinnerCommand

if __name__ == '__main__':
    assert True, 'to be done'
    
    from tqdm import tqdm
    
    print('evaluating accuracy')
    SamplingRate = 48000
    AHMM = CHMM()
    AHMM.Initialize(SamplingRate)
    AHMM.Start()
    ATrainingsDataInterface = TrainingsDataInterface.CTrainingsDataInterface()
    NumberOfTrainingsCorrect = 0
    NumberOfTrainings = 0
    NumberOfTests = 0
    NumberOfTestsCorrect = 0            
    for CommandIndex in range(ATrainingsDataInterface.GetNumberOfCommands()):
        command = ATrainingsDataInterface.GetCommandString(CommandIndex)
        if command in VOCABULARY:
            print(command)
            for InstanceIndex in tqdm(range(ATrainingsDataInterface.GetNumberOfCommandInstances(CommandIndex))):
                x, Fs, bits = ATrainingsDataInterface.GetWaveOfCommandInstance(CommandIndex, InstanceIndex)
                assert np.abs(SamplingRate - Fs) < 1e-3, 'wrong sampling rate'
                mat = TrainingsInterface.SamplesToFeature(x, Fs)
                if not (mat is None):
                    WinnerCommand = AHMM.InputConnector(mat)
                    if InstanceIndex < int(ATrainingsDataInterface.GetNumberOfCommandInstances(CommandIndex) * PercentageOfTrainingsdata / 100):
                        NumberOfTrainings += 1
                        if WinnerCommand == command:
                            NumberOfTrainingsCorrect += 1
                    else:
                        NumberOfTests += 1
                        if WinnerCommand == command:
                            NumberOfTestsCorrect += 1
    print('accuracy training = ', NumberOfTrainingsCorrect / NumberOfTrainings)
    print('accuracy test = ', NumberOfTestsCorrect / NumberOfTests)
    
    #'''