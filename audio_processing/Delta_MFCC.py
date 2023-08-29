import numpy as np

NumberOfMFCC = 12
NumberOfColumns = 70
SampleMFCC = np.random.randn(NumberOfMFCC, NumberOfColumns)

def EvaluateDeltaMFCC(MFCC): 
    DeltaMFCC = np.zeros(MFCC.shape)
    for col in range(MFCC.shape[1]):
        for l in range(-2, 3):
            CurrentCol = col - l
            if CurrentCol < 0:
                CurrentCol = 0
            elif CurrentCol >= SampleMFCC.shape[1]:
                CurrentCol = SampleMFCC.shape[1] - 1
            DeltaMFCC[:, col] += 0.1 * l * MFCC[:, CurrentCol]
    return DeltaMFCC