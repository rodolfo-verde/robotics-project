import numpy as np
from mfcc_processor import mfcc_dataprocessor
import matplotlib.pyplot as plt

# load data and split into trainings and test data
data_mfcc = np.load(f"audio_processing\Train_Data\set_complete_test_mfcc.npy",allow_pickle=True) # load data
data_labels = np.load(f"audio_processing\Train_Data\set_complete_test_label.npy",allow_pickle=True) # load data

print(f"Data shape: {data_mfcc.shape}")
print(f"Labels shape: {data_labels.shape}")

"""print(data_mfcc[1])

# plot first data_mfcc
plt.figure()
plt.imshow(data_mfcc[1])
plt.colorbar()
plt.show()"""
NumberOfMFCC = 11
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

# Calculate Delta MFCC of each mfcc in data_mfcc
data_mfcc_delta = np.zeros((data_mfcc.shape[0], NumberOfMFCC, NumberOfColumns))
for i in range(data_mfcc.shape[0]):
    data_mfcc_delta[i] = EvaluateDeltaMFCC(data_mfcc[i])

# Calculate Delta Delta MFCC of each delta mfcc in data_mfcc_delta
data_mfcc_delta_delta = np.zeros((data_mfcc.shape[0], NumberOfMFCC, NumberOfColumns))
for i in range(data_mfcc.shape[0]):
    data_mfcc_delta_delta[i] = EvaluateDeltaMFCC(data_mfcc_delta[i])

# concatenate data_mfcc, data_mfcc_delta and data_mfcc_delta_delta into one data_mfcc_final
data_mfcc_final = np.concatenate((data_mfcc, data_mfcc_delta, data_mfcc_delta_delta), axis=1)

print(f"Data shape: {data_mfcc_final.shape}")

# plot first data_mfcc_final
plt.figure()
plt.imshow(data_mfcc_final[1])
plt.colorbar()
plt.show()

# save data_mfcc_final
np.save(f"audio_processing\Train_Data\set_complete_test_mfcc_final.npy", data_mfcc_final)


