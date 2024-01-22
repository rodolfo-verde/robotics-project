import numpy as np
from scipy.io import wavfile
import librosa
from scipy.signal import resample


def augmentate_data(mfcc_stored, label_array, output_file):
        # Augmentate mfcc data --> cutout, mixup, sample pairing, specaugment, specmix, vh-mixup, mixed frequency masking
        # cutout
        mfcc_stored_cutout = []
        def cutout(mfcc, size=10):
            # Ensure size is less than mfcc.shape[1]
            if size >= mfcc.shape[1]:
                print("Size is greater than or equal to mfcc.shape[1], reducing size.")
                size = mfcc.shape[1] - 1

            start = np.random.randint(0, mfcc.shape[1] - size)
            mfcc_cutout = mfcc.copy()
            mfcc_cutout[:, start:start + size] = 0
            return mfcc_cutout
        for i in range(mfcc_stored.shape[0]):
            mfcc = mfcc_stored[i,:,:]
            mfcc = cutout(mfcc)
            mfcc_stored_cutout.append(mfcc)
        mfcc_stored_cutout = np.stack(mfcc_stored_cutout)
        # mixup
        mfcc_stored_mixup = []
        def mixup(mfcc1, mfcc2, label, alpha):
            # Generate a random mixup ratio
            l = np.random.beta(alpha, alpha)
            # Create a new MFCC and label by combining the inputs
            mfcc = l * mfcc1 + (1 - l) * mfcc2
            label = l * label + (1 - l) * label
            return mfcc, label
        for i in range(mfcc_stored.shape[0]-1):
            mfcc1 = mfcc_stored[i,:,:]
            mfcc2 = mfcc_stored[i+1,:,:]
            label1 = label_array[i,:]
            label2 = label_array[i+1,:]
            mfcc, label = mixup(mfcc1, mfcc2, label1, 0.2)
            mfcc_stored_mixup.append(mfcc) 
        mfcc_stored_mixup = np.stack(mfcc_stored_mixup)
        mfcc_stored_mixup = np.array(mfcc_stored_mixup)
        # sample pairing
        mfcc_stored_sample_pairing = []
        def sample_pairing(mfcc1, mfcc2):
            # Create a new MFCC by averaging the inputs
            mfcc_sample = (mfcc1 + mfcc2) / 2
            return mfcc_sample
        for i in range(mfcc_stored.shape[0]-1):
            mfcc1 = mfcc_stored[i,:,:]
            mfcc2 = mfcc_stored[i+1,:,:]
            mfcc_sample = sample_pairing(mfcc1, mfcc2)
            mfcc_stored_sample_pairing.append(mfcc_sample)
        mfcc_stored_sample_pairing = np.stack(mfcc_stored_sample_pairing)
        mfcc_stored_sample_pairing = np.array(mfcc_stored_sample_pairing)
        # specaugment
        mfcc_stored_specaugment = []
        def specaugment(mfcc, F, T):
            # Frequency masking
            f = np.random.uniform(low=0.0, high=F)
            f = int(f * mfcc.shape[0])
            f0 = np.random.randint(0, mfcc.shape[0] - f)
            mfcc[f0:f0 + f, :] = 0

            # Time masking
            t = np.random.uniform(low=0.0, high=T)
            t = int(t * mfcc.shape[1])
            t0 = np.random.randint(0, mfcc.shape[1] - t)
            mfcc[:, t0:t0 + t] = 0

            return mfcc
        for i in range(mfcc_stored.shape[0]):
            mfcc = mfcc_stored[i,:,:]
            mfcc = specaugment(mfcc, 0.2, 0.2)
            mfcc_stored_specaugment.append(mfcc)
        mfcc_stored_specaugment = np.stack(mfcc_stored_specaugment)
        mfcc_stored_specaugment = np.array(mfcc_stored_specaugment)
        # specmix
        mfcc_stored_specmix = []
        def mixup2(mfcc1, mfcc2, label1, label2, alpha):
            # Generate a random number for the mixup
            lam = np.random.beta(alpha, alpha)
            # Mixup the MFCCs
            mfcc = lam * mfcc1 + (1 - lam) * mfcc2
            # Mixup the labels
            label = lam * label1 + (1 - lam) * label2
            return mfcc, label
        def specmix(mfcc1, mfcc2, label1, label2, alpha, F, T):
            # Apply mixup
            mfcc, label = mixup2(mfcc1, mfcc2, label1, label2, alpha)
            # Apply specaugment
            mfcc = specaugment(mfcc, F, T)
            return mfcc, label
        for i in range(mfcc_stored.shape[0]-1):
            mfcc1 = mfcc_stored[i,:,:]
            mfcc2 = mfcc_stored[i+1,:,:]
            label1 = label_array[i,:]
            label2 = label_array[i+1,:]
            mfcc, label = specmix(mfcc1, mfcc2, label1, label2, 0.2, 0.2, 0.2)
            mfcc_stored_specmix.append(mfcc)
        mfcc_stored_specmix = np.stack(mfcc_stored_specmix)
        mfcc_stored_specmix = np.array(mfcc_stored_specmix)
        # vh-mixup
        mfcc_stored_vh_mixup = []
        def vh_mixup(mfcc1, mfcc2, mfcc3, mfcc4, alpha):
            # Vertical mixup
            lam_v = np.random.beta(alpha, alpha, size=(mfcc1.shape[0], 1))
            mfcc_v = lam_v * mfcc1 + (1 - lam_v) * mfcc2

            # Horizontal mixup
            lam_h = np.random.beta(alpha, alpha, size=(1, mfcc_v.shape[1]))
            mfcc_h = lam_h * mfcc_v + (1 - lam_h) * mfcc3

            return mfcc_h
        for i in range(mfcc_stored.shape[0]-3):
            mfcc1 = mfcc_stored[i,:,:]
            mfcc2 = mfcc_stored[i+1,:,:]
            mfcc3 = mfcc_stored[i+2,:,:]
            mfcc4 = mfcc_stored[i+3,:,:]
            mfcc = vh_mixup(mfcc1, mfcc2, mfcc3, mfcc4, 0.2)
            mfcc_stored_vh_mixup.append(mfcc)
        mfcc_stored_vh_mixup = np.stack(mfcc_stored_vh_mixup)
        mfcc_stored_vh_mixup = np.array(mfcc_stored_vh_mixup)
        # mixed frequency masking
        mfcc_stored_mixed_frequency_masking = []
        def mixed_frequency_masking(mfcc, F):
            # Determine the number of frequencies to mask
            f = np.random.randint(0, F)

            # Select f frequencies to mask
            freqs_to_mask = np.random.choice(mfcc.shape[0], size=f, replace=False)

            # Mask the selected frequencies
            mfcc[freqs_to_mask, :] = 0

            return mfcc
        for i in range(mfcc_stored.shape[0]):
            mfcc = mfcc_stored[i,:,:]
            mfcc = mixed_frequency_masking(mfcc, 10)
            mfcc_stored_mixed_frequency_masking.append(mfcc)
        mfcc_stored_mixed_frequency_masking = np.stack(mfcc_stored_mixed_frequency_masking)
        #convert to np array
        mfcc_stored_mixed_frequency_masking = np.array(mfcc_stored_mixed_frequency_masking)

        # save all mfcc as npy files
        np.save(output_file+'_mfcc_cutout', mfcc_stored_cutout)
        np.save(output_file+'_mfcc_mixup', mfcc_stored_mixup)
        np.save(output_file+'_mfcc_sample_pairing', mfcc_stored_sample_pairing)
        np.save(output_file+'_mfcc_specaugment', mfcc_stored_specaugment)
        np.save(output_file+'_mfcc_specmix', mfcc_stored_specmix)
        np.save(output_file+'_mfcc_vh_mixup', mfcc_stored_vh_mixup)
        np.save(output_file+'_mfcc_mixed_frequency_masking', mfcc_stored_mixed_frequency_masking)

        print('Shape of mfcc_stored_cutout: ', mfcc_stored_cutout.shape)
        print('Shape of mfcc_stored_mixup: ', mfcc_stored_mixup.shape)
        print('Shape of mfcc_stored_sample_pairing: ', mfcc_stored_sample_pairing.shape)
        print('Shape of mfcc_stored_specaugment: ', mfcc_stored_specaugment.shape)
        print('Shape of mfcc_stored_specmix: ', mfcc_stored_specmix.shape)
        print('Shape of mfcc_stored_vh_mixup: ', mfcc_stored_vh_mixup.shape)
        print('Shape of mfcc_stored_mixed_frequency_masking: ', mfcc_stored_mixed_frequency_masking.shape)

class_names = ['a', 'b', 'c', '1', '2', '3', 'stopp', 'rex', 'other']
# load data
mfcc_data = np.load('audio_processing\Train_Data\set_complete_test_mfcc.npy')
label_data = np.load('audio_processing\Train_Data\set_complete_test_label.npy')

print('Shape of mfcc_data: ', mfcc_data.shape)
print('Shape of label_data: ', label_data.shape)

# Extract all classes into seperate mfcc arrays
mfcc_a = []
mfcc_b = []
mfcc_c = []
mfcc_1 = []
mfcc_2 = []
mfcc_3 = []
mfcc_stopp = []
mfcc_rex = []
mfcc_other = []

for i in range(mfcc_data.shape[0]):
    if label_data[i,0] == 1:
        mfcc_a.append(mfcc_data[i,:,:])
    elif label_data[i,1] == 1:
        mfcc_b.append(mfcc_data[i,:,:])
    elif label_data[i,2] == 1:
        mfcc_c.append(mfcc_data[i,:,:])
    elif label_data[i,3] == 1:
        mfcc_1.append(mfcc_data[i,:,:])
    elif label_data[i,4] == 1:
        mfcc_2.append(mfcc_data[i,:,:])
    elif label_data[i,5] == 1:
        mfcc_3.append(mfcc_data[i,:,:])
    elif label_data[i,6] == 1:
        mfcc_stopp.append(mfcc_data[i,:,:])
    elif label_data[i,7] == 1:
        mfcc_rex.append(mfcc_data[i,:,:])
    elif label_data[i,8] == 1:
        mfcc_other.append(mfcc_data[i,:,:])

mfcc_a = np.stack(mfcc_a)
mfcc_b = np.stack(mfcc_b)
mfcc_c = np.stack(mfcc_c)
mfcc_1 = np.stack(mfcc_1)
mfcc_2 = np.stack(mfcc_2)
mfcc_3 = np.stack(mfcc_3)
mfcc_stopp = np.stack(mfcc_stopp)
mfcc_rex = np.stack(mfcc_rex)
mfcc_other = np.stack(mfcc_other)

print('Shape of mfcc_a: ', mfcc_a.shape)
print('Shape of mfcc_b: ', mfcc_b.shape)
print('Shape of mfcc_c: ', mfcc_c.shape)
print('Shape of mfcc_1: ', mfcc_1.shape)
print('Shape of mfcc_2: ', mfcc_2.shape)
print('Shape of mfcc_3: ', mfcc_3.shape)
print('Shape of mfcc_stopp: ', mfcc_stopp.shape)
print('Shape of mfcc_rex: ', mfcc_rex.shape)
print('Shape of mfcc_other: ', mfcc_other.shape)

# Extract all classes into seperate label arrays
label_a = []
label_b = []
label_c = []
label_1 = []
label_2 = []
label_3 = []
label_stopp = []
label_rex = []
label_other = []

for i in range(label_data.shape[0]):
    if label_data[i,0] == 1:
        label_a.append(label_data[i,:])
    elif label_data[i,1] == 1:
        label_b.append(label_data[i,:])
    elif label_data[i,2] == 1:
        label_c.append(label_data[i,:])
    elif label_data[i,3] == 1:
        label_1.append(label_data[i,:])
    elif label_data[i,4] == 1:
        label_2.append(label_data[i,:])
    elif label_data[i,5] == 1:
        label_3.append(label_data[i,:])
    elif label_data[i,6] == 1:
        label_stopp.append(label_data[i,:])
    elif label_data[i,7] == 1:
        label_rex.append(label_data[i,:])
    elif label_data[i,8] == 1:
        label_other.append(label_data[i,:])

label_a = np.stack(label_a)
label_b = np.stack(label_b)
label_c = np.stack(label_c)
label_1 = np.stack(label_1)
label_2 = np.stack(label_2)
label_3 = np.stack(label_3)
label_stopp = np.stack(label_stopp)
label_rex = np.stack(label_rex)
label_other = np.stack(label_other)

print('Shape of label_a: ', label_a.shape)
print('Shape of label_b: ', label_b.shape)
print('Shape of label_c: ', label_c.shape)
print('Shape of label_1: ', label_1.shape)
print('Shape of label_2: ', label_2.shape)
print('Shape of label_3: ', label_3.shape)
print('Shape of label_stopp: ', label_stopp.shape)
print('Shape of label_rex: ', label_rex.shape)
print('Shape of label_other: ', label_other.shape)

# Create augmentated data for each class
augmentate_data(mfcc_a, label_a, 'audio_processing\Train_Data\set_complete_test_a')
augmentate_data(mfcc_b, label_b, 'audio_processing\Train_Data\set_complete_test_b')
augmentate_data(mfcc_c, label_c, 'audio_processing\Train_Data\set_complete_test_c')
augmentate_data(mfcc_1, label_1, 'audio_processing\Train_Data\set_complete_test_1')
augmentate_data(mfcc_2, label_2, 'audio_processing\Train_Data\set_complete_test_2')
augmentate_data(mfcc_3, label_3, 'audio_processing\Train_Data\set_complete_test_3')
augmentate_data(mfcc_stopp, label_stopp, 'audio_processing\Train_Data\set_complete_test_stopp')
augmentate_data(mfcc_rex, label_rex, 'audio_processing\Train_Data\set_complete_test_rex')
augmentate_data(mfcc_other, label_other, 'audio_processing\Train_Data\set_complete_test_other')

