import numpy as np
from scipy.io import wavfile
import librosa
from scipy.signal import resample
import sys

class Augmentator():

    def __init__(self):
        self.class_names = ['a', 'b', 'c', '1', '2', '3', 'stopp', 'rex', 'other']

    def augmentate_data(self, mfcc_stored, label_array, output_file):
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

            # save all mfcc as npy filess
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
            return mfcc_stored_cutout, mfcc_stored_mixup, mfcc_stored_sample_pairing, mfcc_stored_specaugment, mfcc_stored_specmix, mfcc_stored_vh_mixup, mfcc_stored_mixed_frequency_masking

    #class_names = ['a', 'b', 'c', '1', '2', '3', 'stopp', 'rex', 'other']
    # load data
    #mfcc_data = np.load('audio_processing\Train_Data\set_complete_test_mfcc.npy')
    #label_data = np.load('audio_processing\Train_Data\set_complete_test_label.npy')

    def load_data(self):
        filename = input('Enter the filename of the data: ')
        mfcc_data = np.load("audio_processing/Train_Data/"+filename+"_mfcc.npy")
        label_data = np.load("audio_processing/Train_Data/"+filename+"_label.npy")
        return mfcc_data, label_data

    def separate_data(self, mfcc_data, label_data):
        mfccs = [[] for _ in range(9)]
        labels = [[] for _ in range(9)]
        
        for i in range(mfcc_data.shape[0]):
            for j in range(9):
                if label_data[i,j] == 1:
                    mfccs[j].append(mfcc_data[i,:,:])
                    labels[j].append(label_data[i,:])
        
        for i in range(9):
            mfccs[i] = np.stack(mfccs[i])
            labels[i] = np.stack(labels[i])

        return mfccs, labels

    def print_shapes(self, mfccs, labels):
        for i, (mfcc, label) in enumerate(zip(mfccs, labels)):
            print(f'Shape of mfcc_{i}: ', mfcc.shape)
            print(f'Shape of label_{i}: ', label.shape)

    def label_augmentations(self, output_file, augmentation_file):
        # Create new label arrays for each augmentation depending on the number of augmentations and on the class
        # 0 = a, 1 = b, 2 = c, 3 = 1, 4 = 2, 5 = 3, 6 = stopp, 7 = rex, 8 = other
        # create new label arrays depending on the class (index i) and size of mfcc_stored
        mfcc_stored = np.load(f'{augmentation_file}.npy')
        label_stored = np.zeros((mfcc_stored.shape[0], 9))
        # Find the index of the augmentation file, its the only number in the filename
        i = int(''.join(filter(str.isdigit, output_file)))
        print("Index of the augmentation file: ", i)
        # Load the augmentation data
        print(f'Shape of label_stored_{i}: ', label_stored.shape)
        # set the index i to 1 of the label array
        label_stored[:,i] = 1
        print(" One label is: ", label_stored[0])
        np.save(f'{output_file}_label', label_stored)
        print(f'Shape of label_stored_{i}: ', label_stored.shape)

    def combine_data(self, augmentation, output_file):
        # combine all augmentations into one file
        mfcc_stored_cutout, mfcc_stored_mixup, mfcc_stored_sample_pairing, mfcc_stored_specaugment, mfcc_stored_specmix, mfcc_stored_vh_mixup, mfcc_stored_mixed_frequency_masking = augmentation
        mfcc_stored = np.concatenate((mfcc_stored_cutout, mfcc_stored_mixup, mfcc_stored_sample_pairing, mfcc_stored_specaugment, mfcc_stored_specmix, mfcc_stored_vh_mixup, mfcc_stored_mixed_frequency_masking), axis=0)
        np.save(output_file+"_mfcc", mfcc_stored)
        print('Shape of mfcc_stored: ', mfcc_stored.shape)
        mfcc_file = output_file+"_mfcc"
        return mfcc_file, output_file   
        

    def start_augmentate(self, mfccs, labels):
        output_file = input('Enter the output file name: ')
        # Augmentate the files
        for i, (mfcc, label) in enumerate(zip(mfccs, labels)):
            augmentation = self.augmentate_data(mfcc, label, f'audio_processing/Train_Data/{output_file}_{i}')
            mfcc_file, output_file_2 = self.combine_data(augmentation, f'audio_processing/Train_Data/{output_file}_{i}')
            # label augmentation
            self.label_augmentations(output_file_2, mfcc_file)

    def main(self):
        mfcc_data, label_data = self.load_data()
        print('Shape of mfcc_data: ', mfcc_data.shape)
        print('Shape of label_data: ', label_data.shape)
        
        mfccs, labels = self.separate_data(mfcc_data, label_data)
        self.print_shapes(mfccs, labels)
        self.start_augmentate(mfccs, labels)

if __name__ == "__main__":
    Augmentator = Augmentator()
    Augmentator.main()
