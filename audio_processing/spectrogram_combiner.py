import numpy as np
from data_spectrogramm import get_spectrogram 

# load spectrogram data files, the big ones
spectrogram_all = np.load("audio_processing\Train_Data\set_all_spectrogram.npy",allow_pickle=True) # load data
labels_all = np.load("audio_processing\Train_Data\set_all_label.npy",allow_pickle=True) # load data
spectrogram_augmented = np.load("audio_processing\Train_Data\set_all_spectrogram_augmented.npy",allow_pickle=True) # load data
labels_augmented = np.load("audio_processing\Train_Data\set_all_label_augmented.npy",allow_pickle=True) # load data

# print shapes of data
print(f"Data shape of spectrogram_all: {spectrogram_all.shape}")
print(f"Data shape of labels_all: {labels_all.shape}")
print(f"Data shape of spectrogram_augmented: {spectrogram_augmented.shape}")
print(f"Data shape of labels_augmented: {labels_augmented.shape}")

# combine the data
spectrograms = np.concatenate((spectrogram_all, spectrogram_augmented), axis=0)
labels = np.concatenate((labels_all, labels_augmented), axis=0)

# print shapes of combined data
print(f"Data shape of spectrograms: {spectrograms.shape}")
print(f"Data shape of labels: {labels.shape}")

# shuffle the data in the same order
indices = np.arange(spectrograms.shape[0])
np.random.shuffle(indices)
spectrograms = spectrograms[indices]
labels = labels[indices]

# save the combined data
np.save("audio_processing\Train_Data\set_all_spectrogram_combined.npy", spectrograms)
np.save("audio_processing\Train_Data\set_all_label_combined.npy", labels)

"""# load all spectrogram data files in the folder "Train_Data"
spectrogram_eins = np.load("audio_processing\Train_Data\set_eins_200_spectrogram.npy",allow_pickle=True) # load data
spectrogram_zwei = np.load("audio_processing\Train_Data\set_zwei_200_spectrogram.npy",allow_pickle=True) # load data
spectrogram_drei = np.load("audio_processing\Train_Data\set_drei_200_spectrogram.npy",allow_pickle=True) # load data
spectrogram_a = np.load("audio_processing\Train_Data\set_a_200_spectrogram.npy",allow_pickle=True) # load data
spectrogram_b = np.load("audio_processing\Train_Data\set_b_200_spectrogram.npy",allow_pickle=True) # load data
spectrogram_c = np.load("audio_processing\Train_Data\set_c_200_spectrogram.npy",allow_pickle=True) # load data
spectrogram_stopp = np.load("audio_processing\Train_Data\set_stopp_200_spectrogram.npy",allow_pickle=True) # load data
spectrogram_rex = np.load("audio_processing\Train_Data\set_rex_200_spectrogram.npy",allow_pickle=True) # load data
spectrogram_other = np.load("audio_processing\Train_Data\set_other_200_spectrogram.npy",allow_pickle=True) # load data

# print all the shapes of the data
print(f"Data shape of EINS: {spectrogram_eins.shape}")
print(f"Data shape of ZWEI: {spectrogram_zwei.shape}")
print(f"Data shape of DREI: {spectrogram_drei.shape}")
print(f"Data shape of A: {spectrogram_a.shape}")
print(f"Data shape of B: {spectrogram_b.shape}")
print(f"Data shape of C: {spectrogram_c.shape}")
print(f"Data shape of STOPP: {spectrogram_stopp.shape}")
print(f"Data shape of REX: {spectrogram_rex.shape}")
print(f"Data shape of OTHER: {spectrogram_other.shape}")

# load all the labels of the data files
label_eins = np.load("audio_processing\Train_Data\set_eins_200_label.npy",allow_pickle=True) # load data
label_zwei = np.load("audio_processing\Train_Data\set_zwei_200_label.npy",allow_pickle=True) # load data
label_drei = np.load("audio_processing\Train_Data\set_drei_200_label.npy",allow_pickle=True) # load data
label_a = np.load("audio_processing\Train_Data\set_a_200_label.npy",allow_pickle=True) # load data
label_b = np.load("audio_processing\Train_Data\set_b_200_label.npy",allow_pickle=True) # load data
label_c = np.load("audio_processing\Train_Data\set_c_200_label.npy",allow_pickle=True) # load data
label_stopp = np.load("audio_processing\Train_Data\set_stopp_200_label.npy",allow_pickle=True) # load data
label_rex = np.load("audio_processing\Train_Data\set_rex_200_label.npy",allow_pickle=True) # load data
label_other = np.load("audio_processing\Train_Data\set5_label.npy",allow_pickle=True) # load data
label_other = label_other[:200]

# combine all the data into one spectogram array and one label array
spectrograms = np.concatenate((spectrogram_eins, spectrogram_zwei, spectrogram_drei, spectrogram_a, spectrogram_b, spectrogram_c, spectrogram_stopp, spectrogram_rex, spectrogram_other), axis=0)
labels = np.concatenate((label_eins, label_zwei, label_drei, label_a, label_b, label_c, label_stopp, label_rex, label_other), axis=0)
print(f"Data shape of all spectrograms: {spectrograms.shape}")
print(f"Data shape of all labels: {labels.shape}")

# shuffle the labels and spectrograms in the same order
indices = np.arange(spectrograms.shape[0])
np.random.shuffle(indices)
spectrograms = spectrograms[indices]
labels = labels[indices]

# save the combined data
np.save("audio_processing\Train_Data\set_all_spectrogram.npy", spectrograms)
np.save("audio_processing\Train_Data\set_all_label.npy", labels)"""
