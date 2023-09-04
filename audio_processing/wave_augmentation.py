import numpy as np
import librosa
from data_spectrogramm import get_spectrogram

# load all wave files which should be augmented
wave_eins = np.load("audio_processing\Train_Data\set_eins_200_raw.npy",allow_pickle=True) # load data
wave_zwei = np.load("audio_processing\Train_Data\set_zwei_200_raw.npy",allow_pickle=True) # load data
wave_drei = np.load("audio_processing\Train_Data\set_drei_200_raw.npy",allow_pickle=True) # load data
wave_a = np.load("audio_processing\Train_Data\set_a_200_raw.npy",allow_pickle=True) # load data
wave_b = np.load("audio_processing\Train_Data\set_b_200_raw.npy",allow_pickle=True) # load data
wave_c = np.load("audio_processing\Train_Data\set_c_200_raw.npy",allow_pickle=True) # load data
wave_stopp = np.load("audio_processing\Train_Data\set_stopp_200_raw.npy",allow_pickle=True) # load data
wave_rex = np.load("audio_processing\Train_Data\set_rex_200_raw.npy",allow_pickle=True) # load data
wave_other = np.load("audio_processing\Train_Data\set5_raw.npy",allow_pickle=True) # load data
wave_other = wave_other[:200]

# load all labels of the wave files
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

# define a augmentation function which takes a wave array and returns an augmented wave array, with pitch shift and more but keep the same length
def augment_wave(wave):
    # pitch shift
    wave = librosa.effects.pitch_shift(wave, sr=44100, n_steps=4)
    # add noise
    wn = np.random.randn(len(wave))
    wave = wave + 0.005*wn
    # add random noise
    wave = wave + np.random.normal(0, 0.001, len(wave))
    wave = wave + np.random.normal(0, 0.001, len(wave))
    return wave

# augment all the wave arrays
wave_eins_augmented = []
wave_zwei_augmented = []
wave_drei_augmented = []
wave_a_augmented = []
wave_b_augmented = []
wave_c_augmented = []
wave_stopp_augmented = []
wave_rex_augmented = []
wave_other_augmented = []
for i in range(wave_eins.shape[0]):
    wave_eins_augmented.append(augment_wave(wave_eins[i]))
for i in range(wave_zwei.shape[0]):
    wave_zwei_augmented.append(augment_wave(wave_zwei[i]))
for i in range(wave_drei.shape[0]):
    wave_drei_augmented.append(augment_wave(wave_drei[i]))
for i in range(wave_a.shape[0]):
    wave_a_augmented.append(augment_wave(wave_a[i]))
for i in range(wave_b.shape[0]):
    wave_b_augmented.append(augment_wave(wave_b[i]))
for i in range(wave_c.shape[0]):
    wave_c_augmented.append(augment_wave(wave_c[i]))
for i in range(wave_stopp.shape[0]):
    wave_stopp_augmented.append(augment_wave(wave_stopp[i]))
for i in range(wave_rex.shape[0]):
    wave_rex_augmented.append(augment_wave(wave_rex[i]))
for i in range(wave_other.shape[0]):
    wave_other_augmented.append(augment_wave(wave_other[i]))
wave_eins_augmented = np.array(wave_eins_augmented)
wave_zwei_augmented = np.array(wave_zwei_augmented)
wave_drei_augmented = np.array(wave_drei_augmented)
wave_a_augmented = np.array(wave_a_augmented)
wave_b_augmented = np.array(wave_b_augmented)
wave_c_augmented = np.array(wave_c_augmented)
wave_stopp_augmented = np.array(wave_stopp_augmented)
wave_rex_augmented = np.array(wave_rex_augmented)
wave_other_augmented = np.array(wave_other_augmented)

# print the shapes of the augmented waves
print(f"Data shape of EINS: {wave_eins_augmented.shape}")
print(f"Data shape of ZWEI: {wave_zwei_augmented.shape}")
print(f"Data shape of DREI: {wave_drei_augmented.shape}")
print(f"Data shape of A: {wave_a_augmented.shape}")
print(f"Data shape of B: {wave_b_augmented.shape}")
print(f"Data shape of C: {wave_c_augmented.shape}")
print(f"Data shape of STOPP: {wave_stopp_augmented.shape}")
print(f"Data shape of REX: {wave_rex_augmented.shape}")
print(f"Data shape of OTHER: {wave_other_augmented.shape}")

# convert the augmented wave files into spectrograms
# convert wave to spectrogram
spectrogram_eins = []
spectrogram_zwei = []
spectrogram_drei = []
spectrogram_a = []
spectrogram_b = []
spectrogram_c = []
spectrogram_rex = []
spectrogram_stopp = []
spectrogram_other = []

for i in range(wave_eins_augmented.shape[0]):
    spectrogram_eins.append(get_spectrogram(wave_eins_augmented[i]))
spectrogram_eins = np.array(spectrogram_eins)

for i in range(wave_zwei_augmented.shape[0]):
    spectrogram_zwei.append(get_spectrogram(wave_zwei_augmented[i]))
spectrogram_zwei = np.array(spectrogram_zwei)

for i in range(wave_drei_augmented.shape[0]):
    spectrogram_drei.append(get_spectrogram(wave_drei_augmented[i]))
spectrogram_drei = np.array(spectrogram_drei)

for i in range(wave_a_augmented.shape[0]):
    spectrogram_a.append(get_spectrogram(wave_a_augmented[i]))
spectrogram_a = np.array(spectrogram_a)

for i in range(wave_b_augmented.shape[0]):
    spectrogram_b.append(get_spectrogram(wave_b_augmented[i]))
spectrogram_b = np.array(spectrogram_b)

for i in range(wave_c_augmented.shape[0]):
    spectrogram_c.append(get_spectrogram(wave_c_augmented[i]))
spectrogram_c = np.array(spectrogram_c)

for i in range(wave_rex_augmented.shape[0]):
    spectrogram_rex.append(get_spectrogram(wave_rex_augmented[i]))
spectrogram_rex = np.array(spectrogram_rex)

for i in range(wave_stopp_augmented.shape[0]):
    spectrogram_stopp.append(get_spectrogram(wave_stopp_augmented[i]))
spectrogram_stopp = np.array(spectrogram_stopp)

for i in range(wave_other_augmented.shape[0]):
    spectrogram_other.append(get_spectrogram(wave_other_augmented[i]))
spectrogram_other = np.array(spectrogram_other)

# print the shapes of the spectrograms
print(f"Data shape of EINS: {spectrogram_eins.shape}")
print(f"Data shape of ZWEI: {spectrogram_zwei.shape}")
print(f"Data shape of DREI: {spectrogram_drei.shape}")
print(f"Data shape of A: {spectrogram_a.shape}")
print(f"Data shape of B: {spectrogram_b.shape}")
print(f"Data shape of C: {spectrogram_c.shape}")
print(f"Data shape of STOPP: {spectrogram_stopp.shape}")
print(f"Data shape of REX: {spectrogram_rex.shape}")
print(f"Data shape of OTHER: {spectrogram_other.shape}")

# combine all the spectrograms into one spectrogram array, and all labels into one label array
spectrograms = np.concatenate((spectrogram_eins, spectrogram_zwei, spectrogram_drei, spectrogram_a, spectrogram_b, spectrogram_c, spectrogram_stopp, spectrogram_rex, spectrogram_other), axis=0)
print(f"Data shape of all spectrograms: {spectrograms.shape}")
labels = np.concatenate((label_eins, label_zwei, label_drei, label_a, label_b, label_c, label_stopp, label_rex, label_other), axis=0)
print(f"Data shape of all labels: {labels.shape}")

# save the augmented spectrograms and labels
np.save("audio_processing\Train_Data\set_all_spectrogram_augmented.npy", spectrograms)
np.save("audio_processing\Train_Data\set_all_label_augmented.npy", labels)



