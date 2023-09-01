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

# define a augmentation function which takes a wave array and returns an augmented wave array, with pitch shift and more but keep the same length
def augment_wave(wave):
    # pitch shift
    wave = librosa.effects.pitch_shift(wave, 16000, n_steps=4)
    # add noise
    wn = np.random.randn(len(wave))
    wave = wave + 0.005*wn
    # add random noise
    wave = wave + np.random.normal(0, 0.001, len(wave))
    wave = wave + np.random.normal(0, 0.001, len(wave))

    return wave

# augment all the waves in the wave files and then put them together in one wave file
wave_eins_augmented = augment_wave(wave_eins[0])
for i in range(1, wave_eins.shape[0]):
    wave_eins_augmented = np.concatenate((wave_eins_augmented, augment_wave(wave_eins[i])), axis=0)
wave_zwei_augmented = augment_wave(wave_zwei[0])
for i in range(1, wave_zwei.shape[0]):
    wave_zwei_augmented = np.concatenate((wave_zwei_augmented, augment_wave(wave_zwei[i])), axis=0)
wave_drei_augmented = augment_wave(wave_drei[0])
for i in range(1, wave_drei.shape[0]):
    wave_drei_augmented = np.concatenate((wave_drei_augmented, augment_wave(wave_drei[i])), axis=0)
wave_a_augmented = augment_wave(wave_a[0])
for i in range(1, wave_a.shape[0]):
    wave_a_augmented = np.concatenate((wave_a_augmented, augment_wave(wave_a[i])), axis=0)
wave_b_augmented = augment_wave(wave_b[0])
for i in range(1, wave_b.shape[0]):
    wave_b_augmented = np.concatenate((wave_b_augmented, augment_wave(wave_b[i])), axis=0)
wave_c_augmented = augment_wave(wave_c[0])
for i in range(1, wave_c.shape[0]):
    wave_c_augmented = np.concatenate((wave_c_augmented, augment_wave(wave_c[i])), axis=0)
wave_stopp_augmented = augment_wave(wave_stopp[0])
for i in range(1, wave_stopp.shape[0]):
    wave_stopp_augmented = np.concatenate((wave_stopp_augmented, augment_wave(wave_stopp[i])), axis=0)
wave_rex_augmented = augment_wave(wave_rex[0])
for i in range(1, wave_rex.shape[0]):
    wave_rex_augmented = np.concatenate((wave_rex_augmented, augment_wave(wave_rex[i])), axis=0)
wave_other_augmented = augment_wave(wave_other[0])
for i in range(1, wave_other.shape[0]):
    wave_other_augmented = np.concatenate((wave_other_augmented, augment_wave(wave_other[i])), axis=0)

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
spectrogram_eins = get_spectrogram(wave_eins_augmented)
spectrogram_zwei = get_spectrogram(wave_zwei_augmented)
spectrogram_drei = get_spectrogram(wave_drei_augmented)
spectrogram_a = get_spectrogram(wave_a_augmented)
spectrogram_b = get_spectrogram(wave_b_augmented)
spectrogram_c = get_spectrogram(wave_c_augmented)
spectrogram_stopp = get_spectrogram(wave_stopp_augmented)
spectrogram_rex = get_spectrogram(wave_rex_augmented)
spectrogram_other = get_spectrogram(wave_other_augmented)

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

# combine all the spectrograms into one spectrogram array
spectrograms = np.concatenate((spectrogram_eins, spectrogram_zwei, spectrogram_drei, spectrogram_a, spectrogram_b, spectrogram_c, spectrogram_stopp, spectrogram_rex, spectrogram_other), axis=0)
print(f"Data shape of all spectrograms: {spectrograms.shape}")

# save the augmented spectrograms
np.save("audio_processing\Train_Data\set_all_spectrogram_augmented.npy", spectrograms)



