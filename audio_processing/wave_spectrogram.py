import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as signal
import tensorflow as tf
from data_spectrogramm import get_spectrogram, plot_spectrogram

# load all set_200_raw.npy files for processing

wave_eins = np.load("audio_processing\Train_Data\set_eins_200_raw.npy",allow_pickle=True) # load data
wave_zwei = np.load("audio_processing\Train_Data\set_zwei_200_raw.npy",allow_pickle=True) # load data
wave_drei = np.load("audio_processing\Train_Data\set_drei_200_raw.npy",allow_pickle=True) # load data
wave_a = np.load("audio_processing\Train_Data\set_a_200_raw.npy",allow_pickle=True) # load data
wave_b = np.load("audio_processing\Train_Data\set_b_200_raw.npy",allow_pickle=True) # load data
wave_c = np.load("audio_processing\Train_Data\set_c_200_raw.npy",allow_pickle=True) # load data
wave_rex = np.load("audio_processing\Train_Data\set_rex_200_raw.npy",allow_pickle=True) # load data
wave_stopp = np.load("audio_processing\Train_Data\set_stopp_200_raw.npy",allow_pickle=True) # load data
wave_other = np.load("audio_processing\Train_Data\set5_raw.npy",allow_pickle=True) # load data
# choose the first 200 words of the other file
wave_other = wave_other[:200]

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

for i in range(wave_eins.shape[0]):
    spectrogram_eins.append(get_spectrogram(wave_eins[i]))
spectrogram_eins = np.array(spectrogram_eins)
print(f"Spectrograms shape: {spectrogram_eins.shape}")

for i in range(wave_zwei.shape[0]):
    spectrogram_zwei.append(get_spectrogram(wave_zwei[i]))
spectrogram_zwei = np.array(spectrogram_zwei)
print(f"Spectrograms shape: {spectrogram_zwei.shape}")

for i in range(wave_drei.shape[0]):
    spectrogram_drei.append(get_spectrogram(wave_drei[i]))
spectrogram_drei = np.array(spectrogram_drei)
print(f"Spectrograms shape: {spectrogram_drei.shape}")

for i in range(wave_a.shape[0]):
    spectrogram_a.append(get_spectrogram(wave_a[i]))
spectrogram_a = np.array(spectrogram_a)
print(f"Spectrograms shape: {spectrogram_a.shape}")

for i in range(wave_b.shape[0]):
    spectrogram_b.append(get_spectrogram(wave_b[i]))
spectrogram_b = np.array(spectrogram_b)
print(f"Spectrograms shape: {spectrogram_b.shape}")

for i in range(wave_c.shape[0]):
    spectrogram_c.append(get_spectrogram(wave_c[i]))
spectrogram_c = np.array(spectrogram_c)
print(f"Spectrograms shape: {spectrogram_c.shape}")

for i in range(wave_rex.shape[0]):
    spectrogram_rex.append(get_spectrogram(wave_rex[i]))
spectrogram_rex = np.array(spectrogram_rex)
print(f"Spectrograms shape: {spectrogram_rex.shape}")

for i in range(wave_stopp.shape[0]):
    spectrogram_stopp.append(get_spectrogram(wave_stopp[i]))
spectrogram_stopp = np.array(spectrogram_stopp)
print(f"Spectrograms shape: {spectrogram_stopp.shape}")

for i in range(wave_other.shape[0]):
    spectrogram_other.append(get_spectrogram(wave_other[i]))
spectrogram_other = np.array(spectrogram_other)
print(f"Spectrograms shape: {spectrogram_other.shape}")

# save spectrograms as .npy file
np.save(f"audio_processing\Train_Data\set_eins_200_spectrogram.npy", spectrogram_eins) # X, 251,129 = shape of spectrograms
np.save(f"audio_processing\Train_Data\set_zwei_200_spectrogram.npy", spectrogram_zwei) # X, 251,129 = shape of spectrograms
np.save(f"audio_processing\Train_Data\set_drei_200_spectrogram.npy", spectrogram_drei) # X, 251,129 = shape of spectrograms
np.save(f"audio_processing\Train_Data\set_a_200_spectrogram.npy", spectrogram_a) # X, 251,129 = shape of spectrograms
np.save(f"audio_processing\Train_Data\set_b_200_spectrogram.npy", spectrogram_b) # X, 251,129 = shape of spectrograms
np.save(f"audio_processing\Train_Data\set_c_200_spectrogram.npy", spectrogram_c) # X, 251,129 = shape of spectrograms
np.save(f"audio_processing\Train_Data\set_rex_200_spectrogram.npy", spectrogram_rex) # X, 251,129 = shape of spectrograms
np.save(f"audio_processing\Train_Data\set_stopp_200_spectrogram.npy", spectrogram_stopp) # X, 251,129 = shape of spectrograms
np.save(f"audio_processing\Train_Data\set_other_200_spectrogram.npy", spectrogram_other) # X, 251,129 = shape of spectrograms


"""# load wave file for processing
wave = np.load("audio_processing\Train_Data\set5_raw.npy",allow_pickle=True) # load data
# choose the first 200 words of the wave file
wave = wave[:200]
#labels = np.load("audio_processing\Train_Data\set_eins_200_label.npy",allow_pickle=True) # load data
print(f"Data shape: {wave.shape}")
#print(f"Labels shape: {labels.shape}")

# plot first word of wave
plt.figure()
plt.plot(wave[0])
plt.show()

# convert wave to spectrogram
spectrogram = get_spectrogram(wave[0])
print(f"Spectrogram shape: {spectrogram.shape}")

# plot spectrogram
fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(wave[0].shape[0])
axes[0].plot(timescale, wave[0])
axes[0].set_title('Waveform')

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
#plt.suptitle("Command: Eins")
plt.show()

# convert the whole wave file into spectrograms
spectrograms = []
for i in range(wave.shape[0]):
    spectrograms.append(get_spectrogram(wave[i]))
spectrograms = np.array(spectrograms)
print(f"Spectrograms shape: {spectrograms.shape}")

# save spectrograms as .npy file
np.save(f"audio_processing\Train_Data\set_other_200_spectrogram.npy", spectrograms) # X, 252,129 = shape of spectrograms"""