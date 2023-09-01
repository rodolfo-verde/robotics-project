import numpy as np 
import matplotlib.pyplot as plt
import scipy.signal as signal
import tensorflow as tf
from data_spectrogramm import get_spectrogram, plot_spectrogram

# load wave file for processing
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
axes[0].set_xlim([0, 16000])

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
np.save(f"audio_processing\Train_Data\set_other_200_spectrogram.npy", spectrograms) # X, 252,129 = shape of spectrograms