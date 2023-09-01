import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import tensorflow as tf

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

def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  return spectrogram

# convert wave to spectrogram
spectrogram = get_spectrogram(wave[0])
print(f"Spectrogram shape: {spectrogram.shape}")

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  # Convert the frequencies to log scale and transpose, so that the time is
  # represented on the x-axis (columns).
  # Add an epsilon to avoid taking a log of zero.
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

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

