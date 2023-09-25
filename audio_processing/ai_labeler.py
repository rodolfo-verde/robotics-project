import numpy as np

data = np.load('data.npy')

size = data.shape[0]

labels = np.zeros((size, 1))

for i in range(size):
    labels[i] = "a"

np.save('labels.npy', labels)