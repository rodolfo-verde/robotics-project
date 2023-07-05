import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D
from keras.optimizers import SGD
from keras.regularizers import L2 

#test 
#print(f"Tesorflow version {tf.__version__}")

# load data and split into trainings and test data
data_mfcc = np.load(f"audio_processing\Train_Data\set_complete_test_mfcc.npy",allow_pickle=True) # load data
data_labels = np.load(f"audio_processing\Train_Data\set_complete_test_label.npy",allow_pickle=True) # load data

print(f"Data shape: {data_mfcc.shape}")
print(f"Labels shape: {data_labels.shape}")

split_mfcc = int(len(data_mfcc[:,10,69])*0.8) # 80% trainings data, 20% test data
split_labels = int(len(data_labels[:,8])*0.8) # 80% trainings labels, 20% test labels
X_train = data_mfcc[:split_mfcc] # load mfccs of trainings data, 80% of data
X_test = data_mfcc[split_mfcc:]# load test mfcc data, 20% of data
y_train = data_labels[:split_labels] # load train labels, 80% of labels
y_test = data_labels[split_labels:] # load test labels, 20% of labels

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")



# CNN
model = Sequential()

model.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(11,70,1), padding="same"))
model.add(MaxPooling2D(pool_size=(5, 5), padding="same"))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(10, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))
model.add(Dense(9, activation="softmax"))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]) # optimizer = rmsprop, Adam     loss = categorical_crossentropy, CTCLoss


result = model.fit(
    X_train.reshape(-1, 11, 70, 1), 
    y_train, 
    validation_data = (X_test.reshape(-1, 11, 70, 1), y_test),
    epochs=100, 
    batch_size=100)

model.summary()

# evaluate model
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 11, 70, 1), y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# plot accuracy and loss
plt.figure(1)
plt.subplot(121)
plt.plot(result.history["accuracy"])
plt.plot(result.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.subplot(122)
plt.plot(result.history["loss"])
plt.plot(result.history["val_loss"])
plt.title("Model loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")
plt.show()