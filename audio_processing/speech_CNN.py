import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D
#test 
#print(f"Tesorflow version {tf.__version__}")

# load data and split into trainings and test data
data = np.load(f"audio_processing\Train_Data\set1.npy",allow_pickle=True) # load data
data_mfcc = data[:,0,0] # load mfccs, shape (amount of data --> int)
data_labels = data[:,1,0] # load labels, shape (amount of labels --> int)
split_mfcc = int(len(data_mfcc[:])*0.8) # 80% trainings data, 20% test data
split_labels = int(len(data_labels[:])*0.8) # 80% trainings labels, 20% test labels
X_train = data_mfcc[:split_mfcc] # load mfccs of trainings data, 80% of data
X_test = data_mfcc[split_mfcc:]# load test mfcc data, 20% of data
y_train = data_labels[:split_labels] # load train labels, 80% of labels
y_test = data_labels[split_labels:] # load test labels, 20% of labels

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# reshape data
X_train = X_train[0].reshape(12,70,len(X_train[0]))
X_test = X_test[0].reshape(12,70,len(X_test[0]))
y_train = y_train[0].reshape(9,1,len(y_train[0]))
y_test = y_test[0].reshape(9,1,len(y_test[0]))

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

"""# CTC loss function	not implemented in Keras
def CTCLoss(y_true, y_pred):
    # Compute the training-time loss value
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

    ctc_loss = keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return ctc_loss

# CNN
model = Sequential()


model.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(12,70))) # input shape is ???
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="rmsprop", loss=CTCLoss, metrics=["accuracy"])

model.fit(
    X_train.reshape(12,70), 
    y_train, 
    validation_data = (X_test.reshape(12,70), y_test),
    epochs=30, 
    batch_size=1000)"""