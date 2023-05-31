import pandas as pd
import numpy as np
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization, Dropout, MaxPooling2D
#test 
print(f"Tesorflow version {tf.__version__}")

# load data and split into trainings and test data
data = # load data
split = int(len(data)*0.8) # 80% trainings data, 20% test data
X_train = data# load trainings data of voice samples
y_train = # load labels of trainings data --> commands

# CTC loss function	not implemented in Keras
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


model.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(28, 28, 1))) # input shape is ???
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(10, activation="softmax"))

model.compile(optimizer="rmsprop", loss=CTCLoss, metrics=["accuracy"])

model.fit(
    X_train.reshape(-1, 28, 28, 1), 
    y_train, 
    validation_data = (X_test.reshape(-1, 28, 28, 1), y_test),
    epochs=30, 
    batch_size=1000)