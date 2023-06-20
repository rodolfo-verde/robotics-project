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
data_mfcc = np.load(f"audio_processing\Train_Data\set3_mfcc.npy",allow_pickle=True) # load data
data_labels = np.load(f"audio_processing\Train_Data\set3_label.npy",allow_pickle=True) # load data

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

#tf.nn.ctc_loss(
#    labels,
#    logits,
#    label_length,
#    logit_length,
#    logits_time_major=True,
#   unique=None,
#    blank_index=None,
#    name=None
#)
#ctc_loss = tf.nn.ctc_loss(y_train, X_train, label_length=y_train.shape, logit_length=X_train.shape, logits_time_major=True, unique=None, blank_index=None, name="ctc_loss_dense")

# CNN
model = Sequential()


model.add(Conv2D(10, kernel_size=(3, 3), activation="sigmoid", input_shape=(11,70,1))) 
#model.add(Flatten())
#model.add(BatchNormalization())
#model.add(Dropout(0.02))
#model.add(Dense(10, activation="sigmoid"))
#model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(9, activation="softmax"))

model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"]) # optimizer = rmsprop, Adam         loss = categorical_crossentropy, CTCLoss

model.fit(
    X_train.reshape(-1, 11, 70, 1), 
    y_train, 
    validation_data = (X_test.reshape(-1, 11, 70, 1), y_test),
    epochs=30, 
    batch_size=1)

model.summary()

# evaluate model
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 11, 70, 1), y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# predict
class_names = ["a", "b", "c", "1", "2", "3", "rex", "stopp", "other"]
predict_mfcc = np.load(f"audio_processing\Train_Data\set_test_a1_mfcc.npy",allow_pickle=True) # load data
predict_labels = np.load(f"audio_processing\Train_Data\set_test_a1_label.npy",allow_pickle=True) # load data
print(f"Predict shape: {predict_mfcc.shape}")
print(f"Labels shape: {predict_labels.shape}")
predict = predict_mfcc[0]
prediction = model.predict(predict.reshape(-1, 11, 70, 1))
index_pred = np.argmax(prediction) #tf.argmax geht auch
index_label = np.argmax(predict_labels)
print(f"Prediction: {class_names[index_pred]}")
print(f"Label: {class_names[index_label]}")