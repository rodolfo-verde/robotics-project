import numpy as np
from gmm_hmm_asr.data import DataTuple
from gmm_hmm_asr.trainers import SingleGaussTrainer
from gmm_hmm_asr.trainers import GMMTrainer
from gmm_hmm_asr.trainers import HMMTrainer

import pickle

# load model
with open(f'audio_processing\hmm_model.pkl', 'rb') as f:
    hmm_model = pickle.load(f)
    print("Model loaded")


# load test data
# predict
predict_mfcc = np.load(f"audio_processing\Train_Data\set_complete_test_mfcc.npy",allow_pickle=True) # load data
predict_labels = np.load(f"audio_processing\Train_Data\set_complete_test_label.npy",allow_pickle=True) # load data

labels_string = ["" for x in range(len(predict_labels))]
print(np.size(labels_string))
class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]

# convert one hot encoded labels to class names
for i in range(len(predict_labels)):
    for j in range(len(class_names)):
        if predict_labels[i,j] == 1:
            labels_string[i] = class_names[j]


# create a tuple of mfccs and labels
data = []
for i in range(len(predict_mfcc)):
    data.append((predict_mfcc[i], labels_string[i]))

# create train_data which should be a list of DataTuple(key,feats,label) objects
# key is a unique identifier for each data point
# feats is a numpy array of features
# label is a string containing the label
data = [DataTuple(i, x[0], x[1]) for i, x in enumerate(data)]

print(f"Number of data points: {len(data)}")

# predict
preds = hmm_model.predict(data)
y_pred = [pred[0] for pred in preds]  # predicted labels
y_ll = [pred[1] for pred in preds]  # maximum log-likelihood

# print results
print(f"Predicted labels: {y_pred}")
print(f"Maximum log-likelihood: {y_ll}")

# calculate accuracy
y_true = [x.label for x in data]  # true labels
accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f"Accuracy: {accuracy}")

# calculate accuracy for each class
for i, class_name in enumerate(class_names):
    y_true_class = np.array(y_true) == class_name
    y_pred_class = np.array(y_pred) == class_name
    accuracy_class = np.sum(y_true_class == y_pred_class) / len(y_true_class)
    print(f"Accuracy for class {class_name}: {accuracy_class}")


