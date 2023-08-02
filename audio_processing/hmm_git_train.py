import numpy as np
from gmm_hmm_asr.data import DataTuple
from gmm_hmm_asr.trainers import SingleGaussTrainer
from gmm_hmm_asr.trainers import GMMTrainer
from gmm_hmm_asr.trainers import HMMTrainer


# load data and split into trainings and test data
data_mfcc = np.load(f"audio_processing\Train_Data\set_complete_test_mfcc.npy",allow_pickle=True) # load data
data_labels = np.load(f"audio_processing\Train_Data\set_complete_test_label.npy",allow_pickle=True) # load data

print(f"Data shape: {data_mfcc.shape}")
print(f"Labels shape: {data_labels.shape}")
print(len(data_labels))
labels_string = ["" for x in range(len(data_labels))]
print(np.size(labels_string))
class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]
# convert one hot encoded labels to class names
for i in range(len(data_labels)):
    for j in range(len(class_names)):
        if data_labels[i,j] == 1:
            labels_string[i] = class_names[j]


# create a tuple of mfccs and labels
data = []
for i in range(len(data_mfcc)):
    data.append((data_mfcc[i], labels_string[i]))

# create train_data which should be a list of DataTuple(key,feats,label) objects
# key is a unique identifier for each data point
# feats is a numpy array of features
# label is a string containing the label
data = [DataTuple(i, x[0], x[1]) for i, x in enumerate(data)]

# split data into trainings and test data
split = int(len(data)*0.8) # 80% trainings data, 20% test data
train_data = data[:split] # load mfccs of trainings data, 80% of data
test_data = data[split:] # load mfccs of test data, 20% of data

# All Models

ndim = 70 # dimensionality of features
class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"] # class_names to be recognized

# singe gaussian model
sg_model = SingleGaussTrainer(ndim, class_names)
sg_model.train(train_data)

preds = sg_model.predict(test_data)
y_pred = [pred[0] for pred in preds]  # predicted labels
y_ll = [pred[1] for pred in preds]  # maximum log-likelihood

from gmm_hmm_asr.data import DataTuple
from gmm_hmm_asr.trainers import HMMTrainer

# Gaussian mixture model
nstate = 9 # number of HMM states
ncomp = 8 # number of Gaussian components
niter = 10 # number of training iterations

gmm_model = GMMTrainer(ndim, ncomp, class_names)
gmm_model.train(train_data, niter)

preds = gmm_model.predict(test_data)

hmm_model = GMMTrainer(ndim, nstate, class_names)
hmm_model.train(train_data, niter)

# Hidden markov model
hmm_model = GMMTrainer(ndim, nstate, class_names)
hmm_model.train(train_data, niter)

preds = hmm_model.predict(test_data)


# save model
import pickle
with open('hmm_model.pkl', 'wb') as f:
    pickle.dump(hmm_model, f)
 

# predict
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



