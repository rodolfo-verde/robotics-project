import numpy as np
from tuple_hmm import DataTuple
from single_gaussian_trainer import SingleGaussTrainer
from gaussian_mixture_trainer import GMMTrainer
from hidden_markov_trainer import HMMTrainer
import time
import pickle

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
# feats is a numpy array of features, in this case mfccs with 11 dimensions
# label is a string containing the label of the data point
data = [DataTuple(i, x[0], x[1]) for i, x in enumerate(data)]

print(f"Number of data points: {len(data)}")
print(data[0].feats.shape)

# split data into trainings and test data
split = int(len(data)*0.8) # 80% trainings data, 20% test data
train_data = data[:split] # load mfccs of trainings data, 80% of data
test_data = data[split:] # load mfccs of test data, 20% of data

# parameters for training the model
n_dim = data[0].feats.shape[1] # number of dimensions of the mfccs
n_states = 9 # number of HMM states --> number of classes
n_iter = 10 # number of iterations for training --> variable
n_comp = 8 # number of gaussian components --> variable


## Train the Model ##

# Single Gaussian Model
# start time
start_sg = time.time()

sg_model = SingleGaussTrainer(n_dim, class_names)
sg_model.train(train_data)

preds = sg_model.predict(test_data)
y_pred = [pred[0] for pred in preds]  # predicted labels
y_ll = [pred[1] for pred in preds]  # maximum log-likelihood

# end time
end_sg = time.time()
print("Time taken to train the model (Single Gaussian): ", end_sg - start_sg)

# Gaussian Mixture Model
# start time
start_gmm = time.time()
gmm_model = GMMTrainer(n_dim, n_comp, class_names)
gmm_model.train(train_data, n_iter)

preds = gmm_model.predict(test_data)

# end time
end_gmm = time.time()
print("Time taken to train the model (Gaussian Mixture): ", end_gmm - start_gmm)

# HMM Model
# start time
start_hmm = time.time()

hmm_model = GMMTrainer(n_dim, n_states, class_names)
hmm_model.train(train_data, n_iter)

preds = hmm_model.predict(test_data)

# end time
end_hmm = time.time()
print("Time taken to train the model (HMM): ", end_hmm - start_hmm)

# Evaluation
# predict
y_pred = [pred[0] for pred in preds]  # predicted labels
y_ll = [pred[1] for pred in preds]  # maximum log-likelihood

# print results
print(f"Predicted labels: {y_pred}")
print(f"Maximum log-likelihood: {y_ll}")

"""# calculate accuracy
y_true = [x.label for x in data]  # true labels
accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f"Accuracy: {accuracy}")

# calculate accuracy for each class
for i, class_name in enumerate(class_names):
    y_true_class = np.array(y_true) == class_name
    y_pred_class = np.array(y_pred) == class_name
    accuracy_class = np.sum(y_true_class == y_pred_class) / len(y_true_class)
    print(f"Accuracy for class {class_name}: {accuracy_class}")"""

# save the model
with open(f'audio_processing\speech_hmm_model.pkl', 'wb') as f:
    pickle.dump(hmm_model, f)



