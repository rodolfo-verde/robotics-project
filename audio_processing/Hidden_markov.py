import numpy as np
from tuple_hmm import DataTuple
from single_gaussian_trainer import SingleGaussTrainer
from gaussian_mixture_trainer import GMMTrainer
from hidden_markov_trainer import HMMTrainer
import time
import pickle

# load training data of set_complete_test
data_mfcc = np.load(f"audio_processing\Train_Data\set_complete_test_mfcc.npy",allow_pickle=True) # load data
data_labels = np.load(f"audio_processing\Train_Data\set_complete_test_label.npy",allow_pickle=True) # load data

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
print(data[0])
# create train_data which should be a list of DataTuple(key,feats,label) objects
# key is a unique identifier for each data point
# feats is a numpy array of features, in this case mfccs with 11 dimensions
# label is a string containing the label of the data point
data = [DataTuple(i, x[0], x[1]) for i, x in enumerate(data)]

print(f"Number of data points: {len(data)}")
print(data[0].feats.shape)

#print(data[0])

# split data into trainings and test data
split = int(len(data)*0.8) # 80% trainings data, 20% test data
train_data = data[:split] # load mfccs of trainings data, 80% of data
test_data = data[split:] # load mfccs of test data, 20% of data


# parameters for training the model
n_dim = data[0].feats.shape[1] # number of dimensions of the mfccs
n_states = 9 # number of HMM states --> number of classes
n_iter = 10 # number of iterations for training --> variable

# start timer
start = time.time()

hmm_model = GMMTrainer(n_dim, n_states, class_names)
hmm_model.train(data, n_iter)

# end timer
end = time.time()
print(f"Training took {end - start} seconds")

# predict
#start timer
start = time.time()
preds = hmm_model.predict(data)
# end timer
end = time.time()
print(f"Prediction took {end - start} seconds")

correct = 0
for utt, pred in zip(data, preds):
    if pred[0] == utt.label:
        correct += 1

accuracy = float(correct)/len(data) * 100
print(f"Accuracy: {accuracy}")

# save model
with open(f'audio_processing\speech_hmm_model.pkl', 'wb') as f:
    pickle.dump(hmm_model, f)
    print("Model saved")



