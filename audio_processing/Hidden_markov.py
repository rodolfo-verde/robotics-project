import numpy as np
from tuple_hmm import DataTuple
from single_gaussian_trainer import SingleGaussTrainer
from gaussian_mixture_trainer import GMMTrainer
from hidden_markov_trainer import HMMTrainer
import time
import pickle

# This script will be used to create a Hidden Markov Model for the speech recognition
# The model will be trained on the data set set_complete_test_mfcc.npy
# The model will be saved as speech_hmm_model.pkl


# load data set_complete_test
mfcc_data = np.load("audio_processing\Train_Data\set_complete_test_mfcc.npy", allow_pickle=True)
label_data = np.load("audio_processing\Train_Data\set_complete_test_label.npy", allow_pickle=True)

# create a list of all the different labels
class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]
labels_string = ["" for x in range(len(label_data))]
# convert one hot encoded labels to class names
for i in range(len(label_data)):
    for j in range(len(class_names)):
        if label_data[i,j] == 1:
            labels_string[i] = class_names[j]

# create size variables for length of data and number of dimensions
n_dim_tuple = mfcc_data[0].shape[1]
n_states_tuple = len(class_names)
n_keys_tuple = len(mfcc_data)

random_key = np.random.randint(0, n_keys_tuple)
random_feats = mfcc_data[random_key]
random_label = labels_string[random_key]

data_sets = DataTuple(key=random_key, feats=random_feats, label=random_label)

for i in range(len(class_names)):
    data_sets = DataTuple(n_keys_tuple, n_dim_tuple, class_names)
data_set_0 = DataTuple(key=random_key, feats=random_feats, label=random_label)


for i in range(len(class_names)):
    data_sets[i].feats = mfcc_data[labels_string == class_names[i]]
    data_sets[i].label = labels_string[labels_string == class_names[i]]
    data_sets[i].key = np.arange(len(data_sets[i].label))

# create training and test DataTuple objects for each class, 80% training, 20% test
train_data = DataTuple(n_keys_tuple, n_dim_tuple, class_names)
test_data = DataTuple(n_keys_tuple, n_dim_tuple, class_names)
for i in range(len(class_names)):
    train_data[i].feats = data_sets[i].feats[0:int(0.8*len(data_sets[i].feats))]
    train_data[i].label = data_sets[i].label[0:int(0.8*len(data_sets[i].label))]
    train_data[i].key = np.arange(len(train_data[i].label))
    test_data[i].feats = data_sets[i].feats[int(0.8*len(data_sets[i].feats)):len(data_sets[i].feats)]
    test_data[i].label = data_sets[i].label[int(0.8*len(data_sets[i].label)):len(data_sets[i].label)]
    test_data[i].key = np.arange(len(test_data[i].label))
    print(f"Training data for {class_names[i]}: {len(train_data[i].feats)}")
    print(f"Test data for {class_names[i]}: {len(test_data[i].feats)}")

# parameters for training the model
n_dim = data_sets[0].feats.shape # number of dimensions of the mfccs
n_states = 9 # number of HMM states --> number of classes
n_iter = 10 # number of iterations for training --> variable
n_comp = 8 # number of gaussian components --> variable
# create a HMM for each data set and save it to a file per data set
# replace the DataTuple object with the mfccs and labels for each class

# create a single gaussian model for each class
# start time
start_single = time.time()
for i in range(len(data_sets)):
    print(f"Training single gaussian model for {class_names[i]}")
    single_gauss_model = SingleGaussTrainer(n_dim,class_names[i])
    single_gauss_model.train(train_data[i])
    with open(f'audio_processing\HMM_models\class_{class_names[i]}_single_gauss.pkl', 'wb') as f:
        pickle.dump(single_gauss_model, f)
    print(f"Single gaussian model for {class_names[i]} saved")
end_single = time.time()
print(f"Time to train single gaussian models: {end_single-start_single}")

# create a gaussian mixture model for each class
# start time
start_gmm = time.time()
for i in range(len(data_sets)):
    print(f"Training gaussian mixture model for {class_names[i]}")
    gmm_model = GMMTrainer(n_dim,n_comp,class_names[i])
    gmm_model.train(train_data[i],n_iter)
    with open(f'audio_processing\HMM_models\class_{class_names[i]}_gmm.pkl', 'wb') as f:
        pickle.dump(gmm_model, f)
    print(f"Gaussian mixture model for {class_names[i]} saved")

end_gmm = time.time()
print(f"Time to train gaussian mixture models: {end_gmm-start_gmm}")

# start time
start_hmm = time.time()
for i in range(len(data_sets)):
    print(f"Training HMM for {class_names[i]}")
    hmm_model = HMMTrainer(n_dim,n_states,class_names)
    hmm_model.train(train_data[i],n_iter)
    with open(f'audio_processing\HMM_models\class_{class_names[i]}_hmm.pkl', 'wb') as f:
        pickle.dump(hmm_model, f)
    print(f"HMM for {class_names[i]} saved")
end_hmm = time.time()
print(f"Time to train HMMs: {end_hmm-start_hmm}")

# calculate the accuracy of the model
# load the HMMs
hmm_a = pickle.load(open(f"audio_processing\HMM_models\class_a_hmm.pkl", "rb"))
hmm_b = pickle.load(open(f"audio_processing\HMM_models\class_b_hmm.pkl", "rb"))
hmm_c = pickle.load(open(f"audio_processing\HMM_models\class_c_hmm.pkl", "rb"))
hmm_1 = pickle.load(open(f"audio_processing\HMM_models\class_1_hmm.pkl", "rb"))
hmm_2 = pickle.load(open(f"audio_processing\HMM_models\class_2_hmm.pkl", "rb"))
hmm_3 = pickle.load(open(f"audio_processing\HMM_models\class_3_hmm.pkl", "rb"))
hmm_stopp = pickle.load(open(f"audio_processing\HMM_models\class_stopp_hmm.pkl", "rb"))
hmm_rex = pickle.load(open(f"audio_processing\HMM_models\class_rex_hmm.pkl", "rb"))
hmm_other = pickle.load(open(f"audio_processing\HMM_models\class_other_hmm.pkl", "rb"))

# create a list of all the different HMMs
hmm_list = [hmm_a, hmm_b, hmm_c, hmm_1, hmm_2, hmm_3, hmm_stopp, hmm_rex, hmm_other]

# create a list of all the different test data sets
test_data_sets = []
for i in range(len(class_names)):
    test_data_sets.append([])
for i in range(len(test_data)):
    for j in range(len(test_data[i])):
        test_data_sets[i].append(test_data[i][j].feats)

# calculate the accuracy of the model
# start time
start_acc = time.time()
# create a list of all the different accuracies
acc_list = []
for i in range(len(test_data_sets)):
    acc_list.append([])
# calculate the accuracy for each test data set
for i in range(len(test_data_sets)):
    for j in range(len(test_data_sets[i])):
        acc_list[i].append(hmm_list[i].get_score(test_data_sets[i][j]))
# calculate the average accuracy for each test data set
for i in range(len(acc_list)):
    acc_list[i] = np.mean(acc_list[i])
# end time
end_acc = time.time()
print(f"Time to calculate accuracy: {end_acc-start_acc}")
# print the accuracies
for i in range(len(acc_list)):
    print(f"Accuracy for {class_names[i]}: {acc_list[i]}")



"""# create a GMM for each data set and save it to a file per data set
# start time
start_gmm = time.time()
for i in range(len(data_sets)):
    print(f"Training GMM for {class_names[i]}")
    gmm_model = GMMTrainer(n_dim,n_comp,class_names)
    gmm_model.train(data_sets[i],n_iter)
    with open(f'audio_processing\Gauss_models\class_{class_names[i]}_gmm.pkl', 'wb') as f:
        pickle.dump(gmm_model, f)
    print(f"GMM for {class_names[i]} saved")
end_gmm = time.time()
print(f"Time to train GMMs: {end_gmm-start_gmm}")

# create a HMM out of the GMMs and save it to a file
# load the GMMs
gmm_a = pickle.load(open(f"audio_processing\Gauss_Models\class_a_gmm.pkl", "rb"))
gmm_b = pickle.load(open(f"audio_processing\Gauss_Models\class_b_gmm.pkl", "rb"))
gmm_c = pickle.load(open(f"audio_processing\Gauss_Models\class_c_gmm.pkl", "rb"))
gmm_1 = pickle.load(open(f"audio_processing\Gauss_Models\class_1_gmm.pkl", "rb"))
gmm_2 = pickle.load(open(f"audio_processing\Gauss_Models\class_2_gmm.pkl", "rb"))
gmm_3 = pickle.load(open(f"audio_processing\Gauss_Models\class_3_gmm.pkl", "rb"))
gmm_stopp = pickle.load(open("audio_processing\Gauss_Models\class_stopp_gmm.pkl", "rb"))
gmm_rex = pickle.load(open("audio_processing\Gauss_Models\class_rex_gmm.pkl", "rb"))
gmm_other = pickle.load(open("audio_processing\Gauss_Models\class_other_gmm.pkl", "rb"))

# create a list of all the GMMs
gmm_list = [gmm_a, gmm_b, gmm_c, gmm_1, gmm_2, gmm_3, gmm_stopp, gmm_rex, gmm_other]

# create a HMM out of the GMMs
# start time
print("Training HMM")
hmm = HMMTrainer(n_dim, n_states, class_names)
hmm.train(gmm_list,n_iter)
print("HMM trained")
# end time
end_hmm = time.time()
print(f"Time to train HMM: {end_hmm-end_gmm}")

# save the HMM
hmm.save("audio_processing\speech_hmm_model.pkl")
print("HMM saved")

# load the HMM
hmm = pickle.load(open("audio_processing\speech_hmm_model.pkl", "rb"))

# test the HMM and give the accuracy
# start time
start_test = time.time()
print("Testing HMM")
correct = 0
total = 0
for i in range(len(mfcc_data)):
    if hmm.predict(mfcc_data[i]) == label_data[i]:
        correct += 1
    total += 1
print(f"Accuracy: {correct/total*100}%")
# end time
end_test = time.time()
print(f"Time to test HMM: {end_test-start_test}")

# test the HMM and give the accuracy for each class
# start time
start_test = time.time()
print("Testing HMM")
correct = 0
total = 0
for i in range(len(mfcc_data)):
    if hmm.predict(mfcc_data[i]) == label_data[i]:
        correct += 1
    total += 1
    print(f"Accuracy for {label_data[i]}: {correct/total*100}%")
print(f"Accuracy: {correct/total*100}%")
# end time
end_test = time.time()
print(f"Time to test HMM for each class: {end_test-start_test}")

"""


