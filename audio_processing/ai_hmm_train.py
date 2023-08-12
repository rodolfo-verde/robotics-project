import numpy as np
from ai_hmm import HiddenMarkovModel
import time


# parameters of the hmm
n_states = 9
n_features = 11

# Create an instance of the HiddenMarkovModel class
hmm = HiddenMarkovModel(n_states, n_features)
    
# Load and preprocess your training data (sequences of MFCC features)
data_mfcc = np.load(f"audio_processing\Train_Data\set_complete_test_mfcc.npy",allow_pickle=True) # load data
data_labels = np.load(f"audio_processing\Train_Data\set_complete_test_label.npy",allow_pickle=True) # load data
    
# Train the model
forward_probabilities, backward_probabilities = hmm.forward_backward(data_mfcc)
# start time
start = time.time()
hmm.train(data_mfcc, data_labels)
# end time
end = time.time()
print(f"Training time: {end-start}s")

# Save the model
hmm.save("hmm_model.pkl")

