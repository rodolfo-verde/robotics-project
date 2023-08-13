import numpy as np
from ai_hmm import HiddenMarkovModel
import time

class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]

# load the model
hmm = HiddenMarkovModel.load_model("hmm_model.pkl")

# load the test data
data_mfcc = np.load(f"audio_processing\Train_Data\set_a_30_mfcc.npy",allow_pickle=True) # load data

# start time
start = time.time()
# predict the word
# Predict the most likely state (word/command) for each sequence
predicted_states = hmm.predict(data_mfcc)
    
# Map predicted state indices to class names
predicted_words = [class_names[state] for state in predicted_states]
    
print("Predicted words:", predicted_words)