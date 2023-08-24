import numpy as np
import time
import pickle

class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]

# load model
print("Loading model...")
with open(f'audio_processing\HMM_models\hmm_model.pkl', 'rb') as f:
    hmm_model = pickle.load(f)
    print("Model loaded")

# load the test data
data_mfcc = np.load(f"audio_processing\Train_Data\set_complete_test_mfcc.npy",allow_pickle=True) # load data

# start time
start = time.time()
# predict the word
# Predict the most likely state (word/command) for each sequence
predicted_words = hmm_model.predict(data_mfcc)
endtime = time.time()
# extract the first word of each array in predicted words
# predicted_words = [class_names[i] for i in predicted_words[:,0]]
# end time
print(f"Predicting time: {endtime-start}s --> {(endtime-start)/60} mins --> {(endtime-start)/3600} hours")    
print("shape of predicted words:", predicted_words.shape)