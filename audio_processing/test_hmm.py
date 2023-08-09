print("Importing...")
import numpy as np
from tuple_hmm import DataTuple
import time
import pickle
print("Imported")

# load model
print("Loading model...")
with open(f'audio_processing\speech_hmm_model.pkl', 'rb') as f:
    hmm_model = pickle.load(f)
    print("Model loaded")


# load test data
# predict
mfcc = np.load(f"audio_processing\Train_Data\set_a_30_mfcc.npy",allow_pickle=True) # load data

for i in range(len(mfcc)):
    data = [DataTuple(i, mfcc[i], "")]
    preds = hmm_model.predict(data)
    print(preds)




