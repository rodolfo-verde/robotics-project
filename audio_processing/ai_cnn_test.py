import tensorflow as tf
import numpy as np
import time
from keras.models import load_model

# Load the saved model
print("Loading model...")
loaded_model = load_model('audio_processing\CNN_Models\AI_speech_recognition_model.h5')
print("Model loaded.")

# Assuming you have new data for prediction, preprocess it accordingly
new_data_mfcc = np.load(f"audio_processing\Train_Data\set_complete_test_mfcc.npy",allow_pickle=True) # load data  # Preprocess your new data to match the input shape of the model
new_data_labels = np.load(f"audio_processing\Train_Data\set_complete_test_label.npy",allow_pickle=True) # load data  # Preprocess your new data to match the input shape of the model

# Make predictions using the loaded model
print("Making predictions...")
starttime = time.time()
predictions = loaded_model.predict(new_data_mfcc)
endtime = time.time()
print(f"Predictions took {endtime-starttime} seconds.")

class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]

# Print the predictions
print("Predictions:")
# convert predictions to class names
predicted_classes = [class_names[i] for i in np.argmax(predictions, axis=1)]
print(predicted_classes)

# True labels
print("True labels:")
# convert true labels to class names
new_data_labels = [class_names[i] for i in np.argmax(new_data_labels, axis=1)]
print(new_data_labels)

# Compare the predictions with the true labels and print the accuracy in percent
print("Accuracy:")
print(sum(np.array(predicted_classes) == np.array(new_data_labels))/len(new_data_labels)*100)



# Process the predictions as needed
# For example, you might use np.argmax(predictions, axis=1) to get predicted classes
