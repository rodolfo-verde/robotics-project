import tensorflow as tf
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model('speech_recognition_model.h5')

# Assuming you have new data for prediction, preprocess it accordingly
new_data_mfcc = np.load(f"audio_processing\Train_Data\set_test_a1_mfcc.npy",allow_pickle=True) # load data  # Preprocess your new data to match the input shape of the model

# Make predictions using the loaded model
predictions = loaded_model.predict(new_data_mfcc)

# Process the predictions as needed
# For example, you might use np.argmax(predictions, axis=1) to get predicted classes
