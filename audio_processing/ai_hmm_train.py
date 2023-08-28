import numpy as np
from ai_hmm import HiddenMarkovModel
import time
from sklearn.preprocessing import StandardScaler



# Model improving factors:
# 1. Model Complexity: The Hidden Markov Model might not be suitable for your specific task. HMMs are often used for sequential data, but if your data doesn't exhibit the underlying assumptions of an HMM (such as Markovian behavior), the model might struggle to capture patterns effectively.
# 2. Data Quality: The quality and diversity of your training data play a crucial role in the model's performance. If the training data is noisy, unrepresentative, or lacks enough variation, the model may not generalize well to unseen examples.
# 3. Feature Extraction: The features you're using (MFCCs) may not be capturing the relevant information needed for the classification task. Consider experimenting with different features or preprocessing techniques.
# 4. Model Hyperparameters: The choice of hyperparameters (such as the number of states) can significantly impact the model's performance. You might need to tune these hyperparameters to achieve better results.
# 5. Training Data Size: The amount of training data is also important. If you have a small dataset, the model may struggle to generalize. More data could lead to improved performance.
# 6. Model Initialization: The initial random initialization of the model's parameters (transition matrix, emission matrix, etc.) can affect training convergence. Experiment with different initialization strategies.
# 7. Training Algorithm: The update method you're using to optimize the model's parameters might not be appropriate for your data. There are various optimization algorithms that can be more effective for specific types of data.
# 8. Class Imbalance: If there is a significant class imbalance in your data, the model might bias towards the majority class, leading to poor performance on minority classes.

# Parameters of the HMM
n_states = 9  # Number of states in your HMM
n_features = 11  # Number of features in your extracted MFCC vectors
n_time_steps = 70  # Provide the appropriate value here
class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]

# Create an instance of the HiddenMarkovModel class
hmm = HiddenMarkovModel(n_states, n_features, class_names, n_time_steps)

# Load and preprocess your training data (MFCC images)
data_mfcc = np.load("audio_processing\Train_Data\set_complete_test_mfcc.npy", allow_pickle=True)
# gather first mfcc data for prediction
predict_data = data_mfcc[0, :, :]

print(f"Data shape: {data_mfcc.shape}")
print(f"Predict data shape: {predict_data.shape}")


# Load and preprocess your training labels (strings of letters)
data_labels = np.load("audio_processing\Train_Data\set_complete_test_label.npy", allow_pickle=True)
label_to_index = {label: index for index, label in enumerate(class_names)}
data_labels = [np.argmax(label_row) if np.any(label_row) else label_to_index["other"] for label_row in data_labels]
data_labels = np.array(data_labels)
print(f"Labels shape: {data_labels.shape}")
print("Preprocessing done")
   
# Call this function before training:
print("Starting to fit GMM models")
starttime_fgmm = time.time()
hmm.fit_gmm_models(data_mfcc, data_labels)
endtime_fgmm = time.time()
print(f"Fitting GMM models done. Time: {endtime_fgmm-starttime_fgmm}s")

# Train the model
print("Training started")
# start time
start = time.time()
hmm.train(data_mfcc, data_labels, n_iterations=10)
# end time
end = time.time()
print("Training done")
print(f"Training time: {end-start}s --> {(end-start)/60} mins --> {(end-start)/3600} hours")
 
"""# Predict the most likely state (phoneme) for each sequence
print("Predicting started")
# start time
start = time.time()
predicted_states = hmm.predict(predict_data)
# end time
end = time.time()
print("Predicting done")
print(f"Predicting time: {end-start}s --> {(end-start)/60} mins --> {(end-start)/3600} hours")
print(f"Predicted labels are: {predicted_states}")
print(f"True label is: {data_labels[0]}")"""

# Save the model
hmm.save_model("hmm_model.pkl")

"""# Calculate and print accuracy
overall_accuracy = hmm.calculate_accuracy(predicted_states, data_labels)
print(f"Overall Accuracy: {overall_accuracy:.2f}%")
class_accuracies = hmm.calculate_class_accuracies(predicted_states, data_labels, label_to_index)
# Print class accuracies
for class_name in class_accuracies:
    print(f"Accuracy for {class_name}: {class_accuracies[class_name]:.2f}%")
"""