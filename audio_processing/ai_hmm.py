import numpy as np
import pickle
import time

"""The HiddenMarkovModel class is initialized with the number of hidden states (n_states), the number of observed features (n_features), and a list of class names (class_names). The class initializes the transition matrix, emission matrix, and initial probabilities with random values.

Now, let's explain the methods step by step:

train Method:
This method trains the HMM using the Baum-Welch algorithm (also known as the Forward-Backward algorithm). It iteratively updates the model's parameters to fit the training data.

forward_probabilities and backward_probabilities are computed using the forward_backward method.

The update_model_parameters method updates the model's parameters based on the forward and backward probabilities.

update_model_parameters Method:
This method updates the model parameters (transition matrix, emission matrix, initial probabilities) based on the forward and backward probabilities.

It calculates gamma and xi values using the forward and backward probabilities and updates the model parameters.
forward_backward Method:
This method computes the forward and backward probabilities for each sequence in the training data.

It initializes forward and backward probabilities using initialize_probabilities.
It uses the forward_algorithm and backward_algorithm methods to calculate the probabilities.
initialize_probabilities Method:
This method initializes the forward and backward probabilities for a sequence.

It sets the initial probabilities and the first backward probability.
It returns the initialized probabilities.
forward_algorithm Method:
This method performs the forward algorithm to calculate the forward probabilities for a sequence.

It iteratively computes forward probabilities for each time step and each hidden state.
backward_algorithm Method:
This method performs the backward algorithm to calculate the backward probabilities for a sequence.

It iteratively computes backward probabilities for each time step and each hidden state.
compute_xi Method:
This method calculates the xi values, which are used in updating the transition matrix.

It computes the xi values based on the forward and backward probabilities.
predict Method:
This method predicts the most likely state for each sequence in the test data using the trained model.

It calculates forward probabilities for each sequence and predicts based on the final state probabilities.
calculate_accuracy Method:
This method calculates the overall accuracy of the model's predictions.

It compares predicted labels with true labels and calculates accuracy.
calculate_class_accuracies Method:
This method calculates class-specific accuracies.

It computes accuracy for each class separately and returns a dictionary of class accuracies.
save_model and load_model Methods:
These methods save and load the trained HMM model using pickle.

Each method serves a specific purpose in training, predicting, and evaluating the Hidden Markov Model. 
The training process iteratively updates the model parameters based on the forward and backward probabilities, and the prediction process 
uses the trained model to predict the most likely state for each input sequence. 
The accuracy metrics provide insights into the model's performance on both an overall and class-specific level."""

class HiddenMarkovModel:
    def __init__(self, n_states, n_features, class_names):
        self.n_states = n_states
        self.n_features = n_features
        
        self.transition_matrix = np.random.rand(n_states, n_states)
        self.transition_matrix /= np.sum(self.transition_matrix, axis=1, keepdims=True)
        
        self.emission_matrix = np.random.rand(n_states, n_features)
        self.emission_matrix /= np.sum(self.emission_matrix, axis=1, keepdims=True)
        
        self.initial_probabilities = np.ones(n_states) / n_states

        self.class_names = class_names  # Set the class names attribute

    def train(self, training_data, labels, n_iterations=10):
        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")
            start = time.time()
            forward_probabilities, backward_probabilities = self.forward_backward(training_data)
            self.update_model_parameters(training_data, labels, forward_probabilities, backward_probabilities)
            end = time.time()
            print(f"Training time: {end - start}s for iteration {iteration + 1}")

    def update_model_parameters(self, training_data, labels, forward_probabilities, backward_probabilities):
        n_sequences = len(training_data)
        new_transition_matrix = np.zeros_like(self.transition_matrix)
        new_emission_matrix = np.zeros_like(self.emission_matrix)
        new_initial_probabilities = np.zeros_like(self.initial_probabilities)

        for i in range(n_sequences):
            sequence = training_data[i]
            forward_prob = forward_probabilities[i]
            backward_prob = backward_probabilities[i]
            label = labels[i]
            sequence_length = len(sequence)

            gamma = forward_prob * backward_prob
            xi = self.compute_xi(sequence, forward_prob, backward_prob)

            new_initial_probabilities[label] += gamma[0, label]
            new_transition_matrix[label] += np.sum(xi[:, label, :], axis=0)
            for t in range(sequence_length):
                for j in range(self.n_states):
                    feature_index = np.argmax(sequence[t])
                    new_emission_matrix[j, feature_index] += gamma[t, j]

        self.initial_probabilities = new_initial_probabilities / n_sequences
        self.transition_matrix = new_transition_matrix / np.sum(new_transition_matrix, axis=1, keepdims=True)
        self.emission_matrix = new_emission_matrix / np.sum(new_emission_matrix, axis=1, keepdims=True)


    
    def forward_backward(self, training_data):
        n_sequences = len(training_data)
        forward_probabilities = []
        backward_probabilities = []
        
        for sequence in training_data:
            forward_prob, backward_prob = self.initialize_probabilities(len(sequence))
            forward_prob = self.forward_algorithm(sequence, forward_prob)
            backward_prob = self.backward_algorithm(sequence, backward_prob)
            
            forward_probabilities.append(forward_prob)
            backward_probabilities.append(backward_prob)
        
        return forward_probabilities, backward_probabilities
    
    def initialize_probabilities(self, sequence_length):
        forward_prob = np.zeros((sequence_length, self.n_states))
        backward_prob = np.zeros((sequence_length, self.n_states))
        forward_prob[0] = self.initial_probabilities * self.emission_matrix[:, 0]  # Modify index as needed
        backward_prob[-1] = 1.0
        
        return forward_prob, backward_prob
    
    def forward_algorithm(self, sequence, forward_prob):
        sequence_length = len(sequence)
    
        for t in range(1, sequence_length):
            for j in range(self.n_states):
                probabilities = forward_prob[t - 1] * self.transition_matrix[:, j]
            
                # Calculate the emission index based on the emission probabilities
                emission_index = np.argmax(self.emission_matrix[j, :])
                emission_prob = self.emission_matrix[j, emission_index]
            
                forward_prob[t, j] = np.sum(probabilities) * emission_prob
    
        return forward_prob
    
    def backward_algorithm(self, sequence, backward_prob):
        sequence_length = len(sequence)
    
        for t in range(sequence_length - 2, -1, -1):
            for i in range(self.n_states):
                emission_index = np.argmax(self.emission_matrix[i, :])
                probabilities = self.transition_matrix[i, :] * self.emission_matrix[:, emission_index] * backward_prob[t + 1]
                backward_prob[t, i] = np.sum(probabilities)
        
        return backward_prob
    
    def compute_xi(self, sequence, forward_prob, backward_prob):
        sequence_length = len(sequence)
        xi = np.zeros((sequence_length - 1, self.n_states, self.n_states))
    
        for t in range(sequence_length - 1):
            for i in range(self.n_states):
                emission_index_i = np.argmax(self.emission_matrix[i, :])
                for j in range(self.n_states):
                    emission_index_j = np.argmax(self.emission_matrix[j, :])
                    xi[t, i, j] = forward_prob[t, i] * self.transition_matrix[i, j] * \
                                self.emission_matrix[j, emission_index_j] * backward_prob[t + 1, j]
            xi[t] /= np.sum(xi[t])
    
        return xi
    
    def predict(self, data_mfcc):
        n_sequences, sequence_length, _ = data_mfcc.shape
        predicted_states = []
        
        for i in range(n_sequences):
            sequence = data_mfcc[i]
            forward_prob, _ = self.initialize_probabilities(sequence_length)
            forward_prob = self.forward_algorithm(sequence, forward_prob)
            
            predicted_state = np.argmax(forward_prob[-1])  # Predict based on the final state probabilities
            predicted_states.append(predicted_state)
        
        return predicted_states
    
    def calculate_accuracy(self, predicted_labels, true_labels):
        total_samples = len(predicted_labels)
        correct_predictions = np.sum(predicted_labels == true_labels)
        accuracy = correct_predictions / total_samples * 100
        return accuracy
    
    def calculate_class_accuracies(self, predicted_labels, true_labels, label_to_index):
        class_accuracies = {}
        for class_name in self.class_names:
            class_indices = np.where(true_labels == label_to_index[class_name])[0]
            print(f"Class name: {class_name}")
            print(f"Class indices: {class_indices}")
        
            class_predictions = [predicted_labels[i] for i in class_indices]
            print(f"Class predictions: {class_predictions}")
        
            class_true_labels = [true_labels[i] for i in class_indices]
            print(f"Class true labels: {class_true_labels}")
        
            class_accuracy = self.calculate_accuracy(class_predictions, class_true_labels)
            class_accuracies[class_name] = class_accuracy
        return class_accuracies


    
    def save_model(self, filename):
        with open(f'audio_processing\HMM_models\{filename}', 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(f'audio_processing\HMM_models\{filename}', 'rb') as f:
            model = pickle.load(f)
        return model

