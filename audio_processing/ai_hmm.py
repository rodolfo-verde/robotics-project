import numpy as np
import pickle
import time
from sklearn.mixture import GaussianMixture

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
    def __init__(self, n_states, n_features, class_names, n_components=1):
        self.n_states = n_states
        self.n_features = n_features
        self.n_components = n_components  # Add n_components attribute

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
        new_initial_probabilities = np.zeros_like(self.initial_probabilities)
        
        # Update emission matrix with GMM parameters
        new_emission_matrix = self.update_emission_matrix(training_data, forward_probabilities, backward_probabilities, labels)
        
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

        # Normalize transition matrix and initial probabilities
        self.initial_probabilities = new_initial_probabilities / n_sequences
        self.transition_matrix = new_transition_matrix / np.sum(new_transition_matrix, axis=1, keepdims=True)
        self.emission_matrix = new_emission_matrix / np.sum(new_emission_matrix, axis=1, keepdims=True)

    def update_emission_matrix(self, training_data, forward_probabilities, backward_probabilities, labels):
        new_emission_matrix = np.zeros_like(self.emission_matrix)

        for i in range(len(self.class_names)):
            class_indices = np.where(labels == i)[0]
            class_sequences = [training_data[idx] for idx in class_indices]
            class_forward_probs = [forward_probabilities[idx] for idx in class_indices]
            class_backward_probs = [backward_probabilities[idx] for idx in class_indices]

            gmm = GaussianMixture(n_components=self.n_components, covariance_type='full')
            gmm.fit(np.concatenate(class_sequences))

            for j in range(self.n_states):
                emission_probs = []
                for seq_idx in range(len(class_sequences)):
                    seq_length = len(class_sequences[seq_idx])
                    if j < seq_length:  # Check if j is within the valid range
                        emission_prob = gmm.score_samples(class_sequences[seq_idx][j].reshape(-1, self.n_features))
                        emission_probs.append(np.sum(class_forward_probs[seq_idx][:, j] * class_backward_probs[seq_idx][:, j] * np.exp(emission_prob)))
                new_emission_matrix[i, j] = np.sum(emission_probs)

        return new_emission_matrix

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
    
    def calculate_emission_probability(self, feature_vector, state):
        # Fit a Gaussian Mixture Model (GMM) to the state's data
        gmm = GaussianMixture(n_components=self.n_components, covariance_type='full')
        state_data = feature_vector[state == self.states]
        gmm.fit(state_data)
        
        # Calculate the probability of the feature vector using the GMM
        log_prob = gmm.score_samples(feature_vector.reshape(1, -1))
        emission_prob = np.exp(log_prob)

        return emission_prob
    
    def viterbi_decode(self, sequence):
        sequence_length = len(sequence)
        n_states = self.n_states

        # Initialize variables for Viterbi algorithm
        path_probabilities = np.zeros((sequence_length, n_states))
        best_paths = np.zeros((sequence_length, n_states), dtype=int)

        # Initialize the first step with initial probabilities and emission probabilities
        path_probabilities[0] = self.initial_probabilities * self.calculate_emission_probability(sequence[0])

        # Perform the Viterbi algorithm
        for t in range(1, sequence_length):
            for j in range(n_states):
                transition_probabilities = path_probabilities[t - 1] * self.transition_matrix[:, j]
                best_path = np.argmax(transition_probabilities)
                best_paths[t, j] = best_path
                emission_prob = self.calculate_emission_probability(sequence[t], state=j)  # Provide the state argument
                path_probabilities[t, j] = transition_probabilities[best_path] * emission_prob

        # Backtrack to find the best path
        best_sequence = [np.argmax(path_probabilities[-1])]
        for t in range(sequence_length - 1, 0, -1):
            best_state = best_paths[t, best_sequence[-1]]
            best_sequence.append(best_state)
        best_sequence.reverse()

        return best_sequence


    def predict(self, data_mfcc):
        n_sequences = len(data_mfcc)
        predicted_states = []

        for i in range(n_sequences):
            sequence = data_mfcc[i]
            predicted_path = self.viterbi_decode(sequence)
            predicted_states.append(predicted_path)
    
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

