import numpy as np
import pickle
import time
from sklearn.mixture import GaussianMixture


class HiddenMarkovModel:
    def __init__(self, n_states, n_features, n_time_steps, class_names, n_components=1):
        self.n_states = n_states
        self.n_features = n_features
        self.n_components = n_components  # Add n_components attribute
        self.n_time_steps = n_time_steps  # Add n_time_steps attribute

        self.transition_matrix = np.random.rand(n_states, n_states)
        self.transition_matrix /= np.sum(self.transition_matrix, axis=1, keepdims=True)
        
        self.emission_matrix = np.random.rand(n_states, n_features)
        self.emission_matrix /= np.sum(self.emission_matrix, axis=1, keepdims=True)
        
        self.initial_probabilities = np.ones(n_states) / n_states

        self.class_names = class_names  # Set the class names attribute

        # Initialize the state_gmms attribute as a list of GaussianMixture models
        self.state_gmms = [GaussianMixture(n_components=self.n_components, covariance_type='full') for _ in range(n_states)]

        # Initialize the transition matrix with random values
        self.transition_matrix = np.random.rand(n_states, n_states)
        self.transition_matrix /= np.sum(self.transition_matrix, axis=1, keepdims=True)

    def update_transition_matrix(self, forward_probabilities, backward_probabilities):
        n_sequences, sequence_length, _ = forward_probabilities.shape
        new_transition_matrix = np.zeros_like(self.transition_matrix)

        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = np.sum(forward_probabilities[:, :, i] * self.transition_matrix[i, j] *
                                self.emission_matrix[j] * backward_probabilities[:, :, j])
                denominator = np.sum(forward_probabilities[:, :, i] * backward_probabilities[:, :, i])
                new_transition_matrix[i, j] = numerator / denominator

        # Normalize the new transition matrix
        new_transition_matrix /= np.sum(new_transition_matrix, axis=1, keepdims=True)

        return new_transition_matrix

    def train(self, training_data, labels, n_iterations=10):
        for iteration in range(n_iterations):
            print(f"Iteration {iteration + 1}/{n_iterations}")
            start = time.time()

            forward_probabilities, backward_probabilities = self.forward_backward(training_data)

            forward_probabilities = np.array(forward_probabilities)  # Convert to numpy array
            backward_probabilities = np.array(backward_probabilities)  # Convert to numpy array

            new_transition_matrix = self.update_transition_matrix(forward_probabilities, backward_probabilities)
            new_emission_matrix = self.update_emission_matrix(training_data, forward_probabilities, backward_probabilities, labels)

            self.update_model_parameters(training_data, labels, forward_probabilities, backward_probabilities)

            self.transition_matrix = new_transition_matrix
            self.emission_matrix = new_emission_matrix

            end = time.time()
            print(f"Training time: {end - start}s for iteration {iteration + 1}")

    def update_model_parameters(self, training_data, labels, forward_probabilities, backward_probabilities):
        n_sequences = len(training_data)
    
        if len(forward_probabilities) != n_sequences or len(backward_probabilities) != n_sequences:
            raise ValueError("Number of sequences in forward_probabilities or backward_probabilities does not match training_data.")
    
        # Update initial probabilities
        self.initial_probabilities = np.sum(forward_probabilities[:, 0] * backward_probabilities[:, 0], axis=0)
        self.initial_probabilities /= np.sum(self.initial_probabilities)
    
        # Update transition matrix
        new_transition_matrix = np.zeros_like(self.transition_matrix)
        for i in range(self.n_states):
            for j in range(self.n_states):
                numerator = 0.0
                denominator = 0.0
                for t in range(n_sequences):
                    sequence = training_data[t]
                    forward_prob = forward_probabilities[t]
                    backward_prob = backward_probabilities[t]
                
                    sequence_length = len(sequence)
                
                    if sequence_length <= 1:
                        continue
                
                    for t in range(1, sequence_length):
                        numerator += forward_prob[i, t - 1] * self.transition_matrix[i, j] * \
                                    self.calculate_emission_probability(sequence[t], state=j) * backward_prob[j, t]
                        denominator += forward_prob[i, t - 1] * backward_prob[i, t - 1]
            
                new_transition_matrix[i, j] = numerator / denominator
    
        # Normalize the transition matrix
        self.transition_matrix = new_transition_matrix / np.sum(new_transition_matrix, axis=1, keepdims=True)
    
        # Update emission matrix
        self.update_emission_matrix(training_data, forward_probabilities, backward_probabilities)

    def fit_gmm_models(self, training_data, labels):
        self.state_gmms = {}  # Initialize the state_gmms dictionary
        for state in range(self.n_states):
            state_data = []  # Collect training data for this state
            for i, label in enumerate(labels):
                if label == state:
                    state_data.append(training_data[i])
            state_data = np.vstack(state_data)

            gmm = GaussianMixture(n_components=self.n_components, covariance_type='full')
            gmm.fit(state_data)

            self.state_gmms[state] = gmm




    def update_emission_matrix(self, training_data, forward_probabilities, backward_probabilities, labels):
        n_sequences = len(training_data)
        new_emission_matrix = np.zeros_like(self.emission_matrix)
    
        for i in range(n_sequences):
            sequence = training_data[i]
            forward_prob = forward_probabilities[i]
            backward_prob = backward_probabilities[i]
        
            # Verify the shape and contents of the feature vector
            if sequence.shape != (self.n_features, self.n_time_steps):  # Update the shape condition
                raise ValueError(f"Invalid shape of feature vector in training_data[{i}]")

            for i in range(self.n_states):
                for j in range(self.n_states):
                    emission_prob = self.calculate_emission_probability(sequence, state=j)
                    new_emission_matrix[i, j] += forward_prob[i] * backward_prob[i] * emission_prob

        # Normalize the emission matrix
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
    
    def calculate_emission_probability(self, feature_vector, state):
        print("Feature vector shape = {feature_vector.shape}")  # Add this line to check the shape of the feature vector
        gmm = self.state_gmms[state]
        if feature_vector.shape[0] != self.n_features:
            raise ValueError("Feature vector has incorrect number of features")
        emission_prob = gmm.score_samples(feature_vector.reshape(1, -1))
        return np.exp(emission_prob)

    def viterbi_decode(self, sequence):
        sequence_length = len(sequence)
        n_states = self.n_states

        # Initialize variables for Viterbi algorithm
        path_probabilities = np.zeros((sequence_length, n_states))
        best_paths = np.zeros((sequence_length, n_states), dtype=int)

        # Initialize the first step with initial probabilities and emission probabilities
        for j in range(n_states):
            path_probabilities[0, j] = self.initial_probabilities[j] * self.calculate_emission_probability(sequence[0], state=j)

        # Perform the Viterbi algorithm
        for t in range(1, sequence_length):
            for j in range(n_states):
                transition_probabilities = path_probabilities[t - 1] * self.transition_matrix[:, j]
                best_path = np.argmax(transition_probabilities)
                best_paths[t, j] = best_path
                emission_prob = self.calculate_emission_probability(sequence[t], state=j)
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

