import numpy as np
import pickle

class HiddenMarkovModel:
    def __init__(self, n_states, n_features):
        self.n_states = n_states
        self.n_features = n_features
        
        # Initialize transition matrix with random probabilities
        self.transition_matrix = np.random.rand(n_states, n_states)
        self.transition_matrix /= np.sum(self.transition_matrix, axis=1, keepdims=True)
        
        # Initialize emission matrix with random probabilities
        self.emission_matrix = np.random.rand(n_states, n_features)
        self.emission_matrix /= np.sum(self.emission_matrix, axis=1, keepdims=True)
        
        # Initialize initial state probabilities
        self.initial_probabilities = np.ones(n_states) / n_states
        
    def train(self, training_data, labels, n_iterations=10):
        for _ in range(n_iterations):
            forward_probabilities, backward_probabilities = self.forward_backward(training_data)
            self.update_model_parameters(training_data, labels, forward_probabilities, backward_probabilities)

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
            
            # Compute the expected sufficient statistics
            gamma = forward_prob * backward_prob
            xi = self.compute_xi(sequence, forward_prob, backward_prob)
            
            # Accumulate statistics
            new_initial_probabilities[label] += gamma[0, label]
            new_transition_matrix[label] += np.sum(xi[:, label, :], axis=0)
            for t in range(sequence_length):
                new_emission_matrix[label, sequence[t]] += gamma[t, label]
        
        # Normalize and update model parameters
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
        forward_prob[0] = self.initial_probabilities * self.emission_matrix[:, 0]  # Assuming observation_sequence[0] is the symbol index
        backward_prob[-1] = 1.0
        
        return forward_prob, backward_prob

    def forward_algorithm(self, sequence, forward_prob):
        sequence_length = len(sequence)
        
        for t in range(1, sequence_length):
            for j in range(self.n_states):
                probabilities = forward_prob[t - 1] * self.transition_matrix[:, j]
                forward_prob[t, j] = np.sum(probabilities) * self.emission_matrix[j, sequence[t]]
        
        return forward_prob

    def backward_algorithm(self, sequence, backward_prob):
        sequence_length = len(sequence)
        
        for t in range(sequence_length - 2, -1, -1):
            for i in range(self.n_states):
                probabilities = self.transition_matrix[i, :] * self.emission_matrix[:, sequence[t + 1]] * backward_prob[t + 1]
                backward_prob[t, i] = np.sum(probabilities)
        
        return backward_prob

    def compute_xi(self, sequence, forward_prob, backward_prob):
        sequence_length = len(sequence)
        xi = np.zeros((sequence_length - 1, self.n_states, self.n_states))
        
        for t in range(sequence_length - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = forward_prob[t, i] * self.transition_matrix[i, j] * \
                                  self.emission_matrix[j, sequence[t + 1]] * backward_prob[t + 1, j]
            xi[t] /= np.sum(xi[t])
        
        return xi
    def predict(self, observation_sequence):
        # Viterbi algorithm for finding the most likely state sequence
        n_obs = len(observation_sequence)
        viterbi = np.zeros((n_obs, self.n_states))
        backpointers = np.zeros((n_obs, self.n_states), dtype=int)
        
        # Initialization
        viterbi[0] = self.initial_probabilities * self.emission_matrix[:, observation_sequence[0]]
        
        # Recursion
        for t in range(1, n_obs):
            for j in range(self.n_states):
                probabilities = viterbi[t - 1] * self.transition_matrix[:, j]
                best_state = np.argmax(probabilities)
                viterbi[t, j] = probabilities[best_state] * self.emission_matrix[j, observation_sequence[t]]
                backpointers[t, j] = best_state
        
        # Termination and backtracking
        best_last_state = np.argmax(viterbi[-1])
        best_path = [best_last_state]
        for t in range(n_obs - 1, 0, -1):
            best_last_state = backpointers[t, best_last_state]
            best_path.append(best_last_state)
        
        return best_path[::-1]
    
    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            model = pickle.load(f)
        return model
