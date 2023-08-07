import numpy as np
from collections import namedtuple
from typing import Dict, Type, List, Optional, Tuple
from collections import defaultdict
from itertools import groupby
import warnings
import pickle


# load data and split into trainings and test data
data_mfcc = np.load(f"audio_processing\Train_Data\set_complete_test_mfcc.npy",allow_pickle=True) # load data
data_labels = np.load(f"audio_processing\Train_Data\set_complete_test_label.npy",allow_pickle=True) # load data

print(f"Data shape: {data_mfcc.shape}")
"""# swap first and second dimension of data_mfcc
data_mfcc = np.swapaxes(data_mfcc, 0, 1)
print(f"Data shape: {data_mfcc.shape}")"""
print(f"Labels shape: {data_labels.shape}")
print(len(data_labels))
labels_string = ["" for x in range(len(data_labels))]
print(np.size(labels_string))
class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]
# convert one hot encoded labels to class names
for i in range(len(data_labels)):
    for j in range(len(class_names)):
        if data_labels[i,j] == 1:
            labels_string[i] = class_names[j]


# create a tuple of mfccs and labels
data = []
for i in range(len(data_mfcc)):
    data.append((data_mfcc[i], labels_string[i]))

# create train_data which should be a list of DataTuple(key,feats,label) objects
# key is a unique identifier for each data point
# feats is a numpy array of features, in this case mfccs with 11 dimensions
# label is a string containing the label of the data point
DataTuple = namedtuple('DataTuple', ['key', 'feats', 'label'])
data = [DataTuple(i, x[0], x[1]) for i, x in enumerate(data)]

print(f"Number of data points: {len(data)}")
print(data[0].feats.shape)

# split data into trainings and test data
split = int(len(data)*0.8) # 80% trainings data, 20% test data
train_data = data[:split] # load mfccs of trainings data, 80% of data
test_data = data[split:] # load mfccs of test data, 20% of data

# parameters for training the model
class_names = ["a", "b", "c", "1", "2", "3", "stopp", "rex", "other"]
n_dim = data[0].feats.shape[1] # number of dimensions of the mfccs
n_states = 9 # number of HMM states --> number of classes
n_iter = 10 # number of iterations for training --> variable
n_comp = 8 # number of gaussian components --> variable

# Single Gaussian Model
def elog(x):
    res = np.log(x, where=(x!=0))
    res[np.where(x==0)] = -(10.0**8)
    return (res)

class SingleGauss:
    def __init__(self,
        dim: int
    ) -> None:
        self.dim = dim
        self.mu = None
        self.r = None
    def train(self,
        data: np.ndarray
    ) -> None:
        """
        Train the model on the provided data. The mean and covariance for the
        Gaussian will be estimated from the data.
        :param data: np.array(), shape TxD (T frames of D dimensions)
        """
        T, dim = data.shape
        assert self.dim == dim, f"Expected {self.dim} dimensional data, got {dim}"
        T = data.shape[0]

        self.mu = np.mean(data, axis=0)
        self.r = np.mean(
            np.square(np.subtract(data, self.mu)),
            axis=0
        )
        return     
    def loglike(self,
        data: np.ndarray
    ) -> float:
        """
        Compute log likelihood of the data given model parameters.
        :param data: np.array(), shape TxD
        """
        lls = [
            self._compute_ll(frame) \
                for frame in data.tolist()
        ]
        ll = np.sum(np.array(lls))
        return ll
    def _compute_ll(self,
        data: np.ndarray
    ) -> float:
        ll = (- 0.5*elog(self.r) - np.divide(
                np.square(data - self.mu), 2*self.r) -0.5*np.log(2*np.pi)).sum()
        return ll

# Single Gaussian Trainer
class SingleGaussTrainer:

    def __init__(self,
        dim: int,
        digits: Optional[List[str]] = []
    ) -> None:
        """
        Initialize a single gaussian model for each digit provided. If none are
        provided, then initialization will be done at train time.
        :param dim: Dimensionality of gaussians
        :param digits: list of digits
        """
        self.dim = dim
        self.model = {digit:SingleGauss(dim) for digit in digits}
        self.digits = digits

    def train(self,
        data: List[DataTuple]
    ) -> Dict[str, Type[SingleGauss]]:
        """
        Train a single gaussian model for each digit in the data provided.
        :param data: a list of DataTuple instances
        """
        sorted_data = sorted(data, key=lambda x: x.label)
        label_to_data = defaultdict(list)
        for tpl in sorted_data:
            label_to_data[tpl.label].append(tpl)
        
        for label in label_to_data:
            if label not in self.model:
                warnings.warn(f'Creating new model for uninitialized digit: {label}')
                self.model[label] = SingleGauss(self.dim)
                self.digits.append(label)
            feats = np.vstack(
                [
                    x.feats for x in list(
                        label_to_data[label]
                    )
                ]
            )
            self.model[label].train(feats)


    def predict(self,
        data: List[DataTuple]
    ) -> List[Tuple[str, float]]:
        """
        Compute the predictions and max log likelihoods for each data tuple.
        :param data: list of DataTuple instances
        :returns list of (prediction, log likelihood) 
        """
        preds = []
        for utt in data:
            lls = {}
            for digit in self.model:
                lls[digit] = self.model[digit].loglike(utt.feats)
            
            pred = max(lls, key= lambda x: lls[x])
            log_like = lls[pred]

            preds.append((pred, log_like))

        return preds

# Gaussian Mixture Model
def logSumExp(x, axis=None, keepdims=False):
    x_max = np.max(x, axis=axis, keepdims=keepdims)
    x_diff = x - x_max
    sumexp = np.exp(x_diff).sum(axis=axis, keepdims=keepdims)
    return (x_max + np.log(sumexp))

def exp_normalize(x, axis=None, keepdims=False):
    b = x.max(axis=axis, keepdims=keepdims)
    y = np.exp(x - b)
    return y / y.sum(axis=axis, keepdims=keepdims)

class GMM:

    
    def __init__(self,
        dim: int,
        ncomp: int
    ) -> None:
        
        self.dim = dim
        self.ncomp = ncomp
        self.mu = None
        self.r = None
        self.omega = np.ones(ncomp)/ncomp
        self.sg_init = SingleGauss(dim)

    
    def init_from_sg_model(self,
        sg_model: Type[SingleGauss]
    ) -> None:
        
        self.mu = np.tile(sg_model.mu, (self.ncomp,1))
        for k in range(self.ncomp):
            eps_k = np.random.randn()
            self.mu[k] += 0.01*eps_k*np.sqrt(sg_model.r)
        self.r = np.tile(sg_model.r, (self.ncomp,1))


    def train(self,
        data: np.ndarray
    ) -> None:
        
        gamma = self._e_step(data)
        self._m_step(data, gamma)

    
    def loglike(self,
        data: np.ndarray
    ) -> None:
        
        ll = 0
        for t in range(data.shape[0]):
            ll_t = np.array(
                [
                    np.log(self.omega[k]) + \
                        self._compute_ll(data[t], self.mu[k], self.r[k])
                    for k in range(self.ncomp)
                ]
            )
            ll_t = logSumExp(ll_t)
            ll += ll_t
        return ll


    def _compute_ll(self,
        data: np.ndarray,
        mu: np.ndarray,
        r: np.ndarray
    ) -> float:
        ll = (- 0.5*elog(r) - np.divide(
                np.square(data - mu), 2*r) -0.5*np.log(2*np.pi)).sum()
        return ll


    def _e_step(self,
        data: np.ndarray
    ) -> np.ndarray:
        
        gamma = np.zeros((data.shape[0], self.ncomp))
        for t in range(data.shape[0]):
            log_gamma_t = np.log(self.omega)
            for k in range(self.ncomp):
                log_gamma_t[k] += self._compute_ll(data[t], self.mu[k], self.r[k])
            gamma[t] = exp_normalize(log_gamma_t)
        return gamma

    
    def _m_step(self,
        data: np.ndarray,
        gamma: np.ndarray
    ) -> None:
        
        self.omega = np.sum(gamma, axis=0)/np.sum(gamma)

        denom = np.sum(gamma, axis=0, keepdims=True).T
        
        mu_num = np.zeros_like(self.mu)
        for k in range(self.ncomp):
            mu_num[k] = np.sum(
                np.multiply(
                    data,
                    np.expand_dims(gamma[:,k],axis=1)
                ),
                axis=0
            )
    
        self.mu = np.divide(mu_num, denom)
        
        r_num = np.zeros_like(self.r)
        for k in range(self.ncomp):
            r_num[k] = np.sum(
                np.multiply(
                    np.square(
                        np.subtract(data, self.mu[k])
                    ), 
                    np.expand_dims(gamma[:,k],axis=1)
                ),
                axis=0
            )
    
        self.r = np.divide(r_num, denom)

# GMM Trainer
class GMMTrainer:
    
    def __init__(self,
        dim: int,
        ncomp: int,
        digits: Optional[List[str]] = []
    ) -> None:
        """
        Initialize a GMM model with ncomp components for each digit provided. 
        If none are provided, then initialization will be done at train time.
        :param dim: int, dimensionality of Gaussian components
        :param ncomp: int, number of Gaussian components
        :param digits: (optional) list of digits
        """
        self.gmm_model = {digit:GMM(dim, ncomp) for digit in digits}
        self.sg_model = {digit:SingleGauss(dim) for digit in digits}
        self.digits = digits
        self.dim = dim
        self.ncomp = ncomp

    def train(self,
        data: List[DataTuple],
        niter: Optional[int] = 1
    ) -> None:
        
        sorted_data = sorted(data, key=lambda x: x.label)
        label_to_data = defaultdict(list)
        for tpl in sorted_data:
            label_to_data[tpl.label].append(tpl)
        
        print("Initializing from single gaussian model")
        feats = {}
        for label in label_to_data:
            if label not in self.gmm_model:
                warnings.warn(f'Creating new model for uninitialized digit: {label}')
                self.sg_model[label] = SingleGauss(self.dim)
                self.gmm_model[label] = GMM(self.dim, self.ncomp)
                self.digits.append(label)
            feats[label] = np.vstack(
                [
                    x.feats for x in list(
                        label_to_data[label]
                    )
                ]
            )
            self.sg_model[label].train(feats[label])
            self.gmm_model[label].init_from_sg_model(self.sg_model[label])

        for i in range(niter):
            print(f"Iteration: {i}")
            total_log_like = 0.0
            
            for label in label_to_data:
                self.gmm_model[label].train(feats[label])
                total_log_like += self.gmm_model[label].loglike(feats[label])
            
            print(f"log likelihood: {total_log_like}")


    def predict(self,
        data: List[DataTuple]
    ) -> List[Tuple[str, float]]:
        """
        Compute the predictions and max log likelihoods for each data tuple.
        :param data: list of DataTuple instances
        :returns list of (prediction, log likelihood) 
        """
        preds = []
        for utt in data:
            lls = {}
            for digit in self.gmm_model:
                lls[digit] = self.gmm_model[digit].loglike(utt.feats)
            
            pred = max(lls, key= lambda x: lls[x])
            log_like = lls[pred]

            preds.append((pred, log_like))

        return preds

# Hidden Markov Model
class HMMDataTuple:
    '''Class to store data for HMM training'''
    feats: np.ndarray
    states: List[int]

class HMM:

    
    def __init__(self,
        dim,
        nstate
    ) -> None:
        
        self.pi = np.zeros(nstate) # initial probability
        self.A = np.zeros((nstate,nstate)) # transition matrix
        self.pi[0] = 1
        self.nstate = nstate
        self.dim = dim

        self.mu = None
        self.r = None


    def init_from_sg_model(self,
        sg_model: Type[SingleGauss]
    ) -> None:
        
        self.mu = np.tile(sg_model.mu, (self.nstate,1))
        self.r = np.tile(sg_model.r, (self.nstate,1))

    
    def train(self,
        data: List[HMMDataTuple]
    ) -> None:
        
        self._m_step(data)
        data = self._viterbi(data)
        return data


    def loglike(self,
        feats: np.ndarray
    ) -> float:
        
        T = feats.shape[0]
        log_alpha_t = self._forward(feats)[T-1]
        ll = logSumExp(log_alpha_t)
            
        return ll
        

    def _m_step(self,
        data: List[HMMDataTuple],
    ) -> None:

        self.A = np.zeros_like(self.A)

        gamma_0 = np.zeros(self.nstate)
        gamma_1 = np.zeros((self.nstate, self.dim))
        gamma_2 = np.zeros((self.nstate, self.dim))
        
        for tpl in data:
            feats, states = tpl.feats, tpl.states
            T = len(states)
            gamma = np.zeros((T, self.nstate))

            for t,j in enumerate(states[:-1]):
                self.A[j,states[t+1]] += 1
                gamma[t,j] = 1

            gamma[T-1,self.nstate-1] = 1
            gamma_0 += np.sum(gamma, axis=0)

            for t in range(T):
                gamma_1[states[t]] += feats[t]
                gamma_2[states[t]] += np.square(feats[t])

        gamma_0 = np.expand_dims(gamma_0, axis=1)
        self.mu = gamma_1 / gamma_0
        self.r = (gamma_2 - np.multiply(gamma_0, self.mu**2))/ gamma_0

        for j in range(self.nstate):
            self.A[j] /= np.sum(self.A[j])

    
    def _viterbi(self, 
        data: List[HMMDataTuple]
    ) -> None:
        
        new_data = []
        for tpl in data:
            feats, states = tpl.feats, tpl.states
            T = len(states)
            s_hat = np.zeros(T, dtype=int)
            
            log_delta = np.zeros((T,self.nstate))
            psi = np.zeros((T,self.nstate))
            
            log_delta[0] = elog(self.pi)
            for j in range(self.nstate):
                log_delta[0,j] += self._compute_ll(feats[0], self.mu[j], self.r[j])

            log_A = elog(self.A)
            
            for t in range(1,T):
                for j in range(self.nstate):
                    temp = np.zeros(self.nstate)
                    for i in range(self.nstate):
                        temp[i] = log_delta[t-1,i] + log_A[i,j] + \
                            self._compute_ll(feats[t], self.mu[j], self.r[j])
                    log_delta[t,j] = np.max(temp)
                    psi[t,j] = np.argmax(log_delta[t-1]+log_A[:,j])


            s_hat[T-1] = np.argmax(log_delta[T-1])
            
            for t in reversed(range(T-1)):
                s_hat[t] = psi[t+1,s_hat[t+1]]

            new_data.append(HMMDataTuple(feats=feats, states=s_hat))
        
        return new_data


    def _compute_ll(self,
        data: np.ndarray,
        mu: np.ndarray,
        r: np.ndarray
    ) -> float:
        
        ll = (- 0.5*elog(r) - np.divide(
                np.square(data - mu), 2*r) -0.5*np.log(2*np.pi)).sum()
        return ll


    def _forward(self,
        feats: np.ndarray
    ) -> np.ndarray:
        
        T = feats.shape[0]

        log_alpha = np.zeros((T,self.nstate))
        log_alpha[0] = elog(self.pi)

        log_alpha[0] += np.array(
            [
                self._compute_ll(feats[0], self.mu[j], self.r[j])
                for j in range(self.nstate)
            ]
        )

        for t in range(1, T):
            for j in range(self.nstate):
                log_alpha[t,j] = self._compute_ll(feats[t], self.mu[j], self.r[j]) + \
                    logSumExp(elog(self.A[:,j].T) + log_alpha[t-1])

        return log_alpha

# HMM Trainer
class HMMTrainer:
    
    
    def __init__(self,
        dim: int,
        nstate: int,
        digits: Optional[List[str]] = []
    ) -> None:
        """
        Initialize a HMM model with nstate states for each digit provided. 
        If none are provided, then initialization will be done at train time.
        :param dim: int, dimensionality of each Gaussian output
        :param nstate: int, number of HMM states
        :param digits: (optional) list of digits
        """
        self.hmm_model = {digit:HMM(dim, nstate) for digit in digits}
        self.sg_model = {digit:SingleGauss(dim) for digit in digits}
        self.digits = digits
        self.dim = dim
        self.nstate = nstate


    def train(self,
        data: List[DataTuple],
        niter: Optional[int] = 1
    ) -> None:
        
        sorted_data = sorted(data, key=lambda x: x.label)
        label_to_data = defaultdict(list)
        for tpl in sorted_data:
            label_to_data[tpl.label].append(tpl)
        
        print("Initializing from single gaussian model")
        tuples = {k:[] for k in self.digits}
        
        for label in label_to_data:
            if label not in self.hmm_model:
                warnings.warn(f'Creating new model for uninitialized digit: {label}')
                self.sg_model[label] = SingleGauss(self.dim)
                self.hmm_model[label] = HMM(self.dim, self.nstate)
                self.digits.append(label)
            
            for tpl in label_to_data[label]:
                feats = tpl.feats
                T = feats.shape[0]
                # Initialize state sequence uniformly
                states = np.array([self.nstate*t/T for t in range(T)], dtype=int).tolist()
                tuples[label].append(HMMDataTuple(feats=feats, states=states))
            
            all_feats = np.vstack([x.feats for x in tuples[label]])
            self.sg_model[label].train(all_feats)
            self.hmm_model[label].init_from_sg_model(self.sg_model[label])

        for i in range(niter):
            print(f"Iteration: {i}")
            total_log_like = 0.0
            for label in label_to_data:
                tuples[label] = self.hmm_model[label].train(tuples[label])
                total_log_like += np.sum(
                    [
                        self.hmm_model[label].loglike(x.feats) for x in tuples[label]
                    ]
                )

            print(f"Log likelihood: {total_log_like}")


    def predict(self,
        data: List[DataTuple]
    ) -> List[Tuple[str, float]]:
        """
        Compute the predictions and max log likelihoods for each data tuple.
        :param data: list of DataTuple instances
        :returns list of (prediction, log likelihood) 
        """
        preds = []
        for utt in data:
            lls = {}
            for digit in self.hmm_model:
                lls[digit] = self.hmm_model[digit].loglike(utt.feats)
            
            pred = max(lls, key= lambda x: lls[x])
            log_like = lls[pred]

            preds.append((pred, log_like))

        return preds

# Train the Model

# Single Gaussian Model
sg_model = SingleGaussTrainer(n_dim, class_names)
sg_model.train(train_data)

preds = sg_model.predict(test_data)
y_pred = [pred[0] for pred in preds]  # predicted labels
y_ll = [pred[1] for pred in preds]  # maximum log-likelihood

# Gaussian Mixture Model
gmm_model = GMMTrainer(n_dim, n_comp, class_names)
gmm_model.train(train_data, n_iter)

preds = gmm_model.predict(test_data)

# HMM Model
hmm_model = HMMTrainer(n_dim, n_states, class_names)
hmm_model.train(train_data, n_iter)

preds = hmm_model.predict(test_data)

# Evaluation
# predict
y_pred = [pred[0] for pred in preds]  # predicted labels
y_ll = [pred[1] for pred in preds]  # maximum log-likelihood

# print results
print(f"Predicted labels: {y_pred}")
print(f"Maximum log-likelihood: {y_ll}")

# calculate accuracy
y_true = [x.label for x in data]  # true labels
accuracy = np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
print(f"Accuracy: {accuracy}")

# calculate accuracy for each class
for i, class_name in enumerate(class_names):
    y_true_class = np.array(y_true) == class_name
    y_pred_class = np.array(y_pred) == class_name
    accuracy_class = np.sum(y_true_class == y_pred_class) / len(y_true_class)
    print(f"Accuracy for class {class_name}: {accuracy_class}")

# save the model
with open('speech_hmm_model.pkl', 'wb') as f:
    pickle.dump(hmm_model, f)



