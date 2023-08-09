import numpy as np
from collections import defaultdict
from typing import List, Optional, Tuple
from tuple_hmm import DataTuple, HMMDataTuple
from single_gaussian_model import SingleGauss
from hidden_markov_model import HMM
import warnings

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