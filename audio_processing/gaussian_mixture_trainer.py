import numpy as np
from single_gaussian_model import SingleGauss
from typing import List, Optional, Tuple
from tuple_hmm import DataTuple
from gaussian_mixture_model import GMM
import warnings
from collections import defaultdict


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