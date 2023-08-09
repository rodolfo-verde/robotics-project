import numpy as np
from typing import List, Optional, Dict, Tuple, Type
from collections import defaultdict
from single_gaussian_model import SingleGauss
from tuple_hmm import DataTuple
import warnings

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