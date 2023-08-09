import numpy as np
from typing import List
from functions_hmm import elog


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