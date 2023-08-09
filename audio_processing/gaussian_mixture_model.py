import numpy as np
from functions_hmm import elog, logSumExp, exp_normalize
from single_gaussian_model import SingleGauss
from typing import Type

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