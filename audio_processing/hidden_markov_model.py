import numpy as np
from functions_hmm import elog, logSumExp
from single_gaussian_model import SingleGauss
from typing import Type, List
from tuple_hmm import HMMDataTuple

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