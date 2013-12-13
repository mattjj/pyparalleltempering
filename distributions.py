from __future__ import division
import numpy as np
import abc

import pybasicbayes
from pybasicbayes.util.stats import sample_niw

class _AnnealingMixin(object):
    __metaclass__ = abc.ABCMeta

    @property
    def temperature(self):
        return self._temperature if hasattr(self,'_temperature') else 1.

    @temperature.setter
    def temperature(self,T):
        self._temperature = T

    def log_likelihood(self,x):
        return super(_AnnealingMixin,self).log_likelihood(x) / self.temperature

    def annealing_energy(self,data):
        # NOTE: value is independent of temperature
        if isinstance(data,list):
            return self._log_prior_density_unnorm() + \
                    sum(super(_AnnealingMixin,self).log_likelihood(d) for d in data)
        else:
            return self._log_prior_density_unnorm() + \
                    super(_AnnealingMixin,self).log_likelihood(data)

    @abc.abstractmethod
    def _log_prior_density_unnorm(self):
        pass

    @abc.abstractmethod
    def resample(self,data=[]):
        pass

class AnnealedGaussian(_AnnealingMixin,pybasicbayes.distributions.Gaussian):
    def _log_prior_density_unnorm(self):
        # NOTE: doesn't include (2pi)^(-D/2) in Normal part or any of the
        # normalizer in the InvWish part
        mu, sigma = self.mu, self.sigma
        return -((self.nu_0 + len(mu))/2.+1) * np.linalg.slogdet(sigma)[1] \
                - np.linalg.solve(sigma,self.sigma_0).trace()/2. \
                - 1./2 * mu.dot(np.linalg.solve(sigma/self.kappa_0,mu))

    def resample(self,data=[]):
        D = len(self.mu_0)
        self.mu_mf, self.sigma_mf = self.mu, self.sigma = \
                sample_niw(*self._raise_temperature(
                    *self._posterior_hypparams(*self._get_statistics(data,D))))
        return self

    def _raise_temperature(self,mu_n,sigma_n,kappa_n,nu_n):
        D = len(mu_n)
        return mu_n, sigma_n * self.temperature, kappa_n * self.temperature, \
                (nu_n - D - 1)/self.temperature + D + 1 # NOTE: base measure correction

