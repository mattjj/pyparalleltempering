from __future__ import division
import numpy as np
import abc

import pybasicbayes
from pybasicbayes.util.stats import sample_niw

# TODO swap_sample_with could be generically implemented with parameter
# setters/getters, one more reason for fancy metaclass boilerplate

### mixins

class _AnnealingDistributionMixin(object):
    __metaclass__ = abc.ABCMeta

    @property
    def temperature(self):
        return self._temperature if hasattr(self,'_temperature') else 1.

    @temperature.setter
    def temperature(self,T):
        self._temperature = T

    def log_likelihood(self,x):
        return super(_AnnealingDistributionMixin,self).log_likelihood(x) / self.temperature

    def annealing_energy(self,data):
        # NOTE: value is independent of temperature, since temperature is
        # handled explicitly in the MH probability; this is just the E part
        if isinstance(data,list):
            return self._log_varied_density_unnorm() + \
                    sum(super(_AnnealingDistributionMixin,self).log_likelihood(d) for d in data)
        else:
            return self._log_varied_density_unnorm() + \
                    super(_AnnealingDistributionMixin,self).log_likelihood(data)

    @abc.abstractmethod
    def _log_varied_density_unnorm(self):
        # NOTE: only need the part of the density that is affected by the
        # temperature; other parts just cancel in the MH ratio
        pass

    @abc.abstractmethod
    def resample(self,data=[]):
        # NOTE: this resample method must sample from the heated distribution
        # for exponential family, might just define a _raise_temperature method
        # that transforms the parameters (like scaling the natural params)
        pass

    @abc.abstractmethod
    def swap_sample_with(self,other):
        pass

### distributions

class AnnealedGaussian(_AnnealingDistributionMixin,pybasicbayes.distributions.Gaussian):
    def _log_varied_density_unnorm(self):
        mu, sigma = self.mu, self.sigma
        invwish_part = -1./2 * (self.nu_0) * np.linalg.slogdet(sigma)[1] \
                - 1./2 * np.linalg.solve(sigma,self.sigma_0).trace()
        normal_part = - 1./2 * (
                (mu - self.mu_0).dot(np.linalg.solve(sigma/self.kappa_0,mu - self.mu_0))
                - np.linalg.slogdet(sigma/self.kappa_0)[1])
        return invwish_part + normal_part

    def resample(self,data=[]):
        D = len(self.mu_0)
        self.mu_mf, self.sigma_mf = self.mu, self.sigma = \
                sample_niw(*self._raise_temperature(
                    *self._posterior_hypparams(*self._get_statistics(data,D))))
        return self

    def swap_sample_with(self,other):
        mu, sigma, _sigma_chol = self.mu, self.sigma, self._sigma_chol
        self.mu, self.sigma, self._sigma_chol = other.mu, other.sigma, other._sigma_chol
        other.mu, other.sigma, other._sigma_chol = mu, sigma, _sigma_chol

    def _raise_temperature(self,mu_n,sigma_n,kappa_n,nu_n):
        return mu_n, sigma_n / self.temperature, kappa_n / self.temperature, nu_n

