from __future__ import division
import numpy as np
from matplotlib import pyplot as plt

from pybasicbayes.util import testing
from pybasicbayes.distributions import Gaussian

from controllers import ParallelTempering
from distributions import AnnealedGaussian

class _DistributionToModelMixin(object):
    def __init__(self,*args,**kwargs):
        super(_DistributionToModelMixin,self).__init__(*args,**kwargs)
        self.data = []

    def add_data(self,data):
        self.data.append(data)

    def resample_model(self):
        self.resample(data=self.data)

    def annealing_energy(self):
        return super(_DistributionToModelMixin,self).annealing_energy(self.data)

class AnnealedGaussianModel(_DistributionToModelMixin,AnnealedGaussian):
    pass

# def test_gaussian():
prior_data = 2*np.random.randn(5,2) + np.array([1.,3.])
a = Gaussian().empirical_bayes(prior_data)

# data = a.rvs(10)

gibbs_statistics = []
for itr in range(20000):
    a.resample()
    # a.resample(data)
    gibbs_statistics.append(a.mu)
gibbs_statistics = np.array(gibbs_statistics)

print 'done exact sampling'

b = AnnealedGaussianModel().empirical_bayes(prior_data)
# b.add_data(data)

pt = ParallelTempering(b,[5.])
pt_samples = pt.run(20000,1)
pt_statistics = np.array([m.mu for m in pt_samples])

print 'done ParallelTempering'

fig = plt.figure()
testing.populations_eq_quantile_plot(gibbs_statistics,pt_statistics,fig=fig)
plt.savefig('gaussian_test.png')
plt.show()

