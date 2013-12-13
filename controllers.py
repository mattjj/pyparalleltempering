from __future__ import division
import numpy as np

class ParallelTempering(object):
    def __init__(self,model,temperatures):
        temperatures = [1.] + temperatures
        self.models = [model.copy_sample() for T in temperatures]
        for m,T in zip(self.models,temperatures):
            m.temperature = T

    @property
    def unit_temp_model(self):
        return self.models[0]

    @property
    def temperatures(self):
        return [m.temperature for m in self.models]

    @property
    def energies(self):
        return [m.annealing_energy() for m in self.models]

    @property
    def triples(self):
        return zip(self.models,self.energies,self.temperatures)

    def step(self,intermediate_resamples):
        for m in self.models:
            for itr in xrange(intermediate_resamples):
                m.resample_model()

        for (M1,E1,T1), (M2,E2,T2) in zip(self.triples[:-1],self.triples[1:]):
            swap_logprob = min(0., (E1-E2)*(1./T1 - 1./T2) )
            if np.log(np.random.random()) < swap_logprob:
                M1.swap_sample_with(M2)

    def run(self,niter,intermediate_resamples):
        samples = []
        for itr in xrange(niter):
            self.step(intermediate_resamples)
            samples.append(self.unit_temp_model.copy_sample())
        return samples
