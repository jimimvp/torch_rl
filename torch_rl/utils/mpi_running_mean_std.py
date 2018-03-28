from mpi4py import MPI
from torch_rl.utils import to_tensor
import torch as tor
import numpy as np


class RunningMeanStd(object):
    """
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        This is a parallel implementation in pytorch to the implementation in tensorflow 
        from https://github.com/openai/baselines.
    """


    def __init__(self, epsilon=1e-2, shape=()):

        if shape == ():
            shape = 1
        self._sum = tor.zeros(shape).type(tor.double).detach()
        self._sumsq = tor.from_numpy(np.full(shape, epsilon)).type(tor.double).detach()
        self._count = tor.zeros(shape).type(tor.double).detach()
        self.shape = shape



    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n*2+1, 'float64')
        addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)],dtype='float64')])
        MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)

        self._sum += tor.from_numpy(totalvec[0:n].reshape(self.shape))
        self._sumsq += tor.from_numpy(totalvec[n:2*n].reshape(self.shape))
        self._count += totalvec[2*n]

        self._mean = self._sum / self._count
        print(self._sumsq)
        self._std = tor.sqrt(tor.max(self._sumsq / self._count - self._mean**2 , tor.zeros_like(self._sumsq) + 1e-2))
       


    @property
    def mean(self):
        return self._mean.data.numpy()

    @property
    def std(self):
        return self._std.data.numpy()




# for (x1, x2, x3) in [
#     (np.random.randn(3), np.random.randn(4), np.random.randn(5)),
#     (np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),
#     ]:

#     rms = RunningMeanStd(epsilon=0.0, shape=x1.shape[1:])

#     x = np.concatenate([x1, x2, x3], axis=0)
#     ms1 = np.asarray([x.mean(axis=0), x.std(axis=0)])
#     print(x1.shape)
#     rms.update(x1)
#     rms.update(x2)
#     rms.update(x3)
#     ms2 = np.hstack([rms.mean, rms.std])


#     print(ms1, "#")
#     print(ms2, "#")

#     assert np.allclose(ms1, ms2.reshape(ms1.shape))