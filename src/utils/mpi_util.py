import os
import subprocess
import sys

from mpi4py import MPI
import tensorflow as tf
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import src.utils.tf_util as tf_util


def mpi_fork(n, bind_to_core=False):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n<=1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        args += [sys.executable] + sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        return "child"

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)

def mpi_mean(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    if comm is None: comm = MPI.COMM_WORLD
    xsum = x.sum(axis=axis, keepdims=keepdims)
    n = xsum.size
    localsum = np.zeros(n+1, x.dtype)
    localsum[:n] = xsum.ravel()
    localsum[n] = x.shape[axis]
    globalsum = np.zeros_like(localsum)
    comm.Allreduce(localsum, globalsum, op=MPI.SUM)
    return globalsum[:n].reshape(xsum.shape) / globalsum[n], globalsum[n]

def mpi_moments(x, axis=0, comm=None, keepdims=False):
    x = np.asarray(x)
    assert x.ndim > 0
    mean, count = mpi_mean(x, axis=axis, comm=comm, keepdims=True)
    sqdiffs = np.square(x - mean)
    meansqdiff, count1 = mpi_mean(sqdiffs, axis=axis, comm=comm, keepdims=True)
    assert count1 == count
    std = np.sqrt(meansqdiff)
    if not keepdims:
        newshape = mean.shape[:axis] + mean.shape[axis+1:]
        mean = mean.reshape(newshape)
        std = std.reshape(newshape)
    return mean, std, count


def test_runningmeanstd():
    import subprocess
    subprocess.check_call(['mpirun', '-np', '3',
        'python','-c',
        'from imagine.utils.mpi_moments import _helper_runningmeanstd; _helper_runningmeanstd()'])

def _helper_runningmeanstd():
    comm = MPI.COMM_WORLD
    np.random.seed(0)
    for (triple,axis) in [
        ((np.random.randn(3), np.random.randn(4), np.random.randn(5)),0),
        ((np.random.randn(3,2), np.random.randn(4,2), np.random.randn(5,2)),0),
        ((np.random.randn(2,3), np.random.randn(2,4), np.random.randn(2,4)),1),
        ]:


        x = np.concatenate(triple, axis=axis)
        ms1 = [x.mean(axis=axis), x.std(axis=axis), x.shape[axis]]


        ms2 = mpi_moments(triple[comm.Get_rank()],axis=axis)

        for (a1,a2) in zipsame(ms1, ms2):
            print(a1, a2)
            assert np.allclose(a1, a2)
            print("ok!")

class MpiAdam(object):
    def __init__(self, var_list, *, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=True, comm=None):
        self.var_list = var_list
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs
        size = sum(tf_util.numel(v) for v in var_list)
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        self.setfromflat = tf_util.SetFromFlat(var_list)
        self.getflat = tf_util.GetFlat(var_list)
        self.comm = MPI.COMM_WORLD if comm is None else comm

    def surgery_on_gradients(self, list_of_gradients):
        indexes = list(range(len(list_of_gradients)))
        np.random.shuffle(indexes)
        for i in indexes:
            random_indexes_but_i = indexes.copy()
            random_indexes_but_i.remove(i)
            inds = np.arange((len(list_of_gradients) - 1))
            np.random.shuffle(inds)
            random_indexes_but_i = np.array(random_indexes_but_i)[inds]
            for j in random_indexes_but_i:
                gi = list_of_gradients[i]
                gj = list_of_gradients[j]
                if cosine_similarity(gi.reshape(1, -1), gj.reshape(1, -1))[0][0] < 0:
                    gi = gi - np.dot(gi, gj) / np.linalg.norm(gi) ** 2 * gj
                    list_of_gradients[i] = gi
        return list_of_gradients

    def update(self, localg, stepsize):
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        globalg = np.zeros_like(localg)
        # print(self.comm.Get_rank(), 'bef :',  localg.sum())
        # list_of_gradient = self.comm.gather(localg, root=0)
        # if self.comm.Get_rank() == 0:
        #     list_of_gradient = self.surgery_on_gradients(list_of_gradient.copy())
        # localg = self.comm.scatter(list_of_gradient, root=0)
        self.comm.Allreduce(localg, globalg, op=MPI.SUM)
        if self.scale_grad_by_procs:
            globalg /= self.comm.Get_size()

        self.t += 1
        a = stepsize * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.setfromflat(self.getflat() + step)

    def sync(self):
        theta = self.getflat()
        self.comm.Bcast(theta, root=0)
        self.setfromflat(theta)

    def check_synced(self):
        if self.comm.Get_rank() == 0: # this is root
            theta = self.getflat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.getflat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)

@tf_util.in_session
def test_MpiAdam():
    np.random.seed(0)
    tf.set_random_seed(0)

    a = tf.Variable(np.random.randn(3).astype('float32'))
    b = tf.Variable(np.random.randn(2,5).astype('float32'))
    loss = tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.sin(b))

    stepsize = 1e-2
    update_op = tf.train.AdamOptimizer(stepsize).minimize(loss)
    do_update = tf_util.function([], loss, updates=[update_op])

    tf.get_default_session().run(tf.global_variables_initializer())
    for i in range(10):
        print(i,do_update())

    tf.set_random_seed(0)
    tf.get_default_session().run(tf.global_variables_initializer())

    var_list = [a,b]
    lossandgrad = tf_util.function([], [loss, tf_util.flatgrad(loss, var_list)], updates=[update_op])
    adam = MpiAdam(var_list)

    for i in range(10):
        l,g = lossandgrad()
        adam.update(g, stepsize)
        print(i,l)