import os
import subprocess
import sys
import importlib
import json
import inspect
import functools
from copy import deepcopy
from subprocess import CalledProcessError

import torch
import numpy as np
from mpi4py import MPI

from src.utils.mpi_util import mpi_moments
import random


def get_time_tracker():
    class TimeTracker:
        def __init__(self):
            self.time_stats =  dict(time_rollout=0,
                                    time_store=0,
                                    time_eval=0,
                                    time_batch=0,
                                    time_update=0,
                                    time_train=0,
                                    time_epoch=0,
                                    time_info=0,
                                    time_reset=0,
                                    time_env=0,
                                    time_social_partner=0,
                                    time_goal_sampler=0,
                                    time_get_classif_samples=0,
                                    time_encoding=0,
                                    time_replay=0,
                                    time_recompute_reward=0,
                                    time_sample_1=0,
                                    time_sample_shuffle=0,
                                    time_transition_batch=0,
                                    time_run_batch=0,
                                    time_buffer_sample=0,
                                    time_reward_func_replay=0,
                                    time_argwhere=0,
                                    time_random=0,
                                    time_pre_replay=0,
                                    time_comm=0,
                                    time_sample_ind=0,
                                    time_reward_func_update=0,
                                    time_data_process=0,
                                    time_sample_goal=0,
                                    time_infer_social_partner=0,
                                    time_scatter_data=0,
                                    time_store_policy=0)

        def add(self, **kwargs):
            for k in kwargs.keys():
                self.time_stats[k] += kwargs[k]

        def update(self, **kwargs):
            for k in kwargs.keys():
                self.time_stats[k] = kwargs[k]

    return TimeTracker()

def get_stat_func(line='mean', err='std'):

    if line == 'mean':
        def line_f(a):
            return np.nanmean(a, axis=0)
    elif line == 'median':
        def line_f(a):
            return np.nanmedian(a, axis=0)
    else:
        raise NotImplementedError

    if err == 'std':
        def err_plus(a):
            return line_f(a) + np.nanstd(a, axis=0)
        def err_minus(a):
            return line_f(a) - np.nanstd(a, axis=0)
    elif err == 'sem':
        def err_plus(a):
            return line_f(a) + np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
        def err_minus(a):
            return line_f(a) - np.nanstd(a, axis=0) / np.sqrt(a.shape[0])
    elif err == 'range':
        def err_plus(a):
            return np.nanmax(a, axis=0)
        def err_minus(a):
            return np.nanmin(a, axis=0)
    elif err == 'interquartile':
        def err_plus(a):
            return np.nanpercentile(a, q=75, axis=0)
        def err_minus(a):
            return np.nanpercentile(a, q=25, axis=0)
    else:
        raise NotImplementedError

    return line_f, err_minus, err_plus

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False

def clean_dict_for_json(a_dict):
    if is_jsonable(a_dict):
        return deepcopy(a_dict)
    else:
        if isinstance(a_dict, dict):
            new_dict = dict()
            for k in a_dict.keys():
                new_dict[k] = clean_dict_for_json(a_dict[k])
        else:
            return None
        return new_dict



def fork(num_cpu):
    # Fork for multi-CPU MPI implementation.
    if num_cpu > 1:
        try:
            whoami = mpi_fork(num_cpu, ['--bind-to', 'core:overload-allowed'])
            # whoami = mpi_fork(num_cpu, ['--bind-to', 'core:overload-allowed'])
        except CalledProcessError:
            # fancy version of mpi call failed, try simple version
            whoami = mpi_fork(num_cpu)

        if whoami == 'parent':
            sys.exit(0)

    rank = MPI.COMM_WORLD.Get_rank()
    return rank

def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper

def mpi_average(value):
    if value == []:
        value = [0.]
    if not isinstance(value, list):
        value = [value]
    return mpi_moments(np.array(value))[0]

def find_save_path(dir, trial_id):
    """
    Create a directory to save notebooks and arguments. Adds 100 to the trial id if a directory already exists.

    Params
    ------
    - dir (str)
        Main saving directory
    - trial_id (int)
        Trial identifier
    """
    i=0
    while True:
        save_dir = dir+str(trial_id+i*100)+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            break
        i+=1
    return save_dir

def set_global_seeds(i):
    torch.manual_seed(i)
    np.random.seed(i)
    random.seed(i)

def import_function(spec):
    """Import a function identified by a string like "pkg.module:fn_name".
    """
    mod_name, fn_name = spec.split(':')
    module = importlib.import_module(mod_name)
    fn = getattr(module, fn_name)
    return fn



def var_shape(x):
    out = x.get_shape().as_list()
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))

def install_mpi_excepthook():
    import sys
    from mpi4py import MPI
    old_hook = sys.excepthook

    def new_hook(a, b, c):
        old_hook(a, b, c)
        sys.stdout.flush()
        sys.stderr.flush()
        MPI.COMM_WORLD.Abort()
    sys.excepthook = new_hook


def mpi_fork(n, extra_mpi_args=[]):
    """Re-launches the current script with workers
    Returns "parent" for original parent, "child" for MPI children
    """
    if n <= 1:
        return "child"
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        # "-bind-to core" is crucial for good performance
        args = ["mpirun", "-np", str(n)] + \
            extra_mpi_args + \
            [sys.executable]

        args += sys.argv
        subprocess.check_call(args, env=env)
        return "parent"
    else:
        install_mpi_excepthook()
        return "child"


def transitions_in_episode_batch(episode_batch):
    """Number of transitions in a given episode batch.
    """
    shape = episode_batch['u'].shape
    return shape[0] * shape[1]

