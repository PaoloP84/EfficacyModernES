#!/usr/bin/python

"""
This class implements the policy.

"""
import numpy as np
import tensorflow as tf
import gym
from gym import spaces
import configparser
import time
import sys

# Dummy class
class NN:
    def __init__(self):
        self.dummy = 0

    def updateNormalizationVectors(self):
        self.dummy += 1

# Create a new interactive session
def make_session(single_threaded):
    if not single_threaded:
        return tf.InteractiveSession()
    return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))

# ================================================================
# Global session
# ================================================================

# Returns a default session for tensorflow
def get_session():
    return tf.get_default_session()

# Initialize variables (with the default initializer glorot_uniform_initializer?)
ALREADY_INITIALIZED = set()
def initialize():
    new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
    # Initialize all variables
    get_session().run(tf.variables_initializer(new_variables))
    ALREADY_INITIALIZED.update(new_variables)

# ================================================================
# Model components
# ================================================================

# Initializer based on the xavier/he method for setting initial parameters
def xavier_initializer(std=0.0): #std parameter is useless
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        # Compute the limits on the basis of the inputs and outputs of the layer
        limit = np.sqrt(2.0 / np.sum(shape))
        out = np.random.randn(*shape) * limit
        out = out.astype(np.float32)
        return tf.constant(out)
    return _initializer

# Initializer based on a normal distribution shaped by the <std> parameter
def normc_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        # Extract values from Gaussian distribution with mean 0.0 and std deviation 1.0
        out = np.random.randn(*shape).astype(np.float32)
        # Scale values according to <std> and the sum of square values (i.e., it reduces values of a factor about 50.0-100.0)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

# Compute the net-input of the neurons in the layer of size <size>, given the inputs <x> to the layer
def dense(x, size, name, weight_init=None, bias=True):
    # Get the weights
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    # Compute the net-input
    ret = tf.matmul(x, w)
    if bias:
        # Add bias
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer)
        return ret + b
    else:
        return ret

# ================================================================
# Basic Stuff
# ================================================================

def function(inputs, outputs, updates=None, givens=None):
    if isinstance(outputs, list):
        return _Function(inputs, outputs, updates, givens=givens)
    elif isinstance(outputs, dict):
        f = _Function(inputs, outputs.values(), updates, givens=givens)
        return lambda *inputs : dict(zip(outputs.keys(), f(*inputs)))
    else:
        f = _Function(inputs, [outputs], updates, givens=givens)
        return lambda *inputs : f(*inputs)[0]

# This class is responsible for performing all computations concerning the policy.
# More specifically, it is called for initializing variables, performing the spreading
# of the neural network, etc.
class _Function(object):
    def __init__(self, inputs, outputs, updates, givens, check_nan=False):
        assert all(len(i.op.inputs)==0 for i in inputs), "inputs should all be placeholders"
        # Set the inputs (they must be placeholders/tensors)
        self.inputs = inputs
        # Set the update operation (if passed)
        updates = updates or []
        # Create an op that groups multiple operations defined as tensors (see tf.group documentation)
        self.update_group = tf.group(*updates)
        # Operation(s) to be run on the inputs with their values (see __call__ method)
        self.outputs_update = list(outputs) + [self.update_group] # Outputs can be tensors too!
        self.givens = {} if givens is None else givens
        self.check_nan = check_nan
    def __call__(self, *inputvals):
        assert len(inputvals) == len(self.inputs)
        # Create a dict where inputs are filled with inputvals (required for run outputs_update)
        feed_dict = dict(zip(self.inputs, inputvals))
        feed_dict.update(self.givens) # ??? (givens is None, probably it does nothing...)
        # Performs the operations defined in outputs_update with the values specified in feed_dict
        # N.B.: it ignores the last element of results (operation [:-1])
        results = get_session().run(self.outputs_update, feed_dict=feed_dict)[:-1] # Results are tensors
        if self.check_nan:
            if any(np.isnan(r).any() for r in results):
                raise RuntimeError("Nan detected")
        return results

# ================================================================
# Flat vectors
# ================================================================

def var_shape(x):
    out = [k.value for k in x.get_shape()]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def numel(x):
    return intprod(var_shape(x))

def intprod(x):
    return int(np.prod(x))

# Set parameters
class SetFromFlat(object):
    def __init__(self, var_list, dtype=tf.float32):
        assigns = []
        shapes = list(map(var_shape, var_list))
        total_size = np.sum([intprod(shape) for shape in shapes])
        # Parameters are defined as placeholders/tensors
        self.theta = theta = tf.placeholder(dtype,[total_size])
        # Define the assignment operation to assign values to parameters
        start=0
        assigns = []
        for (shape,v) in zip(shapes,var_list):
            size = intprod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start+size],shape)))
            start+=size
        assert start == total_size
        self.op = tf.group(*assigns)
    def __call__(self, theta):
        get_session().run(self.op, feed_dict={self.theta:theta})

# Get parameters (i.e., it returns a tensor obtained by concatenating the input tensors)
class GetFlat(object):
    def __init__(self, var_list):
        self.op = tf.concat([tf.reshape(v, [numel(v)]) for v in var_list], 0)
    def __call__(self):
        return get_session().run(self.op)

# Class storing/updating mean and standard deviation of the observations
class RunningStat(object):
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count

def bins(x, dim, num_bins, name, initializer):
    scores = dense(x, dim * num_bins, name, initializer(0.01))
    scores_nab = tf.reshape(scores, [-1, dim, num_bins])
    return tf.argmax(scores_nab, 2)  # 0 ... num_bins-1

# Policy/neural network
class PolicyTf(object):
    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        # Initialize policy
        self.scope = self._initialize(*args, **kwargs)
        # List of all variables
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.name)
        # List of trainable variables only
        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        self.nparams = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        # Set parameters function
        self._setfromflat = SetFromFlat(self.trainable_variables)
        # Get parameters function
        self._getflat = GetFlat(self.trainable_variables)
        # Define a placeholder for each variable
        placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
        # Assign a value to each variable?
        self.set_all_vars = function(
                inputs=placeholders,
                outputs=[],
                updates=[tf.group(*[v.assign(p) for v, p in zip(self.all_variables, placeholders)])]
            )

    def initTfVars(self):
        initialize()

    # Policy reset (TO BE CAREFULLY ANALYZED)
    def reset(self):
        # Initialize normalization vector
        if self.normalize == 1:
            self.normvector = np.arange((self.ob_space.shape[0] * 2), dtype=np.float64)
        else:
            self.normvector = None
        # List of all variables
        self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.name)
        # List of trainable variables only
        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
        self.nparams = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
        # Set parameters function
        self._setfromflat = SetFromFlat(self.trainable_variables)
        # Get parameters function
        self._getflat = GetFlat(self.trainable_variables)
        # Define a placeholder for each variable
        placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
        # Assign a value to each variable?
        self.set_all_vars = function(
                inputs=placeholders,
                outputs=[],
                updates=[tf.group(*[v.assign(p) for v, p in zip(self.all_variables, placeholders)])]
            )
        # Reset tensorflow variables
        self.initTfVars()
        # Reset stat (in case of normalization)
        if self.normalize == 1:
            self.initStat()
            self.statcnt = 0

    def initStat(self):
        self.ob_stat = RunningStat(
                self.ob_space.shape,
                eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
             )

    # Update number of evaluations per generation (required by CMA-ES, sNES and xNES algorithms
    # if the batchSize is not read from the configuration file and is computed by the algorithm)
    def updaten(self, n):
        self.n = n

    # Set the seed for the environment
    def setSeed(self, seed):
        self.env.seed(seed)

    # === Rollouts/training ===
    # virtual function, implemented in sub-classes
    def rollout(self, render=False, timestep_limit=None):
        raise NotImplementedError

    def set_trainable_flat(self, x):
        self._setfromflat(x)

    def get_trainable_flat(self):
        return self._getflat()

    def readConfig(self, filename):
        # parse the [POLICY] section of the configuration file
        config = configparser.ConfigParser()
        config.read(filename)
        options = config.options("POLICY")
        for o in options:
            found = 0
            if o == "ntrials":
                self.ntrials = config.getint("POLICY","ntrials")
                found = 1
            if o == "nttrials":
                self.nttrials = config.getint("POLICY","nttrials")
                found = 1
            if o == "maxsteps":
                self.maxsteps = config.getint("POLICY","maxsteps")
                found = 1
            if o == "nhiddens":
                self.nhiddens = config.getint("POLICY","nhiddens")
                found = 1
            if o == "nlayers":
                self.nlayers = config.getint("POLICY","nlayers")
                found = 1
            if o == "bias":
                self.bias = config.getint("POLICY","bias")
                found = 1
            if o == "out_type":
                self.out_type = config.getint("POLICY","out_type")
                found = 1
            if o == "nbins":
                self.nbins = config.getint("POLICY","nbins")
                found = 1
            if o == "architecture":
                self.architecture = config.getint("POLICY","architecture")
                found = 1
            if o == "afunction":
                self.afunction = config.getint("POLICY","afunction")
                found = 1
            if o == "winit":
                self.winit = config.getint("POLICY","winit")
                found = 1
            if o == "action_noise":
                self.action_noise = config.getint("POLICY","action_noise")
                found = 1
            if o == "normalize":
                self.normalize = config.getint("POLICY","normalize")
                found = 1
            if o == "clip":
                self.clip = config.getint("POLICY","clip")
                found = 1
            if o == "wrange":
                self.wrange = config.getint("POLICY","wrange")
                found = 1
            if found == 0:
                print("\033[1mOption %s in section [POLICY] of %s file is unknown\033[0m" % (o, filename))
                sys.exit()

    def doGeneralization(self, test):
        self.genTest = test

    def runEvo(self):
        self.test = False

    def runTest(self):
        self.test = True

    @property
    def get_seed(self):
        return self.seed

    @property
    def generalize(self):
        return (self.nttrials > 0)

    def save(self, filename):
        # Save the policy. More specifically, it stores:
        # - ob_mean
        # - ob_std
        # - weights (for all hidden layers and outputs)
        # - biases (for all hidden layers and outputs)
        assert filename.endswith('.h5')
        with h5py.File(filename, 'w') as f:
            for v in self.all_variables:
                f[v.name] = v.eval()
            # TODO: it would be nice to avoid pickle, but it's convenient to pass Python objects to _initialize
            # (like Gym spaces or numpy arrays)
            f.attrs['name'] = type(self).__name__
            f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))

    @classmethod
    def Load(cls, filename, extra_kwargs=None):
        with h5py.File(filename, 'r') as f:
            args, kwargs = pickle.loads(f.attrs['args_and_kwargs'].tostring())
            if extra_kwargs:
                kwargs.update(extra_kwargs)
            policy = cls(*args, **kwargs)
            policy.set_all_vars(*[f[v.name][...] for v in policy.all_variables])
        return policy

    def _initialize(self, env, ninputs, noutputs, low, high, n, filename, seed):
        # Copy parameters from configuration
        self.env = env
        # Observation and action spaces
        # Check if the environment has already its own observation space
        if hasattr(self.env, 'observation_space'):
            self.ob_space = env.observation_space
        else:
            # Define a new observation space
            ob_high = np.inf * np.ones(ninputs)
            ob_low = -ob_high
            self.ob_space = spaces.Box(low=ob_low, high=ob_high, dtype=np.float32)
        # Check if the environment has already its own action space
        if hasattr(self.env, 'action_space'):
            self.ac_space = env.action_space
            self.low = self.ac_space.low
            self.high = self.ac_space.high
        else:
            # Define a new action space
            self.low = low * np.ones(noutputs)
            self.high = high * np.ones(noutputs)
            self.ac_space = spaces.Box(low=self.low, high=self.high, dtype=np.float32)
        # Define other variables
        self.seed = seed
        self.environment = None
        self.saveeach = 0
        self.ob_stat = None
        self.statcnt = 0
        self.n = n
        # Initialize parameters to default values
        self.ntrials = 1        # evaluation triala
        self.nttrials = 0       # post-evaluation trials
        self.maxsteps = 1000    # max number of steps (used for ERPolicy only)
        self.nhiddens = 50      # number of hiddens
        self.nlayers = 1        # number of hidden layers 
        self.bias = 0           # whether or not we have biases
        self.out_type = 2       # output type (actFunct, linear, binary, bins)
        self.nbins = 1          # number of bins
        self.architecture = 0   # Feed-forward, recurrent, or full-recurrent network
        self.afunction = 2      # activation function
        self.winit = 0          # weight initialization: Xavier, normc, uniform
        self.action_noise = 1   # noise applied to actions
        self.normalize = 0      # normalize observations
        self.clip = 0           # clip observation
        self.wrange = 1.0       # weight range (used for uniform initialization only)
        self.genTest = False    # generalization test flag
        self.test = False       # test flag
        # Read configuration file
        self.readConfig(filename)
        # Initialize normalization vector
        if self.normalize == 1:
            self.normvector = np.arange((self.ob_space.shape[0] * 2), dtype=np.float64)
        else:
            self.normvector = None

        # Hidden dimensions
        self.hidden_dims = []
        if self.nlayers == 1:
            # One layer only
            self.hidden_dims = [ self.nhiddens ]
        else:
            # Multiple layers
            for layer in range(self.nlayers):
                hidden_dims.append(self.nhiddens)

        # Dummy object (for the interaction with algorithms)
        self.nn = NN()

        assert len(self.ob_space.shape) == len(self.ac_space.shape) == 1
        assert np.all(np.isfinite(self.low)) and np.all(np.isfinite(self.high)), \
                'Action bounds required'

        self.nonlin = {2: tf.tanh, 1: tf.sigmoid}[self.afunction]
        self.initializer = {1: normc_initializer, 0: xavier_initializer}[self.winit]

        with tf.variable_scope(type(self).__name__, reuse=tf.AUTO_REUSE) as scope:
            # Observation normalization
            ob_mean = tf.get_variable(
                'ob_mean', self.ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            ob_std = tf.get_variable(
                'ob_std', self.ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
            in_mean = tf.placeholder(tf.float32, self.ob_space.shape)
            in_std = tf.placeholder(tf.float32, self.ob_space.shape)
            # This should normalize observations based on the updated mean and standard deviation
            self._set_ob_mean_std = function([in_mean, in_std], [], updates=[
                tf.assign(ob_mean, in_mean),
                tf.assign(ob_std, in_std),
            ])

            # Policy network
            o = tf.placeholder(tf.float32, [None] + list(self.ob_space.shape))
            if self.normalize == 1:
                # Normalized observations are converted to values in the range [-5.0,5.0] (N.B.: values outside the range are truncated in the range!!!)
                a = self._make_net(tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0)) # We ignore self.clip parameter
            else:
                # Observations are not normalized
                a = self._make_net(o)
            # Perform the activation of the network (all neurons) and determines the action
            self._act = function([o], a)
        return scope

    def _make_net(self, o):
        # Process observation
        if self.architecture == 0:
            # Feed forward network
            x = o
            for ilayer, hd in enumerate(self.hidden_dims):
                x = self.nonlin(dense(x, hd, 'l{}'.format(ilayer), weight_init=self.initializer(1.0), bias=True))
        else:
            raise NotImplementedError(self.architecture)

        # Map to action
        adim, ahigh, alow = self.ac_space.shape[0], self.high, self.low

        if self.out_type == 5:
            # Uniformly spaced bins, from ac_space.low to ac_space.high
            num_ac_bins = self.nbins
            aidx_na = bins(x, adim, num_ac_bins, 'out', self.initializer)  # 0 ... num_ac_bins-1
            ac_range_1a = (ahigh - alow)[None, :]
            a = 1. / (num_ac_bins - 1.) * tf.to_float(aidx_na) * ac_range_1a + alow[None, :]
        elif self.out_type >= 1 and self.out_type <= 3:
            if self.out_type == 3:
                # Linear output(s)
                a = dense(x, adim, 'out', weight_init=self.initializer(0.01), bias=True)
            else:
                # Either logistic or tanh output(s)
                a = self.nonlin(dense(x, adim, 'out', weight_init=self.initializer(0.01), bias=True))
        elif self.out_type == 4:
            # Binary output(s)
            a = dense(x, adim, 'out', weight_init=self.initializer(0.01), bias=True)
            # Apply binary outputs
            myones = tf.fill(tf.shape(a), 1.0)
            myzeros = tf.fill(tf.shape(a), 0.0)
            bin_ac = tf.where(tf.greater_equal(a, 0.0), myones, myzeros)
            a = bin_ac
        else:
            raise NotImplementedError(self.out_type)

        return a

    def act(self, ob):
        a = self._act(ob)
        # Add noise to action if random_stream is not None
        if self.action_noise == 1:
            a += np.random.randn(*a.shape) * 0.01
        return a

    @property
    def needs_ob_stat(self):
        return True
    
    # Normalize observations
    def set_ob_stat(self, ob_mean, ob_std):
        self._set_ob_mean_std(ob_mean, ob_std)

    def initialize_from(self, filename, ob_stat=None):
        """
        Initializes weights from another policy, which must have the same architecture (variable names),
        but the weight arrays can be smaller than the current policy.
        """
        with h5py.File(filename, 'r') as f:
            f_var_names = []
            f.visititems(lambda name, obj: f_var_names.append(name) if isinstance(obj, h5py.Dataset) else None)
            assert set(v.name for v in self.all_variables) == set(f_var_names), 'Variable names do not match'

            init_vals = []
            for v in self.all_variables:
                shp = v.get_shape().as_list()
                f_shp = f[v.name].shape
                assert len(shp) == len(f_shp) and all(a >= b for a, b in zip(shp, f_shp)), \
                    'This policy must have more weights than the policy to load'
                init_val = v.eval()
                # ob_mean and ob_std are initialized with nan, so set them manually
                if 'ob_mean' in v.name:
                    init_val[:] = 0
                    init_mean = init_val
                elif 'ob_std' in v.name:
                    init_val[:] = 0.001
                    init_std = init_val
                # Fill in subarray from the loaded policy
                init_val[tuple([np.s_[:s] for s in f_shp])] = f[v.name]
                init_vals.append(init_val)
            self.set_all_vars(*init_vals)

        if ob_stat is not None:
            ob_stat.set_from_init(init_mean, init_std, init_count=1e5)

class GymPolicyTf(PolicyTf):
    def __init__(self, env, ninputs, noutputs, low, high, n, filename, seed):
        PolicyTf.__init__(self, env, ninputs, noutputs, low, high, n, filename, seed)

    # === Rollouts/training ===
    def rollout(self, render=False, timestep_limit=None):
        # We update the stat counter
        if self.normalize == 1 and self.statcnt == 0:
            # We set normalization mean and standard deviation
            self.set_ob_stat(self.ob_stat.mean, self.ob_stat.std)
            self.normvector[0:self.ob_space.shape[0]] = np.copy(self.ob_stat.mean)
            self.normvector[self.ob_space.shape[0]:(self.ob_space.shape[0] * 2)] = np.copy(self.ob_stat.std)
        self.statcnt += 1

        env_timestep_limit = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps') # This might be fixed...
        if env_timestep_limit is None:
            env_timestep_limit = 1000
        timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
        if timestep_limit is None:
            timestep_limit = self.maxsteps
        rews = 0.0
        steps = 0
        # Set the number of trials depending on whether or not test flag is set to True
        ntrials = self.ntrials
        if self.genTest:
            ntrials = self.nttrials
        # Loop over the number of trials
        for trial in range(ntrials):
            # Typically we do not save observations
            save_obs = False
            # Observations must be saved if and only if normalization
            # flag is set to True and we are not in test phase
            if self.normalize == 1 and not self.test:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    # Save observations
                    save_obs = True
            if save_obs:
                obs = []
            # Reset environment
            ob = self.env.reset()
            # Reward for current trial
            crew = 0.0
            # Perform the steps
            t = 0
            while t < timestep_limit:
                # Compute the action (noisy if random_stream is not None)
                # action ranges are defined in MujocoEnv class, in __init__ method
                # (see https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py
                # lines 46-49), but I cannot find where action values are set in the ranges!!!)
                ac = self.act(ob[None])[0]
                # Save observations
                if save_obs:
                    obs.append(ob)
                # Perform a step
                ob, rew, done, _ = self.env.step(ac) # mujoco internally scales actions in the proper ranges!!!
                # Append the reward
                crew += rew
                t += 1
                if render:
                    self.env.render()
                if done:
                    break
            # Print fitness for each trial during test phase
            if self.test:
                print("Trial %d - fitness %lf" % (trial, crew))
            # Update overall reward
            rews += crew
            # Update steps
            steps += t
            # Update stat if we stored observations for current trial
            if save_obs:
                obs = np.array(obs)
                self.ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
        if self.statcnt == self.n:
            self.statcnt = 0
        # Normalize reward by the number of trials
        rews /= ntrials
        return rews, steps

class ErPolicyTf(PolicyTf):
    def __init__(self, env, ninputs, noutputs, low, high, n, ob, ac, done, filename, seed):
        PolicyTf.__init__(self, env, ninputs, noutputs, low, high, n, filename, seed)
        # Copy pointers
        self.ob = ob
        self.ac = ac
        self.done = done

    # === Rollouts/training ===
    def rollout(self, render=False, timestep_limit=None):
        # We update the stat counter
        if self.normalize == 1 and self.statcnt == 0:
            # We set normalization mean and standard deviation
            self.set_ob_stat(self.ob_stat.mean, self.ob_stat.std)
            self.normvector[0:self.ob_space.shape[0]] = np.copy(self.ob_stat.mean)
            self.normvector[self.ob_space.shape[0]:(self.ob_space.shape[0] * 2)] = np.copy(self.ob_stat.std)
        self.statcnt += 1

        rews = 0.0
        steps = 0
        # Set the number of trials depending on whether or not test flag is set to True
        ntrials = self.ntrials
        if self.genTest:
            ntrials = self.nttrials
        # Loop over the number of trials
        for trial in range(ntrials):
            # Typically we do not save observations
            save_obs = False
            # Observations must be saved if and only if normalization
            # flag is set to True and we are not in test phase
            if self.normalize == 1 and not self.test:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    # Save observations
                    save_obs = True
            if save_obs:
                obs = []
            # Reset environment
            self.env.reset()
            # Reward for current trial
            crew = 0.0
            # Perform the steps
            t = 0
            while t < self.maxsteps:
                # Compute the action (noisy if random_stream is not None)
                # action ranges are defined in MujocoEnv class, in __init__ method
                # (see https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py
                # lines 46-49), but I cannot find where action values are set in the ranges!!!)
                ac = self.act(self.ob[None])[0]
                # We need to copy back the action(s) to action pointer (Tensorflow returns a different pointer!!!)
                # The for-loop may slow down program speed
                for i in range(self.ac_space.shape[0]):
                    self.ac[i] = ac[i] # Now action pointer contains the right action(s)
                # Save observations
                if save_obs:
                    obs.append(self.ob)
                # Perform a step
                rew = self.env.step() # mujoco internally scales actions in the proper ranges!!!
                # Append the reward
                crew += rew
                t += 1
                if render:
                    self.env.render()
                if self.done:
                    break
            # Print fitness for each trial during test phase
            if self.test:
                print("Trial %d - fitness %lf" % (trial, crew))
            # Update overall reward
            rews += crew
            # Update steps
            steps += t
            # Update stat if we stored observations for current trial
            if save_obs:
                obs = np.array(obs)
                self.ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
        if self.statcnt == self.n:
            self.statcnt = 0
        # Normalize reward by the number of trials
        rews /= ntrials
        return rews, steps

