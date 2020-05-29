# Humanoid task

# Libraries to be imported
import gym
import numpy as np
import tensorflow as tf
from numpy import floor, log, eye, zeros, array, sqrt, sum, dot, tile, outer, real
from numpy import exp, diag, power, ravel
from numpy.linalg import eig, norm
from numpy.random import randn
import math
import random
import time
from scipy import zeros, ones
from scipy.linalg import expm
import configparser
import sys
import os
from six import integer_types
import struct
import pickle
import h5py

# Directory of the script .py
scriptdirname = os.path.dirname(os.path.realpath(__file__))
# Directory where files will be saved
filedir = None
# Global variables
center = None						   # the solution center
sample = None						   # the solution samples
# Network
ac_bins = "continuous:"
ac_noise_std = 0.01
connection_type = "ff"
hidden_dims = None
nonlin_type = "sigmoid"
nonlin_out = False
init_type = "normc"
out_type = 0
obSpace = None
acSpace = None
# Evaluation
ntrials = 1
envChangeEvery = 1
numHiddens = 0
numHiddenLayers = 1
environment = None
fullyRandom = False
storingRate = 1000
biasCorr = True
# Id algo
id_algo = 1
# Specific parameter for ES with Adam optimizer
stepsize = 0.01
noiseStdDev = 0.02
sampleSize = 20

# Array of seeds (to be generated at each generation) used to change initial position
currSeeds = None

# Methods copied from tf_util

# ================================================================
# Global session
# ================================================================

def get_session():
	return tf.get_default_session()

# Initialize variables (with the default initializer glorot_uniform_initializer?)
ALREADY_INITIALIZED = set()
def initialize():
	new_variables = set(tf.global_variables()) - ALREADY_INITIALIZED
	get_session().run(tf.variables_initializer(new_variables))
	ALREADY_INITIALIZED.update(new_variables)

# ================================================================
# Model components
# ================================================================

# Initializer based on the xavier/he method for setting initial parameters
def xavierhe_initializer(std=0.0): #std parameter is useless
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
		print(shape)
		# Extract values from Gaussian distribution with mean 0.0 and std deviation 1.0
		out = np.random.randn(*shape).astype(np.float32)
		# Scale values according to <std> and the sum of square values (i.e., it reduces values of a factor about 50.0-100.0)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer

# Compute the net-input of the neurons in the layer of size <size>, given the inputs <x> to the layer
def dense(x, size, name, weight_init=None, bias=True):
	w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
	# Compute the net-input
	ret = tf.matmul(x, w)
	if bias:
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

def bins(x, dim, num_bins, name):
	scores = dense(x, dim * num_bins, name, normc_initializer(0.01))
	scores_nab = tf.reshape(scores, [-1, dim, num_bins])
	return tf.argmax(scores_nab, 2)  # 0 ... num_bins-1

# Policy/neural network
class Policy:
	def __init__(self, *args, **kwargs):
		self.args, self.kwargs = args, kwargs
		# Initialize policy
		self.scope = self._initialize(*args, **kwargs)
		# List of all variables
		self.all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope.name)
		# List of trainable variables only
		self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
		self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
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

	# === Rollouts/training ===

	def rollout(self, env, render=False, timestep_limit=None, save_obs=False, random_stream=None, trial=0, seed=None):
		global obSpace
		global acSpace
		"""
		If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
		Otherwise, no action noise will be added.
		"""
		env_timestep_limit = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
		if env_timestep_limit is None:
			env_timestep_limit = 1000
		timestep_limit = env_timestep_limit if timestep_limit is None else min(timestep_limit, env_timestep_limit)
		if timestep_limit is None:
			timestep_limit = 1000
		rews = []
		t = 0
		if save_obs:
			obs = []
		# To ensure replicability (we always pass a valid seed, even if fully-random evaluation is going to be run)
		if seed is not None:
			env.seed(seed)
		ob = env.reset()
		currOb = ob
		if "Dict" in obSpace:
			currOb = ob["observation"]
		for _ in range(timestep_limit):
			# Compute the action (noisy if random_stream is not None)
			# action ranges are defined in MujocoEnv class, in __init__ method
			# (see https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py
			# lines 46-49), but I cannot find where action values are set in the ranges!!!)
			ac = self.act(currOb[None], random_stream=random_stream)[0]
			# Perform checks on the action to be performed
			if not "Discrete" in acSpace:
				# Check action range
				if self.ac_bins == 'binary:' or self.nonlin_out:
					# We truncate outputs out of range [-1,1] when they are binary or come from tanh
					for i in range(ac.shape[0]):
						if ac[i] > 1.0:
							ac[i] = 1.0
						if ac[i] < -1.0:
							ac[i] = -1.0
			else:
				# In case of discrete action, select the one corresponding to the neuron with highest activation
				ac = np.argmax(ac)
			# Save observations
			if save_obs:
				obs.append(ob)
			# Perform a step
			ob, rew, done, _ = env.step(ac) # mujoco internally scales actions in the proper ranges!!!
			# Append the reward
			rews.append(rew)
			t += 1
			if render:
				env.render()
			if done:
				break
		# Transform the list of rewards into an array
		rews = np.array(rews, dtype=np.float32)
		if save_obs:
			# Return the observations too!!!
			return rews, t, np.array(obs)
		return rews, t

	def set_trainable_flat(self, x):
		self._setfromflat(x)

	def get_trainable_flat(self):
		return self._getflat()

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

	def _initialize(self, ob_space, ac_space, ac_bins, ac_noise_std, nonlin_type, hidden_dims, connection_type, nonlin_out, init_type):
		global acSpace
		# Copy parameters from configuration
		self.ac_space = ac_space
		self.ac_bins = ac_bins
		self.ac_noise_std = ac_noise_std
		self.hidden_dims = hidden_dims
		self.connection_type = connection_type
		# Added the possibility to have non linear activation function for the output and a bias initializer
		self.nonlin_out = nonlin_out

		if not "Discrete" in acSpace:
			assert len(ob_space.shape) == len(self.ac_space.shape) == 1
			assert np.all(np.isfinite(self.ac_space.low)) and np.all(np.isfinite(self.ac_space.high)), \
				'Action bounds required'

		self.nonlin = {'tanh': tf.tanh, 'sigmoid': tf.sigmoid}[nonlin_type]
		self.initializer = {'normc': normc_initializer, 'xavierhe': xavierhe_initializer}[init_type]

		with tf.variable_scope(type(self).__name__, reuse=tf.AUTO_REUSE) as scope:
			# Observation normalization
			ob_mean = tf.get_variable(
				'ob_mean', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
			ob_std = tf.get_variable(
				'ob_std', ob_space.shape, tf.float32, tf.constant_initializer(np.nan), trainable=False)
			in_mean = tf.placeholder(tf.float32, ob_space.shape)
			in_std = tf.placeholder(tf.float32, ob_space.shape)
			# This should normalize observations based on the updated mean and standard deviation
			self._set_ob_mean_std = function([in_mean, in_std], [], updates=[
				tf.assign(ob_mean, in_mean),
				tf.assign(ob_std, in_std),
			])

			# Policy network
			o = tf.placeholder(tf.float32, [None] + list(ob_space.shape))
			# Normalized observations are converted to values in the range [-5.0,5.0] (N.B.: values outside the range are truncated in the range!!!)#
			a = self._make_net(tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0))
			# Perform the activation of the network (all neurons) and determines the action
			self._act = function([o], a)
		return scope

	def _make_net(self, o):
		global acSpace
		# Process observation
		if self.connection_type == 'ff':
			x = o
			for ilayer, hd in enumerate(self.hidden_dims):
				x = self.nonlin(dense(x, hd, 'l{}'.format(ilayer), weight_init=self.initializer(1.0), bias=True))
		else:
			raise NotImplementedError(self.connection_type)

		# Map to action
		if not "Discrete" in acSpace:
			adim, ahigh, alow = self.ac_space.shape[0], self.ac_space.high, self.ac_space.low
		else:
			adim, ahigh, alow = self.ac_space.n, None, None
		assert isinstance(self.ac_bins, str)
		ac_bin_mode, ac_bin_arg = self.ac_bins.split(':')

		if ac_bin_mode == 'uniform':
			# Uniformly spaced bins, from ac_space.low to ac_space.high
			num_ac_bins = int(ac_bin_arg)
			aidx_na = bins(x, adim, num_ac_bins, 'out')  # 0 ... num_ac_bins-1
			ac_range_1a = (ahigh - alow)[None, :]
			a = 1. / (num_ac_bins - 1.) * tf.to_float(aidx_na) * ac_range_1a + alow[None, :]
		elif ac_bin_mode == 'continuous':
			if not self.nonlin_out:
				a = dense(x, adim, 'out', weight_init=self.initializer(0.01), bias=True)
			else:
				a = self.nonlin(dense(x, adim, 'out', weight_init=self.initializer(0.01), bias=True))
		elif ac_bin_mode == 'binary':
			if not self.nonlin_out:
				a = dense(x, adim, 'out', weight_init=self.initializer(0.01), bias=True)
			else:
				a = self.nonlin(dense(x, adim, 'out', weight_init=self.initializer(0.01), bias=True))
			# Apply binary outputs
			myones = tf.fill(tf.shape(a), 1.0)
			myzeros = tf.fill(tf.shape(a), 0.0)
			bin_ac = tf.where(tf.greater_equal(a, 0.0), myones, myzeros)
			a = bin_ac
		else:
			raise NotImplementedError(ac_bin_mode)

		return a

	def act(self, ob, random_stream=None):
		a = self._act(ob)
		# Add noise to action if random_stream is not None
		if random_stream is not None and self.ac_noise_std != 0:
			a += random_stream.randn(*a.shape) * self.ac_noise_std
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

# Create a new interactive session
def make_session(single_threaded):
	if not single_threaded:
		return tf.InteractiveSession()
	return tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1))

# Sorting functions

# Descendent sorting
def descendent_sort(vect):
	# Copy of the vector
	tmpv = np.copy(vect)
	n = len(tmpv)
	# Index list
	index = np.arange(n, dtype=np.int32)
	i = 0
	while i < n:
		# Look for maximum
		maxv = tmpv[0]
		maxi = 0
		j = 1
		while j < n:
			if tmpv[j] > maxv:
				maxv = tmpv[j]
				maxi = j
			j += 1
		vect[i] = tmpv[maxi]
		index[i] = maxi
		i += 1
		# Set invalid value
		tmpv[maxi] = -999999999999.0
	return vect, index

# Ascendent sorting
def ascendent_sort(vect):
	# Copy of the vector
	tmpv = np.copy(vect)
	n = len(tmpv)
	# Index list
	index = np.arange(n, dtype=np.int32)
	i = 0
	while i < n:
		# Look for maximum
		minv = tmpv[0]
		mini = 0
		j = 1
		while j < n:
			if tmpv[j] < minv:
				minv = tmpv[j]
				mini = j
			j += 1
		vect[i] = tmpv[mini]
		index[i] = mini
		i += 1
		# Set invalid value
		tmpv[mini] = 999999999999.0
	return vect, index

# average fitness of the samples
def AverageFit(fitness):
	avef = 0.0
	for i in range(len(fitness)):
		avef = avef + fitness[i]
	avef = avef / len(fitness)
	return avef

# Evolve with CMA-ES algorithm
def evolve_CMAES(env, policy, ob_stat, seed, nevals, ntrials):
	global center
	global sample
	global currSeeds
	global fullyRandom
	global storingRate
	global filedir

	# Get parameters to be trained (once only!!!)
	center = policy.get_trainable_flat()
	# Extract the number of parameters
	nparams = len(center)

	# setting parameters
	batchSize = int(4 + floor(3 * log(nparams))) # population size, offspring number
	mu = int(floor(batchSize / 2)) # number of parents/points for recombination
	weights = log(mu + 1) - log(array(range(1, mu + 1))) # use array
	weights /= sum(weights)	# normalize recombination weights array
	muEff = sum(weights) ** 2 / sum(power(weights, 2)) # variance-effective size of mu
	cumCov = 4 / float(nparams + 4)	# time constant for cumulation for covariance matrix
	cumStep = (muEff + 2) / (nparams + muEff + 3) # t-const for cumulation for Size control
	muCov = muEff # size of mu used for calculating learning rate covLearningRate
	covLearningRate = ((1 / muCov) * 2 / (nparams + 1.4) ** 2 + (1 - 1 / muCov) *	   # learning rate for
			((2 * muEff - 1) / ((nparams + 2) ** 2 + 2 * muEff)))		   # covariance matrix
	dampings = 1 + 2 * max(0, sqrt((muEff - 1) / (nparams + 1)) - 1) + cumStep	   
			# damping for stepSize usually close to 1 former damp == dampings/cumStep

	# allocate the solution sampled
	if sample is None:
		sample = np.arange(nparams*batchSize, dtype=np.float64).reshape(batchSize,nparams)

	# Allocate space for current seeds
	if currSeeds is None:
		currSeeds = [0] * ntrials
		
	# initialize statitiscs
	stat = np.arange(0, dtype=np.float64)

	bestsol = None
	bestfit = -9999.0			# best fitness achieved so far
	ceval = 0				# current evaluation
	cgen = 0				# current generation
	start_time = time.time()

	# Initialize dynamic (internal) strategy parameters and constants
	covPath = zeros(nparams)
	stepPath = zeros(nparams)		# evolution paths for C and stepSize
	B = eye(nparams, nparams)		# B defines the coordinate system
	D = eye(nparams, nparams)		# diagonal matrix D defines the scaling
	C = dot(dot(B, D), dot(B, D).T)		# covariance matrix
	chiN = nparams ** 0.5 * (1 - 1. / (4. * nparams) + 1 / (21. * nparams ** 2))
						# expectation of ||numParameters(0,I)|| == norm(randn(numParameters,1))
	stepSize = 0.5

	# RandomState for perturbing the performed actions (used only for samples, not for centroid)
	rs = np.random.RandomState(seed)				

	print("CMA-ES: seed %d batchSize %d stepSize %lf nparams %d" % (seed, batchSize, stepSize, nparams))

	# main loop
	elapsed = 0
	while (ceval <= nevals):
		cgen = cgen + 1

		# Create the current seeds
		if fullyRandom:
			# Dummy seeds (not used)
			for t in range(ntrials):
				currSeeds[t] = t
		else:
			if cgen % envChangeEvery == 0:
				print("Environment changed")
				for t in range(ntrials):
					currSeeds[t] = rs.randint(1,10001)

		# Generate and evaluate lambda offspring
		arz = rs.randn(nparams, batchSize)
		arx = tile(center.reshape(nparams, 1), (1, batchSize))\
							+ stepSize * dot(dot(B, D), arz)
		fitness = zeros(batchSize)
		# Mean and standard deviation
		currMean = ob_stat.mean
		currStd = ob_stat.std
		policy.set_ob_stat(currMean, currStd)
		cbestfit = -999999999999.0
		for k in range(batchSize):
			for g in range(nparams):
				sample[k][g] = arx[g,k]
			# Store initial values of mean, std and count for current sample
			for t in range(ntrials):
				# Evaluation: noiseless weights and noiseless actions
				policy.set_trainable_flat(sample[k]) # Parameters must be updated by the algorithm!!
				currSeed = currSeeds[t]
				if fullyRandom:
					# We do not set <currSeed> to None since we want to replicate the evaluation in test phase!
					currSeed = rs.randint(1,10001)
				if rs.rand() < 0.01:
					eval_rews, eval_length, obs = policy.rollout(env, render=False, trial=t, seed=currSeed, save_obs=True, random_stream=rs)  # eval rollouts don't obey task_data.timestep_limit
					# Update observation mean, std and count for successive normalizations
					ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
				else:
					eval_rews, eval_length = policy.rollout(env, render=False, trial=t, seed=currSeed, random_stream=rs)  # eval rollouts don't obey task_data.timestep_limit
				ceval = ceval + eval_length
				cfit = eval_rews.sum()
				if cfit > cbestfit:
					cbestfit = cfit
					# We check if a new best fitness is found here!!!
					if cbestfit > bestfit:
						# Found a new best sample, store:
						# - fitness
						# - seed used for the evaluation
						# - sample
						# - seed
						# - mean and standard deviation
						bestfit = cbestfit
						bestsol = np.copy(sample[k])
						# Save best sample found
						fname = filedir + "/bestgS" + str(seed)
						np.save(fname, bestsol)
						# Save the policy too!!!
						fname = filedir + "/policyS" + str(seed) + ".h5"
						policy.save(fname)
						# Save the seed
						fname = filedir + "/bestSeedS" + str(seed) + ".txt"
						fp = open(fname, "w")
						fp.write("%d" % currSeed)
						fp.close()
						# Save mean and standard deviation
						fname = filedir + "/obMeanS" + str(seed)
						np.save(fname, currMean)
						fname = filedir + "/obStdS" + str(seed)
						np.save(fname, currStd)
				fitness[k] += cfit
			fitness[k] /= ntrials
			
		# Sort by fitness and compute weighted mean into center
		fitness, index = descendent_sort(fitness)
		arz = arz[:, index]
		arx = arx[:, index]
		arzsel = arz[:, range(mu)]
		arxsel = arx[:, range(mu)]
		arxmut = arxsel - tile(center.reshape(nparams, 1), (1, mu))

		zmean = dot(arzsel, weights)
		center = dot(arxsel, weights)

		# Cumulation: Update evolution paths
		stepPath = (1 - cumStep) * stepPath \
				+ sqrt(cumStep * (2 - cumStep) * muEff) * dot(B, zmean)		 # Eq. (4)
		hsig = norm(stepPath) / sqrt(1 - (1 - cumStep) ** (2 * ceval / float(batchSize))) / chiN \
				< 1.4 + 2. / (nparams + 1)
		covPath = (1 - cumCov) * covPath + hsig * \
				sqrt(cumCov * (2 - cumCov) * muEff) * dot(dot(B, D), zmean) # Eq. (2)

		# Adapt covariance matrix C
		C = ((1 - covLearningRate) * C					# regard old matrix   % Eq. (3)
				 + covLearningRate * (1 / muCov) * (outer(covPath, covPath) # plus rank one update
				 + (1 - hsig) * cumCov * (2 - cumCov) * C)
				 + covLearningRate * (1 - 1 / muCov)				 # plus rank mu update
				 * dot(dot(arxmut, diag(weights)), arxmut.T)
			)

		# Adapt step size
		stepSize *= exp(( cumStep /  dampings) * (norm( stepPath) /  chiN - 1)) # Eq. (5)

		# Update B and D from C
		# This is O(n^3). When strategy internal CPU-time is critical, the
		# next three lines should be executed only every (alpha/covLearningRate/N)-th
		# iteration, where alpha is e.g. between 0.1 and 10
		C = (C + C.T) / 2 # enforce symmetry
		Ev, B = eig(C)		  # eigen decomposition, B==normalized eigenvectors
		Ev = real(Ev)	   # enforce real value
		D = diag(sqrt(Ev))	  #diag(ravel(sqrt(Ev))) # D contains standard deviations now
		B = real(B)

		averagef = AverageFit(fitness)
		stat = np.append(stat,[ceval, bestfit, averagef, fitness[0]])
		elapsed = (time.time() - start_time)
		
		# We store the best sample either at the first generation, or every <storingRate> generations, or when the evolution ends
		if cgen == 1 or cgen % storingRate == 0 or ceval >= nevals:
			# Save the centroid
			fname = filedir + "/centroidS" + str(seed)
			np.save(fname, center)
			# Save stat file for fitness curve generation
			fname = filedir + "/statS" + str(seed)
			statsize = np.shape(stat)[0]
			statsize = statsize / 4
			statsize = int(statsize)
			stat.resize(statsize, 4)
			stat = stat.transpose()
			np.save(fname, stat)
																  
		print('Seed %d Gen %d Steps %d Bestfit %.2f bestsam %.2f Avg %.2f Elapsed %d' % (seed, cgen, ceval, bestfit, fitness[0], averagef, elapsed))

	end_time = time.time()
	print('Simulation took %d seconds' % (end_time - start_time))

	# Stat
	fname = filedir + "/S" + str(seed) + ".fit"
	fp = open(fname, "w")
	fp.write('Seed %d Gen %d Evaluat %d Bestfit %.2f bestoffspring %.2f Average %.2f Runtime %d\n' % (seed, cgen, ceval, bestfit, fitness[0], averagef, (time.time() - start_time)))
	fp.close() 
	# Stat file
	fname = filedir + "/statS" + str(seed)
	statsize = np.shape(stat)[0]
	statsize = statsize / 4
	statsize = int(statsize)
	stat.resize(statsize, 4)
	stat = stat.transpose()
	stat = stat.transpose()
	np.save(fname, stat)

# Evolve with ES algorithm taken from Salimans et al. (2017)
def evolve_ES(env, policy, ob_stat, seed, nevals, ntrials):
	global center
	global currSeeds
	global fullyRandom
	global stepsize
	global noiseStdDev
	global sampleSize
	global storingRate
	global biasCorr
	global numHiddens
	global filedir

	# initialize the solution center
	center = policy.get_trainable_flat()
	# Extract the number of parameters
	nparams = len(center)
	# setting parameters
	batchSize = sampleSize
	# Symmetric weights in the range [-0.5,0.5]
	weights = zeros(batchSize)

	# Allocate space for current seeds
	if currSeeds is None:
		currSeeds = [0] * ntrials
	
	# initialize statitiscs
	stat = np.arange(0, dtype=np.float64)

	bestsol = None
	bestfit = -9999.0                       # best fitness achieved so far
	ceval = 0			        # current evaluation
	cgen = 0				# current generation
	start_time = time.time()
	# Parameters for Adam policy
	m = zeros(nparams)
	v = zeros(nparams)
	epsilon = 1e-08 # To avoid numerical issues with division by zero...
	beta1 = 0.9
	beta2 = 0.999
	
	# RandomState for perturbing the performed actions (used only for samples, not for centroid)
	rs = np.random.RandomState(seed)

	print("ES: seed %d batchSize %d stepsize %lf noiseStdDev %lf nparams %d" % (seed, batchSize, stepsize, noiseStdDev, nparams))

	# main loop
	elapsed = 0
	while (ceval <= nevals):
		cgen = cgen + 1

		# Create the current seeds
		if fullyRandom:
			# Dummy seeds (not used)
			for t in range(ntrials):
				currSeeds[t] = t
		else:
			if cgen % envChangeEvery == 0:
				print("Environment changed")
				for t in range(ntrials):
					currSeeds[t] = rs.randint(1,10001)

		# Extract half samples from Gaussian distribution with mean 0.0 and standard deviation 1.0
		samples = rs.randn(batchSize, nparams)
		# We generate simmetric variations for the offspring
		symmSamples = zeros((batchSize * 2,nparams))
		for i in range(batchSize):
			sampleIdx = 2 * i
			for g in range(nparams):
				symmSamples[sampleIdx,g] = samples[i,g]
				symmSamples[sampleIdx + 1,g] = -samples[i,g]
		# Generate offspring
		z = tile(center.reshape(1,nparams),(batchSize * 2,1)) + noiseStdDev * symmSamples
		# Evaluate offspring
		fitness = zeros(batchSize * 2)
		# Mean and standard deviation
		currMean = ob_stat.mean
		currStd = ob_stat.std
		policy.set_ob_stat(currMean, currStd)
		cbestfit = -999999999999.0
		""" Append batch size new samples and evaluate them. """
		for k in range(batchSize * 2):
			# Store initial values of mean, std and count for current sample
			for t in range(ntrials):
				# Evaluation: noiseless weights and noiseless actions
				policy.set_trainable_flat(z[k]) # Parameters must be updated by the algorithm!!
				currSeed = currSeeds[t]
				if fullyRandom:
					# We do not set <currSeed> to None since we want to replicate the evaluation in test phase!
					currSeed = rs.randint(1,10001)
				if rs.rand() < 0.01:
					eval_rews, eval_length, obs = policy.rollout(env, render=False, trial=t, seed=currSeed, save_obs=True, random_stream=rs)  # eval rollouts don't obey task_data.timestep_limit
					# Update observation mean, std and count for successive normalizations
					ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
				else:
					eval_rews, eval_length = policy.rollout(env, render=False, trial=t, seed=currSeed, random_stream=rs)  # eval rollouts don't obey task_data.timestep_limit
				ceval = ceval + eval_length
				cfit = eval_rews.sum()
				if cfit > cbestfit:
					cbestfit = cfit
					# We check if a new best fitness is found here!!!
					if cbestfit > bestfit:
						# Found a new best sample, store:
						# - fitness
						# - sample
						# - seed
						# - mean and standard deviation
						bestfit = cbestfit
						bestsol = np.copy(z[k])
						# Save best sample found
						fname = filedir + "/bestgS" + str(seed)
						np.save(fname, bestsol)
						# Save the policy too!!!
						fname = filedir + "/policyS" + str(seed) + ".h5"
						policy.save(fname)
						# Save the seed
						fname = filedir + "/bestSeedS" + str(seed) + ".txt"
						fp = open(fname, "w")
						fp.write("%d" % currSeed)
						fp.close()
						# Save mean and standard deviation
						fname = filedir + "/obMeanS" + str(seed)
						np.save(fname, currMean)
						fname = filedir + "/obStdS" + str(seed)
						np.save(fname, currStd)
				fitness[k] += cfit
			fitness[k] /= ntrials
		
		# Sort by fitness and compute weighted mean into center
		fitness, index = ascendent_sort(fitness)
		# Now me must compute the symmetric weights in the range [-0.5,0.5]
		utilities = zeros(batchSize * 2)
		for i in range(batchSize * 2):
			utilities[index[i]] = i
		utilities /= (batchSize * 2 - 1)
		utilities -= 0.5
		# Now we assign the weights to the samples
		for i in range(batchSize):
			idx = 2 * i
			weights[i] = (utilities[idx] - utilities[idx + 1]) # pos - neg

		# Compute the gradient
		g = 0.0
		i = 0
		while i < batchSize:
			gsize = -1
			if batchSize - i < 500:
				gsize = batchSize - i
			else:
				gsize = 500
			g += dot(weights[i:i + gsize], samples[i:i + gsize,:]) # weights * samples
			i += gsize
		# Normalization over the number of samples
		g /= (batchSize * 2)
		# Global gradient takes into account the centroid, though only partially
		globalg = -g + 0.005 * center
		# ADAM policy
		# Compute how much the center moves
		a = stepsize
		if biasCorr:
			a *= sqrt(1.0 - beta2 ** cgen) / (1.0 - beta1 ** cgen)
		m = beta1 * m + (1.0 - beta1) * globalg
		v = beta2 * v + (1.0 - beta2) * (globalg * globalg)
		dCenter = -a * m / (sqrt(v) + epsilon)
		# update center
		center += dCenter

		# Compute average fitness of the offspring
		averagef = AverageFit(fitness)
		stat = np.append(stat, [ceval, bestfit, averagef, fitness[batchSize * 2 - 1]])

		# We store the best sample either at the first generation, or every <storingRate> generations, or when the evolution ends
		if cgen == 1 or cgen % storingRate == 0 or ceval >= nevals:
			# Save the centroid (overwrite old one)
			fname = filedir + "/centroidS" + str(seed)
			np.save(fname, center)
			# Save stat file for fitness curve generation
			fname = filedir + "/statS" + str(seed)
			statsize = np.shape(stat)[0]
			statsize = statsize / 4
			statsize = int(statsize)
			stat.resize(statsize, 4)
			stat = stat.transpose()
			np.save(fname, stat)

		# Compute the elapsed time (i.e., how much time the generation lasted)
		elapsed = (time.time() - start_time)

            	# Print information
		print('Seed %d gen %d steps %d bestfit %.2f bestsam %.2f Avg %.2f weightSize %.2f Elapsed %d' % (seed, cgen, ceval, bestfit, fitness[batchSize * 2 - 1], averagef, np.average(np.absolute(center)), elapsed))

	end_time = time.time()
	print('Simulation took %d seconds' % (end_time - start_time))

	# Stat
	fname = filedir + "/S" + str(seed) + ".fit"
	fp = open(fname, "w")
	fp.write('Seed %d gen %d eval %d bestfit %.2f bestoffspring %.2f average %.2f runtime %d\n' % (seed, cgen, ceval, bestfit, fitness[batchSize * 2 - 1], averagef, (time.time() - start_time)))
	fp.close()
	# Stat file
	fname = filedir + "/statS" + str(seed)
	statsize = np.shape(stat)[0]
	statsize = statsize / 4
	statsize = int(statsize)
	stat.resize(statsize, 4)
	stat = stat.transpose()
	np.save(fname, stat)

# Evolve with xNES algorithm
def evolve_xNES(env, policy, ob_stat, seed, nevals, ntrials):
	global center
	global sample
	global currSeeds
	global fullyRandom
	global storingRate
	global filedir

	# initialize the solution center
	center = policy.get_trainable_flat()
	# Extract the number of parameters
	nparams = len(center)
	# setting parameters
	centerLearningRate = 1.0
	covLearningRate = 0.5 * min(1.0 / nparams, 0.25) # from MATLAB # covLearningRate = 0.6*(3+log(ngenes))/ngenes/sqrt(ngenes)
	batchSize = int(4 + floor(3 * log(nparams)))
	mu = int(floor(batchSize / 2))          # number of parents/points for recombination
	weights = log(mu + 1) - log(array(range(1, mu + 1)))	  # use array
	weights /= sum(weights)	        # normalize recombination weights array

	# allocate the solution sampled
	if sample is None:
		sample = np.arange(nparams*batchSize, dtype=np.float64).reshape(batchSize,nparams)

	# initialize covariance and identity matrix
	_A = zeros((nparams,nparams))     # square root of covariance matrix
	_I = eye(nparams)		# Identity matrix

	# Allocate space for current seeds
	if currSeeds is None:
		currSeeds = [0] * ntrials
		
	# initialize statitiscs
	stat = np.arange(0, dtype=np.float64)

	bestsol = None
	bestfit = -999999999.0			# best fitness achieved so far
	bestseed = -1
	bestgen = 0	
	ceval = 0				# current evaluation
	cgen = 0				# current generation
	start_time = time.time()

	# RandomState for perturbing the performed actions (used only for samples, not for centroid)
	rs = np.random.RandomState(seed)	

	print("xNES: seed %d batchSize %d nparams %d" % (seed, batchSize, nparams))

	# main loop
	elapsed = 0
	while (ceval <= nevals):
		cgen = cgen + 1

		# Create the current seeds
		if fullyRandom:
			# Dummy seeds (not used)
			for t in range(ntrials):
				currSeeds[t] = t
		else:
			if cgen % envChangeEvery == 0:
				print("Environment changed")
				for t in range(ntrials):
					currSeeds[t] = rs.randint(1,10001)

		# Compute the exponential of the covariance matrix
		_expA = expm(_A)
		# Generate and evaluate lambda offspring
		samples = rs.randn(nparams, batchSize)
		z = tile(center.reshape(nparams, 1), (1, batchSize)) + _expA.dot(samples)
		fitness = zeros(batchSize)
		# Mean and standard deviation
		currMean = ob_stat.mean
		currStd = ob_stat.std
		policy.set_ob_stat(currMean, currStd)
		cbestfit = -999999999999.0
		for k in range(batchSize):
			for g in range(nparams):
				sample[k][g] = z[g,k]
			# Store initial values of mean, std and count for current sample
			for t in range(ntrials):
				# Evaluation: noiseless weights and noiseless actions
				policy.set_trainable_flat(sample[k]) # Parameters must be updated by the algorithm!!
				currSeed = currSeeds[t]
				if fullyRandom:
					# We do not set <currSeed> to None since we want to replicate the evaluation in test phase!
					currSeed = rs.randint(1,10001)
				if rs.rand() < 0.01:
					eval_rews, eval_length, obs = policy.rollout(env, render=False, trial=t, seed=currSeed, save_obs=True, random_stream=rs)  # eval rollouts don't obey task_data.timestep_limit
					# Update observation mean, std and count for successive normalizations
					ob_stat.increment(obs.sum(axis=0), np.square(obs).sum(axis=0), len(obs))
				else:
					eval_rews, eval_length = policy.rollout(env, render=False, trial=t, seed=currSeed, random_stream=rs)  # eval rollouts don't obey task_data.timestep_limit
				cfit = eval_rews.sum()
				if cfit > cbestfit:
					cbestfit = cfit
					# We check if a new best fitness is found here!!!
					if cbestfit > bestfit:
						# Found a new best sample, store:
						# - fitness
						# - seed used for the evaluation
						# - sample
						# - seed
						bestfit = cbestfit
						bestsol = np.copy(z[:,k])
						# Save best sample found
						fname = filedir + "/bestgS" + str(seed)
						np.save(fname, bestsol)
						# Save the policy too!!!
						fname = filedir + "/policyS" + str(seed) + ".h5"
						policy.save(fname)
						# Save the seed
						fname = filedir + "/bestSeedS" + str(seed) + ".txt"
						fp = open(fname, "w")
						fp.write("%d" % currSeed)
						fp.close()
						# Save mean and standard deviation
						fname = filedir + "/obMeanS" + str(seed)
						np.save(fname, currMean)
						fname = filedir + "/obStdS" + str(seed)
						np.save(fname, currStd)
				fitness[k] += cfit
				ceval = ceval + eval_length
			fitness[k] /= ntrials
		
		# Sort by fitness and compute weighted mean into center
		fitness, index = descendent_sort(fitness)
		# Utilities
		utilities = zeros(batchSize)
		uT = zeros((batchSize,1))
		for i in range(mu):
			utilities[index[i]] = weights[i]
			uT[index[i],0] = weights[i]
			
		# Compute gradients
		U = zeros((nparams,batchSize))
		for i in range(nparams):
			for j in range(batchSize):
				U[i][j] = utilities[j]

		us = zeros((nparams,batchSize))
		for i in range(nparams):
			for j in range(batchSize):
				us[i][j] = U[i][j] * samples[i][j]
		G = us.dot(samples.transpose()) - sum(utilities) * _I
		dCenter = centerLearningRate * _expA.dot(samples.dot(uT))
		deltaCenter = zeros(nparams)
		for g in range(nparams):
			deltaCenter[g] = dCenter[g,0]
		dA = covLearningRate * G

		# Update
		center += deltaCenter
		_A += dA

		averagef = AverageFit(fitness)
		stat = np.append(stat,[ceval, bestfit, averagef, fitness[0]])
		elapsed = (time.time() - start_time)
		
		# We store the best sample either at the first generation, or every <storingRate> generations, or when the evolution ends
		if cgen == 1 or cgen % storingRate == 0 or ceval >= nevals:
			# Save the centroid of the current generation
			fname = filedir + "/centroidS" + str(seed)
			np.save(fname, center)
			# Save stat file for fitness curve generation
			fname = filedir + "/statS" + str(seed)
			statsize = np.shape(stat)[0]
			statsize = statsize / 4
			statsize = int(statsize)
			stat.resize(statsize, 4)
			stat = stat.transpose()
			np.save(fname, stat)
																  
		print('Seed %d Gen %d Steps %d Bestfit %.2f bestsam %.2f Avg %.2f Elapsed %d' % (seed, cgen, ceval, bestfit, fitness[0], averagef, elapsed))

	end_time = time.time()
	print('Simulation took %d seconds' % (end_time - start_time))

	# Stat
	fname = filedir + "/S" + str(seed) + ".fit"
	fp = open(fname, "w")
	fp.write('Seed %d Gen %d Evaluat %d Bestfit %.2f bestoffspring %.2f Average %.2f Runtime %d\n' % (seed, cgen, ceval, bestfit, fitness[0], averagef, (time.time() - start_time)))
	fp.close() 
	fname = filedir + "/statS" + str(seed)
	statsize = np.shape(stat)[0]
	statsize = statsize / 4
	statsize = int(statsize)
	stat.resize(statsize, 4)
	stat = stat.transpose()
	np.save(fname, stat)
	

def evolve(env, policy, ob_stat, seed, nevals, ntrials):
	global id_algo

	# Call the evolve method associated to the algorithm whose identifier is <id_algo>
	if id_algo == 0:
		evolve_xNES(env, policy, ob_stat, seed, nevals, ntrials)
	elif id_algo == 1:
		evolve_CMAES(env, policy, ob_stat, seed, nevals, ntrials)
	elif id_algo == 2:
		evolve_ES(env, policy, ob_stat, seed, nevals, ntrials)
	else:
		# Invalid identifier!!!
		print("Invalid <id_algo> %d... Abort!!!" % id_algo)
		sys.exit(-1)

# Test evolved individual
def test(env, policy, ob_stat, seed, ntrials, centroidTest):
	global filedir
	# Load best individual
	if centroidTest:
		fname = filedir + "/centroidS" + str(seed) + ".npy"
		# Centroid must be tested on random trials (we overwrite <ntrials> parameter)
		ntrials = 10
	else:
		fname = filedir + "/bestgS" + str(seed) + ".npy"
	bestgeno = np.load(fname)
	# Load policy
	fname = filedir + "/policyS" + str(seed) + ".h5"
	policy.initialize_from(fname, ob_stat)
	# Load mean and standard deviation (to avoid issues with stored policy...)
	fname = filedir + "/obMeanS" + str(seed) + ".npy"
	obMean = np.load(fname)
	fname = filedir + "/obStdS" + str(seed) + ".npy"
	obStd = np.load(fname)
	fit = 0.0
	# Test the loaded individual
	policy.set_trainable_flat(bestgeno) # Parameters must be updated by the algorithm!!
	policy.set_ob_stat(ob_stat.mean, ob_stat.std)
	if ntrials == 1:
		# Load the seed
		fname = filedir + "/bestSeedS" + str(seed) + ".txt"
		cseed = np.loadtxt(fname)
		cseed = int(cseed)
		print("testing seed %d" % cseed)
		env.render()
		eval_rews, eval_length = policy.rollout(env, render=True, timestep_limit=1000, trial=0, seed=cseed)  # eval rollouts don't obey task_data.timestep_limit
		fit = eval_rews.sum()
		#env.close()
	else:
		seeds = np.arange(ntrials)
		if not centroidTest:
			seed = int(seed)
			np.random.seed(seed)
			# Load the seed
			fname = filedir + "/bestSeedS" + str(seed) + ".txt"
			cseed = np.loadtxt(fname)
			cseed = int(cseed)
			seeds[0] = cseed
			i = 1
			while i < ntrials:
				seeds[i] = np.random.randint(1,10001)
				i += 1
		for t in range(ntrials):
			cseed = int(seeds[t])
			env.render()
			eval_rews, eval_length = policy.rollout(env, render=True, timestep_limit=1000, trial=t, seed=cseed)  # eval rollouts don't obey task_data.timestep_limit
			eval_return = eval_rews.sum()
			print("trial %d - fit %lf" % (t, eval_return))
			fit += eval_return
			#env.close()
		fit /= ntrials
	print("total fitness %lf" % fit)

# Parsing configuration file
# read parameters from the configuration file
def parseConfigFile(filename):
	global id_algo
	global ntrials
	global nevals
	global envChangeEvery
	global numHiddens
	global environment
	global fullyRandom
	global storingRate
	global biasCorr
	global numHiddenLayers
	global stepsize
	global noiseStdDev
	global sampleSize
	global ac_bins
	global ac_noise_std
	global connection_type
	global hidden_dims
	global nonlin_type
	global nonlin_out
	global init_type
	global out_type

	# The configuration file must have the following sections:
	# [DEFAULT]: contains the default setting for the other sections
	# [EVAL]: parameters for the algorithm
	# [POLICY]: parameters of the policy

	config = configparser.ConfigParser()
	config.read(filename)
    
	# Section EVAL
	if (config.has_option("EVAL", "id_algo")):
		id_algo = config.getint("EVAL","id_algo")
	if (config.has_option("EVAL", "nevals")):
		nevals = config.getint("EVAL","nevals")
	if (config.has_option("EVAL", "ntrials")):
		ntrials = config.getint("EVAL","ntrials")
	if (config.has_option("EVAL", "envChangeEvery")):
		envChangeEvery = config.getint("EVAL","envChangeEvery")
	if (config.has_option("EVAL", "storingRate")):
		storingRate = config.getint("EVAL","storingRate")
	if (config.has_option("EVAL", "biasCorr")):
		biasCorr = config.getboolean("EVAL","biasCorr")
	if (config.has_option("EVAL", "numHiddens")):
		numHiddens = config.getint("EVAL","numHiddens")
	if (config.has_option("EVAL", "environment")):
		environment = config.get("EVAL","environment")
	if (config.has_option("EVAL", "fullyRandom")):
		fullyRandom = config.getboolean("EVAL","fullyRandom")
	if (config.has_option("EVAL", "numHiddenLayers")):
		numHiddenLayers = config.getint("EVAL","numHiddenLayers")
	if (config.has_option("EVAL", "stepsize")):
		stepsize = config.getfloat("EVAL","stepsize")
	if (config.has_option("EVAL", "noiseStdDev")):
		noiseStdDev = config.getfloat("EVAL","noiseStdDev")
	if (config.has_option("EVAL", "sampleSize")):
		sampleSize = config.getint("EVAL","sampleSize")
	# Section POLICY
	"""
	if (config.has_option("POLICY", "ac_bins")):
		ac_bins = config.get("POLICY","ac_bins")
	"""
	if (config.has_option("POLICY", "out_type")):
		out_type = config.getint("POLICY","out_type")
	if (config.has_option("POLICY", "ac_noise_std")):
		ac_noise_std = config.getfloat("POLICY","ac_noise_std")
	if (config.has_option("POLICY", "connection_type")):
		connection_type = config.get("POLICY","connection_type")
	if (config.has_option("POLICY", "nonlin_type")):
		nonlin_type = config.get("POLICY","nonlin_type")
	if (config.has_option("POLICY", "nonlin_out")):
		nonlin_out = config.getboolean("POLICY","nonlin_out")
	if (config.has_option("POLICY", "init_type")):
		init_type = config.get("POLICY","init_type")
	# Convert out_type into corresponding ac_bins
	if out_type == 0:
		ac_bins = "continuous:"
	elif out_type == 1:
		ac_bins = "binary:"
	elif out_type > 1:
		ac_bins = "uniform:" + str(out_type)
	else:
		print("Invalid out_type %d" % out_type)
		sys.exit(-1)
	# Hidden dimensions
	if environment == "hu":
		environment = 'Humanoid-v2'
		hidden_dims = []
		if numHiddenLayers == 1:
			hidden_dims = [ numHiddens ]
		else:
			for layer in range(numHiddenLayers):
				hidden_dims.append(numHiddens)
	elif environment == "2d":
		environment = 'Walker2d-v2'
		hidden_dims = []
		if numHiddenLayers == 1:
			hidden_dims = [ numHiddens ]
		else:
			for layer in range(numHiddenLayers):
				hidden_dims.append(numHiddens)
	elif environment == "sw":
		environment = 'Swimmer-v2'
		hidden_dims = []
		if numHiddenLayers == 1:
			hidden_dims = [ numHiddens ]
		else:
			for layer in range(numHiddenLayers):
				hidden_dims.append(numHiddens)
	elif environment == "ho":
		environment = 'Hopper-v2'
		hidden_dims = []
		if numHiddenLayers == 1:
			hidden_dims = [ numHiddens ]
		else:
			for layer in range(numHiddenLayers):
				hidden_dims.append(numHiddens)
	elif environment == "ip":
		environment = 'InvertedDoublePendulum-v2'
		hidden_dims = []
		if numHiddenLayers == 1:
			hidden_dims = [ numHiddens ]
		else:
			for layer in range(numHiddenLayers):
				hidden_dims.append(numHiddens)
	elif environment == "bw":
		environment = 'BipedalWalker-v2'
		hidden_dims = []
		if numHiddenLayers == 1:
			hidden_dims = [ numHiddens ]
		else:
			for layer in range(numHiddenLayers):
				hidden_dims.append(numHiddens)
	elif environment == "hc":
		environment = 'HalfCheetah-v2'
		hidden_dims = []
		if numHiddenLayers == 1:
			hidden_dims = [ numHiddens ]
		else:
			for layer in range(numHiddenLayers):
				hidden_dims.append(numHiddens)
	elif environment == "hme":
		environment = 'HandManipulateEgg-v0'
		hidden_dims = []
		if numHiddenLayers == 1:
			hidden_dims = [ numHiddens ]
		else:
			for layer in range(numHiddenLayers):
				hidden_dims.append(numHiddens)
	else:
		# Default environment is cart-pole
		print("Invalid environment %s --> cart-pole is going to be run" % environment)
		environment = 'CartPole-v0'
		hidden_dims = [ numHiddens ]

# Main code
def main(argv):
	global ac_bins
	global ac_noise_std
	global connection_type
	global hidden_dims
	global nonlin_type
	global nonlin_out
	global init_type
	global id_algo
	global ntrials
	global nevals
	global numHiddens
	global numHiddenLayers
	global environment
	global fullyRandom
	global envChangeEvery
	global storingRate
	global biasCorr
	global stepsize
	global noiseStdDev
	global sampleSize
	global filedir
	global obSpace
	global acSpace

	# Processing command line argument
	argc = len(argv)

	# Parameters:
	# - seed (default is 1);
	# - number of replications (default is 1);
	# - configuration file;
	# - test option
	# - generation to be tested
	filename = None
	cseed = 1
	nreplications = 1
	doTest = False
	centroidTest = False
	filedir = None
	
	i = 1
	while (i < argc):
		if (argv[i] == "-f"):
			i += 1
			if (i < argc):
				filename = argv[i]
				i += 1
		elif (argv[i] == "-s"):
			i += 1
			if (i < argc):
				cseed = int(argv[i])
				i += 1
		elif (argv[i] == "-n"):
			i += 1
			if (i < argc):
				nreplications = int(argv[i])
				i += 1
		elif (argv[i] == "-t"):
			doTest = True
			i += 1
		elif (argv[i] == "-c"):
			if doTest:
				centroidTest = True
			i += 1
		elif (argv[i] == "-d"):
			i += 1
			if (i < argc):
				filedir = argv[i]
				i += 1
		else:
			# We simply ignore the argument
			print("Invalid argument %s" % argv[i])
			i += 1

	if filename is None:
		# Warning if configuration file is not passed
		print("WARNING - configuration file not specified, we use the default parameters!!!")
		# Set default
		cseed = 1
		nreplications = 1
		filedir = scriptdirname
		nevals = 1000000
		ntrials = 1
		id_algo = 2 # ES algorithm
		envChangeEvery = 1
		numHiddens = 0
		numHiddenLayers = 0
		environment = "CartPole-v0" # Default task is the simple cart-pole
		fullyRandom = False
		storingRate = 1000
		biasCorr = True
		stepsize = 0.01
		noiseStdDev = 0.02
		sampleSize = 20
		# Default policy
		ac_bins = "continuous:"
		ac_noise_std = 0.01
		connection_type = "ff"
		hidden_dims = [ numHiddens ]
		nonlin_type = "tanh"
		nonlin_out = False
		init_type = "normc"
		print("Default settings: id_algo %d nevals %d ntrials %d numHiddens %d environment %s ac_bins %s ac_noise_std %lf connection_type %s nonlin_type %s nonlin_out %d init_type %s" % (id_algo, nevals, ntrials, numHiddens, environment, ac_bins, ac_noise_std, connection_type, nonlin_type, nonlin_out, init_type))
	else:
		# Parse configuration file
		parseConfigFile(filename)
	if filedir is None:
		# Default directory is that of the .py file
		filedir = scriptdirname

	# Get the environment
	env = gym.make(environment)
	obSpace = str(type(env.observation_space))
	acSpace = str(type(env.action_space))
	currObSpace = None
	# Parsing ob_space and action_space
	if "Dict" in obSpace:
		# Dict
		currObSpace = env.observation_space.spaces['observation']
	elif "Box" in obSpace:
		# Box
		currObSpace = env.observation_space
	else:
		print("Invalid obSpace " + str(obSpace))
		sys.exit(-1)
	# Consistency check
	if "Discrete" in acSpace and "uniform" in ac_bins:
		raise "Cannot use discrete action space with uniform output!!!"
	# Create a new session
	session = make_session(single_threaded=True)
	# Create policy
	policy = Policy(currObSpace, env.action_space, ac_bins, ac_noise_std, nonlin_type, hidden_dims, connection_type, nonlin_out, init_type)
	# Initialize
	initialize()
	ob_stat = RunningStat(
				currObSpace.shape,
				eps=1e-2  # eps to prevent dividing by zero at the beginning when computing mean/stdev
			     )

	# Print the network structure
	ninputs = currObSpace.shape[0]
	noutputs = None
	if "Box" in acSpace:
		# Box
		noutputs = env.action_space.shape[0]
	elif "Discrete" in acSpace:
		# Discrete
		noutputs = env.action_space.n
	else:
		print("Invalid acSpace " + str(acSpace))
		sys.exit(-1)
	oidx = ac_bins.index(':')
	output_type = ac_bins[:oidx]
	print("ANN:")
	if connection_type == "ff":
		print("- feedforward")
	else:
		print("- fully-recurrent")
	print("- number of layers: %d" % numHiddenLayers)
	print("- number of hidden per layer: %d" % numHiddens)
	print("ninputs: %d" % ninputs)
	print("nhiddens: %d" % (numHiddens * numHiddenLayers))
	print("noutputs: %d" % noutputs)
	print("activation function: %s" % nonlin_type)
	print("output type: %s" % output_type)
	if "uniform" in ac_bins:
		nbins = ac_bins[(oidx+1):]
		print("number of bins: %s" % nbins)

	if not doTest:
		# Evolution
		print("evolution: env %s nreplications %d id_algo %d nevals %d ntrials %d numHiddens %d" % (environment, nreplications, id_algo, nevals, ntrials, numHiddens))
		evolve(env, policy, ob_stat, cseed, nevals, ntrials)
	else:
		# Test
		print("test: env %s id_algo %d ntrials %d test centroid %d" % (environment, id_algo, ntrials, centroidTest))
		test(env, policy, ob_stat, cseed, ntrials, centroidTest)
	# Close environment
	#env.close()
	# Close session
	session.close()

if __name__ == "__main__":
	main(sys.argv)
