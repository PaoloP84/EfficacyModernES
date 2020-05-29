#!/usr/bin/python

"""
This class implements the policy.

"""
import numpy as np
import net
import configparser
import time
import renderWorld
import sys

class Policy(object):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed):
        # Copy environment
        self.environment = None
        self.env = env
        self.seed = seed
        self.ninputs = ninputs 
        self.noutputs = noutputs
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
        self.displayneurons = 0 # Gym policies can display or the robot or the neurons activations
        # Read configuration file
        self.readConfig(filename)
        # Initialize the neural network
        self.nn = net.PyEvonet(self.ninputs, (self.nhiddens * self.nlayers), self.noutputs, self.nlayers, self.bias, self.architecture, self.afunction, self.out_type, self.winit, self.clip, self.normalize, self.action_noise, self.wrange, self.nbins, low, high)
        # Initialize policy parameters
        self.nparams = self.nn.computeParameters()
        self.params = np.arange(self.nparams, dtype=np.float64)
        # allocate neuron activation vector
        self.nact = np.arange((self.ninputs + (self.nhiddens * self.nlayers) + self.noutputs), dtype=np.float64)                             
        # Initialize normalization vector
        if self.normalize == 1:
            self.normvector = np.arange((self.ninputs * 2), dtype=np.float64)
        else:
            self.normvector = None
        # Allocate space for observation and action
        self.ob = ob
        self.ac = ac
        # Copy pointers
        self.nn.copyGenotype(self.params)
        self.nn.copyInput(self.ob)
        self.nn.copyOutput(self.ac)
        self.nn.copyNeuronact(self.nact)
        if self.normalize == 1:
            self.nn.copyNormalization(self.normvector)
        # Initialize weights
        self.nn.seed(self.seed)
        self.nn.initWeights()
        print("Policy: ntrials %d nttrials %d maxsteps %d normalize %d clip %d action_noise %d" % (self.ntrials, self.nttrials, self.maxsteps, self.normalize, self.clip, self.action_noise))

    def reset(self):
        self.nn.seed(self.seed)
        self.nn.initWeights()
        if self.normalize == 1:
            self.nn.resetNormalizationVectors()

    def setSeed(self, seed):
        self.env.seed(seed)
        self.nn.seed(seed)

    # === Rollouts/training ===
    # virtual function, implemented in sub-classes
    def rollout(self, render=False, timestep_limit=None):
        raise NotImplementedError

    def set_trainable_flat(self, x):
        self.params = np.copy(x)
        self.nn.copyGenotype(self.params)

    def get_trainable_flat(self):
        return self.params

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

class GymPolicy(Policy):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed):
        Policy.__init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed)
        # we allocate the vector containing the objects to be rendered by the 2D-Renderer
        self.objs = np.arange(10, dtype=np.float64) # DEBUG SIZE TO BE FIXED
        self.objs[0] = -1                           # to indicate that as default the list contains no objects
    
    # === Rollouts/training ===
    def rollout(self, render=False, timestep_limit=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        env_timestep_limit = self.env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
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
            self.nn.normPhase(0)
            # Observations must be saved if and only if normalization
            # flag is set to True and we are not in test phase
            if self.normalize == 1 and not self.test:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    # Save observations
                    self.nn.normPhase(1)
            # Reset environment
            self.ob = self.env.reset()
            # Reset network
            self.nn.resetNet()
            # Reward for current trial
            crew = 0.0
            # Perform the steps
            t = 0
            while t < timestep_limit:
                # Copy the input pointer to the network
                self.nn.copyInput(self.ob)
                # Activate network
                self.nn.updateNet()
                # Perform a step
                self.ob, rew, done, _ = self.env.step(self.ac)
                # Append the reward
                crew += rew
                t += 1
                if render:
                    if  self.displayneurons == 0:
                        self.env.render(mode="human")
                        time.sleep(0.05)
                    else:
                        info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, rew, rews)
                        renderWorld.update(self.objs, info, self.ob, self.ac, self.nact[self.ninputs:len(self.nact)-self.noutputs])
                if done:
                    break
            # Print fitness for each trial during test phase
            if self.test:
                print("Trial %d - fitness %lf" % (trial, crew))
            # Update overall reward
            rews += crew
            # Update steps
            steps += t
        # Normalize reward by the number of trials
        rews /= ntrials
        return rews, steps

class ErPolicy(Policy):
    def __init__(self, env, ninputs, noutputs, low, high, ob, ac, done, filename, seed):
        Policy.__init__(self, env, ninputs, noutputs, low, high, ob, ac, filename, seed)
        self.done = done
        # we allocate the vector containing the objects to be rendered by the 2D-Renderer
        self.objs = np.arange(1000, dtype=np.float64) # DEBUG SIZE TO BE FIXED
        self.objs[0] = -1                             # to indicate that as default the list contains no objects
        self.env.copyDobj(self.objs)
    
    # === Rollouts/training ===
    def rollout(self, render=False, timestep_limit=None):
        """
        If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
        Otherwise, no action noise will be added.
        """
        rews = 0.0
        steps = 0
        # Set the number of trials depending on whether or not test flag is set to True
        ntrials = self.ntrials
        if self.genTest:
            ntrials = self.nttrials
        # Loop over the number of trials
        for trial in range(ntrials):
            self.nn.normPhase(0)
            # Observations must be saved if and only if normalization
            # flag is set to True and we are not in test phase
            if self.normalize == 1 and not self.test:
                if np.random.uniform(low=0.0, high=1.0) < 0.01:
                    # Save observations
                    self.nn.normPhase(1)
            # Reset environment
            self.env.reset()
            # Reset network
            self.nn.resetNet()
            # Reward for current trial
            crew = 0.0
            # Perform the steps
            t = 0
            while t < self.maxsteps:
                # Activate network
                self.nn.updateNet()
                # Perform a step
                rew = self.env.step()
                # Append the reward
                crew += rew
                t += 1
                if render:
                    self.env.render()
                    info = 'Trial %d Step %d Fit %.2f %.2f' % (trial, t, rew, rews)
                    renderWorld.update(self.objs, info, self.ob, self.ac, self.nact[self.ninputs:len(self.nact)-self.noutputs])
                if self.done:
                    break
            # Print fitness for each trial during test phase
            if self.test:
                print("Trial %d - fitness %lf" % (trial, crew))
            # Update overall reward
            rews += crew
            # Update steps
            steps += t
        # Normalize reward by the number of trials
        rews /= ntrials
        return rews, steps
        
