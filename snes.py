#!/usr/bin/python

# Libraries to be imported
import gym
from gym import spaces
import numpy as np
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
import net
from policy import ErPolicy, GymPolicy
from evoalgo import EvoAlgo
from utils import ascendent_sort

# Evolve with sNES algorithm (Wierstra, Schaul, Peters and Schmidhuber, 2008)
class sNES(EvoAlgo):
    def __init__(self, env, policy, seed, filedir):
        EvoAlgo.__init__(self, env, policy, seed, filedir)

    def run(self, maxsteps):

        start_time = time.time()

        # initialize the solution center
        self.center = self.policy.get_trainable_flat()
        
        # Extract the number of parameters
        nparams = self.policy.nparams
        # setting parameters
        centerLearningRate = 1.0
        covLearningRate = 0.6 * (3 + log(nparams)) / 3.0 / sqrt(nparams)
        if self.batchSize == 0:
            # Use default value: 4 + floor(3 * log(N)), where N is the number of parameters
            self.batchSize = int(4 + floor(3 * log(nparams))) # population size, offspring number
            if "Tf" in type(self.policy).__name__:
                # Update the number of rollout calls in policy
                self.policy.updaten(self.batchSize)
        initVar = 1.0
        mu = int(floor(self.batchSize / 2))                       # number of parents/points for recombination
        self.stepsize = 1.0 / mu
        weights = zeros(self.batchSize)
        w = self.stepsize
        for i in range(mu):
            weights[self.batchSize - mu + i] = w
            w += self.stepsize
        weights /= sum(weights)	                                  # normalize recombination weights array
        # initialize variance array
        _sigmas = ones(nparams) * initVar

        ceval = 0               # current evaluation
        cgen = 0                # current generation
    
        # RandomState for perturbing the performed actions (used only for samples, not for centroid)
        np.random.seed(self.seed)

        print("sNES: seed %d maxmsteps %d batchSize %d stepsize %.2f sameEnvCond %d nparams %d" % (self.seed, maxsteps / 1000000, self.batchSize, self.stepsize, self.sameenvcond, nparams))

        # Set evolution mode
        self.policy.runEvo()

        # main loop
        elapsed = 0
        while ceval < maxsteps:
            cgen += 1

            # Extract half samples from Gaussian distribution with mean 0.0 and standard deviation 1.0
            samples = np.random.randn(self.batchSize, nparams)
            S = samples.transpose()
            # Generate offspring
            offspring = tile(self.center.reshape(1, nparams), (self.batchSize, 1)) + tile(_sigmas.reshape(1, nparams), (self.batchSize, 1)) * samples
            # Evaluate offspring
            fitness = zeros(self.batchSize)
            # If normalize=1 we update the normalization vectors
            if self.policy.normalize == 1:
                self.policy.nn.updateNormalizationVectors()
            # Reset environmental seed every generation
            self.policy.setSeed(self.policy.get_seed + cgen)
            # Set generalization flag to False
            self.policy.doGeneralization(False)
            # Evaluate offspring
            for k in range(self.batchSize):
                # Set policy parameters (corresponding to the current offspring)
                self.policy.set_trainable_flat(offspring[k])
                # Sample of the same generation experience the same environmental conditions
                if self.sameenvcond == 1:
                    self.policy.setSeed(self.policy.get_seed + cgen)
                # Evaluate the offspring
                eval_rews, eval_length = self.policy.rollout(timestep_limit=1000)
                # Get the fitness
                fitness[k] = eval_rews
                # Update the number of evaluations
                ceval += eval_length
                # Update data if the current offspring is better than current best
                self.updateBest(fitness[k], offspring[k])

            # Sort by fitness and compute weighted mean into center
            fitness, index = ascendent_sort(fitness)
            S = S[:, index]

            # Update center
            dCenter = dot(weights, S.transpose())
            self.center += dCenter

            # Update variances
            Ssq = S * S
            SsqMinusOne = Ssq - ones((nparams, self.batchSize))
            covGrad = dot(weights, SsqMinusOne.transpose())
            dSigma = 0.5 * covLearningRate * covGrad
            _sigmas = _sigmas * exp(dSigma).transpose()

            centroidfit = -999999999.0
            if self.evalCenter != 0:
                # Evaluate the centroid
                self.policy.set_trainable_flat(self.center)
                if self.sameenvcond == 1:
                    self.policy.setSeed(self.policy.get_seed + cgen)
                eval_rews, eval_length = self.policy.rollout(timestep_limit=1000)
                centroidfit = eval_rews
                ceval += eval_length
                # Update data if the centroid is better than current best
                self.updateBest(centroidfit, self.center)

            # Now perform generalization
            if self.policy.generalize:
                candidate = None
                if centroidfit > fitness[self.batchSize - 1]:
                    # Centroid undergoes generalization test
                    candidate = np.copy(self.center)
                else:
                    # Best sample undergoes generalization test
                    bestsamid = index[self.batchSize - 1]
                    candidate = np.copy(offspring[bestsamid])
                # Set the seed
                self.policy.set_trainable_flat(candidate) # Parameters must be updated by the algorithm!!
                self.policy.setSeed(self.policy.get_seed + 1000000)
                self.policy.doGeneralization(True)
                eval_rews, eval_length = self.policy.rollout(timestep_limit=1000)
                gfit = eval_rews
                ceval += eval_length
                # Update data if the candidate is better than current best generalizing individual
                self.updateBestg(gfit, candidate)

            # Compute the elapsed time (i.e., how much time the generation lasted)
            elapsed = (time.time() - start_time)

            # Update information
            self.updateInfo(cgen, ceval, fitness, self.center, centroidfit, fitness[self.batchSize - 1], elapsed, maxsteps)

        # save data
        self.save(cgen, ceval, centroidfit, self.center, fitness[self.batchSize - 1], (time.time() - start_time))

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

