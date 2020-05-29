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
from utils import descendent_sort

# Evolve with xNES algorithm (Wierstra, Schaul, Peters and Schmidhuber, 2008)
class xNES(EvoAlgo):
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
        covLearningRate = 0.5 * min(1.0 / nparams, 0.25)          # from MATLAB # covLearningRate = 0.6*(3+log(ngenes))/ngenes/sqrt(ngenes)
        if self.batchSize == 0:
            # Use default value: 4 + floor(3 * log(N)), where N is the number of parameters
            self.batchSize = int(4 + floor(3 * log(nparams))) # population size, offspring number
            if "Tf" in type(self.policy).__name__:
                # Update the number of rollout calls in policy
                self.policy.updaten(self.batchSize)
        mu = int(floor(self.batchSize / 2))                       # number of parents/points for recombination
        weights = log(mu + 1) - log(array(range(1, mu + 1)))	  # use array
        weights /= sum(weights)	                                  # normalize recombination weights array
        # initialize covariance and identity matrix
        _A = zeros((nparams,nparams))                             # square root of covariance matrix
        _I = eye(nparams)		                          # Identity matrix

        ceval = 0               # current evaluation
        cgen = 0                # current generation
    
        # RandomState for perturbing the performed actions (used only for samples, not for centroid)
        np.random.seed(self.seed)

        print("xNES: seed %d maxmsteps %d batchSize %d sameEnvCond %d nparams %d" % (self.seed, maxsteps / 1000000, self.batchSize, self.sameenvcond, nparams))

        # Set evolution mode
        self.policy.runEvo()

        # main loop
        elapsed = 0
        while ceval < maxsteps:
            cgen += 1

            # Compute the exponential of the covariance matrix
            _expA = expm(_A)
            # Extract half samples from Gaussian distribution with mean 0.0 and standard deviation 1.0
            samples = np.random.randn(nparams, self.batchSize)
            # Generate offspring
            offspring = tile(self.center.reshape(nparams, 1), (1, self.batchSize)) + _expA.dot(samples)
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
                self.policy.set_trainable_flat(offspring[:,k])
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
                self.updateBest(fitness[k], offspring[:,k])

            # Sort by fitness and compute weighted mean into center
            fitness, index = descendent_sort(fitness)
            # Utilities
            utilities = zeros(self.batchSize)
            uT = zeros((self.batchSize,1))
            for i in range(mu):
                utilities[index[i]] = weights[i]
                uT[index[i],0] = weights[i]
			
            # Compute gradients
            U = zeros((nparams,self.batchSize))
            for i in range(nparams):
                for j in range(self.batchSize):
                    U[i][j] = utilities[j]

            us = zeros((nparams,self.batchSize))
            for i in range(nparams):
                for j in range(self.batchSize):
                    us[i][j] = U[i][j] * samples[i][j]
            G = us.dot(samples.transpose()) - sum(utilities) * _I
            dCenter = centerLearningRate * _expA.dot(samples.dot(uT))
            deltaCenter = zeros(nparams)
            for g in range(nparams):
                deltaCenter[g] = dCenter[g,0]
            dA = covLearningRate * G

            # Update
            self.center += deltaCenter
            _A += dA

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
                if centroidfit > fitness[0]:
                    # Centroid undergoes generalization test
                    candidate = np.copy(self.center)
                else:
                    # Best sample undergoes generalization test
                    bestsamid = index[0]
                    candidate = np.copy(offspring[:,bestsamid])
                # Set the seed
                self.policy.set_trainable_flat(candidate) # Parameters must be updated by the algorithm!!
                self.policy.setSeed(self.policy.get_seed + 1000000)
                self.policy.doGeneralization(True)
                eval_rews, eval_length = self.policy.rollout(timestep_limit=1000)
                gfit = eval_rews
                ceval += eval_length
                # Update data if the candidate is better than current best generalizing individual
                self.updateBestg(gfit, candidate)

            # Compute the average value in the covariance matrix
            covSize = 0.0
            for g in range(nparams):
                for gg in range(nparams):
                    covSize += abs(_A[g,gg])
            covSize /= nparams
            if covSize >= 100.0:
                # Reset variables when covariance matrix diverges
                print("Reset xNES: covsize %.2f" % covSize)
                _A = zeros((nparams,nparams))

            # Compute the elapsed time (i.e., how much time the generation lasted)
            elapsed = (time.time() - start_time)

            # Update information
            self.updateInfo(cgen, ceval, fitness, self.center, centroidfit, fitness[0], elapsed, maxsteps)

        # save data
        self.save(cgen, ceval, centroidfit, self.center, fitness[0], (time.time() - start_time))

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

