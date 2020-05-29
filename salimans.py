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

# Evolve with ES algorithm taken from Salimans et al. (2017)
class Salimans(EvoAlgo):
    def __init__(self, env, policy, seed, filedir):
        EvoAlgo.__init__(self, env, policy, seed, filedir)

    def run(self, maxsteps):

        start_time = time.time()

        # initialize the solution center
        self.center = self.policy.get_trainable_flat()
        
        # Extract the number of parameters
        nparams = self.policy.nparams
        # setting parameters
        if self.batchSize == 0:
            # 4 + floor(3 * log(N))
            self.batchSize = int(4 + math.floor(3 * math.log(nparams)))
        # Symmetric weights in the range [-0.5,0.5]
        weights = zeros(self.batchSize)

        ceval = 0               # current evaluation
        cgen = 0                # current generation
        # Parameters for Adam policy
        m = zeros(nparams)
        v = zeros(nparams)
        epsilon = 1e-08 # To avoid numerical issues with division by zero...
        beta1 = 0.9
        beta2 = 0.999
    
        # RandomState for perturbing the performed actions (used only for samples, not for centroid)
        np.random.seed(self.seed)

        print("Salimans: seed %d maxmsteps %d batchSize %d stepsize %lf noiseStdDev %lf wdecay %d sameEnvCond %d nparams %d" % (self.seed, maxsteps / 1000000, self.batchSize, self.stepsize, self.noiseStdDev, self.wdecay, self.sameenvcond, nparams))

        # Set evolution mode
        self.policy.runEvo()

        # main loop
        elapsed = 0
        while ceval < maxsteps:
            cgen += 1

            # Extract half samples from Gaussian distribution with mean 0.0 and standard deviation 1.0
            samples = np.random.randn(self.batchSize, nparams)
            # We generate simmetric variations for the offspring
            symmSamples = zeros((self.batchSize * 2, nparams))
            for i in range(self.batchSize):
                sampleIdx = 2 * i
                for g in range(nparams):
                    symmSamples[sampleIdx,g] = samples[i,g]
                    symmSamples[sampleIdx + 1,g] = -samples[i,g]
            # Generate offspring
            offspring = tile(self.center.reshape(1, nparams), (self.batchSize * 2, 1)) + self.noiseStdDev * symmSamples
            # Evaluate offspring
            fitness = zeros(self.batchSize * 2)
            # If normalize=1 we update the normalization vectors
            if self.policy.normalize == 1:
                self.policy.nn.updateNormalizationVectors()
            # Reset environmental seed every generation
            self.policy.setSeed(self.policy.get_seed + cgen)
            # Set generalization flag to False
            self.policy.doGeneralization(False)
            # Evaluate offspring
            for k in range(self.batchSize * 2):
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
            # Now me must compute the symmetric weights in the range [-0.5,0.5]
            utilities = zeros(self.batchSize * 2)
            for i in range(self.batchSize * 2):
                utilities[index[i]] = i
            utilities /= (self.batchSize * 2 - 1)
            utilities -= 0.5
            # Now we assign the weights to the samples
            for i in range(self.batchSize):
                idx = 2 * i
                weights[i] = (utilities[idx] - utilities[idx + 1]) # pos - neg

            # Compute the gradient
            g = 0.0
            i = 0
            while i < self.batchSize:
                gsize = -1
                if self.batchSize - i < 500:
                    gsize = self.batchSize - i
                else:
                    gsize = 500
                g += dot(weights[i:i + gsize], samples[i:i + gsize,:]) # weights * samples
                i += gsize
            # Normalization over the number of samples
            g /= (self.batchSize * 2)
            # Weight decay
            if (self.wdecay == 1):
                globalg = -g + 0.005 * self.center
            else:
                globalg = -g
            # ADAM policy
            # Compute how much the center moves
            a = self.stepsize * sqrt(1.0 - beta2 ** cgen) / (1.0 - beta1 ** cgen)
            m = beta1 * m + (1.0 - beta1) * globalg
            v = beta2 * v + (1.0 - beta2) * (globalg * globalg)
            dCenter = -a * m / (sqrt(v) + epsilon)
            # update center
            self.center += dCenter

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
                if centroidfit > fitness[self.batchSize * 2 - 1]:
                    # Centroid undergoes generalization test
                    candidate = np.copy(self.center)
                else:
                    # Best sample undergoes generalization test
                    bestsamid = index[self.batchSize * 2 - 1]
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
            self.updateInfo(cgen, ceval, fitness, self.center, centroidfit, fitness[self.batchSize * 2 - 1], elapsed, maxsteps)

        # save data
        self.save(cgen, ceval, centroidfit, self.center, fitness[self.batchSize * 2 - 1], (time.time() - start_time))

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

