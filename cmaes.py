#!/usr/bin/python

# Libraries to be imported
import gym
from gym import spaces
import numpy as np
from numpy import floor, ceil, log, eye, zeros, array, sqrt, sum, dot, tile, outer, real
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

# Evolve with CMA-ES algorithm (Hansen and Ostermeier, 2001)
# This code comes from PyBrain (Schaul et al., 2010) with
# some custom modifications
class CMAES(EvoAlgo):
    def __init__(self, env, policy, seed, filedir):
        EvoAlgo.__init__(self, env, policy, seed, filedir)
        #self.updateCovEveryGen = True

    def run(self, maxsteps):

        start_time = time.time()

        # initialize the solution center
        self.center = self.policy.get_trainable_flat()
        
        # Extract the number of parameters
        nparams = self.policy.nparams
        # setting parameters
        if self.batchSize == 0:
            # Use default value: 4 + floor(3 * log(N)), where N is the number of parameters
            self.batchSize = int(4 + floor(3 * log(nparams))) # population size, offspring number
            if "Tf" in type(self.policy).__name__:
                # Update the number of rollout calls in policy (the initial value has been set based on configuration file)
                self.policy.updaten(self.batchSize)
            
        mu = int(floor(self.batchSize / 2)) # number of parents/points for recombination
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
        # Initialize dynamic (internal) strategy parameters and constants
        covPath = zeros(nparams)
        stepPath = zeros(nparams)               # evolution paths for C and stepSize
        B = eye(nparams, nparams)               # B defines the coordinate system
        D = eye(nparams, nparams)               # diagonal matrix D defines the scaling
        C = dot(dot(B, D), dot(B, D).T)         # covariance matrix
        chiN = nparams ** 0.5 * (1 - 1. / (4. * nparams) + 1 / (21. * nparams ** 2))
                                                # expectation of ||numParameters(0,I)|| == norm(randn(numParameters,1))
        self.stepsize = 0.5

        ceval = 0               # current evaluation
        cgen = 0                # current generation
    
        # RandomState for perturbing the performed actions (used only for samples, not for centroid)
        np.random.seed(self.seed)

        print("CMA-ES: seed %d maxmsteps %d batchSize %d stepsize %.2f sameEnvCond %d nparams %d" % (self.seed, maxsteps / 1000000, self.batchSize, self.stepsize, self.sameenvcond, nparams))

        # Set evolution mode
        self.policy.runEvo()

        """
        updateCovMatRate = 1
        if not self.updateCovEveryGen:
            updateCovMatRate = 0.1 / covLearningRate / nparams
            decPart = math.modf(updateCovMatRate)[0]
            if decPart >= 0.5:
                updateCovMatRate = ceil(updateCovMatRate)
            else:
                updateCovMatRate = floor(updateCovMatRate)
        updateCovMatRate = int(updateCovMatRate)
        """

        # main loop
        elapsed = 0
        while ceval < maxsteps:
            cgen += 1

            # Extract half samples from Gaussian distribution with mean 0.0 and standard deviation 1.0
            samples = np.random.randn(nparams, self.batchSize)
            # Generate offspring
            offspring = tile(self.center.reshape(nparams, 1), (1, self.batchSize)) + self.stepsize * dot(dot(B, D), samples)
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
            # Re-organize samples according to indices
            samples = samples[:, index]
            # Do the same for offspring
            offspring = offspring[:, index]
            # Select best <mu> samples and offspring for computing new center and cumulation paths
            samsel = samples[:, range(mu)]
            offsel = offspring[:, range(mu)]
            offmut = offsel - tile(self.center.reshape(nparams, 1), (1, mu))

            samplemean = dot(samsel, weights)
            self.center = dot(offsel, weights)

            # Cumulation: Update evolution paths
            stepPath = (1 - cumStep) * stepPath \
                     + sqrt(cumStep * (2 - cumStep) * muEff) * dot(B, samplemean)		 # Eq. (4)
            hsig = norm(stepPath) / sqrt(1 - (1 - cumStep) ** (2 * ceval / float(self.batchSize))) / chiN \
                     < 1.4 + 2. / (nparams + 1)
            covPath = (1 - cumCov) * covPath + hsig * \
                     sqrt(cumCov * (2 - cumCov) * muEff) * dot(dot(B, D), samplemean) # Eq. (2)

            # Adapt covariance matrix C
            C = ((1 - covLearningRate) * C					# regard old matrix   % Eq. (3)
                     + covLearningRate * (1 / muCov) * (outer(covPath, covPath) # plus rank one update
                     + (1 - hsig) * cumCov * (2 - cumCov) * C)
                     + covLearningRate * (1 - 1 / muCov)				 # plus rank mu update
                     * dot(dot(offmut, diag(weights)), offmut.T)
                )

            # Adapt step size
            self.stepsize *= exp((cumStep / dampings) * (norm(stepPath) / chiN - 1)) # Eq. (5)

            # Update B and D from C
            # This is O(n^3). When strategy internal CPU-time is critical, the
            # next three lines should be executed only every (alpha/covLearningRate/N)-th
            # iteration, where alpha is e.g. between 0.1 and 10
            C = (C + C.T) / 2     # enforce symmetry
            Ev, B = eig(C)        # eigen decomposition, B==normalized eigenvectors
            Ev = real(Ev)	  # enforce real value
            D = diag(sqrt(Ev))    #diag(ravel(sqrt(Ev))) # D contains standard deviations now
            B = real(B)

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
                    covSize += abs(C[g,gg])
            covSize /= nparams
            if self.stepsize >= 10.0 or covSize >= 100.0 or (self.stepsize >= 5.0 and covSize >= 20.0):
                # Reset variables when either stepsize or covariance matrix diverges
                print("Reset CMAES: stepsize %.2f covsize %.2f" % (self.stepsize, covSize))
                covPath = zeros(nparams)
                stepPath = zeros(nparams)
                B = eye(nparams, nparams)
                D = eye(nparams, nparams)
                C = dot(dot(B, D), dot(B, D).T)
                self.stepsize = 0.5

            # Compute the elapsed time (i.e., how much time the generation lasted)
            elapsed = (time.time() - start_time)

            # Update information
            self.updateInfo(cgen, ceval, fitness, self.center, centroidfit, fitness[0], elapsed, maxsteps)

        # save data
        self.save(cgen, ceval, centroidfit, self.center, fitness[0], (time.time() - start_time))

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

