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

# Evolve with sNES algorithm (Wierstra, Schaul, Peters and Schmidhuber, 2008)
class SSS(EvoAlgo):
    def __init__(self, env, policy, seed, filedir):
        EvoAlgo.__init__(self, env, policy, seed, filedir)
        self.pop = None

    def run(self, maxsteps):

        start_time = time.time()

        # initialize the solution center (here the centroid is used to generate
        # random individuals)
        self.center = self.policy.get_trainable_flat()
        
        # Extract the number of parameters
        nparams = self.policy.nparams
        # setting parameters

        ceval = 0               # current evaluation
        cgen = 0                # current generation
    
        # RandomState for perturbing the performed actions (used only for samples, not for centroid)
        np.random.seed(self.seed)

        print("SSS: seed %d maxmsteps %d batchSize %d sameEnvCond %d nparams %d" % (self.seed, maxsteps / 1000000, self.batchSize, self.sameenvcond, nparams))

        # Set evolution mode
        self.policy.runEvo()

        # Population
        self.pop = tile(self.center.reshape(1, nparams), (self.batchSize, 1))
        # Apply random variations to solution center
        for i in range(self.batchSize):
            for j in range(nparams):
                self.pop[i,j] += np.random.random() * 0.2 - 0.1

        # Allocate offspring
        offspring = np.zeros((self.batchSize, nparams))

        # Here centroid is useless
        centroidfit = -999999999.0

        # main loop
        elapsed = 0
        while ceval < maxsteps:
            cgen += 1

            fitness = zeros(self.batchSize * 2)
            # If normalize=1 we update the normalization vectors
            if self.policy.normalize == 1:
                self.policy.nn.updateNormalizationVectors()
            # Reset environmental seed every generation
            self.policy.setSeed(self.policy.get_seed + cgen)
            # Set generalization flag to False
            self.policy.doGeneralization(False)
            # Evaluate parents and offspring
            for k in range(self.batchSize):
                # Set policy parameters (corresponding to the current offspring)
                self.policy.set_trainable_flat(self.pop[k])
                # Sample of the same generation experience the same environmental conditions
                if self.sameenvcond == 1:
                    self.policy.setSeed(self.policy.get_seed + cgen)
                # Evaluate the parents
                eval_rews, eval_length = self.policy.rollout(timestep_limit=1000)
                # Get the fitness
                fitness[k] = eval_rews
                # Update the number of evaluations
                ceval += eval_length
                # Update data if the current parent is better than current best
                self.updateBest(fitness[k], self.pop[k])
                # Generate the offspring
                for j in range(nparams):
                    offspring[k,j] = self.pop[k,j]
                    if np.random.uniform(low=0.0, high=1.0) < 0.03:
                        # Extract a random number to perform either weight
                        # replacemente or weight perturbation
                        if np.random.uniform(low=0.0, high=1.0) < 0.5:
                            # Weight replacement
                            offspring[k,j] = np.random.random() * (self.policy.wrange * 2.0) - self.policy.wrange
                        else:
                            # Weight perturbation
                            offspring[k,j] += np.random.random() * 0.2 - 0.1
                self.policy.set_trainable_flat(offspring[k])
                # Sample of the same generation experience the same environmental conditions
                if self.sameenvcond == 1:
                    self.policy.setSeed(self.policy.get_seed + cgen)
                # Evaluate the offspring
                eval_rews, eval_length = self.policy.rollout(timestep_limit=1000)
                # Get the fitness
                fitness[self.batchSize + k] = eval_rews
                # Update the number of evaluations
                ceval += eval_length
                # Update data if the current offspring is better than current best
                self.updateBest(fitness[self.batchSize + k], offspring[k])

            # Selection
            parentfit = np.copy(fitness[0:self.batchSize])
            # Add noise to parent fitness
            """
            for i in range(self.batchSize):
                noise = np.random.random() * self.noiseStdDev * 2.0 - self.noiseStdDev
                parentfit[i] += (noise * parentfit[i])
            """
            offspringfit = np.copy(fitness[self.batchSize:(self.batchSize * 2)])
            # Add noise to offspring fitness
            """
            for i in range(self.batchSize):
                noise = np.random.random() * self.noiseStdDev * 2.0 - self.noiseStdDev
                offspringfit[i] += (noise * offspringfit[i])
            """
            # Sort parent and offspring based on fitness (descending mode)
            parentfit, parentidx = descendent_sort(parentfit)
            offspringfit, offspringidx = descendent_sort(offspringfit)
            # Population index
            k = 0
            # Parent index
            p = 0
            # Offspring index
            o = 0
            while k < self.batchSize:
                if parentfit[p] > offspringfit[o]:
                    p += 1
                else:
                    # Offspring replaces worst parent
                    wp = parentidx[self.batchSize - 1 - o]
                    bo = offspringidx[o]
                    self.pop[wp] = np.copy(offspring[bo])
                    fitness[wp] = fitness[self.batchSize + bo]
                    o += 1
                k += 1

            # Get the best individual (of the current generation)
            bfit = np.copy(fitness[0:self.batchSize])
            bfit, bidx = descendent_sort(bfit)
            bidx = bidx[0]

            # Now perform generalization
            if self.policy.generalize:
                candidate = np.copy(self.pop[bidx])
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
            self.updateInfo(cgen, ceval, fitness[0:self.batchSize], self.center, centroidfit, fitness[bidx], elapsed, maxsteps)

        # save data
        self.save(cgen, ceval, centroidfit, self.center, fitness[bidx], (time.time() - start_time))

        # print simulation time
        end_time = time.time()
        print('Simulation time: %dm%ds ' % (divmod(end_time - start_time, 60)))

