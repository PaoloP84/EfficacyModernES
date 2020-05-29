#!/usr/bin/python

import numpy as np
import time

# average fitness of the samples
def averageFit(fitness):
    avef = 0.0
    for i in range(len(fitness)):
        avef = avef + fitness[i]
    avef = avef / len(fitness)
    return avef

class EvoAlgo(object):
    def __init__(self, env, policy, seed, filedir):
        # Copy the environment
        self.env = env
        # Copy the policy
        self.policy = policy
        # Copy the seed
        self.seed = seed
        # Copy the directory where files will be saved
        self.filedir = filedir
        # Set fitness initially to a very low value
        self.bestfit = -999999999.0
        # Initialize best solution found
        self.bestsol = None
        # Set generalization initially to a very low value
        self.bestgfit = -999999999.0
        # Initialize best generalizing solution found
        self.bestgsol = None
        # Initialize stat information
        self.stat = np.arange(0, dtype=np.float64)
        # Initialize average fitness
        self.avgfit = 0.0
        # Time from last save
        self.last_save_time = time.time()
        # Variables for evolution
        self.center = None
        self.stepsize = 0.0
        self.noiseStdDev = 0.0
        self.batchSize = 0
        self.sameenvcond = 0
        self.wdecay = 0
        self.evalCenter = 0
        
    def reset(self):
         # Set fitness initially to a very low value
        self.bestfit = -999999999.0
        # Initialize best solution found
        self.bestsol = None
        # Set generalization initially to a very low value
        self.bestgfit = -999999999.0
        # Initialize best generalizing solution found
        self.bestgsol = None
        # Initialize stat information
        self.stat = np.arange(0, dtype=np.float64)
        # Initialize average fitness
        self.avgfit = 0.0
        # Time from last save
        self.last_save_time = time.time()

    # Set evolutionary variables like batchSize, step size, etc.
    def setEvoVars(self, sampleSize, stepsize, noiseStdDev, sameenvcond, wdecay, evalCenter):
        self.batchSize = sampleSize
        self.stepsize = stepsize
        self.noiseStdDev = noiseStdDev
        self.sameenvcond = sameenvcond
        self.wdecay = wdecay
        self.evalCenter = evalCenter          

    # Implemented by sub-classes
    def run(self, maxsteps):
        # Run method depends on the algorithm
        raise NotImplementedError

    def test(self, testfile):
        """
        # Extract the seed from the testfile
        import re
        testseed = [int(s) for s in re.findall("(\d+)", testfile)]
        testseed = testseed[0]
        """
        # Set the seed
        self.policy.setSeed(self.policy.get_seed + 1000000)
        # necessary initialization of the renderer
        if self.policy.displayneurons == 0 and not "Er" in self.policy.environment:
            self.env.render(mode="human")
        if testfile is not None:
            if self.filedir.endswith("/"):
                fname = self.filedir + testfile
            else:
                fname = self.filedir + "/" + testfile
            # Load the individual to be tested
            if self.policy.normalize == 0:
                # bestgS*.npy (or bestS*.npy) contains the individual only
                bestgeno = np.load(fname)
            else:
                # The numpy file contains both the individual and the normalization vectors
                geno = np.load(fname)
                for i in range(self.policy.ninputs * 2):
                    self.policy.normvector[i] = geno[self.policy.nparams + i]
                bestgeno = geno[0:self.policy.nparams]
                # Set normalization vectors
                self.policy.nn.setNormalizationVectors()
            # Test the loaded individual
            self.policy.set_trainable_flat(bestgeno)
        else:
            self.policy.reset()
        # Set test mode
        self.policy.runTest()
        # If the number of generalization trials is 0, we set it to the
        # number of evaluation trials
        if self.policy.nttrials == 0:
            self.policy.nttrials = self.policy.ntrials
        self.policy.doGeneralization(True)
        # During test phase, render is active
        # Test loaded individual
        eval_rews, eval_length = self.policy.rollout(render=True, timestep_limit=1000)  # eval rollouts don't obey task_data.timestep_limit
        print("Test: steps %d reward %lf" % (eval_length, eval_rews))

    def updateBest(self, fit, ind):
        if fit > self.bestfit:
            self.bestfit = fit
            # in case of normalization store also the normalization vectors
            if  self.policy.normalize == 0:
                self.bestsol = np.copy(ind)
            else:
                self.bestsol = np.append(ind, self.policy.normvector)

    def updateBestg(self, fit, ind):
        if fit > self.bestgfit:
            self.bestgfit = fit
            # in case of normalization store also the normalization vectors
            if self.policy.normalize == 0:
                self.bestgsol = np.copy(ind)
            else:
                self.bestgsol = np.append(ind, self.policy.normvector)
                
    # called at the end of every generation to display and store data
    def updateInfo(self, gen, steps, fitness, centroid, centroidfit, bestsam, elapsed, nevals):
        self.computeAvg(fitness)
        self.stat = np.append(self.stat, [steps, self.bestfit, self.bestgfit, self.avgfit, centroidfit, bestsam])
        print('Seed %d (%.1f%%) gen %d msteps %d bestfit %.2f bestgfit %.2f centroid %.2f bestsam %.2f avg %.2f weightsize %.2f' %
                      (self.seed, steps / float(nevals) * 100, gen, steps / 1000000, self.bestfit, self.bestgfit, centroidfit, bestsam, self.avgfit, np.average(np.absolute(centroid))))
        # Save data every <saveeach> minutes
        if (gen > 1) and ((time.time() - self.last_save_time) > (self.policy.saveeach * 60)):
        #if gen % 10 == 0:
            self.save(gen, steps, centroidfit, centroid, bestsam, elapsed)
            self.last_save_time = time.time()        

    def save(self, gen, steps, centroidfit, centroid, bestsam, elapsed):
        # save best, bestg, and last centroid
        fname = self.filedir + "/bestS" + str(self.seed)
        np.save(fname, self.bestsol)
        if self.bestgsol is not None:
            fname = self.filedir + "/bestgS" + str(self.seed)
            np.save(fname, self.bestgsol)
        fname = self.filedir + "/centroidS" + str(self.seed)
        if self.policy.normalize == 0:
            np.save(fname, centroid)
        else:
            np.save(fname, np.append(centroid, self.policy.normvector))
        # save statistics
        fname = self.filedir + "/statS" + str(self.seed)
        np.save(fname, self.stat)
        # save summary statistics
        fname = self.filedir + "/S" + str(self.seed) + ".fit"
        fp = open(fname, "w")
        fp.write('Seed %d gen %d eval %d bestfit %.2f bestgfit %.2f centroid %.2f bestsam %.2f average %.2f runtime %.2f\n' % (self.seed, gen, steps, self.bestfit, self.bestgfit, centroidfit, bestsam, self.avgfit, elapsed))
        fp.close()

    def computeAvg(self, fitness):
        self.avgfit = averageFit(fitness)
