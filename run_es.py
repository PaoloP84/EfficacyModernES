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
from policyt import ErPolicyTf, GymPolicyTf, make_session
from evoalgo import EvoAlgo
from salimans import Salimans
from cmaes import CMAES
from xnes import xNES
from snes import sNES
from sss import SSS

# Directory of the script .py
scriptdirname = os.path.dirname(os.path.realpath(__file__))
# Directory where files will be saved
filedir = None
# Global variables
center = None                           # the solution center
sample = None                           # the solution samples
# Evaluation
environment = None
# Specific parameter for ES with Adam optimizer
stepsize = 0.01
noiseStdDev = 0.02
sampleSize = 20
wdecay = 0
sameenvcond = 0
maxsteps = 1000000
evalCenter = 0
saveeach = 60
# Algorithms dict
algodict = {
    'CMAES': CMAES,
    'Salimans': Salimans,
    'xNES': xNES,
    'sNES': sNES,
    'SSS': SSS
}

# Parse the [ADAPT] section of the configuration file
def parseConfigFile(filename):
    global maxsteps
    global envChangeEvery
    global environment
    global fullyRandom
    global stepsize
    global noiseStdDev
    global sampleSize
    global wdecay
    global sameenvcond
    global evalCenter
    global saveeach

    if os.path.isfile(filename):

        config = configparser.ConfigParser()
        config.read(filename)

        # Section EVAL
        options = config.options("ADAPT")
        for o in options:
            found = 0
            if o == "maxmsteps":
                maxsteps = config.getint("ADAPT","maxmsteps") * 1000000
                found = 1
            if o == "environment":
                environment = config.get("ADAPT","environment")
                found = 1
            if o == "stepsize":
                stepsize = config.getfloat("ADAPT","stepsize")
                found = 1
            if o == "noisestddev":
                noiseStdDev = config.getfloat("ADAPT","noiseStdDev")
                found = 1
            if o == "samplesize":
                sampleSize = config.getint("ADAPT","sampleSize")
                found = 1
            if o == "wdecay":
                wdecay = config.getint("ADAPT","wdecay")
                found = 1
            if o == "sameenvcond":
                sameenvcond = config.getint("ADAPT","sameenvcond")
                found = 1
            if o == "evalcenter":
                evalCenter = config.getint("ADAPT","evalcenter")
                found = 1
            if o == "saveeach":
                saveeach = config.getint("ADAPT","saveeach")
                found = 1
              
            if found == 0:
                print("\033[1mOption %s in section [ADAPT] of %s file is unknown\033[0m" % (o, filename))
                sys.exit()
    else:
        print("\033[1mERROR: configuration file %s does not exist\033[0m" % (filename))
        sys.exit()

def helper():
    print("Main()")
    print("Program Arguments: ")
    print("-f [filename]             : the file containing the parameters shown below (mandatory)")
    print("-s [integer]              : the number used to initialize the seed")
    print("-n [integer]              : the number of replications to be run")
    print("-a [algorithm]            : the algorithm used for the evolution (algorithms available: CMAES, Salimans or xNES)")
    print("-t [filename]             : the file containing the adapted policy to be tested")
    print("-T [filename]             : the file containing the adapted policy to be tested, display neurons")    
    print("-d [directory]            : the directory where all output files are stored (default current dir)")
    print("-tf                       : use tensorflow policy (valid only for gym and pybullet")
    print("")
    print("The .ini file contains the following [ADAPT] and [POLICY] parameters:")
    print("[ADAPT]")
    print("environment [string]      : environment (default 'CartPole-v0'")
    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
    print("sampleSize [integer]      : number of samples (default 20)")
    print("stepsize [float]          : learning stepsize (default 0.01)")
    print("noiseStdDev [float]       : samples noise (default 0.02)")
    print("wdecay [0/1]              : weight decay (defaul 0)")
    print("sameenvcond [0/1]         : samples experience the same environmental conditions")
    print("evalCenter [0/1]          : whether or not centroid is evaluated")
    print("saveeach [integer]        : save data each n minutes (default 60)")
    print("[POLICY]")
    print("ntrials [integer]         : number of evaluation episodes (default 1)")
    print("nttrials [integer]        : number of post-evaluation episodes (default 0)")
    print("maxsteps [integer]        : number of evaluation steps (default 1000), for EREnvs only")
    print("nhiddens [integer]        : number of hidden x layer (default 50)")
    print("nlayers [integer]         : number of hidden layers (default 1)")
    print("bias [0/1]                : whether we have biases (default 0)")
    print("out_type [integer]        : type of output: 1=logistic, 2=tanh, 3=linear, 4=binary, 5=uniform with bins (default 2)")
    print("nbins [integer]           : number of bins, for uniform output type only")
    print("architecture [0/1/2]      : network architecture 0=feedforward 1=recurrent 2=fullrecurrent (default 0)")
    print("afunction [1/2]           : the activation function of neurons 1=logistic 2=tanh (default 2)")
    print("winit [0/1/2]             : weight initialization 0=xavier 1=normc 2=uniform (default 0)")
    print("action_noise [0/1]        : noise applied to actions (default 1)")
    print("normalized [0/1]          : whether or not the input observations are normalized (default 1)")
    print("clip [0/1]                : whether we clip observation in [-5,5] (default 0)")
    print("")
    print("This is the help output")
    sys.exit()

# Main code
def main(argv):
    global maxsteps
    global environment
    global filedir
    global saveeach

    # Processing command line argument
    argc = len(argv)

    # Print help information
    if (argc == 1):
                helper()

    # Parameters:
    # - configuration file;
    # - seed (default is 1);
    # - number of replications (default is 1);
    # - test option
    # - centroid test option
    # - directory where files will be stored (default is the directory containing this file)
    filename = None
    cseed = 0
    nreplications = 1
    algorithm = None
    filedir = './'
    testfile = None
    test = False
    displayneurons = 0
    useTf = False
    
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
        elif (argv[i] == "-a"):
            i += 1
            if (i < argc):
                algorithm = argv[i]
                i += 1
        elif (argv[i] == "-t"):
            i += 1
            test = True
            if (i < argc):
                testfile = argv[i]
                i += 1
        elif (argv[i] == "-T"):
            i += 1
            test = True
            if (i < argc):
                testfile = argv[i]
                displayneurons = 1
                i += 1   
        elif (argv[i] == "-d"):
            i += 1
            if (i < argc):
                filedir = argv[i]
                i += 1
        elif (argv[i] == "-tf"):
            i += 1
            useTf = True
        else:
            # We simply ignore the argument
            print("WARNING: unrecognized argument %s" % argv[i])
            i += 1

    if filename is not None:
        # Parse configuration file
        parseConfigFile(filename)
    else:
        print("File %s does not exist... It is mandatory!!! Stop!" % filename)
        sys.exit(-1)
    if filedir is None:
        # Default directory is that of the .py file
        filedir = scriptdirname
    if algorithm is None:
        # If the algorithm is not specified, we print out a warning message
        print("WARNING: unspecified algorithm with option -a... We use Salimans algorithm (default)!")
        # Salimans algorithm is the default evolutionary method
        algorithm = "Salimans"
    if algorithm not in algodict:
        # The passed algorithm is unknown! We stop the program!
        print("Algorithm %s is unknown!!!" % algorithm)
        print("Please use one of the following algorithms:")
        for a in algodict:
            print("%s" % a)
        sys.exit(-1)

    print("Environment %s nreplications %d maxmsteps %d " % (environment, nreplications, maxsteps / 1000000))
    env = None
    policy = None
    # Environment can be either of evorobot type or from gym
    if "Er" in environment:
        # import the problem library
        ErProblem = __import__(environment)
        # Create a new evorobot environment
        env = ErProblem.PyErProblem()
        # Evorobot environment
        # Define the objects required (they depend on the environment)
        ob = np.arange(env.ninputs, dtype=np.float64)
        ac = np.arange(env.noutputs, dtype=np.float64)
        done = np.arange(1, dtype=np.float64)
        env.copyObs(ob)
        env.copyAct(ac)
        env.copyDone(done)
        if useTf:
            if algorithm == "Salimans":
                size = sampleSize * 2
            else:
                size = sampleSize
            # Create a new session
            session = make_session(single_threaded=True)
            # Use policy with Tensorflow
            policy = ErPolicyTf(env, env.ninputs, env.noutputs, env.low, env.high, size, ob, ac, done, filename, cseed)
            # Initialize tensorflow variables
            policy.initTfVars()
            # Initialize stat
            if policy.normalize == 1:
                policy.initStat()
        else:       
            # Define the policy
            policy = ErPolicy(env, env.ninputs, env.noutputs, env.low, env.high, ob, ac, done, filename, cseed)
    else:
        # Gym (or pybullet) environment
        # Check if the environment comes from pybullet
        if "Bullet" in environment:
            # Import pybullet
            import pybullet
            import pybullet_envs
        # Get the environment
        env = gym.make(environment)
        if useTf:
            if algorithm == "Salimans":
                size = sampleSize * 2
            else:
                size = sampleSize
            # Create a new session
            session = make_session(single_threaded=True)
            # Use policy with Tensorflow
            policy = GymPolicyTf(env, env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low[0], env.action_space.high[0], size, filename, cseed)
            # Initialize tensorflow variables
            policy.initTfVars()
            # Initialize stat
            if policy.normalize == 1:
                policy.initStat()
        else:
            # Define the objects required (they depend on the environment)
            ob = np.arange(env.observation_space.shape[0], dtype=np.float64)
            ac = np.arange(env.action_space.shape[0], dtype=np.float64)
            # Define the policy
            policy = GymPolicy(env, env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low[0], env.action_space.high[0], ob, ac, filename, cseed)

    policy.environment = environment
    policy.saveeach = saveeach
    # Create the algorithm class
    algo = algodict[algorithm](env, policy, cseed, filedir)
    # Set evolutionary variables (i.e., batchSize, stepSize, sameenvcond, wdecay, etc.)
    algo.setEvoVars(sampleSize, stepsize, noiseStdDev, sameenvcond, wdecay, evalCenter)

    if (test):
        # Test
        print("Run Test: Environment %s testfile %s" % (environment, testfile))
        policy.displayneurons = displayneurons
        algo.test(testfile)
    else:
        if (cseed != 0):
            print("Run Evolve: Environment %s Seed %d Nreplications %d" % (environment, cseed, nreplications))
            for r in range(nreplications):
                # Run <maxsteps> evaluations
                algo.run(maxsteps)
                # Update seed
                algo.seed += 1
                policy.seed += 1
                # Reset both algorithm and policy for next replication
                algo.reset()
                policy.reset()
        else:
            print("Please indicate the seed to run evolution")

if __name__ == "__main__":
    main(sys.argv)
