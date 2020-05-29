#!/usr/bin/python

# Libraries to be imported
import configparser
import sys
import os
from pathlib import Path

# Directory of the script .py
scriptdirname = os.path.dirname(os.path.realpath(__file__))
# Directory where files will be saved
filedir = None
# Evaluation
environment = None
# Evaluation
maxsteps = 1000000
# Policy
nhiddens = 32
nlayers = 2

# Parse the [ADAPT] and [POLICY] sections of the configuration file
def parseConfigFile(filename):
    global maxsteps
    global environment
    global nhiddens
    global nlayers

    if os.path.isfile(filename):

        config = configparser.ConfigParser()
        config.read(filename)

        # Section ADAPT
        options = config.options("ADAPT")
        for o in options:
            if o == "maxmsteps":
                maxsteps = config.getint("ADAPT","maxmsteps") * 1000000
            if o == "environment":
                environment = config.get("ADAPT","environment")
        # Section POLICY
        options = config.options("POLICY")
        for o in options:
            if o == "nhiddens":
                nhiddens = config.getint("POLICY","nhiddens")
            if o == "nlayers":
                nlayers = config.getint("POLICY","nlayers")
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
    print("-d [directory]            : the directory where all output files are stored (default current dir)")
    print("")
    print("The .ini file contains the following [ADAPT] and [POLICY] parameters:")
    print("[ADAPT]")
    print("environment [string]      : environment (default 'CartPole-v0'")
    print("maxmsteps [integer]       : max number of (million) steps (default 1)")
    print("[POLICY]")
    print("nhiddens [integer]        : number of hidden x layer (default 50)")
    print("nlayers [integer]         : number of hidden layers (default 1)")
    print("All other parameters in the .ini file are ignored!")
    print("")
    print("IMPORTANT!!! run_reinforce.py should be run from baselines directory")
    print("")
    print("This is the help output")
    sys.exit()

# Main code
def main(argv):
    global maxsteps
    global environment
    global filedir
    global nhiddens
    global nlayers

    # Processing command line argument
    argc = len(argv)

    if argc == 1:
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
    filedir = './'
    
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
        elif (argv[i] == "-d"):
            i += 1
            if (i < argc):
                filedir = argv[i]
                i += 1
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
    if algorithm == "PPO":
        algorithm = "ppo2"

    print("Environment %s nreplications %d maxmsteps %d - Network: nhiddens %d nlayers %d" % (environment, nreplications, maxsteps / 1000000, nhiddens, nlayers))

    # Get the home directory for current user (for baselines)
    home = str(Path.home())
    # Build baselines path
    baselines_path = home + "/baselines"
    
    # Launch the command with the right arguments
    if (cseed != 0):
        print("Run Reinforce: Environment %s Seed %d Nreplications %d" % (environment, cseed, nreplications))
        for r in range(nreplications):
            #OPENAI_LOGDIR=./logs/h20/cartpole-ppo/$CSEED OPENAI_LOG_FORMAT=csv python3.5 -m baselines.run --alg=ppo2 --env=CartPole-v0 --network=mlp --num_hidden=20 --num_layers=1 --num_timesteps=1e6 --seed=$CSEED --save_path=./models/h20/cartpole_ppo/$CSEED
            # We create the bash command to be run
            bashCmd = "OPENAI_LOGDIR=.logs/" + str(filedir) + "/" + str(cseed) + " "
            bashCmd += "OPENAI_LOG_FORMAT=csv "
            bashCmd += "python3.5 -m "
            bashCmd += "baselines.run "
            bashCmd += "--alg=" + str(algorithm) + " "
            bashCmd += "--env=" + str(environment) + " "
            bashCmd += "--network=mlp "
            bashCmd += "--num_hidden=" + str(nhiddens) + " "
            bashCmd += "--num_layers=" + str(nlayers) + " "
            bashCmd += "--num_timesteps=" + str(maxsteps) + " "
            bashCmd += "--seed=" + str(cseed) + " "
            bashCmd += "--save_path=.models/" + str(filedir) + "/" + str(cseed)
            # Run the command
            os.system(bashCmd)
            # Update the seed
            cseed += 1
    else:
        print("Please indicate the seed to run evolution")

if __name__ == "__main__":
    main(sys.argv)
