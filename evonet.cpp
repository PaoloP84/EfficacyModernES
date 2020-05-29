#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <ctype.h>
#include <string.h>
#include "evonet.h"
#include "utilities.h"

#define MAX_BLOCKS 20
#define MAXN 10000
#define CLIP_VALUE 5.0

// Local random number generator
RandomGenerator* netRng;

// Pointers to genotype, observation, action, neuron activation, normalization vectors
double *cgenotype = NULL;
double *cobservation = NULL;
double *caction = NULL;
double *neuronact = NULL;
double *cnormalization = NULL;

// Weight range (for uniform initialization only)
double netWrange = 1.0; // Default weight range is [-1.0,1.0]

// Actual number of outputs (used for bins)
int nouts = 0;

/*
 * standard logistic
 */
double logistic(double f)
{
    return ((double) (1.0 / (1.0 + exp(-f))));
}

/*
 * hyperbolic tangent
 */
double tanh(double f)
{
    if (f > 10.0)
        return 1.0;
    else if (f < - 10.0)
        return -1.0;
    else
        return ((double) ((1.0 - exp(-2.0 * f)) / (1.0 + exp(-2.0 * f))));    
}

// constructor
Evonet::Evonet()
{
    m_ninputs = 0;
    m_nhiddens = 0;
    m_noutputs = 0;
    m_nneurons = (m_ninputs + m_nhiddens + m_noutputs);
    m_nlayers = 1;
    m_bias = 0;
    m_netType = 0; // Feed-forward network
    m_actFunct = 2; // Tanh
    m_outType = 2; // Tanh output
    m_wInit = 0; // Xavier initializer
    m_clip = 0; // Do not clip inputs
    m_normalize = 0; // Do not normalize inputs
    m_randAct = 0; // Deterministic actions
    m_nbins = 1;
    m_low = -1.0;
    m_high = 1.0;
    m_normPhase = 0;
    netRng = new RandomGenerator(time(NULL));
    nouts = 0;
    m_act = new double[MAXN];
    m_netinput = new double[MAXN];
    m_netblock = new int[MAX_BLOCKS * 5];
    m_nblocks = 0;
    m_neurontype = new int[MAXN];
    m_mean = NULL;
    m_std = NULL;
    m_sum = NULL;
    m_sumsq = NULL;
    m_count = 0.0;
}

/*
 * set the seed
 */
void Evonet::seed(int s)
{
    netRng->setSeed(s);
}
    
Evonet::Evonet(int ninputs, int nhiddens, int noutputs, int nlayers, int bias, int netType, int actFunct, int outType, int wInit, int clip, int normalize, int randAct, double wrange, int nbins, double low, double high)
{
    int i;
    int l;

    // set variables
    m_ninputs = ninputs;
    m_nhiddens = nhiddens;
    m_noutputs = noutputs;
    m_nlayers = nlayers;
    if ((m_nhiddens % m_nlayers) != 0)
    {
        printf("WARNING: invalid combination for number of hiddens %d and number of layers %d --> division has remainder %d... We set m_nlayers to 1\n", m_nhiddens, m_nlayers, (m_nhiddens % m_nlayers));
        m_nlayers = 1;
    }
    m_bias = bias;
    m_netType = netType;
    if (m_netType > 0)
        // Only feed-forward network can have more than one hidden layer
        m_nlayers = 1;
    m_actFunct = actFunct;
    m_outType = outType;
    m_wInit = wInit;
    m_clip = clip;
    m_normalize = normalize;
    m_randAct = randAct;
    m_nbins = nbins;
    if (m_outType != 5)
        // Overwrite nbins: bins can be used only in case of <outType> == 5
        m_nbins = 1;
    m_low = low;
    m_high = high;
    netWrange = wrange;
    m_normPhase = 0;
    // Initialize random number generator
    netRng = new RandomGenerator(time(NULL));
    // Store the actual number of outputs
    nouts = m_noutputs;
    // Check how many outputs the network has
    if (m_nbins > 1)
        // In case of bins, the network has (noutputs * nbins) outputs
        m_noutputs = (nouts * nbins);
    // Set number of neurons
    m_nneurons = (m_ninputs + m_nhiddens + m_noutputs);
    // display info and check parameters are in range
    printf("Network %d->", m_ninputs);
    for (l = 0; l < nlayers; l++)
        printf("%d->", m_nhiddens / m_nlayers);
    printf("%d->", m_noutputs);
    if (m_netType < 0 || m_netType > 2)
        m_netType = 0; // Default is feed-forward
    if (m_netType == 0)
        printf("feedforward ");
    else if (m_netType == 1)
        printf("recurrent ");
    else if (m_netType == 2)
        printf("fully recurrent ");
    if (m_bias)
        printf("with bias ");
    if (m_actFunct < 1 || m_actFunct > 2)
        m_actFunct = 2; // Default is tanh activation function
    if (m_actFunct == 1)
        printf("logistic ");
    else
        printf("tanh ");
    if (m_outType < 1 || m_outType > 5)
        m_outType = 2; // Default is tanh output
    switch (m_outType)
    {
        case 1:
            printf("out_type:logistic ");
            break;
        case 2:
            printf("out_type:tanh ");
            break;
        case 3:
            printf("out_type:linear ");
            break;
        case 4:
            printf("out_type:binary ");
            break;
        case 5:
            printf("out_type:uniform with bins ");
            break;
        default:
            // Code should never enter here!!!
            m_outType = 2;
            printf("out_type:tanh ");
            break;
    }
    if (m_wInit < 0 || m_wInit > 2)
        m_wInit = 0; // Default is Xavier initializer
    if (m_wInit == 0)
        printf("init:xavier ");
    else if (m_wInit == 1)
        printf("init:normc ");
    else if (m_wInit == 2)
        printf("init:uniform ");
    if (m_normalize < 0 || m_normalize > 1)
        m_normalize = 0; // Default is no normalization of the inputs
    if (m_normalize == 1)
        printf("input-normalization ");
    if (m_clip < 0 || m_clip > 1)
        m_clip = 0; // Default is no clipping of the inputs
    if (m_clip == 1)
        printf("clip ");
    if (m_randAct < 0 || m_randAct > 1)
        m_randAct = 0; // Default is no randomization of the actions
    if (m_randAct == 1)
        printf("action-noise ");
    printf("\n");

    // In case of normalization we allocate variables
    // for observation mean and standard deviation
    if (m_normalize == 1)
    {
        m_mean = new double[m_ninputs];
        m_std = new double[m_ninputs];
        m_sum = new double[m_ninputs];
        m_sumsq = new double[m_ninputs];
        for (i = 0; i < m_ninputs; i++)
        {
            m_mean[i] = 0.0;
            m_std[i] = 1.0;
            m_sum[i] = 0.0;
            m_sumsq[i] = 0.01;
        }
        m_count = 0.01;
    }
    
    // allocate variables
    m_nblocks = 0;
    m_act = new double[m_nneurons];
    m_netinput = new double[m_nneurons];
    m_netblock = new int[MAX_BLOCKS * 5];
    m_neurontype = new int[m_nneurons];
    // Initialize network architecture
    initNetArchitecture();
}
    
Evonet::~Evonet()
{
}

// reset net
void Evonet::resetNet()
{
    int i;
    // Reset both activation and net-input for all neurons
    for (i = 0; i < m_nneurons; i++)
    {
        m_act[i] = 0.0;
        m_netinput[i] = 0.0;
    }
    // Reset neuron activation (for graphic purpose)
    for (i = 0; i < (m_ninputs + m_nhiddens + nouts); i++)
        neuronact[i] = 0.0;
}

// store pointer to weights vector
void Evonet::copyGenotype(double* genotype)
{
    cgenotype = genotype;
}

// store pointer to observation vector
void Evonet::copyInput(double* input)
{
    cobservation = input;
}

// store pointer to update vector
void Evonet::copyOutput(double* output)
{
    caction = output;
}

// store pointer to neuron activation vector
void Evonet::copyNeuronact(double* na)
{
    neuronact = na;
}

// store pointer to neuron activation vector
void Evonet::copyNormalization(double* no)
{
    cnormalization = no;
}

// update net
void Evonet::updateNet()
{
    double* p;
    double* a;
    double* ni;
    int i;
    int t;
    int b;
    int* nbl;
    int* nt;
    int j;

    p = cgenotype;

    // Normalize input
    if (m_normalize == 1)
    {
        // Before normalizing, we check whether or not
        // we must save new observations
        if (m_normPhase == 1)
            // Save observations
            collectNormalizationData();
        // Normalize observations
        // The following lines correspond to the author original code:
        // a = self._make_net(tf.clip_by_value((o - ob_mean) / ob_std, -5.0, 5.0))
        // We can observe that observations are first normalized and then clipped!!!
        for (j = 0; j < m_ninputs; j++)
        {
            *(cobservation + j) = (*(cobservation + j) - m_mean[j]) / m_std[j];
            // Clip input values
            if (m_clip == 1)
            {
                // Observations are clipped in the range [-CLIP_VALUE, CLIP_VALUE]
                // (i.e., values outside the range are truncated to the boundaries)
                if (*(cobservation + j) < -CLIP_VALUE)
                    *(cobservation + j) = -CLIP_VALUE;
                if (*(cobservation + j) > CLIP_VALUE)
                    *(cobservation + j) = CLIP_VALUE;
            }
        }
    }

    // compute biases
    if (m_bias == 1)
    {
        // Only non-input neurons have bias
        for(i = 0, ni = m_netinput; i < m_nneurons; i++, ni++)
        {
            if (i >= m_ninputs)
            {
                *ni = *p;
                p++;
            }
            else
                *ni = 0.0;
        }
    }

    // blocks
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
    {
        // connection block
        if (*nbl == 0)
        {
            for (t = 0, ni = (m_netinput + *(nbl + 1)); t < *(nbl + 2); t++, ni++)
            {
                for (i = 0, a = (m_act + *(nbl + 3)); i < *(nbl + 4); i++, a++)
                {
                    *ni += *a * *p;
                    p++;
                }
            }
        }

        // update block
        if (*nbl == 1)
        {
            for(t = *(nbl + 1), a = (m_act + *(nbl + 1)), ni = (m_netinput + *(nbl + 1)), nt = (m_neurontype + *(nbl + 1)); t < (*(nbl + 1) + *(nbl + 2)); t++, a++, ni++, nt++)
            {
                switch (*nt)
                {
                    case 0:
                        // input neurons are simple rely units
                        *a = *(cobservation + t);
                        break;
                    case 1:
                        // Logistic
                        *a = logistic(*ni);
                        break;
                    case 2:
                        // Tanh
                        *a = tanh(*ni);
                        break;
                    case 3:
                    // Linear outputs (just copy the net-input)
                    case 5:
                    // Uniformly spaced bins: copy the net-input (the output is processed in the getOutput() method!!!)
                        *a = *ni;
                        break;
                    case 4:
                        // Binary outputs
                        if (*ni >= 0.5)
                            *a = 1.0;
                        else
                            *a = -1.0;
                        break;
                    default:
                        // Invalid neuron type (code should never enter here!!!)
                        printf("Invalid neuron type %d\n", *nt);
                        // We force the use of the activation function
                        *a = tanh(*ni);
                        break;
                }
            }
        }
        nbl = (nbl + 5);
    }
    // Store the action
    getAction(caction);
    // Copy back the activations (for graphic purposes)
    for (i = 0, a = neuronact; i < (m_ninputs + m_nhiddens + nouts); i++, a++)
    {
        if (i < m_ninputs + m_nhiddens)
            // Copy activations for inputs and internal units
            *a = m_act[i];
        else
            // Outputs correspond to actions (i.e., values from bins in case of bins, activations otherwise!!!)
            *a = caction[(i - (m_ninputs + m_nhiddens))];
    }
}

// copy the output and eventually add noise
void Evonet::getAction(double* output)
{
    int i;
    // Get output activation
    if (m_nbins > 1)
        // Bins
        getOutputBins(output);
    else
        // Other cases
        getOutput(output);
    // Check whether or not action is stochastic
    if (m_randAct == 1)
    {
        // Randomize action
        for (i = 0; i < nouts; i++)
            // Add Gaussian noise
            output[i] += (netRng->getGaussian(1.0, 0.0) * 0.01);
    }
}

// copy the output
void Evonet::getOutput(double* output)
{
    int i;
    // Get output activation
    for (i = 0; i < nouts; i++)
        output[i] = m_act[m_ninputs + m_nhiddens + i];
}

// copy the output and eventually add noise
void Evonet::getOutputBins(double* output)
{
    // In case of bins, apply an action based on the bin with highest activation
    int i;
    int j;
    double cact;
    int cidx;

    // For each output, we look for the bin with the highest activation
    for (i = 0; i < nouts; i++)
    {
        // Current best activation
        cact = -9999.0;
        // Index of the current best activation
        cidx = -1;
        for (j = 0; j < m_nbins; j++)
        {
            if (m_act[m_ninputs + m_nhiddens + ((i * m_nbins) + j)] > cact)
            {
                // Found a new best, update
                cact = m_act[m_ninputs + m_nhiddens + ((i * m_nbins) + j)];
                cidx = j;
            }
        }
        output[i] = 1.0 / ((double)m_nbins - 1.0) * (double)cidx * (m_high - m_low) + m_low;
    }
}

// compute the number of required parameters
int Evonet::computeParameters()
{
    int nparams;
    int i;
    int t;
    int b;
    int* nbl;

    nparams = 0;
    
    // biases
    if (m_bias)
        nparams += (m_nhiddens + m_noutputs); // Only hiddens and outputs can have bias
    
    // blocks
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
    {
        // connection block
        if (*nbl == 0)
        {
            for (t = 0; t < *(nbl + 2); t++)
            {
                for (i = 0; i < *(nbl + 4); i++)
                    nparams++;
            }
        }
        nbl = (nbl + 5);
    }
    return nparams;
}

// initialize the architecture description
void Evonet::initNetArchitecture()
{
    int* nbl;
    int* nt;
    int n;
    
    m_nblocks = 0;
    nbl = m_netblock;

    // neurons' type
    for (n = 0, nt = m_neurontype; n < m_nneurons; n++, nt++)
    {
         if (n < m_ninputs)
             *nt = 0; // Inputs correspond to neuron type 0
         else
         {
             if (n < (m_ninputs + m_nhiddens))
                 *nt = m_actFunct; // Hiddens have type dependent on the activation function
             else
                 // The neuron type for outputs matches the output type
                 *nt = m_outType;
         }
    }
    
    // input update block
    *nbl = 1; nbl++;
    *nbl = 0; nbl++;
    *nbl = m_ninputs; nbl++;
    *nbl = 0; nbl++;
    *nbl = 0; nbl++;
    m_nblocks++;
    
    // Fully-recurrent network
    if (m_netType == 2)
    {
        // hiddens and outputs receive connections from input, hiddens and outputs
        *nbl = 0; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens + m_noutputs; nbl++;
        *nbl = 0; nbl++;
        *nbl = m_ninputs + m_nhiddens + m_noutputs; nbl++;
        m_nblocks++;
        
        // hidden update block
        *nbl = 1; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens + m_noutputs; nbl++;
        *nbl = 0; nbl++;
        *nbl = 0; nbl++;
        m_nblocks++;
    }
    // recurrent network with 1 layer
    else if (m_netType == 1)
    {
        // input-hidden connections
        *nbl = 0; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        *nbl = 0; nbl++;
        *nbl = m_ninputs; nbl++;
        m_nblocks++;
    
        // hidden-hidden connections
        *nbl = 0; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        m_nblocks++;
    
        // hidden update block
        *nbl = 1; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        *nbl = 0; nbl++;
        *nbl = 0; nbl++;
        m_nblocks++;

        // hidden-output connections
        *nbl = 0; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        *nbl = m_ninputs; nbl++;
        *nbl = m_nhiddens; nbl++;
        m_nblocks++;
      
        // output-output connections
        *nbl = 0; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        m_nblocks++;
    
        // output update block
        *nbl = 1; nbl++;
        *nbl = m_ninputs + m_nhiddens; nbl++;
        *nbl = m_noutputs; nbl++;
        *nbl = 0; nbl++;
        *nbl = 0; nbl++;
        m_nblocks++;
    }
    else
    {
        // Feed-forward network
        if (m_nhiddens == 0)
        {
            // Sensory-motor network
            *nbl = 0; nbl++;
            *nbl = m_ninputs; nbl++;
            *nbl = m_noutputs; nbl++;
            *nbl = 0; nbl++;
            *nbl = m_ninputs; nbl++;
            m_nblocks++;

            // output update block
            *nbl = 1; nbl++;
            *nbl = m_ninputs; nbl++;
            *nbl = m_noutputs; nbl++;
            *nbl = 0; nbl++;
            *nbl = 0; nbl++;
            m_nblocks++;
        }
        else
        {
            // input-hidden connections
            if (m_nlayers == 1)
            {
                *nbl = 0; nbl++;
                *nbl = m_ninputs; nbl++;
                *nbl = m_nhiddens; nbl++;
                *nbl = 0; nbl++;
                *nbl = m_ninputs; nbl++;
                m_nblocks++;
                
                // hidden update block
                *nbl = 1; nbl++;
                *nbl = m_ninputs; nbl++;
                *nbl = m_nhiddens; nbl++;
                *nbl = 0; nbl++;
                *nbl = 0; nbl++;
                m_nblocks++;

                // hidden-output connections
                *nbl = 0; nbl++;
                *nbl = m_ninputs + m_nhiddens; nbl++;
                *nbl = m_noutputs; nbl++;
                *nbl = m_ninputs; nbl++;
                *nbl = m_nhiddens; nbl++;
                m_nblocks++;

                // output update block
                *nbl = 1; nbl++;
                *nbl = m_ninputs + m_nhiddens; nbl++;
                *nbl = m_noutputs; nbl++;
                *nbl = 0; nbl++;
                *nbl = 0; nbl++;
                m_nblocks++;
            }
            else
            {
                int nhiddenperlayer;
                int start;
                int end;
                int prev;
                int i;
                nhiddenperlayer = m_nhiddens / m_nlayers;
                // input-hidden connections
                *nbl = 0; nbl++;
                *nbl = m_ninputs; nbl++;
                *nbl = nhiddenperlayer; nbl++;
                *nbl = 0; nbl++;
                *nbl = m_ninputs; nbl++;
                m_nblocks++;
                // hidden update block
                *nbl = 1; nbl++;
                *nbl = m_ninputs; nbl++;
                *nbl = nhiddenperlayer; nbl++;
                *nbl = 0; nbl++;
                *nbl = 0; nbl++;
                m_nblocks++;
                start = m_ninputs + nhiddenperlayer;
                end = nhiddenperlayer;
                prev = m_ninputs;
                i = 1;
                while (i < m_nlayers)
                {
                    // hidden-hidden connections
                    *nbl = 0; nbl++;
                    *nbl = start; nbl++;
                    *nbl = end; nbl++;
                    *nbl = prev; nbl++;
                    *nbl = end; nbl++;
                    m_nblocks++;
                    // hidden update block
                    *nbl = 1; nbl++;
                    *nbl = start; nbl++;
                    *nbl = end; nbl++;
                    *nbl = 0; nbl++;
                    *nbl = 0; nbl++;
                    m_nblocks++;
                    i++;
                    prev = start;
                    start += nhiddenperlayer;
                }

                // hidden-output connections
                *nbl = 0; nbl++;
                *nbl = start; nbl++;
                *nbl = m_noutputs; nbl++;
                *nbl = prev; nbl++;
                *nbl = nhiddenperlayer; nbl++;
                m_nblocks++;

                // output update block
                *nbl = 1; nbl++;
                *nbl = m_ninputs + m_nhiddens; nbl++;
                *nbl = m_noutputs; nbl++;
                *nbl = 0; nbl++;
                *nbl = 0; nbl++;
                m_nblocks++;
            }
        }
    }
}

// initialize weights
void Evonet::initWeights()
{
    int i;
    int j;
    int t;
    int b;
    int* nbl;
    double range;
 
    // cparameter
    j = 0;
    // Initialize biases to 0.0
    if (m_bias)
    {
        // Bias are initialized to 0.0
        for (i = 0; i < (m_nhiddens + m_noutputs); i++)
        {
            cgenotype[j] = 0.0;
            j++;
        }
    }
    // Initialize weights of connection blocks
    for (b = 0, nbl = m_netblock; b < m_nblocks; b++)
    {
        // connection block
        if (*nbl == 0)
        {
            switch (m_wInit)
            {
                // xavier initialization
                // gaussian distribution with range = (radq(2.0 / (ninputs + noutputs))
                case 0:
                {
                    int nin;
                    int nout;
                    // Number of inputs based on the incoming connections for the current block (temporary value)
                    nin = *(nbl + 4);
                    // Number of outputs based on the outging connections for the current block (final value)
                    nout = *(nbl + 2); // This does not change independently of the neighbouring blocks
                    // Check if neighbouring blocks refer to the same layer (we check only previous and next block)
                    if ((*(nbl + 5) == 0) && ((*(nbl + 1) == *(nbl + 6)) && (*(nbl + 2) == *(nbl + 7))))
                        nin += *(nbl + 9);
                    else if ((*(nbl - 5) == 0) && ((*(nbl - 4) == *(nbl + 1)) && (*(nbl - 3) == *(nbl + 2))))
                        nin += *(nbl - 1);
                    // Define the range as sum of number of incoming and outgoing connections
                    range = sqrt(2.0 / (nin + nout));
                    for (t = 0; t < *(nbl + 2); t++)
                    {
                        for (i = 0; i < *(nbl + 4); i++)
                        {
                            cgenotype[j] = netRng->getGaussian(range, 0.0); // It corresponds to getGaussian(1.0, 0.0) * range
                            j++;
                        }
                    }
                    break;
                }
                // normc, range=1.0 first layers, range=0.01 last layer
                // we assume that the last layer corresponds to the last connection block followed by the last update block
                case 1:
                {
                    double* wSqSum = new double[*(nbl + 2)];
                    int k;
                    int cnt;
                    if (b == (m_nblocks - 2))
                        range = 0.01;
                    else
                        range = 1.0;
                    // we have to compute the squared sum of gaussian numbers in order to scale the weights
                    // Here below the original python code from Salimans:
                    // out = np.random.randn(*shape).astype(np.double32)
                    // out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
                    // where randn extract samples from Gaussian distribution with mean 0.0 and std 1.0
                    // std is either 1.0 or 0.01 depending on the layer
                    // np.square(out).sum(axis=0, keepdims=True) computes the squared sum of the elements in out
                    for (t = 0; t < *(nbl + 2); t++)
                        wSqSum[t] = 0.0;
                    // Index storing the genotype block to be normalized (i.e., the starting index)
                    k = j;
                    // Counter of weights
                    cnt = 0;
                    for (t = 0; t < *(nbl + 2); t++)
                    {
                        for (i = 0; i < *(nbl + 4); i++)
                        {
                            // Extract weights from Gaussian distribution with mean 0.0 and std 1.0
                            cgenotype[j] = netRng->getGaussian(1.0, 0.0);
                            // Update square sum of weights
                            wSqSum[t] += (cgenotype[j] * cgenotype[j]);
                            j++;
                            // Update counter of weights
                            cnt++;
                        }
                    }
                    // Normalize weights
                    j = k;
                    t = 0;
                    i = 0;
                    while (j < (k + cnt))
                    {
                        cgenotype[j] *= (range / sqrt(wSqSum[t])); // Normalization factor
                        j++;
                        i++;
                        if (i % *(nbl + 4) == 0)
                            // Move to next sum
                            t++;
                    }
                    // We delete the pointer
                    delete wSqSum;
                    break;
                }
                // uniform
                case 2:
                {
                    // the range is specified manually and is the same for all layers
                    range = netWrange;
                    for (t = 0; t < *(nbl + 2); t++)
                    {
                        for (i = 0; i < *(nbl + 4); i++)
                        {
                            cgenotype[j] = netRng->getDouble(-range, range); // Values drawn from interval [-range, range]
                            j++;
                        }
                    }
                    break;
                }
                default:
                    // Code should never enter here!!!
                    printf("Invalid initializer with id %d!!!\n", m_wInit);
                    break;
            }
        }
        nbl = (nbl + 5);
    }
}

// set the normalization phase (0 = do nothing, 1 = collect data to update normalization vectors)
void Evonet::normPhase(int phase)
{
    m_normPhase = phase;
}

// collect data for normalization
void Evonet::collectNormalizationData()
{
    int i;
    // Update both observation's sum and observation's squared sum
    for (i = 0; i < m_ninputs; i++)
    {
        m_sum[i] += cobservation[i];
        m_sumsq[i] += (cobservation[i] * cobservation[i]);
    }
    // Update counter
    m_count++;
}

// compute normalization vectors
void Evonet::updateNormalizationVectors()
{
    int i;
    int ii;
    double cStd;
	
    for (i = 0; i < m_ninputs; i++)
    {
        // Compute new mean
        m_mean[i] = m_sum[i] / m_count;
        // Compute new standard deviation
        cStd = (m_sumsq[i] / m_count - (m_mean[i] * m_mean[i]));
        if (cStd < 0.01)
            // Minimum variance is 0.01 --> minimum standard deviation is sqrt(0.01) = 0.1
            cStd = 0.01;
        m_std[i] = sqrt(cStd);
    }
    // copy normalization vectors on the cnormalization vector that is used for saving data
    ii = 0;
    for (i = 0; i < m_ninputs; i++)
    {
        cnormalization[ii] = m_mean[i];
        ii++;
    }
    for (i = 0; i < m_ninputs; i++)
    {
        cnormalization[ii] = m_std[i];
        ii++;
    }
}

// reset vectors for normalization
void Evonet::resetNormalizationVectors()
{
    int i;
    for (i = 0; i < m_ninputs; i++)
    {
        m_mean[i] = 0.0;
        m_std[i] = 1.0;
        m_sum[i] = 0.0;
        m_sumsq[i] = 0.01;
    }
    m_count = 0.01;
}

// restore a loaded normalization vector
void Evonet::setNormalizationVectors()
{

    int i;
    int ii;
	
    if (m_normalize == 1)
    {
        ii = 0;
        for (i = 0; i < m_ninputs; i++)
        {
            m_mean[i] = cnormalization[ii];
            ii++;
        }
        for (i = 0; i < m_ninputs; i++)
        {
            m_std[i] = cnormalization[ii];
            ii++;
        }
    }
}
