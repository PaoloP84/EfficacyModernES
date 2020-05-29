#ifndef EVONET_H
#define EVONET_H

class Evonet
{

public:
    // Void constructor
    Evonet();
    // Other constructor
    Evonet(int ninputs, int nhiddens, int noutputs, int nlayers=1, int bias=0, int netType=0, int actFunct=2, int outType=2, int wInit=0, int clip=0, int normalize=0, int randAct=0, double wrange=1.0, int nbins=1, double low=-1.0, double high=1.0);
    // Destructor
    ~Evonet();
    // Init network from file
    void initNet(char* filename);
    // set the seed
    void seed(int s);
    // Reset network
    void resetNet();
    // Copy genotype pointer
    void copyGenotype(double* genotype);
    // Copy input pointer
    void copyInput(double* input);
    // Copy output pointer
    void copyOutput(double* output);
    // Copy neuro activation pointer
    void copyNeuronact(double* na);
    // Copy normalization vector pointer
    void copyNormalization(double* no);
    // Activate network
    void updateNet();
    // Get the output(s)
    void getAction(double* output);
    // Init network architecture
    void initNetArchitecture();
    // Get the number of parameters
    int computeParameters();
    // Initialize weights
    void initWeights();
    // set the normalization phase (0 = do nothing, 1 = collect data to update normalization vectors)
    void normPhase(int phase);
    // update normalization vectors
    void updateNormalizationVectors();
    // reset normalization vectors
    void resetNormalizationVectors();
    // retrive normalization vectors from free parameters
    void setNormalizationVectors();

    // Number of inputs
    int m_ninputs;
    // Number of hiddens
    int m_nhiddens;
    // Number of outputs
    int m_noutputs;
    // Number of neurons
    int m_nneurons;
    // Number of layers
    int m_nlayers;
    // Bias
    int m_bias;
    // Activation of the neurons
    double* m_act;
    // Net-input of the neurons
    double* m_netinput;
    // Network type: 0 = 'ff'; 1 = 'rec'; 2 = 'fully-rec'
    int m_netType;
    // Activation function type: 1 = 'logistic'; 2 = 'tanh'; 3 when <outType> flag is set to 1
    int m_actFunct;
    // Output type: 1 = logistic; 2 = tanh; 3 = linear; 4 = binary; 5 = bins
    int m_outType;
    // Weight initializer: 0 = 'xavier'; 1 = 'normc'
    int m_wInit;
    // Clip values flag
    int m_clip;
    // Normalize input flag
    int m_normalize;
    // Random actions
    int m_randAct;
    // Number of bins
    int m_nbins;
    // Minimum value for action
    double m_low;
    // Maximum value for action
    double m_high;
    // Network architecture (block structure)
    int* m_netblock;
    // Number of blocks
    int m_nblocks;
    // Type of neurons: 0 = input; 1 = logistic; 2 = tanh; 3 = linear; 4 = binary; >4 = bins
    int* m_neurontype;
    // Normalization phase: 0 = no normalization; 1 = normalization
    int m_normPhase;

private:
    // Get the output(s) for all cases but uniformly spaced bins
    void getOutput(double* output);
    // Get the output(s) for uniformly spaced bins
    void getOutputBins(double* output);
    // collect normalization data
    void collectNormalizationData();

    // normalization mean
    double* m_mean;
    // normalization stdv
    double* m_std;
    // normalization sum
    double* m_sum;
    // normalization squared sum
    double* m_sumsq;
    // normalization data number
    double m_count;
};

#endif
