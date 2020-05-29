cdef extern from "utilities.cpp":
    pass

cdef extern from "evonet.cpp":
    pass

# Declare the class with cdef
cdef extern from "evonet.h":
    cdef cppclass Evonet:
        Evonet() except +
        Evonet(int, int, int, int, int, int, int, int, int, int, int, int, double, int, double, double) except +
        int m_ninputs, m_nhiddens, m_noutputs, m_nneurons, m_nlayers, m_netType, m_actFunct, m_bias, m_outType, m_wInit, m_clip, m_normalize, m_randAct, m_nbins, m_normPhase
        double m_low
        double m_high
        double* m_netinput
        void resetNet()
        void seed(int s)
        void copyGenotype(double* genotype)
        void copyInput(double* input)
        void copyOutput(double* output)
        void copyNeuronact(double* na)
        void copyNormalization(double* no)
        void updateNet()
        void getAction(double* output)
        int computeParameters()
        void initWeights()
        void normPhase(int phase)
        void updateNormalizationVectors()
        void setNormalizationVectors()
        void resetNormalizationVectors()

