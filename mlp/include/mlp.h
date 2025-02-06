#pragma once

#include <vector>
#include <string>
#include "Perceptron.h"

#ifdef _WIN32
#ifdef MLP_EXPORT
#define MLP_API __declspec(dllexport)
#else
#define MLP_API __declspec(dllimport)
#endif
#else
#define MLP_API
#endif

class MLP_API MLP
{
public:
    /**
     * Constructor for a network with multiple hidden layers.
     * @param inputSize Number of input neurons.
     * @param hiddenSizes A vector containing the size of each hidden layer.
     * @param outputSize Number of output neurons.
     * @param learningRate Learning rate for training.
     */
    MLP(int inputSize, const std::vector<int> &hiddenSizes,
        int outputSize, double learningRate = 0.1);

    // Forward pass: returns the network output for given inputs.
    std::vector<double> forward(const std::vector<double> &inputs);

    // Training with early stopping.
    void startTraining(const std::vector<std::vector<double>> &trainingInputs,
                       const std::vector<std::vector<double>> &trainingTargets,
                       int epochs, int patience = 5,
                       double minimalImprovement = 0.0001);

    // Save and load the model.
    void saveModel(const std::string &filename);
    void loadModel(const std::string &filename);

private:
    // PIMPL–style internal implementation.
    struct Layers;
    Layers *m_Layers;

    // A backpropagation training step.
    void train(const std::vector<double> &inputs,
               const std::vector<double> &targets);

    // Compute the output of a vector of Perceptron (a layer), given the input.
    std::vector<double> computeLayerOutput(const std::vector<Perceptron> &layer,
                                           const std::vector<double> &inputs);
};
