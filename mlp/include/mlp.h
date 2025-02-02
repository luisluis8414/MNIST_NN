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
    MLP(int inputSize, int hiddenSize, int outputSize, double learningRate = 0.1);

    // Forward pass
    std::vector<double> forward(const std::vector<double> &inputs);

    // Training
    void train(const std::vector<double> &inputs, const std::vector<double> &targets);
    void startTraining(const std::vector<std::vector<double>> &trainingInputs,
                       const std::vector<std::vector<double>> &trainingTargets,
                       int epochs, int patience = 5, double minimalImprovement = 0.0001);

    // Model saving/loading
    void saveModel(const std::string &filename);
    void loadModel(const std::string &filename);

private:
    // PIMPL: forward-declare the internal implementation structure.
    struct Layers;
    Layers *m_Layers;

    // Helper functions
    std::vector<double> computeLayerOutput(const std::vector<Perceptron> &layer,
                                           const std::vector<double> &inputs);
    double meanSquaredError(const std::vector<double> &outputs,
                            const std::vector<double> &targets);
};
