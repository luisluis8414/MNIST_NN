#include "../include/mlp.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <limits>

struct MLP::Layers
{
    std::vector<Perceptron> hiddenLayer;
    std::vector<Perceptron> outerLayer;
};

// Constructor: Initialize the MLP with the given sizes and learning rate
MLP::MLP(int inputSize, int hiddenSize, int outputSize, double learningRate) : m_Layers(new Layers())
{
    for (int i = 0; i < hiddenSize; i++)
    {
        m_Layers->hiddenLayer.emplace_back(inputSize, learningRate);
    }
    for (int i = 0; i < outputSize; i++)
    {
        m_Layers->outerLayer.emplace_back(hiddenSize, learningRate);
    }
}

// Compute the output of a single layer
std::vector<double> MLP::computeLayerOutput(const std::vector<Perceptron> &layer,
                                            const std::vector<double> &inputs)
{
    if (layer.empty())
    {
        throw std::runtime_error("Layer is empty.");
    }
    if (inputs.size() != layer[0].getWeights().size())
    {
        throw std::invalid_argument("Size of inputs doesn't match perceptron input size");
    }

    std::vector<double> outputs(layer.size(), 0.0);
    for (size_t i = 0; i < layer.size(); i++)
    {
        outputs[i] = layer[i].calcOutput(inputs);
    }
    return outputs;
}

// Forward pass
std::vector<double> MLP::forward(const std::vector<double> &inputs)
{
    std::vector<double> hiddenOutputs = computeLayerOutput(m_Layers->hiddenLayer, inputs);
    return computeLayerOutput(m_Layers->outerLayer, hiddenOutputs);
}

// Mean squared error calculation
double MLP::meanSquaredError(const std::vector<double> &outputs,
                             const std::vector<double> &targets)
{
    if (outputs.size() != targets.size())
    {
        throw std::invalid_argument("outputs size doesn't match targets size");
    }
    double error = 0.0;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        error += std::pow(outputs[i] - targets[i], 2);
    }
    return error / outputs.size();
}

// Backpropagation training step
void MLP::train(const std::vector<double> &inputs, const std::vector<double> &targets)
{
    std::vector<double> hiddenOutputs = computeLayerOutput(m_Layers->hiddenLayer, inputs);
    std::vector<double> outerOutputs = computeLayerOutput(m_Layers->outerLayer, hiddenOutputs);

    // Compute output layer error gradients
    std::vector<double> outputDeltas(m_Layers->outerLayer.size());
    for (size_t i = 0; i < m_Layers->outerLayer.size(); i++)
    {
        double error = outerOutputs[i] - targets[i];
        double derivative = outerOutputs[i] * (1.0 - outerOutputs[i]); // Sigmoid derivative
        outputDeltas[i] = error * derivative;
    }

    // Update outer layer weights
    for (size_t i = 0; i < m_Layers->outerLayer.size(); i++)
    {
        m_Layers->outerLayer[i].updateWeights(hiddenOutputs, outputDeltas[i]);
    }

    // Compute hidden layer gradients
    std::vector<double> hiddenDeltas(m_Layers->hiddenLayer.size());
    for (size_t i = 0; i < m_Layers->hiddenLayer.size(); i++)
    {
        double error = 0.0;
        for (size_t j = 0; j < m_Layers->outerLayer.size(); j++)
        {
            error += outputDeltas[j] * m_Layers->outerLayer[j].getWeights()[i];
        }
        double derivative = hiddenOutputs[i] * (1.0 - hiddenOutputs[i]);
        hiddenDeltas[i] = error * derivative;
    }

    // Update hidden layer weights
    for (size_t i = 0; i < m_Layers->hiddenLayer.size(); i++)
    {
        m_Layers->hiddenLayer[i].updateWeights(inputs, hiddenDeltas[i]);
    }
}

// Training loop with early stopping
void MLP::startTraining(const std::vector<std::vector<double>> &trainingInputs,
                        const std::vector<std::vector<double>> &trainingTargets,
                        int epochs, int patience, double minimalImprovement)
{
    double bestMSE = std::numeric_limits<double>::max();
    int epochsWithoutImprovement = 0;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double totalMSE = 0.0;
        int correctPredictions = 0;

        for (size_t i = 0; i < trainingInputs.size(); i++)
        {
            train(trainingInputs[i], trainingTargets[i]);

            std::vector<double> hiddenOutputs = computeLayerOutput(m_Layers->hiddenLayer, trainingInputs[i]);
            std::vector<double> outerOutputs = computeLayerOutput(m_Layers->outerLayer, hiddenOutputs);

            totalMSE += meanSquaredError(outerOutputs, trainingTargets[i]);

            size_t predictedLabel = std::distance(
                outerOutputs.begin(),
                std::max_element(outerOutputs.begin(), outerOutputs.end()));
            size_t targetLabel = std::distance(
                trainingTargets[i].begin(),
                std::max_element(trainingTargets[i].begin(), trainingTargets[i].end()));
            if (predictedLabel == targetLabel)
                correctPredictions++;
        }

        double avgMSE = totalMSE / trainingInputs.size();
        double accuracy = static_cast<double>(correctPredictions) / trainingInputs.size();

        std::cout << "Epoch " << epoch + 1 << " - Average MSE: " << avgMSE
                  << ", Accuracy: " << (accuracy * 100.0) << "%" << '\n';

        if (avgMSE < bestMSE - minimalImprovement)
        {
            bestMSE = avgMSE;
            epochsWithoutImprovement = 0;
        }
        else
        {
            epochsWithoutImprovement++;
        }

        if (epochsWithoutImprovement >= patience)
        {
            std::cout << "Early stopping triggered after " << epoch + 1
                      << " epochs." << std::endl;
            break;
        }
    }
}

// Save the model to a file
void MLP::saveModel(const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
    {
        throw std::runtime_error("Unable to open file for saving: " + filename);
    }

    size_t hiddenSize = m_Layers->hiddenLayer.size();
    ofs.write(reinterpret_cast<const char *>(&hiddenSize), sizeof(hiddenSize));
    for (const auto &perceptron : m_Layers->hiddenLayer)
    {
        perceptron.save(ofs);
    }

    size_t outerSize = m_Layers->outerLayer.size();
    ofs.write(reinterpret_cast<const char *>(&outerSize), sizeof(outerSize));
    for (const auto &perceptron : m_Layers->outerLayer)
    {
        perceptron.save(ofs);
    }
    ofs.close();
}

// Load the model from a file
void MLP::loadModel(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
    {
        throw std::runtime_error("Unable to open file for loading: " + filename);
    }

    size_t hiddenSize;
    ifs.read(reinterpret_cast<char *>(&hiddenSize), sizeof(hiddenSize));
    m_Layers->hiddenLayer.resize(hiddenSize);
    for (size_t i = 0; i < hiddenSize; i++)
    {
        m_Layers->hiddenLayer[i].load(ifs);
    }

    size_t outerSize;
    ifs.read(reinterpret_cast<char *>(&outerSize), sizeof(outerSize));
    m_Layers->outerLayer.resize(outerSize);
    for (size_t i = 0; i < outerSize; i++)
    {
        m_Layers->outerLayer[i].load(ifs);
    }
    ifs.close();
}
