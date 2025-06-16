#include "../include/mlp.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <limits>

// The Layers structure now contains multiple hidden layers.
struct MLP::Layers
{
    // Each hidden layer is represented as a vector of Perceptron.
    std::vector<std::vector<Perceptron>> hiddenLayers;
    // Outer (output) layer.
    std::vector<Perceptron> outerLayer;
};

// Constructor: builds the network from input -> (multiple hidden layers) -> output.
MLP::MLP(int inputSize, const std::vector<int> &hiddenSizes,
         int outputSize, double learningRate)
    : m_Layers(new Layers())
{
    int previousSize = inputSize;
    // Create each hidden layer.
    for (int size : hiddenSizes)
    {
        std::vector<Perceptron> layer;
        layer.reserve(size);
        for (int i = 0; i < size; i++)
        {
            layer.emplace_back(previousSize, learningRate);
        }
        m_Layers->hiddenLayers.push_back(layer);
        previousSize = size;
    }
    // Create the output (outer) layer.
    for (int i = 0; i < outputSize; i++)
    {
        m_Layers->outerLayer.emplace_back(previousSize, learningRate);
    }
}

// Helper: calculates the output of a single layer.
std::vector<double>
MLP::computeLayerOutput(const std::vector<Perceptron> &layer,
                        const std::vector<double> &inputs,
                        bool skipActivation)
{
    if (layer.empty())
    {
        throw std::runtime_error("Layer is empty.");
    }
    if (inputs.size() != layer[0].getWeights().size())
    {
        throw std::invalid_argument(
            "Size of inputs doesn't match perceptron input size");
    }

    std::vector<double> outputs(layer.size(), 0.0);
    for (size_t i = 0; i < layer.size(); i++)
    {
        outputs[i] = layer[i].calcOutputRaw(inputs); // Get raw output
        if (!skipActivation)
        {
            outputs[i] = 1.0 / (1.0 + std::exp(-outputs[i])); // Apply sigmoid only for hidden layers
        }
    }
    return outputs;
}

// Helper: applies softmax to a vector of values
std::vector<double> MLP::applySoftmax(const std::vector<double> &inputs)
{
    std::vector<double> output(inputs.size());

    // Find max for numerical stability
    double maxVal = *std::max_element(inputs.begin(), inputs.end());

    // Calculate exp(x - max) and sum
    double sum = 0.0;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        output[i] = std::exp(inputs[i] - maxVal);
        sum += output[i];
    }

    // Normalize
    for (size_t i = 0; i < inputs.size(); i++)
    {
        output[i] /= sum;
    }

    return output;
}

// Forward pass: propagate input through every hidden layer then the output layer.
std::vector<double> MLP::forward(const std::vector<double> &inputs)
{
    std::vector<double> activations = inputs;
    // Pass through each hidden layer with sigmoid activation
    for (const auto &hiddenLayer : m_Layers->hiddenLayers)
    {
        activations = computeLayerOutput(hiddenLayer, activations, false);
    }
    // Pass through the output layer without activation, then apply softmax
    return applySoftmax(computeLayerOutput(m_Layers->outerLayer, activations, true));
}

// Training step: performs forward propagation (storing all activations)
// and then backward propagation updating weights for all layers.
void MLP::train(const std::vector<double> &inputs,
                const std::vector<double> &targets)
{
    // Store activations for each layer; index 0 holds the network input.
    std::vector<std::vector<double>> layerActivations;
    layerActivations.push_back(inputs);

    // Forward pass through all hidden layers with sigmoid
    for (const auto &hiddenLayer : m_Layers->hiddenLayers)
    {
        std::vector<double> activation =
            computeLayerOutput(hiddenLayer, layerActivations.back(), false);
        layerActivations.push_back(activation);
    }

    // Compute raw outputs and softmax for the output layer
    std::vector<double> rawOutputs = computeLayerOutput(m_Layers->outerLayer, layerActivations.back(), true);
    std::vector<double> softmaxOutputs = applySoftmax(rawOutputs);

    // Calculate deltas for the output layer using softmax derivative
    std::vector<double> outputDeltas(m_Layers->outerLayer.size());
    for (size_t i = 0; i < m_Layers->outerLayer.size(); i++)
    {
        // For softmax + cross-entropy loss, the gradient simplifies to (output - target)
        outputDeltas[i] = softmaxOutputs[i] - targets[i];
    }

    // Update weights for the output layer
    for (size_t i = 0; i < m_Layers->outerLayer.size(); i++)
    {
        m_Layers->outerLayer[i].updateWeights(layerActivations.back(),
                                              outputDeltas[i]);
    }

    // Propagate error backwards through the hidden layers using sigmoid derivative
    std::vector<double> nextDeltas = outputDeltas;
    for (int layerIndex = static_cast<int>(m_Layers->hiddenLayers.size()) - 1;
         layerIndex >= 0; layerIndex--)
    {
        std::vector<Perceptron> &currentLayer =
            m_Layers->hiddenLayers[layerIndex];
        std::vector<double> &currentActivations =
            layerActivations[layerIndex + 1];

        std::vector<double> currentDeltas(currentLayer.size());
        for (size_t i = 0; i < currentLayer.size(); i++)
        {
            double error = 0.0;
            if (layerIndex == static_cast<int>(m_Layers->hiddenLayers.size()) - 1)
            {
                // Next layer is the output layer
                for (size_t k = 0; k < m_Layers->outerLayer.size(); k++)
                {
                    error += m_Layers->outerLayer[k].getWeights()[i] *
                             nextDeltas[k];
                }
            }
            else
            {
                // Next layer is another hidden layer
                for (size_t k = 0; k < m_Layers->hiddenLayers[layerIndex + 1].size(); k++)
                {
                    error += m_Layers->hiddenLayers[layerIndex + 1][k]
                                 .getWeights()[i] *
                             nextDeltas[k];
                }
            }
            // Use sigmoid derivative for hidden layers
            double derivative = currentActivations[i] * (1.0 - currentActivations[i]);
            currentDeltas[i] = error * derivative;
        }

        // Update weights for the current hidden layer
        for (size_t i = 0; i < currentLayer.size(); i++)
        {
            currentLayer[i].updateWeights(layerActivations[layerIndex],
                                          currentDeltas[i]);
        }
        nextDeltas = currentDeltas;
    }
}

// A helper for computing mean squared error over one training example.
double meanSquaredError(const std::vector<double> &outputs,
                        const std::vector<double> &targets)
{
    if (outputs.size() != targets.size())
    {
        throw std::invalid_argument(
            "Output size doesn't match targets size");
    }
    double error = 0.0;
    for (size_t i = 0; i < outputs.size(); i++)
    {
        error += std::pow(outputs[i] - targets[i], 2);
    }
    return error / outputs.size();
}

// Training loop with early stopping based on minimal improvement.
void MLP::startTraining(const std::vector<std::vector<double>> &trainingInputs,
                        const std::vector<std::vector<double>> &trainingTargets,
                        int epochs, int patience,
                        double minimalImprovement)
{
    double bestMSE = std::numeric_limits<double>::max();
    int epochsWithoutImprovement = 0;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        double totalMSE = 0.0;

        for (size_t i = 0; i < trainingInputs.size(); i++)
        {
            train(trainingInputs[i], trainingTargets[i]);

            // Compute output for MSE calculation.
            std::vector<double> output = forward(trainingInputs[i]);
            totalMSE += meanSquaredError(output, trainingTargets[i]);
        }

        double avgMSE = totalMSE / trainingInputs.size();
        std::cout << "Epoch " << epoch + 1 << " - Average MSE: " << avgMSE << '\n';

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

// Save the network model to a file in binary format.
void MLP::saveModel(const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
    {
        throw std::runtime_error("Unable to open file for saving: " + filename);
    }

    // Save the number and configuration of hidden layers.
    size_t numHiddenLayers = m_Layers->hiddenLayers.size();
    ofs.write(reinterpret_cast<const char *>(&numHiddenLayers),
              sizeof(numHiddenLayers));
    for (const auto &hiddenLayer : m_Layers->hiddenLayers)
    {
        size_t layerSize = hiddenLayer.size();
        ofs.write(reinterpret_cast<const char *>(&layerSize),
                  sizeof(layerSize));
        for (const Perceptron &perceptron : hiddenLayer)
        {
            perceptron.save(ofs);
        }
    }

    // Save the outer (output) layer.
    size_t outerSize = m_Layers->outerLayer.size();
    ofs.write(reinterpret_cast<const char *>(&outerSize), sizeof(outerSize));
    for (const Perceptron &perceptron : m_Layers->outerLayer)
    {
        perceptron.save(ofs);
    }
    ofs.close();
}

// Load a saved network model from a file.
void MLP::loadModel(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
    {
        throw std::runtime_error("Unable to open file for loading: " + filename);
    }

    // Load hidden layers.
    size_t numHiddenLayers;
    ifs.read(reinterpret_cast<char *>(&numHiddenLayers),
             sizeof(numHiddenLayers));
    m_Layers->hiddenLayers.resize(numHiddenLayers);
    for (size_t i = 0; i < numHiddenLayers; i++)
    {
        size_t layerSize;
        ifs.read(reinterpret_cast<char *>(&layerSize), sizeof(layerSize));
        m_Layers->hiddenLayers[i].resize(layerSize);
        for (size_t j = 0; j < layerSize; j++)
        {
            m_Layers->hiddenLayers[i][j].load(ifs);
        }
    }

    // Load outer (output) layer.
    size_t outerSize;
    ifs.read(reinterpret_cast<char *>(&outerSize), sizeof(outerSize));
    m_Layers->outerLayer.resize(outerSize);
    for (size_t i = 0; i < outerSize; i++)
    {
        m_Layers->outerLayer[i].load(ifs);
    }
    ifs.close();
}
