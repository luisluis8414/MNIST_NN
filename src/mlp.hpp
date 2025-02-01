#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <sstream>
#include <algorithm>
#include <limits>
#include "Perceptron.hpp"

class MLP
{
private:
    // We'll assume the inner layer is not used.
    std::vector<Perceptron> m_hiddenLayer;
    std::vector<Perceptron> m_outerLayer;

public:
    MLP(int inputSize, int hiddenSize, int outputSize, double learningRate = 0.1)
    {
        // Construct hidden layer: each perceptron has "inputSize" weights.
        for (int i = 0; i < hiddenSize; i++)
        {
            m_hiddenLayer.emplace_back(inputSize, learningRate);
        }
        // Construct outer layer: each perceptron has "hiddenSize" weights.
        for (int i = 0; i < outputSize; i++)
        {
            m_outerLayer.emplace_back(hiddenSize, learningRate);
        }
    }

    // Default constructor for loading
    MLP() {}

    // Calculate outputs of one layer.
    std::vector<double>
    computeLayerOutput(const std::vector<Perceptron> &layer,
                       const std::vector<double> &inputs)
    {
        if (layer.empty())
        {
            throw std::runtime_error("Layer is empty.");
        }
        // layer[0] because weights size of one perceptron is constant in the layer.
        if (inputs.size() != layer[0].getWeights().size())
        {
            throw std::invalid_argument(
                "Size of inputs doesn't match perceptron input size");
        }

        std::vector<double> outputs(layer.size(), 0.0);
        for (size_t i = 0; i < layer.size(); i++)
        {
            outputs[i] = layer[i].calcOutput(inputs);
        }
        return outputs;
    }

    // Forward pass.
    std::vector<double> forward(const std::vector<double> &inputs)
    {
        std::vector<double> hiddenOutputs = computeLayerOutput(m_hiddenLayer, inputs);
        std::vector<double> outerOutputs =
            computeLayerOutput(m_outerLayer, hiddenOutputs);
        return outerOutputs;
    }

    double meanSquaredError(const std::vector<double> &outputs,
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

    // Backpropagation training step.
    void train(const std::vector<double> &inputs,
               const std::vector<double> &targets)
    {
        std::vector<double> hiddenOutputs = computeLayerOutput(m_hiddenLayer, inputs);
        std::vector<double> outerOutputs =
            computeLayerOutput(m_outerLayer, hiddenOutputs);

        // Compute output layer error gradients.
        std::vector<double> outputDeltas(m_outerLayer.size());
        for (size_t i = 0; i < m_outerLayer.size(); i++)
        {
            double error = outerOutputs[i] - targets[i];
            double derivative = outerOutputs[i] * (1.0 - outerOutputs[i]); // Sigmoid derivative.
            outputDeltas[i] = error * derivative;
        }

        // Update outer layer weights.
        for (size_t i = 0; i < m_outerLayer.size(); i++)
        {
            m_outerLayer[i].updateWeights(hiddenOutputs, outputDeltas[i]);
        }

        // Compute hidden layer gradients.
        std::vector<double> hiddenDeltas(m_hiddenLayer.size());
        for (size_t i = 0; i < m_hiddenLayer.size(); i++)
        {
            double error = 0.0;
            for (size_t j = 0; j < m_outerLayer.size(); j++)
            {
                error += outputDeltas[j] * m_outerLayer[j].getWeights()[i];
            }
            double derivative = hiddenOutputs[i] * (1.0 - hiddenOutputs[i]);
            hiddenDeltas[i] = error * derivative;
        }
        // Update hidden layer weights.
        for (size_t i = 0; i < m_hiddenLayer.size(); i++)
        {
            m_hiddenLayer[i].updateWeights(inputs, hiddenDeltas[i]);
        }
    }

    // patience: number of epochs to wait for improvement
    // minimalImprovement: minimal difference in MSE to count as an improvement.
    void startTraining(const std::vector<std::vector<double>> &trainingInputs,
                       const std::vector<std::vector<double>> &trainingTargets,
                       int epochs,
                       int patience = 5,
                       double minimalImprovement = 0.0001)
    {
        double bestMSE = std::numeric_limits<double>::max();
        int epochsWithoutImprovement = 0;

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalMSE = 0.0;
            int correctPredictions = 0;
            // Process each training sample.
            for (size_t i = 0; i < trainingInputs.size(); i++)
            {
                // Training step.
                train(trainingInputs[i], trainingTargets[i]);

                // Forward pass for metrics.
                std::vector<double> hiddenOutputs =
                    computeLayerOutput(m_hiddenLayer, trainingInputs[i]);
                std::vector<double> outerOutputs =
                    computeLayerOutput(m_outerLayer, hiddenOutputs);

                totalMSE += meanSquaredError(outerOutputs, trainingTargets[i]);

                // Calculate accuracy assuming one-hot encoded targets.
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

            // Determine if there's any improvement.
            if (avgMSE < bestMSE - minimalImprovement)
            {
                bestMSE = avgMSE;
                epochsWithoutImprovement = 0;
            }
            else
            {
                std::cout << "no improvment this epoch" << std::endl;
                epochsWithoutImprovement++;
            }

            // Early stopping triggered if improvement has stalled.
            if (epochsWithoutImprovement >= patience)
            {
                std::cout << "Early stopping triggered after " << epoch + 1
                          << " epochs." << std::endl;
                break;
            }
        }
    }

    // Save the entire model (hidden and outer layers) to a binary file.
    void saveModel(const std::string &filename)
    {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs)
        {
            throw std::runtime_error("Unable to open file for saving: " + filename);
        }

        // Save number of perceptrons in the hidden layer.
        size_t hiddenSize = m_hiddenLayer.size();
        ofs.write(reinterpret_cast<const char *>(&hiddenSize), sizeof(hiddenSize));
        // Save each perceptron in the hidden layer.
        for (const auto &perceptron : m_hiddenLayer)
        {
            perceptron.save(ofs);
        }

        // Save number of perceptrons in the outer layer.
        size_t outerSize = m_outerLayer.size();
        ofs.write(reinterpret_cast<const char *>(&outerSize), sizeof(outerSize));
        // Save each perceptron in the outer layer.
        for (const auto &perceptron : m_outerLayer)
        {
            perceptron.save(ofs);
        }
        ofs.close();
    }

    // Load the entire model (hidden and outer layers) from a binary file.
    void loadModel(const std::string &filename)
    {
        std::ifstream ifs(filename, std::ios::binary);
        if (!ifs)
        {
            throw std::runtime_error("Unable to open file for loading: " + filename);
        }

        // Load hidden layer.
        size_t hiddenSize;
        ifs.read(reinterpret_cast<char *>(&hiddenSize), sizeof(hiddenSize));
        m_hiddenLayer.resize(hiddenSize);
        for (size_t i = 0; i < hiddenSize; i++)
        {
            m_hiddenLayer[i].load(ifs);
        }

        // Load outer layer.
        size_t outerSize;
        ifs.read(reinterpret_cast<char *>(&outerSize), sizeof(outerSize));
        m_outerLayer.resize(outerSize);
        for (size_t i = 0; i < outerSize; i++)
        {
            m_outerLayer[i].load(ifs);
        }
        ifs.close();
    }
};
