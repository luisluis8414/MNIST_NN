#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "csv_reader.hpp"
#include "../mlp/include/mlp.h"

//============================================================================
// Parameters
//============================================================================
const double LEARNING_RATE = 0.01;
const int EPOCHS = 100;
const int TRAINING_SAMPLES = 60000;
const int HIDDEN_NEURONS_LAYER1 = 128;
const int HIDDEN_NEURONS_LAYER2 = 64;

// Input and output sizes for the MNIST dataset
const int INPUT_SIZE = 784; // 28x28 pixels
const int OUTPUT_SIZE = 10; // Digits 0 to 9

//============================================================================
// Helper Functions
//============================================================================

// One-hot encode an integer label (assumes classes 0 to 9)
std::vector<double> oneHotEncode(int label, int numClasses = OUTPUT_SIZE)
{
    std::vector<double> encoded(numClasses, 0.0);
    if (label >= 0 && label < numClasses)
    {
        encoded[label] = 1.0;
    }
    else
    {
        throw std::runtime_error("Label out of range for one-hot encoding.");
    }
    return encoded;
}

// Convert vector of char pixel values [0,255] to normalized double values [0,1]
std::vector<double> normalizePixels(const std::vector<unsigned char> &pixels)
{
    std::vector<double> normalized;
    normalized.reserve(pixels.size());
    for (uchar p : pixels)
    {
        normalized.push_back(static_cast<double>(p) / 255.0);
    }
    return normalized;
}

// Build model file path based on parameters
std::string buildModelPath()
{
    std::ostringstream oss;
    oss << "models/model_"
        << LEARNING_RATE << "_"
        << EPOCHS << "_"
        << TRAINING_SAMPLES << "_"
        << HIDDEN_NEURONS_LAYER1 << "_"
        << HIDDEN_NEURONS_LAYER2;
    return oss.str();
}

//============================================================================
// Training Functionality
//============================================================================

void train()
{
    std::string csvTrainingFile = "resources/training_data/mnist_train.csv";
    std::string csvTestingFile = "resources/training_data/mnist_test.csv";

    CsvReader trainReader(csvTrainingFile);
    std::vector<std::vector<double>> trainingInputs;
    std::vector<std::vector<double>> trainingTargets;

    for (int i = 0; i < TRAINING_SAMPLES && !trainReader.eof(); ++i)
    {
        auto [label, pixels] = trainReader.getLabelAndPixels();
        trainingInputs.push_back(normalizePixels(pixels));
        trainingTargets.push_back(oneHotEncode(label));
    }

    // two hidden layers
    std::vector<int> hiddenLayers = {HIDDEN_NEURONS_LAYER1, HIDDEN_NEURONS_LAYER2};

    // create mlp
    MLP mlp(INPUT_SIZE, hiddenLayers, OUTPUT_SIZE, LEARNING_RATE);

    std::cout << "Starting training for " << EPOCHS << " epochs on "
              << TRAINING_SAMPLES << " samples." << std::endl;
    mlp.startTraining(trainingInputs, trainingTargets, EPOCHS);
    std::cout << "Training completed." << std::endl;

    std::string modelPath = buildModelPath();
    mlp.saveModel(modelPath);
    std::cout << "Model saved to: " << modelPath << std::endl;
}

//============================================================================
// Evaluation Functionality
//============================================================================

void loadModel(std::string modelPath = buildModelPath())
{
    std::string csvTestingFile = "resources/training_data/mnist_test.csv";

    // two hidden layers
    std::vector<int> hiddenLayers = {HIDDEN_NEURONS_LAYER1, HIDDEN_NEURONS_LAYER2};

    // Note: The learning rate here is not used in inference.
    MLP mlp(INPUT_SIZE, hiddenLayers, OUTPUT_SIZE, 0.01);

    mlp.loadModel(modelPath);
    std::cout << "Model loaded successfully from file: " << modelPath
              << std::endl;

    CsvReader testReader(csvTestingFile);
    std::cout << "\n----- Testing on 20 samples -----" << std::endl;

    const int testSamples = 20;
    for (int i = 0; i < testSamples && !testReader.eof(); ++i)
    {
        auto [testLabel, testPixels] = testReader.getLabelAndPixels();
        std::vector<double> testInput = normalizePixels(testPixels);
        auto output = mlp.forward(testInput);

        std::cout << "\nSample " << i + 1 << ":" << std::endl;
        std::cout << "Expected Label: " << testLabel << std::endl;
        std::cout << "MLP Output: { ";
        for (const auto &val : output)
        {
            std::cout << val << " ";
        }
        std::cout << "}" << std::endl;
    }
}

//============================================================================
// Main Entry
//============================================================================

int main()
{
    try
    {
        // Uncomment the following line to train a new model.
        // train();

        loadModel("models/model_0.01_100_60000_128_64");
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
