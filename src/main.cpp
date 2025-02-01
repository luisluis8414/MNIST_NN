#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>

#include <opencv2/opencv.hpp>

#include "csv_reader.hpp"
#include "perceptron.hpp"
#include "mlp.hpp"

//============================================================================
// Parameters
//============================================================================
const double LEARNING_RATE = 0.05;
const int EPOCHS = 100;
const int TRAINING_SAMPLES = 10000;
const int HIDDEN_NEURONS = 128;

// Input and output sizes for the MNIST dataset
const int INPUT_SIZE = 784;
// Numbers 0 to 9
const int OUTPUT_SIZE = 10;

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
        << HIDDEN_NEURONS;
    return oss.str();
}

//============================================================================
// Training Functionality
//============================================================================

void train()
{
    // Paths to CSV files (update with your own paths)
    std::string csvTrainingFile = "resources/training_data/mnist_train.csv";
    std::string csvTestingFile = "resources/training_data/mnist_test.csv";

    // Open the training CSV file
    CsvReader trainReader(csvTrainingFile);
    std::vector<std::vector<double>> trainingInputs;
    std::vector<std::vector<double>> trainingTargets;

    // Load the training instances based on our defined training samples.
    for (int i = 0; i < TRAINING_SAMPLES && !trainReader.eof(); ++i)
    {
        auto [label, pixels] = trainReader.getLabelAndPixels();
        trainingInputs.push_back(normalizePixels(pixels));
        trainingTargets.push_back(oneHotEncode(label));
    }

    // Create an MLP with specified dimensions and learning rate.
    MLP mlp(INPUT_SIZE, HIDDEN_NEURONS, OUTPUT_SIZE, LEARNING_RATE);
    std::cout << "Starting training for " << EPOCHS << " epochs on "
              << TRAINING_SAMPLES << " samples." << std::endl;
    mlp.startTraining(trainingInputs, trainingTargets, EPOCHS);
    std::cout << "Training completed." << std::endl;

    // Save the model for later use.
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

    // Create an empty MLP with the same architecture as when training.
    MLP mlp(INPUT_SIZE, HIDDEN_NEURONS, OUTPUT_SIZE, 0.01);

    // Load the model parameters.
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
        // Uncomment train() to train and save a model.
        // train();

        // Load the pre-trained model and evaluate on test samples.
        loadModel("models/best_so_far/model");
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
