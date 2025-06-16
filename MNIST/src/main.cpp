#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <sstream>
#include <algorithm>

#include <opencv2/opencv.hpp>

#include "csv_reader.hpp"
#include "../mlp/include/mlp.h"

//============================================================================
// Parameters
//============================================================================
const double LEARNING_RATE = 0.01;
const int EPOCHS = 100; // 100 epochs but early stopping stopps way earlier
const int TRAINING_SAMPLES = 60000;
const int HIDDEN_NEURONS_LAYER1 = 128;
const int HIDDEN_NEURONS_LAYER2 = 64;

// Input and output sizes for the MNIST dataset
const int INPUT_SIZE = 784; // 28x28 pixels
const int OUTPUT_SIZE = 10; // Digits 0 to 9

//============================================================================
// Helper Functions
//============================================================================

// Get predicted class from network output (index of maximum value)
int getPredictedClass(const std::vector<double> &output)
{
    return static_cast<int>(std::max_element(output.begin(), output.end()) - output.begin());
}

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
    std::vector<std::vector<double>> allInputs;
    std::vector<std::vector<double>> allTargets;

    // Load all training data
    for (int i = 0; i < TRAINING_SAMPLES && !trainReader.eof(); ++i)
    {
        auto [label, pixels] = trainReader.getLabelAndPixels();
        allInputs.push_back(normalizePixels(pixels));
        allTargets.push_back(oneHotEncode(label));
    }

    // Split into training (80%) and validation (20%) sets
    size_t totalSamples = allInputs.size();
    size_t validationSize = totalSamples / 5; // 20% for validation
    size_t trainingSize = totalSamples - validationSize;

    std::vector<std::vector<double>> trainingInputs(allInputs.begin(), allInputs.begin() + trainingSize);
    std::vector<std::vector<double>> trainingTargets(allTargets.begin(), allTargets.begin() + trainingSize);
    std::vector<std::vector<double>> validationInputs(allInputs.begin() + trainingSize, allInputs.end());
    std::vector<std::vector<double>> validationTargets(allTargets.begin() + trainingSize, allTargets.end());

    // two hidden layers
    std::vector<int> hiddenLayers = {HIDDEN_NEURONS_LAYER1, HIDDEN_NEURONS_LAYER2};

    // create mlp
    MLP mlp(INPUT_SIZE, hiddenLayers, OUTPUT_SIZE, LEARNING_RATE);

    std::cout << "Starting training with " << trainingSize << " training samples and "
              << validationSize << " validation samples." << std::endl;

    mlp.startTraining(trainingInputs, trainingTargets, validationInputs, validationTargets, EPOCHS);
    std::cout << "Training completed." << std::endl;

    std::string modelPath = buildModelPath();
    mlp.saveModel(modelPath);
    std::cout << "Model saved to: " << modelPath << std::endl;
}

//============================================================================
// Evaluation Functionality
//============================================================================

void evaluateModel(std::string modelPath = buildModelPath())
{
    std::string csvTestingFile = "resources/training_data/mnist_test.csv";

    // two hidden layers
    std::vector<int> hiddenLayers = {HIDDEN_NEURONS_LAYER1, HIDDEN_NEURONS_LAYER2};

    // Note: The learning rate here is not used in inference.
    MLP mlp(INPUT_SIZE, hiddenLayers, OUTPUT_SIZE, 0.01);

    mlp.loadModel(modelPath);
    std::cout << "Model loaded successfully from file: " << modelPath << std::endl;

    CsvReader testReader(csvTestingFile);

    // Comprehensive evaluation on full test set
    std::cout << "\n----- Evaluating on full test set -----" << std::endl;

    int totalSamples = 0;
    int correctPredictions = 0;
    std::vector<int> classCorrect(10, 0); // Track correct predictions per digit
    std::vector<int> classTotal(10, 0);   // Track total samples per digit

    try
    {
        while (!testReader.eof())
        {
            auto [testLabel, testPixels] = testReader.getLabelAndPixels();
            std::vector<double> testInput = normalizePixels(testPixels);
            auto output = mlp.forward(testInput);

            int predictedClass = getPredictedClass(output);

            totalSamples++;
            classTotal[testLabel]++;

            if (predictedClass == testLabel)
            {
                correctPredictions++;
                classCorrect[testLabel]++;
            }

            // Show progress every 1000 samples
            if (totalSamples % 1000 == 0)
            {
                std::cout << "Processed " << totalSamples << " samples..." << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cout << "Evaluation completed early due to: " << e.what() << std::endl;
        std::cout << "Total samples processed: " << totalSamples << std::endl;
    }

    // Calculate and display results
    double accuracy = static_cast<double>(correctPredictions) / totalSamples * 100.0;

    std::cout << "\n==================== EVALUATION RESULTS ====================" << std::endl;
    std::cout << "Total samples tested: " << totalSamples << std::endl;
    std::cout << "Correct predictions: " << correctPredictions << std::endl;
    std::cout << "Overall accuracy: " << accuracy << "%" << std::endl;

    std::cout << "\n----- Per-digit accuracy -----" << std::endl;
    for (int digit = 0; digit < 10; digit++)
    {
        if (classTotal[digit] > 0)
        {
            double digitAccuracy = static_cast<double>(classCorrect[digit]) / classTotal[digit] * 100.0;
            std::cout << "Digit " << digit << ": " << digitAccuracy << "% ("
                      << classCorrect[digit] << "/" << classTotal[digit] << ")" << std::endl;
        }
    }
    std::cout << "==========================================================" << std::endl;
}

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

        int predictedClass = getPredictedClass(output);
        bool isCorrect = (predictedClass == testLabel);

        std::cout << "\nSample " << i + 1 << ":" << std::endl;
        std::cout << "Expected Label: " << testLabel << std::endl;
        std::cout << "Predicted Label: " << predictedClass << std::endl;
        std::cout << "Correct: " << (isCorrect ? "YES" : "NO") << std::endl;
        std::cout << "Confidence: " << output[predictedClass] << std::endl;
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
        train();

        // Quick test on 20 samples with detailed output
        loadModel("models/model_0.01_100_60000_128_64");

        // Comprehensive evaluation on full test set
        evaluateModel("models/model_0.01_100_60000_128_64");
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
