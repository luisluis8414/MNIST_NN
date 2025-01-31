#include <vector>
#include <cmath>
#include <iostream>
#include <random>

class Perceptron
{
private:
    std::vector<double> m_weights;
    double m_bias;
    double m_learningRate;

public:
    // n = length of weights vector
    Perceptron(int n, double learningRate) : m_weights(n, 0.0), m_bias(0.0), m_learningRate(learningRate)
    {
        // Initialize weights with random values between -1.0 and 1.0
        std::random_device dev;
        std::mt19937 rng(dev());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (size_t i = 0; i < m_weights.size(); i++)
        {
            m_weights[i] = dist(rng);
        }

        // Initialize bias with a random value between -1.0 and 1.0
        m_bias = dist(rng);
    }

    // calc the sum of all weights at index i with input same index
    double calcOutput(const std::vector<double> &inputs) const
    {
        if (inputs.size() != m_weights.size())
        {
            throw std::invalid_argument("length of weights not matching inputs");
        }

        double sum = m_bias;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            sum += m_weights[i] * inputs[i];
        }

        return activate(sum);
    }

    // sigmoid activation function std::exp(-x) = e^-x
    double activate(double x) const
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    void updateWeights(const std::vector<double> &inputs, double delta)
    {
        for (size_t i = 0; i < m_weights.size(); i++)
        {
            m_weights[i] -= m_learningRate * delta * inputs[i];
        }
        m_bias -= m_learningRate * delta;
    }

    const std::vector<double> &getWeights() const
    {
        return m_weights;
    }

    double getBias() const
    {
        return m_bias;
    }
};

class MLP
{
private:
    std::vector<Perceptron> m_innerLayer;
    std::vector<Perceptron> m_hiddenLayer;
    std::vector<Perceptron> m_outerLayer;

public:
    MLP(int inputSize, int hiddenSize, int outputSize, double learningRate = 0.1)
    {
        // for (int i = 0; i < inputSize; i++)
        // {
        //     m_innerLayer.emplace_back(1, learningRate);
        // }

        for (int i = 0; i < hiddenSize; i++)
        {
            m_hiddenLayer.emplace_back(inputSize, learningRate);
        }

        for (int i = 0; i < outputSize; i++)
        {
            m_outerLayer.emplace_back(hiddenSize, learningRate);
        }
    }

    // takes a layer and the inputs to that layer and calc the output of each perceptron of that layer
    std::vector<double> computeLayerOutput(const std::vector<Perceptron> &layer, const std::vector<double> &inputs)
    {
        // The input size should match the size of the weights of each perceptron
        if (inputs.size() != layer[0].getWeights().size())
        {
            throw std::invalid_argument("size of inputs doesn't match perceptron input size");
        }

        // Calculate the output for each perceptron in the layer
        std::vector<double> outputs(layer.size(), 0.0);
        for (size_t i = 0; i < layer.size(); i++)
        {
            outputs[i] = layer[i].calcOutput(inputs);
        }

        return outputs;
    }

    // forward pass calc and passes further outputs of each layer
    std::vector<double> forward(const std::vector<double> &inputs)
    {
        // std::vector<double> innerOutputs = computeLayerOutput(m_innerLayer, inputs);

        std::vector<double> hiddenOutputs = computeLayerOutput(m_hiddenLayer, inputs);

        std::vector<double> outerOutputs = computeLayerOutput(m_outerLayer, hiddenOutputs);

        return outerOutputs;
    }

    double meanSquaredError(const std::vector<double> &outputs, const std::vector<double> &targets)
    {
        if (outputs.size() != targets.size())
        {
            throw std::invalid_argument("outputs size doesnt match targets size");
        }
        double error = 0.0;
        for (size_t i = 0; i < outputs.size(); i++)
        {
            error += std::pow(outputs[i] - targets[i], 2);
        }

        return error / outputs.size();
    }

    // backprobagation
    void train(const std::vector<double> &inputs, const std::vector<double> &targets)
    {
        // std::vector<double> innerOutputs = computeLayerOutput(m_innerLayer, inputs);
        std::vector<double> hiddenOutputs = computeLayerOutput(m_hiddenLayer, inputs);
        std::vector<double> outerOutputs = computeLayerOutput(m_outerLayer, hiddenOutputs);

        // Compute output layer error gradients
        std::vector<double> outputDeltas(m_outerLayer.size());
        for (size_t i = 0; i < m_outerLayer.size(); i++)
        {
            double error = outerOutputs[i] - targets[i];
            double derivative = outerOutputs[i] * (1.0 - outerOutputs[i]); // Sigmoid derivative
            outputDeltas[i] = error * derivative;
        }

        // Update output layer weights and biases
        for (size_t i = 0; i < m_outerLayer.size(); i++)
        {
            m_outerLayer[i].updateWeights(hiddenOutputs, outputDeltas[i]);
        }

        // Compute hidden layer error gradients
        std::vector<double> hiddenDeltas(m_hiddenLayer.size());
        for (size_t i = 0; i < m_hiddenLayer.size(); i++)
        {
            double error = 0.0;
            for (size_t j = 0; j < m_outerLayer.size(); j++)
            {
                error += outputDeltas[j] * m_outerLayer[j].getWeights()[i];
            }
            double derivative = hiddenOutputs[i] * (1.0 - hiddenOutputs[i]); // Sigmoid derivative
            hiddenDeltas[i] = error * derivative;
        }

        // Update hidden layer weights and biases
        for (size_t i = 0; i < m_hiddenLayer.size(); i++)
        {
            m_hiddenLayer[i].updateWeights(inputs, hiddenDeltas[i]);
        }

        // Compute inner layer error gradients
        std::vector<double> innerDeltas(m_innerLayer.size());
        for (size_t i = 0; i < m_innerLayer.size(); i++)
        {
            double error = 0.0;
            for (size_t j = 0; j < m_hiddenLayer.size(); j++)
            {
                error += hiddenDeltas[j] * m_hiddenLayer[j].getWeights()[i];
            }
            double derivative = inputs[i] * (1.0 - inputs[i]); // Sigmoid derivative
            innerDeltas[i] = error * derivative;
        }

        // Update inner layer weights and biases
        for (size_t i = 0; i < m_innerLayer.size(); i++)
        {
            m_innerLayer[i].updateWeights(inputs, innerDeltas[i]);
        }
    }

    void startTraining(const std::vector<std::vector<double>> &trainingInputs,
                       const std::vector<std::vector<double>> &trainingTargets,
                       int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double totalError = 0.0;

            for (size_t i = 0; i < trainingInputs.size(); i++)
            {
                // Train on a single example
                train(trainingInputs[i], trainingTargets[i]);

                // Forward pass to calculate the output
                // std::vector<double> innerOutputs = computeLayerOutput(m_innerLayer, trainingInputs[i]);
                std::vector<double> hiddenOutputs = computeLayerOutput(m_hiddenLayer, trainingInputs[i]);
                std::vector<double> outerOutputs = computeLayerOutput(m_outerLayer, hiddenOutputs);

                // Calculate the error for this example
                totalError += meanSquaredError(outerOutputs, trainingTargets[i]);
            }

            // Print the average error for this epoch
            std::cout << "Epoch " << epoch + 1 << ", MSE: " << totalError / trainingInputs.size() << std::endl;
        }
    }
};
