#include "../include/perceptron.h"
#include <cmath>
#include <stdexcept>
#include <random>

// Constructor: Initialize weights and bias with random values
Perceptron::Perceptron(int n, double learningRate)
    : m_weights(n, 0.0), m_bias(0.0), m_learningRate(learningRate)
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

// Default constructor for loading from file
Perceptron::Perceptron() : m_bias(0.0), m_learningRate(0.1) {}

// Calculate raw output (weighted sum + bias) without activation
double Perceptron::calcOutputRaw(const std::vector<double> &inputs) const
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
    return sum;
}

// Calculate output with activation function
double Perceptron::calcOutput(const std::vector<double> &inputs) const
{
    return activate(calcOutputRaw(inputs));
}

// Sigmoid activation function
double Perceptron::activate(double x) const
{
    return 1.0 / (1.0 + std::exp(-x));
}

// Update weights and bias using the delta value
void Perceptron::updateWeights(const std::vector<double> &inputs, double delta)
{
    for (size_t i = 0; i < m_weights.size(); i++)
    {
        m_weights[i] -= m_learningRate * delta * inputs[i];
    }
    m_bias -= m_learningRate * delta;
}

// Getters
const std::vector<double> &Perceptron::getWeights() const
{
    return m_weights;
}

double Perceptron::getBias() const
{
    return m_bias;
}

double Perceptron::getLearningRate() const
{
    return m_learningRate;
}

// Save Perceptron parameters to binary file
void Perceptron::save(std::ofstream &ofs) const
{
    // Save size of the weights vector
    size_t size = m_weights.size();
    ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
    // Save weights
    ofs.write(reinterpret_cast<const char *>(m_weights.data()), size * sizeof(double));
    // Save bias and learning rate
    ofs.write(reinterpret_cast<const char *>(&m_bias), sizeof(m_bias));
    ofs.write(reinterpret_cast<const char *>(&m_learningRate), sizeof(m_learningRate));
}

// Load Perceptron parameters from binary file
void Perceptron::load(std::ifstream &ifs)
{
    size_t size;
    ifs.read(reinterpret_cast<char *>(&size), sizeof(size));
    m_weights.resize(size);
    ifs.read(reinterpret_cast<char *>(m_weights.data()), size * sizeof(double));
    ifs.read(reinterpret_cast<char *>(&m_bias), sizeof(m_bias));
    ifs.read(reinterpret_cast<char *>(&m_learningRate), sizeof(m_learningRate));
}
