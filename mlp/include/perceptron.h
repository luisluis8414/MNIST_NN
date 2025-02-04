#pragma once

#include <vector>
#include <fstream>

class Perceptron
{
public:
    Perceptron(int n, double learningRate);
    Perceptron();

    // Forward pass
    double calcOutput(const std::vector<double> &inputs) const;

    // Activation function
    double activate(double x) const;

    // Update weights and bias
    void updateWeights(const std::vector<double> &inputs, double delta);

    // Getters
    const std::vector<double> &getWeights() const;
    double getBias() const;
    double getLearningRate() const;

    // Save and load perceptron parameters
    void save(std::ofstream &ofs) const;
    void load(std::ifstream &ifs);

private:
    std::vector<double> m_weights;
    double m_bias;
    double m_learningRate;
};
