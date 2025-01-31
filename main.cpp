#include <vector>
#include <iostream>
#include <cmath>

class Perceptron
{
private:
    std::vector<double> weights;
    double learningRate;
    double bias;

public:
    Perceptron(int n, double lr = 0.1) : weights(n, 0.0),
                                         bias(0.0),
                                         learningRate(lr)
    {
    }

    // Heaviside-Funktion
    int activate(double sum)
    {
        return sum > 0 ? 1 : 0;
    }

    std::vector<double> getWeights()
    {
        return weights;
    };

    int predict(const std::vector<double> &inputs)
    {
        if (inputs.size() != weights.size())
        {
            throw std::invalid_argument("Eingabedimension stimmt nicht überein!");
        }

        double sum = bias;
        for (int i = 0; i < inputs.size(); i++)
        {
            sum += inputs[i] * weights[i];
        }
        return activate(sum);
    }

    void train(const std::vector<double> &inputs, int target)
    {
        int prediction = predict(inputs);
        int error = target - prediction;

        if (error != 0)
        {
            for (int i = 0; i < weights.size(); i++)
            {
                weights[i] += learningRate * error * inputs[i];
            }
            bias += learningRate * error;
        }
    }
};

int main()
{
    Perceptron p(2);

    std::vector<std::vector<double>> training_inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<int> training_outputs = {1, 0, 0, 1};

    for (int epoch = 0; epoch < 100; epoch++)
    {
        for (size_t i = 0; i < training_inputs.size(); i++)
        {
            p.train(training_inputs[i], training_outputs[i]);
        }
    }

    for (const auto &input : training_inputs)
    {
        std::cout << input[0] << " AND " << input[1] << " = "
                  << p.predict(input) << std::endl;
    }

    for (const double weight : p.getWeights())
    {
        std::cout << "weight: " << weight << std::endl;
    }

    // std::cout << "bias: " << p.bias << std::endl;

    return 0;
}
