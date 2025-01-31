#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

class MLP
{
private:
    std::vector<std::vector<double>> hidden_weights; // Gewichte der versteckten Schicht
    std::vector<double> hidden_bias;                 // Bias der versteckten Schicht
    std::vector<double> output_weights;              // Gewichte der Ausgabeschicht
    double output_bias;                              // Bias der Ausgabeschicht
    double learning_rate;

    // Sigmoid-Aktivierungsfunktion
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + std::exp(-x));
    }

    // Ableitung der Sigmoid-Funktion
    double sigmoid_derivative(double x)
    {
        return x * (1.0 - x);
    }

public:
    // Konstruktor: Initialisiert die Gewichte und Biases
    MLP(int n_inputs, int n_hidden, double lr = 0.1)
        : hidden_weights(n_hidden, std::vector<double>(n_inputs)),
          hidden_bias(n_hidden, 0.0),
          output_weights(n_hidden, 0.0),
          output_bias(0.0),
          learning_rate(lr)
    {
        // Initialisiere die Gewichte und Biases mit Zufallswerten
        std::srand(std::time(0));
        for (auto &weights : hidden_weights)
        {
            for (auto &weight : weights)
            {
                weight = ((double)std::rand() / RAND_MAX) * 2 - 1; // Werte zwischen -1 und 1
            }
        }
        for (auto &weight : output_weights)
        {
            weight = ((double)std::rand() / RAND_MAX) * 2 - 1;
        }
        output_bias = ((double)std::rand() / RAND_MAX) * 2 - 1;
    }

    // Vorwärtsdurchlauf
    double forward(const std::vector<double> &inputs)
    {
        // Berechnung der versteckten Schicht
        std::vector<double> hidden_outputs(hidden_weights.size());
        for (size_t i = 0; i < hidden_weights.size(); i++)
        {
            double sum = hidden_bias[i];
            for (size_t j = 0; j < inputs.size(); j++)
            {
                sum += inputs[j] * hidden_weights[i][j];
            }
            hidden_outputs[i] = sigmoid(sum);
        }

        // Berechnung der Ausgabeschicht
        double output_sum = output_bias;
        for (size_t i = 0; i < hidden_outputs.size(); i++)
        {
            output_sum += hidden_outputs[i] * output_weights[i];
        }
        return sigmoid(output_sum);
    }

    // Rückwärtsdurchlauf (Backpropagation)
    void backward(const std::vector<double> &inputs, double target)
    {
        // Vorwärtsdurchlauf, um die aktuellen Ausgaben zu berechnen
        std::vector<double> hidden_outputs(hidden_weights.size());
        for (size_t i = 0; i < hidden_weights.size(); i++)
        {
            double sum = hidden_bias[i];
            for (size_t j = 0; j < inputs.size(); j++)
            {
                sum += inputs[j] * hidden_weights[i][j];
            }
            hidden_outputs[i] = sigmoid(sum);
        }

        double output_sum = output_bias;
        for (size_t i = 0; i < hidden_outputs.size(); i++)
        {
            output_sum += hidden_outputs[i] * output_weights[i];
        }
        double output = sigmoid(output_sum);

        // Fehler in der Ausgabeschicht
        double output_error = target - output;
        double output_delta = output_error * sigmoid_derivative(output);

        // Fehler in der versteckten Schicht
        std::vector<double> hidden_deltas(hidden_weights.size());
        for (size_t i = 0; i < hidden_weights.size(); i++)
        {
            hidden_deltas[i] = output_delta * output_weights[i] * sigmoid_derivative(hidden_outputs[i]);
        }

        // Aktualisierung der Gewichte und Biases der Ausgabeschicht
        for (size_t i = 0; i < output_weights.size(); i++)
        {
            output_weights[i] += learning_rate * output_delta * hidden_outputs[i];
        }
        output_bias += learning_rate * output_delta;

        // Aktualisierung der Gewichte und Biases der versteckten Schicht
        for (size_t i = 0; i < hidden_weights.size(); i++)
        {
            for (size_t j = 0; j < hidden_weights[i].size(); j++)
            {
                hidden_weights[i][j] += learning_rate * hidden_deltas[i] * inputs[j];
            }
            hidden_bias[i] += learning_rate * hidden_deltas[i];
        }
    }

    // Training
    void train(const std::vector<std::vector<double>> &training_inputs,
               const std::vector<double> &training_outputs, int epochs)
    {
        for (int epoch = 0; epoch < epochs; epoch++)
        {
            double total_loss = 0.0;
            for (size_t i = 0; i < training_inputs.size(); i++)
            {
                double output = forward(training_inputs[i]);
                double error = training_outputs[i] - output;
                total_loss += error * error;
                backward(training_inputs[i], training_outputs[i]);
            }
            if ((epoch + 1) % 1000 == 0 || epoch == 0)
            {
                std::cout << "Epoch " << epoch + 1 << ", Loss: " << total_loss / training_inputs.size() << std::endl;
            }
        }
    }

    // Vorhersage
    double predict(const std::vector<double> &inputs)
    {
        return forward(inputs) > 0.5 ? 1.0 : 0.0;
    }
};

// int main()
// {
//     // XOR-Problem
//     std::vector<std::vector<double>> training_inputs = {
//         {0, 0},
//         {0, 1},
//         {1, 0},
//         {1, 1}};
//     std::vector<double> training_outputs = {0, 1, 1, 0};

//     // Initialisiere das MLP mit 2 Eingaben, 2 Neuronen in der versteckten Schicht und einer Lernrate von 0.1
//     MLP mlp(2, 2, 0.1);

//     // Trainiere das MLP
//     mlp.train(training_inputs, training_outputs, 10000);

//     // Teste das MLP
//     std::cout << "Testing the MLP on XOR problem:" << std::endl;
//     for (const auto &input : training_inputs)
//     {
//         std::cout << input[0] << " XOR " << input[1] << " = " << mlp.predict(input) << std::endl;
//     }

//     return 0;
// }
