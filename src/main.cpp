#include <iostream>
#include "file_reader.hpp"
#include "perceptron.hpp"

std::vector<double> oneHotEncode(int label, int size = 9)
{
    std::vector<double> encoded(size, 0.0);
    encoded[label - 1] = 1.0; // Label 1 maps to index 0, etc.
    return encoded;
}

int main()
{
    FileReader fileReader(100);
    MLP mlp(100 * 100, 50, 9, 0.01);

    try
    {
        std::vector<std::vector<double>> data = fileReader.splitImage("resources/1_9.png", 9);

        std::vector<std::vector<double>> inputs;
        std::vector<std::vector<double>> targets;
        for (int i = 0; i < data.size(); i++)
        {
            inputs.push_back(data[i]);
            targets.push_back(oneHotEncode(i + 1));
        }

        mlp.startTraining(inputs, targets, 1000);

        auto output = mlp.forward(data[0]);

        std::cout << "{";
        for (auto value : output)
        {
            std::cout << value << ", ";
        }
        std::cout << "}";
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
