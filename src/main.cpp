#include <iostream>
#include "file_reader.hpp"
#include "perceptron.hpp"

std::vector<double> oneHotEncode(int label, int size = 9)
{
    std::vector<double> encoded(size, 0.0);
    encoded[label - 1] = 1.0; // Label 1 maps to index 0, etc.
    return encoded;
}

void printVec(std::vector<double> vec)
{
    std::cout << "{";
    for (auto value : vec)
    {
        std::cout << value << ", ";
    }
    std::cout << "}" << std::endl;
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

        std::vector<std::vector<double>> data2 = fileReader.splitImage("resources/1_9.2.png", 9);

        {
            auto output1 = mlp.forward(data[0]);

            auto output2 = mlp.forward(data[1]);

            auto output3 = mlp.forward(data[2]);

            auto output4 = mlp.forward(data[3]);

            auto output5 = mlp.forward(data[4]);

            auto output6 = mlp.forward(data[5]);

            auto output7 = mlp.forward(data[6]);

            printVec(output1);
            printVec(output2);
            printVec(output3);
            printVec(output4);
            printVec(output5);
            printVec(output6);
            printVec(output7);
        }
        std::cout << "----------------------------------------" << std::endl;
        {
            {
                auto output1 = mlp.forward(data2[0]);

                auto output2 = mlp.forward(data2[1]);

                auto output3 = mlp.forward(data2[2]);

                auto output4 = mlp.forward(data2[3]);

                auto output5 = mlp.forward(data2[4]);

                auto output6 = mlp.forward(data2[5]);

                auto output7 = mlp.forward(data2[6]);

                printVec(output1);
                printVec(output2);
                printVec(output3);
                printVec(output4);
                printVec(output5);
                printVec(output6);
                printVec(output7);
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
