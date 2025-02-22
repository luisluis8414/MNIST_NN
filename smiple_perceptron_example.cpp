#include <iostream>
#include <iomanip>

float w1 = 0.1;
float w2 = -0.4;

float bias = 0.f;

float lrnR = 0.2f;

int activate(float x)
{
    return x > 0 ? 1 : 0;
}

float predict(float x1, float x2)
{
    return activate(x1 * w1 + x2 * w2 + bias);
}

void train(float x1, float x2, float target)
{
    // if (x1 == 0)
    // {
    //     x1 = -1;
    // }

    // if (x2 == 0)
    // {
    //     x2 = -1;
    // }

    float prediction = predict(x1, x2);

    float error = target - prediction;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Training on Input: (" << x1 << ", " << x2 << "), Target: " << target << "\n";
    std::cout << "  Prediction: " << prediction << ", Error: " << error << "\n";
    std::cout << "  Weights before update: w1 = " << w1 << ", w2 = " << w2 << ", Bias = " << bias << "\n";

    w1 += lrnR * error * x1;
    w2 += lrnR * error * x2;
    bias += lrnR * error;

    std::cout << "  Weights after update:  w1 = " << w1 << ", w2 = " << w2 << ", Bias = " << bias << "\n";
    std::cout << "--------------------------------------------------\n";
}

int main()
{
    for (int i = 0; i < 10; i++)
    {
        std::cout << "Epoch " << i + 1 << ":\n";
        train(0, 0, 0);
        train(0, 1, 0);
        train(1, 0, 0);
        train(1, 1, 1);
        std::cout << "==================================================\n";
    }

    std::cout << "Final Predictions:\n";
    std::cout << "  (0, 0): " << predict(0, 0) << std::endl;
    std::cout << "  (0, 1): " << predict(0, 1) << std::endl;
    std::cout << "  (1, 0): " << predict(1, 0) << std::endl;
    std::cout << "  (1, 1): " << predict(1, 1) << std::endl;
}
