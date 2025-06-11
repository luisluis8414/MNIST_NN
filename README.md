# Neural Network for MNIST Handwritten Digit Recognition

A C++ implementation of a Multi-Layer Perceptron (MLP) for recognizing handwritten digits using the MNIST dataset.

## 🏆 Performance Results

**Achieved Accuracy: 96.29%** on test set (10,000 samples)

### Per-Digit Accuracy Breakdown:
| Digit | Accuracy | Correct/Total |
|-------|----------|---------------|
| 0     | 98.47%   | 965/980       |
| 1     | 98.50%   | 1118/1135     |
| 2     | 96.03%   | 991/1032      |
| 3     | 95.45%   | 964/1010      |
| 4     | 96.44%   | 947/982       |
| 5     | 94.96%   | 847/892       |
| 6     | 97.29%   | 932/958       |
| 7     | 95.82%   | 985/1028      |
| 8     | 94.97%   | 925/974       |
| 9     | 94.65%   | 955/1009      |

## 🧠 Network Architecture

- **Input Layer**: 784 neurons (28×28 pixel images)
- **Hidden Layer 1**: 128 neurons
- **Hidden Layer 2**: 64 neurons  
- **Output Layer**: 10 neurons (digits 0-9)
- **Activation Function**: Sigmoid
- **Learning Rate**: 0.01
- **Training Epochs**: 100

## 🔬 Training Algorithm

### Backpropagation with Gradient Descent

1. **Forward Pass**: Compute the outputs of the network for a given input
2. **Compute Error**: Compare network output with expected output using loss function
3. **Backward Pass (Backpropagation)**: Compute gradients of loss with respect to weights and biases
4. **Update Weights**: Adjust weights and biases using gradient descent

### Early Stopping
- **Patience**: 5 epochs
- **Minimal Improvement**: 0.0001
- **Metric**: Mean Squared Error on training data

## 📁 Project Components

- **MNIST Training (`MNIST/`)**:
    - Purpose: Handles the training and evaluation of neural network models using the MNIST dataset.
    - Key files: `MNIST/src/main.cpp` is the main entry point for training-related tasks. Model configurations and outputs are typically stored in `MNIST/models/`.

- **Interactive Prediction (`draw_and_predict/`)**:
    - Purpose: An SFML-based graphical application that allows users to draw a digit with the mouse and have a pre-trained model predict the digit.
    - Key files: `draw_and_predict/src/main.cpp` contains the logic for the interactive drawing and prediction interface.

- **MLP Library (`mlp/`)**:
    - Purpose: A foundational library providing the implementation of a Multi-Layer Perceptron. This includes the core neural network structures like perceptrons, layers, and the backpropagation algorithm.
    - Key files: Key interface is `mlp/include/mlp.h` and its implementation `mlp/src/mlp.cpp`. The basic building block, the perceptron, is defined in `mlp/include/perceptron.h`.

## 🚀 Getting Started

### Build the Project

The project uses Premake5 for build configuration.

#### Option 1: Using Premake5 directly
```bash
premake5 vs2022
msbuild MultiLayerPerception.sln /p:Configuration=Debug /p:Platform=x64
```

#### Option 2: Using build script (Windows)
```bash
scripts/build.bat
```

### Train a New Model
1. Uncomment `train();` in `MNIST/src/main.cpp`
2. Build and run the MNIST project

### Evaluate Pre-trained Model
```bash
cd bin/Debug/MNIST
./MNIST.exe
```

## 📊 Dataset

**Training Set**: 60,000 images from `mnist_train.csv`  
**Test Set**: 10,000 images from `mnist_test.csv`

Data source: [MNIST in CSV format](https://pjreddie.com/projects/mnist-in-csv/)

## 🛠️ Dependencies

- [OpenCV](https://github.com/opencv/opencv) 4.11.0 - Computer vision library
- [SFML](https://www.sfml-dev.org/) 3.0.0 - Simple and Fast Multimedia Library (for interactive drawing app)
- Visual Studio 2019/2022 - C++ compiler
- Premake5 - Build system generator

## 📁 Project Structure

```
NN/
├── mlp/                    # Core neural network library
├── MNIST/                  # MNIST training & evaluation
├── draw_and_predict/       # Interactive digit drawing app
├── scripts/                # Build scripts
└── README.md
```
