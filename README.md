# Handwritten Digit Recognition Project

## Overview
This project focuses on handwritten digit recognition using a custom-built Multi-Layer Perceptron (MLP). It includes components for training a model on the MNIST dataset and an interactive application to draw digits and see real-time predictions.

## Components
- **MNIST Training (`MNIST/`)**:
    - Purpose: Handles the training and evaluation of neural network models using the MNIST dataset.
    - Key files: `MNIST/src/main.cpp` is the main entry point for training-related tasks. Model configurations and outputs are typically stored in `MNIST/models/`.
- **Interactive Prediction (`draw_and_predict/`)**:
    - Purpose: An SFML-based graphical application that allows users to draw a digit with the mouse and have a pre-trained model predict the digit.
    - Key files: `draw_and_predict/src/main.cpp` contains the logic for the interactive drawing and prediction interface. Example images or UI elements might be found in `MNIST/resources/`.
- **MLP Library (`mlp/`)**:
    - Purpose: A foundational library providing the implementation of a Multi-Layer Perceptron. This includes the core neural network structures like perceptrons, layers, and the backpropagation algorithm.
    - Key files: Key interface is `mlp/include/mlp.h` and its implementation `mlp/src/mlp.cpp`. The basic building block, the perceptron, is defined in `mlp/include/perceptron.h`.

## Dependencies
- SFML (Simple and Fast Multimedia Library) 3.0.0: Used for graphical interface in the `draw_and_predict` application.
- OpenCV (Open Source Computer Vision Library) 4.11.0: Used for image processing tasks, potentially for handling the input from the drawing canvas and interacting with MNIST data.
- MNIST in CSV format: The dataset used for training and testing, sourced from [https://pjreddie.com/projects/mnist-in-csv/](https://pjreddie.com/projects/mnist-in-csv/).

## How to Build
The project uses Premake5 for build configuration.

A `scripts/build.bat` script is provided which likely runs Premake5 to generate project files (e.g., Visual Studio solutions) and then potentially invokes a compiler like MSBuild or Make.

To build the project:
1. Ensure Premake5 is installed and in your PATH.
2. Run `premake5 <action>` (e.g., `premake5 vs2019`) in the root directory to generate project files for your preferred toolchain.
3. Alternatively, run the `scripts/build.bat` script if you are on Windows and have a compatible compiler setup.
