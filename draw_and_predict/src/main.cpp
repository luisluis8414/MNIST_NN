// #include <algorithm>
#include <iostream>
// #include <stdexcept>
// #include <string>
// #include <vector>

// #include <opencv2/opencv.hpp>
// #include "../../mlp/include/mlp.h"

// //============================================================================
// // Parameters
// //============================================================================

// // For the processed (preprocessed) image, we use 28x28 = 784 pixels.
// const int DRAW_INPUT_WIDTH = 28;
// const int DRAW_INPUT_HEIGHT = 28;
// const int DRAW_INPUT_SIZE = DRAW_INPUT_WIDTH * DRAW_INPUT_HEIGHT;

// const int OUTPUT_SIZE = 10;

// // Define two hidden layers for the MLP.
// const int HIDDEN_NEURONS_LAYER1 = 128;
// const int HIDDEN_NEURONS_LAYER2 = 64;

// // Brush settings for drawing (10 pixels wide)
// const int BRUSH_WIDTH = 10;
// const int BRUSH_RADIUS = BRUSH_WIDTH / 2; // 10/2 = 5

// //============================================================================
// // Helper Functions
// //============================================================================

// // Convert a vector of pixel values [0,255] to normalized double values [0,1]
// std::vector<double> normalizePixels(const std::vector<unsigned char> &pixels)
// {
//     std::vector<double> normalized;
//     normalized.reserve(pixels.size());
//     for (uchar p : pixels)
//     {
//         normalized.push_back(static_cast<double>(p) / 255.0);
//     }
//     return normalized;
// }

// //============================================================================
// // Global Drawing Canvas and Mouse Callback Function
// //============================================================================

// // Global canvas where the user can draw.
// cv::Mat drawCanvas;

// // Mouse callback: Draw a white filled circle (our brush) when the left button
// // is pressed or dragged.
// void onMouse(int event, int x, int y, int flags, void *userdata)
// {
//     if (event == cv::EVENT_LBUTTONDOWN ||
//         (event == cv::EVENT_MOUSEMOVE && (flags & cv::EVENT_FLAG_LBUTTON)))
//     {
//         cv::circle(drawCanvas, cv::Point(x, y), BRUSH_RADIUS,
//                    cv::Scalar(255), -1);
//         cv::imshow("Draw", drawCanvas);
//     }
// }

// //============================================================================
// // Improved Preprocessing and Forward Pass on the Drawn Image
// //============================================================================

// void forwardDraw(const std::string &modelPath = "models/best_so_far/model")
// {
//     // 1. Create a large black drawing canvas.
//     drawCanvas = cv::Mat::zeros(200, 200, CV_8UC1);
//     cv::namedWindow("Draw", cv::WINDOW_AUTOSIZE);
//     cv::setMouseCallback("Draw", onMouse, nullptr);

//     std::cout << "Draw a digit using the mouse.\n"
//               << "Press ESC to finish drawing." << std::endl;

//     // Let the user draw until ESC is pressed.
//     while (true)
//     {
//         cv::imshow("Draw", drawCanvas);
//         int key = cv::waitKey(20);
//         if (key == 27)
//         { // ESC key
//             break;
//         }
//     }

//     // 2. Post-process the drawing: Extract the digit region.
//     std::vector<cv::Point> nonZeroPoints;
//     cv::findNonZero(drawCanvas, nonZeroPoints);

//     if (nonZeroPoints.empty())
//     {
//         std::cerr << "No drawing detected. Exiting." << std::endl;
//         return;
//     }

//     // Get the bounding box of the drawn digit.
//     cv::Rect bbox = cv::boundingRect(nonZeroPoints);

//     // 3. (Optional) Add some padding to the bounding box.
//     int padding = 10; // adjust as needed
//     int x = std::max(bbox.x - padding, 0);
//     int y = std::max(bbox.y - padding, 0);
//     int width = std::min(bbox.width + 2 * padding, drawCanvas.cols - x);
//     int height = std::min(bbox.height + 2 * padding, drawCanvas.rows - y);
//     cv::Rect paddedBBox(x, y, width, height);
//     cv::Mat cropped = drawCanvas(paddedBBox);

//     // 4. Resize the cropped (ROI) image to 20x20.
//     cv::Mat resizedROI;
//     cv::resize(cropped, resizedROI, cv::Size(20, 20));

//     // 5. Create a new 28x28 image and embed the 20x20 resized ROI
//     // at the center (this mimics MNIST preprocessing).
//     cv::Mat processed = cv::Mat::zeros(DRAW_INPUT_HEIGHT, DRAW_INPUT_WIDTH, CV_8UC1);
//     int xOffset = (DRAW_INPUT_WIDTH - 20) / 2;
//     int yOffset = (DRAW_INPUT_HEIGHT - 20) / 2;
//     resizedROI.copyTo(processed(cv::Rect(xOffset, yOffset, resizedROI.cols, resizedROI.rows)));

//     // 6. Convert the processed image into a vector and normalize.
//     std::vector<unsigned char> pixels(processed.datastart, processed.dataend);
//     std::vector<double> normalized = normalizePixels(pixels);

//     // 7. Initialize the MLP with multiple hidden layers and load the pre-trained model.
//     std::vector<int> hiddenLayers = {HIDDEN_NEURONS_LAYER1, HIDDEN_NEURONS_LAYER2};
//     MLP mlp(DRAW_INPUT_SIZE, hiddenLayers, OUTPUT_SIZE, 0.01);

//     mlp.loadModel(modelPath);
//     std::cout << "Model loaded successfully from file: " << modelPath << std::endl;

//     // 8. Perform a forward pass.
//     std::vector<double> output = mlp.forward(normalized);

//     std::cout << "\nMLP Output: { ";
//     for (const auto &val : output)
//     {
//         std::cout << val << " ";
//     }
//     std::cout << "}" << std::endl;

//     // 9. Check if any output exceeds the confidence threshold of 0.7.
//     bool isConfident = false;
//     for (const double val : output)
//     {
//         if (val > 0.5)
//         {
//             isConfident = true;
//             break;
//         }
//     }
//     if (!isConfident)
//     {
//         std::cerr << "Error: Model is not confident enough. No output exceeds 0.7." << std::endl;
//         return;
//     }

//     // 10. Determine and display the predicted digit.
//     auto maxIt = std::max_element(output.begin(), output.end());
//     int predicted = std::distance(output.begin(), maxIt);
//     std::cout << "Predicted digit: " << predicted << std::endl;
// }

// //============================================================================
// // Main Entry Point
// //============================================================================

int main()
{
    std::cout << "empty" << std::endl;
    // try
    // {
    //     forwardDraw();
    // }
    // catch (const std::exception &e)
    // {
    //     std::cerr << "Error: " << e.what() << std::endl;
    //     return 1;
    // }
    return 0;
}
