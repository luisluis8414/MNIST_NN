#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

#include <opencv2/opencv.hpp>
#include "../../mlp/include/mlp.h"

//============================================================================
// Parameters
//============================================================================

// For the processed (preprocessed) image, we use 28x28 = 784 pixels.
const int DRAW_INPUT_WIDTH = 28;
const int DRAW_INPUT_HEIGHT = 28;
const int DRAW_INPUT_SIZE = DRAW_INPUT_WIDTH * DRAW_INPUT_HEIGHT;

const int OUTPUT_SIZE = 10;

// Define two hidden layers for the MLP.
const int HIDDEN_NEURONS_LAYER1 = 128;
const int HIDDEN_NEURONS_LAYER2 = 64;

// Improved brush settings - smaller, softer strokes
const float BRUSH_RADIUS = 3.0f;   // Smaller radius for finer strokes
const float BRUSH_SOFTNESS = 1.5f; // Controls edge softness
const int CANVAS_SIZE = 280;       // Larger canvas for better resolution
const float MIN_OPACITY = 0.3f;    // Minimum brush opacity
const float MAX_OPACITY = 0.9f;    // Maximum brush opacity

//============================================================================
// Helper Functions
//============================================================================

// Convert a vector of pixel values [0,255] to normalized double values [0,1]
std::vector<double> normalizePixels(const std::vector<unsigned char> &pixels)
{
    std::vector<double> normalized;
    normalized.reserve(pixels.size());
    for (uchar p : pixels)
    {
        normalized.push_back(static_cast<double>(p) / 255.0);
    }
    return normalized;
}

// Calculate distance between two points
float distance(float x1, float y1, float x2, float y2)
{
    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// Soft brush function - creates anti-aliased, gradient brush strokes
void drawSoftBrush(cv::Mat &canvas, float x, float y, float opacity = MAX_OPACITY)
{
    int minX = std::max(0, (int)(x - BRUSH_RADIUS - BRUSH_SOFTNESS));
    int maxX = std::min(canvas.cols - 1, (int)(x + BRUSH_RADIUS + BRUSH_SOFTNESS));
    int minY = std::max(0, (int)(y - BRUSH_RADIUS - BRUSH_SOFTNESS));
    int maxY = std::min(canvas.rows - 1, (int)(y + BRUSH_RADIUS + BRUSH_SOFTNESS));

    for (int py = minY; py <= maxY; py++)
    {
        for (int px = minX; px <= maxX; px++)
        {
            float dist = distance(x, y, px, py);

            if (dist <= BRUSH_RADIUS + BRUSH_SOFTNESS)
            {
                float alpha;
                if (dist <= BRUSH_RADIUS)
                {
                    // Core brush area - full opacity
                    alpha = opacity;
                }
                else
                {
                    // Soft edge - gradient falloff
                    float falloff = (BRUSH_RADIUS + BRUSH_SOFTNESS - dist) / BRUSH_SOFTNESS;
                    alpha = opacity * falloff;
                }

                // Blend with existing pixel value
                uchar currentValue = canvas.at<uchar>(py, px);
                float currentNormalized = currentValue / 255.0f;
                float newValue = std::min(1.0f, currentNormalized + alpha);
                canvas.at<uchar>(py, px) = (uchar)(newValue * 255);
            }
        }
    }
}

// Draw line between two points with soft brush (for continuous strokes)
void drawSoftLine(cv::Mat &canvas, float x1, float y1, float x2, float y2, float opacity = MAX_OPACITY)
{
    float dist = distance(x1, y1, x2, y2);
    int steps = std::max(1, (int)(dist / 0.5f)); // Sample every 0.5 pixels

    for (int i = 0; i <= steps; i++)
    {
        float t = (float)i / steps;
        float x = x1 + t * (x2 - x1);
        float y = y1 + t * (y2 - y1);
        drawSoftBrush(canvas, x, y, opacity);
    }
}

// Apply Gaussian blur to smooth the drawing (similar to pen ink spreading)
void smoothDrawing(cv::Mat &canvas)
{
    cv::Mat temp;
    cv::GaussianBlur(canvas, temp, cv::Size(3, 3), 0.5);
    temp.copyTo(canvas);
}

// Enhanced preprocessing that better matches MNIST characteristics
cv::Mat preprocessDrawing(const cv::Mat &rawCanvas)
{
    cv::Mat canvas = rawCanvas.clone();

    // 1. Apply smoothing to simulate ink spreading
    smoothDrawing(canvas);

    // 2. Find the bounding box of non-zero pixels
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(canvas, nonZeroPoints);

    if (nonZeroPoints.empty())
    {
        return cv::Mat::zeros(DRAW_INPUT_HEIGHT, DRAW_INPUT_WIDTH, CV_8UC1);
    }

    cv::Rect bbox = cv::boundingRect(nonZeroPoints);

    // 3. Add padding (MNIST digits have some padding)
    int padding = std::max(10, (int)(std::max(bbox.width, bbox.height) * 0.1));
    int x = std::max(bbox.x - padding, 0);
    int y = std::max(bbox.y - padding, 0);
    int width = std::min(bbox.width + 2 * padding, canvas.cols - x);
    int height = std::min(bbox.height + 2 * padding, canvas.rows - y);

    cv::Mat cropped = canvas(cv::Rect(x, y, width, height));

    // 4. Make the crop square by adding padding to the shorter dimension
    int maxDim = std::max(cropped.rows, cropped.cols);
    cv::Mat square = cv::Mat::zeros(maxDim, maxDim, CV_8UC1);

    int offsetX = (maxDim - cropped.cols) / 2;
    int offsetY = (maxDim - cropped.rows) / 2;
    cropped.copyTo(square(cv::Rect(offsetX, offsetY, cropped.cols, cropped.rows)));

    // 5. Resize to 20x20 (MNIST standard)
    cv::Mat resized;
    cv::resize(square, resized, cv::Size(20, 20), 0, 0, cv::INTER_AREA);

    // 6. Center in 28x28 image
    cv::Mat processed = cv::Mat::zeros(DRAW_INPUT_HEIGHT, DRAW_INPUT_WIDTH, CV_8UC1);
    int xOffset = (DRAW_INPUT_WIDTH - 20) / 2;
    int yOffset = (DRAW_INPUT_HEIGHT - 20) / 2;
    resized.copyTo(processed(cv::Rect(xOffset, yOffset, 20, 20)));

    // 7. Normalize intensity to match MNIST range
    double maxVal;
    cv::minMaxLoc(processed, nullptr, &maxVal);
    if (maxVal > 0)
    {
        processed.convertTo(processed, CV_8UC1, 255.0 / maxVal);
    }

    return processed;
}

//============================================================================
// Global Drawing Canvas and Mouse State
//============================================================================

cv::Mat drawCanvas;
bool isDrawing = false;
float lastX = -1, lastY = -1;

// Enhanced mouse callback with continuous stroke support
void onMouse(int event, int x, int y, int flags, void *userdata)
{
    float fx = (float)x;
    float fy = (float)y;

    if (event == cv::EVENT_LBUTTONDOWN)
    {
        isDrawing = true;
        lastX = fx;
        lastY = fy;
        drawSoftBrush(drawCanvas, fx, fy);
        cv::imshow("Draw Digit", drawCanvas);
    }
    else if (event == cv::EVENT_MOUSEMOVE && isDrawing)
    {
        // Draw continuous line from last position
        drawSoftLine(drawCanvas, lastX, lastY, fx, fy);
        lastX = fx;
        lastY = fy;
        cv::imshow("Draw Digit", drawCanvas);
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        isDrawing = false;
    }
}

//============================================================================
// Main Drawing and Prediction Function
//============================================================================

void forwardDraw(const std::string &modelPath = "models/model_0.01_100_60000_128_64")
{
    // 1. Create a larger canvas for better resolution
    drawCanvas = cv::Mat::zeros(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1);
    cv::namedWindow("Draw Digit", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Draw Digit", onMouse, nullptr);

    std::cout << "=== MNIST Digit Drawing Tool ===" << std::endl;
    std::cout << "Instructions:" << std::endl;
    std::cout << "- Draw a digit (0-9) using your mouse" << std::endl;
    std::cout << "- Try to draw naturally, like writing with a pen" << std::endl;
    std::cout << "- Press SPACE to predict the digit" << std::endl;
    std::cout << "- Press 'c' to clear the canvas" << std::endl;
    std::cout << "- Press ESC to exit" << std::endl;
    std::cout << "=================================" << std::endl;

    // Main drawing loop
    while (true)
    {
        cv::imshow("Draw Digit", drawCanvas);
        int key = cv::waitKey(30);

        if (key == 27) // ESC key
        {
            break;
        }
        else if (key == 'c' || key == 'C') // Clear canvas
        {
            drawCanvas = cv::Mat::zeros(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1);
            cv::imshow("Draw Digit", drawCanvas);
            std::cout << "Canvas cleared. Draw a new digit." << std::endl;
        }
        else if (key == ' ') // Space bar - predict
        {
            // Check if anything was drawn
            double maxVal;
            cv::minMaxLoc(drawCanvas, nullptr, &maxVal);
            if (maxVal == 0)
            {
                std::cout << "No drawing detected. Please draw a digit first." << std::endl;
                continue;
            }

            // Preprocess the drawing
            cv::Mat processed = preprocessDrawing(drawCanvas);

            // Show preprocessed image for debugging
            cv::Mat preview;
            cv::resize(processed, preview, cv::Size(140, 140), 0, 0, cv::INTER_NEAREST);
            cv::imshow("Preprocessed (28x28)", preview);

            // Convert to vector and normalize
            std::vector<unsigned char> pixels(processed.datastart, processed.dataend);
            std::vector<double> normalized = normalizePixels(pixels);

            // Load model and predict
            try
            {
                std::vector<int> hiddenLayers = {HIDDEN_NEURONS_LAYER1, HIDDEN_NEURONS_LAYER2};
                MLP mlp(DRAW_INPUT_SIZE, hiddenLayers, OUTPUT_SIZE, 0.01);
                mlp.loadModel(modelPath);

                std::vector<double> output = mlp.forward(normalized);

                // Find the predicted digit
                auto maxIt = std::max_element(output.begin(), output.end());
                int predicted = std::distance(output.begin(), maxIt);
                double confidence = *maxIt;

                std::cout << "\n=== PREDICTION RESULTS ===" << std::endl;
                std::cout << "Predicted digit: " << predicted << std::endl;
                std::cout << "Confidence: " << (confidence * 100) << "%" << std::endl;

                // Show all probabilities
                std::cout << "All probabilities:" << std::endl;
                for (int i = 0; i < 10; i++)
                {
                    std::cout << "  " << i << ": " << (output[i] * 100) << "%" << std::endl;
                }
                std::cout << "==========================" << std::endl;
                std::cout << "Draw another digit or press ESC to exit." << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error loading model or making prediction: " << e.what() << std::endl;
            }
        }
    }

    cv::destroyAllWindows();
}

//============================================================================
// Main Entry Point
//============================================================================

int main()
{
    try
    {
        forwardDraw();
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
