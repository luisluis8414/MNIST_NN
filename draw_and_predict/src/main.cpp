#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>

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

// Improved brush settings
const float BRUSH_RADIUS = 3.0f;
const float BRUSH_SOFTNESS = 1.5f;
const float MIN_OPACITY = 0.3f;
const float MAX_OPACITY = 0.9f;

// Single window layout - optimized dimensions and spacing
const int CANVAS_SIZE = 300;        // Slightly smaller canvas
const int MAIN_WINDOW_WIDTH = 850;  // Wider window
const int MAIN_WINDOW_HEIGHT = 600; // Taller window
const int BUTTON_WIDTH = 90;        // Slightly wider buttons
const int BUTTON_HEIGHT = 35;

//============================================================================
// Global State
//============================================================================

cv::Mat mainWindow;
cv::Mat drawCanvas;
bool isDrawing = false;
float lastX = -1, lastY = -1;

// Results state
std::vector<double> lastPrediction(10, 0.0);
int predictedDigit = -1;
double confidence = 0.0;
bool hasPrediction = false;

// UI regions - reorganized without preview
cv::Rect canvasRect(25, 70, CANVAS_SIZE, CANVAS_SIZE);            // More space from top
cv::Rect predictButtonRect(25, 385, BUTTON_WIDTH, BUTTON_HEIGHT); // Below canvas with margin
cv::Rect clearButtonRect(125, 385, BUTTON_WIDTH, BUTTON_HEIGHT);  // Next to predict button
cv::Rect resultsRect(380, 70, 420, 350);                          // Right side with more space

//============================================================================
// Helper Functions
//============================================================================

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

float distance(float x1, float y1, float x2, float y2)
{
    return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

// Create gradient background
cv::Mat createGradient(int width, int height, cv::Scalar color1, cv::Scalar color2)
{
    cv::Mat gradient(height, width, CV_8UC3);
    for (int y = 0; y < height; y++)
    {
        double alpha = (double)y / height;
        cv::Scalar currentColor = color1 * (1.0 - alpha) + color2 * alpha;
        cv::line(gradient, cv::Point(0, y), cv::Point(width, y), currentColor);
    }
    return gradient;
}

// Draw a modern button
void drawButton(cv::Mat &img, cv::Rect rect, const std::string &text, cv::Scalar color, bool pressed = false)
{
    cv::Scalar btnColor = pressed ? color * 0.8 : color;
    cv::Scalar borderColor = cv::Scalar(100, 100, 100);

    // Draw button background
    cv::rectangle(img, rect, btnColor, -1);
    cv::rectangle(img, rect, borderColor, 2);

    // Add inner highlight
    if (!pressed)
    {
        cv::rectangle(img, cv::Rect(rect.x + 2, rect.y + 2, rect.width - 4, rect.height - 4),
                      color * 1.2, 1);
    }

    // Draw text
    int baseline = 0;
    cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.55, 2, &baseline);
    cv::Point textPos(rect.x + (rect.width - textSize.width) / 2,
                      rect.y + (rect.height + textSize.height) / 2);
    cv::putText(img, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 2);
}

// Draw probability bar
void drawProbabilityBar(cv::Mat &img, int x, int y, int digit, double probability, bool isSelected)
{
    int barWidth = 200; // Good width for bars
    int barHeight = 16;

    // Background bar
    cv::rectangle(img, cv::Rect(x + 30, y - 5, barWidth, barHeight), cv::Scalar(240, 240, 240), -1);
    cv::rectangle(img, cv::Rect(x + 30, y - 5, barWidth, barHeight), cv::Scalar(150, 150, 150), 1);

    // Probability bar
    int fillWidth = static_cast<int>(probability * barWidth);
    cv::Scalar barColor = isSelected ? cv::Scalar(80, 200, 80) : cv::Scalar(120, 120, 200);
    if (fillWidth > 0)
    {
        cv::rectangle(img, cv::Rect(x + 30, y - 5, fillWidth, barHeight), barColor, -1);
    }

    // Digit label
    cv::putText(img, std::to_string(digit) + ":", cv::Point(x + 5, y + 7),
                cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(50, 50, 50), 1);

    // Percentage
    std::ostringstream probStream;
    probStream << std::fixed << std::setprecision(1) << (probability * 100) << "%";
    cv::putText(img, probStream.str(), cv::Point(x + 240, y + 7),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(50, 50, 50), 1);
}

// Update the complete main window
void updateMainWindow()
{
    // Create main window background
    mainWindow = createGradient(MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT,
                                cv::Scalar(250, 250, 255), cv::Scalar(240, 240, 250));

    // Title with more space
    cv::putText(mainWindow, "MNIST Digit Recognition", cv::Point(25, 35),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(50, 50, 100), 2);

    // Drawing area label with proper spacing
    cv::putText(mainWindow, "Drawing Area:", cv::Point(canvasRect.x, canvasRect.y - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 80, 80), 1);

    // Draw canvas border and background
    cv::rectangle(mainWindow, canvasRect, cv::Scalar(255, 255, 255), -1);
    cv::rectangle(mainWindow, canvasRect, cv::Scalar(100, 100, 100), 2);

    // Copy drawing canvas to main window
    if (!drawCanvas.empty())
    {
        cv::Mat canvasColor;
        cv::cvtColor(drawCanvas, canvasColor, cv::COLOR_GRAY2BGR);
        canvasColor.copyTo(mainWindow(canvasRect));
    }

    // Control buttons with proper spacing
    drawButton(mainWindow, predictButtonRect, "PREDICT", cv::Scalar(80, 150, 80));
    drawButton(mainWindow, clearButtonRect, "CLEAR", cv::Scalar(150, 80, 80));

    // Instructions below buttons with proper spacing
    std::vector<std::string> instructions = {
        "Instructions:",
        "Draw digit (0-9) with mouse",
        "Click PREDICT for recognition",
        "Click CLEAR to restart",
        "Press ESC to exit"};

    int instructionStartY = 440; // Well below buttons
    for (size_t i = 0; i < instructions.size(); i++)
    {
        cv::Scalar color = (i == 0) ? cv::Scalar(60, 60, 60) : cv::Scalar(90, 90, 90);
        int fontWeight = (i == 0) ? 2 : 1;
        cv::putText(mainWindow, instructions[i], cv::Point(25, instructionStartY + static_cast<int>(i) * 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, color, fontWeight);
    }

    // Results section with proper spacing
    cv::putText(mainWindow, "Recognition Results:", cv::Point(resultsRect.x, resultsRect.y - 15),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(50, 50, 80), 2);

    if (hasPrediction)
    {
        // Main prediction result
        std::ostringstream predStream;
        predStream << "Predicted Digit: " << predictedDigit;
        cv::putText(mainWindow, predStream.str(), cv::Point(resultsRect.x + 10, resultsRect.y + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 120, 0), 2);

        std::ostringstream confStream;
        confStream << "Confidence: " << std::fixed << std::setprecision(1) << (confidence * 100) << "%";
        cv::putText(mainWindow, confStream.str(), cv::Point(resultsRect.x + 10, resultsRect.y + 55),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 150), 1);

        // Probability bars with proper spacing
        cv::putText(mainWindow, "All Probabilities:", cv::Point(resultsRect.x + 10, resultsRect.y + 85),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(80, 80, 80), 1);

        for (int i = 0; i < 10; i++)
        {
            int y = resultsRect.y + 110 + i * 22; // Good spacing between bars
            drawProbabilityBar(mainWindow, resultsRect.x, y, i, lastPrediction[i], i == predictedDigit);
        }
    }
    else
    {
        cv::putText(mainWindow, "Draw a digit and click PREDICT", cv::Point(resultsRect.x + 10, resultsRect.y + 35),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(120, 120, 120), 1);
        cv::putText(mainWindow, "to see recognition results", cv::Point(resultsRect.x + 10, resultsRect.y + 55),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(120, 120, 120), 1);
    }
}

// Optimized function to update only the drawing canvas area
void updateCanvasArea()
{
    // Only update the canvas area for faster real-time drawing
    if (!drawCanvas.empty())
    {
        cv::Mat canvasColor;
        cv::cvtColor(drawCanvas, canvasColor, cv::COLOR_GRAY2BGR);
        canvasColor.copyTo(mainWindow(canvasRect));
    }
}

//============================================================================
// Drawing Functions
//============================================================================

void drawSoftBrush(cv::Mat &canvas, float x, float y, float opacity = MAX_OPACITY)
{
    int minX = std::max(0, static_cast<int>(x - BRUSH_RADIUS - BRUSH_SOFTNESS));
    int maxX = std::min(canvas.cols - 1, static_cast<int>(x + BRUSH_RADIUS + BRUSH_SOFTNESS));
    int minY = std::max(0, static_cast<int>(y - BRUSH_RADIUS - BRUSH_SOFTNESS));
    int maxY = std::min(canvas.rows - 1, static_cast<int>(y + BRUSH_RADIUS + BRUSH_SOFTNESS));

    for (int py = minY; py <= maxY; py++)
    {
        for (int px = minX; px <= maxX; px++)
        {
            float dist = distance(x, y, static_cast<float>(px), static_cast<float>(py));

            if (dist <= BRUSH_RADIUS + BRUSH_SOFTNESS)
            {
                float alpha;
                if (dist <= BRUSH_RADIUS)
                {
                    alpha = opacity;
                }
                else
                {
                    float falloff = (BRUSH_RADIUS + BRUSH_SOFTNESS - dist) / BRUSH_SOFTNESS;
                    alpha = opacity * falloff;
                }

                uchar currentValue = canvas.at<uchar>(py, px);
                float currentNormalized = currentValue / 255.0f;
                float newValue = std::min(1.0f, currentNormalized + alpha);
                canvas.at<uchar>(py, px) = static_cast<uchar>(newValue * 255);
            }
        }
    }
}

void drawSoftLine(cv::Mat &canvas, float x1, float y1, float x2, float y2, float opacity = MAX_OPACITY)
{
    float dist = distance(x1, y1, x2, y2);
    int steps = std::max(1, static_cast<int>(dist / 0.5f));

    for (int i = 0; i <= steps; i++)
    {
        float t = static_cast<float>(i) / steps;
        float x = x1 + t * (x2 - x1);
        float y = y1 + t * (y2 - y1);
        drawSoftBrush(canvas, x, y, opacity);
    }
}

void smoothDrawing(cv::Mat &canvas)
{
    cv::Mat temp;
    cv::GaussianBlur(canvas, temp, cv::Size(3, 3), 0.5);
    temp.copyTo(canvas);
}

cv::Mat preprocessDrawing(const cv::Mat &rawCanvas)
{
    cv::Mat canvas = rawCanvas.clone();
    smoothDrawing(canvas);

    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(canvas, nonZeroPoints);

    if (nonZeroPoints.empty())
    {
        return cv::Mat::zeros(DRAW_INPUT_HEIGHT, DRAW_INPUT_WIDTH, CV_8UC1);
    }

    cv::Rect bbox = cv::boundingRect(nonZeroPoints);

    int padding = std::max(10, static_cast<int>(std::max(bbox.width, bbox.height) * 0.1));
    int x = std::max(bbox.x - padding, 0);
    int y = std::max(bbox.y - padding, 0);
    int width = std::min(bbox.width + 2 * padding, canvas.cols - x);
    int height = std::min(bbox.height + 2 * padding, canvas.rows - y);

    cv::Mat cropped = canvas(cv::Rect(x, y, width, height));

    int maxDim = std::max(cropped.rows, cropped.cols);
    cv::Mat square = cv::Mat::zeros(maxDim, maxDim, CV_8UC1);

    int offsetX = (maxDim - cropped.cols) / 2;
    int offsetY = (maxDim - cropped.rows) / 2;
    cropped.copyTo(square(cv::Rect(offsetX, offsetY, cropped.cols, cropped.rows)));

    cv::Mat resized;
    cv::resize(square, resized, cv::Size(20, 20), 0, 0, cv::INTER_AREA);

    cv::Mat processed = cv::Mat::zeros(DRAW_INPUT_HEIGHT, DRAW_INPUT_WIDTH, CV_8UC1);
    int xOffset = (DRAW_INPUT_WIDTH - 20) / 2;
    int yOffset = (DRAW_INPUT_HEIGHT - 20) / 2;
    resized.copyTo(processed(cv::Rect(xOffset, yOffset, 20, 20)));

    double maxVal;
    cv::minMaxLoc(processed, nullptr, &maxVal);
    if (maxVal > 0)
    {
        processed.convertTo(processed, CV_8UC1, 255.0 / maxVal);
    }

    return processed;
}

//============================================================================
// Mouse Callback
//============================================================================

void onMouse(int event, int x, int y, int flags, void *userdata)
{
    // Check if click is on drawing canvas
    if (x >= canvasRect.x && x < canvasRect.x + canvasRect.width &&
        y >= canvasRect.y && y < canvasRect.y + canvasRect.height)
    {
        float fx = static_cast<float>(x - canvasRect.x);
        float fy = static_cast<float>(y - canvasRect.y);

        if (event == cv::EVENT_LBUTTONDOWN)
        {
            isDrawing = true;
            lastX = fx;
            lastY = fy;
            drawSoftBrush(drawCanvas, fx, fy);
            updateCanvasArea(); // Use optimized update for real-time drawing
            cv::imshow("MNIST Digit Recognizer", mainWindow);
        }
        else if (event == cv::EVENT_MOUSEMOVE && isDrawing)
        {
            drawSoftLine(drawCanvas, lastX, lastY, fx, fy);
            lastX = fx;
            lastY = fy;
            updateCanvasArea(); // Use optimized update for real-time drawing
            cv::imshow("MNIST Digit Recognizer", mainWindow);
        }
        else if (event == cv::EVENT_LBUTTONUP)
        {
            isDrawing = false;
        }
    }
    // Check button clicks
    else if (event == cv::EVENT_LBUTTONDOWN)
    {
        // PREDICT button
        if (x >= predictButtonRect.x && x < predictButtonRect.x + predictButtonRect.width &&
            y >= predictButtonRect.y && y < predictButtonRect.y + predictButtonRect.height)
        {
            double maxVal;
            cv::minMaxLoc(drawCanvas, nullptr, &maxVal);
            if (maxVal == 0)
            {
                std::cout << "No drawing detected. Please draw a digit first." << std::endl;
                return;
            }

            try
            {
                cv::Mat processed = preprocessDrawing(drawCanvas);

                std::vector<unsigned char> pixels(processed.datastart, processed.dataend);
                std::vector<double> normalized = normalizePixels(pixels);

                std::vector<int> hiddenLayers = {HIDDEN_NEURONS_LAYER1, HIDDEN_NEURONS_LAYER2};
                MLP mlp(DRAW_INPUT_SIZE, hiddenLayers, OUTPUT_SIZE, 0.01);
                mlp.loadModel("models/model_0.01_100_60000_128_64");

                lastPrediction = mlp.forward(normalized);

                auto maxIt = std::max_element(lastPrediction.begin(), lastPrediction.end());
                predictedDigit = static_cast<int>(std::distance(lastPrediction.begin(), maxIt));
                confidence = *maxIt;
                hasPrediction = true;

                updateMainWindow();
                cv::imshow("MNIST Digit Recognizer", mainWindow);

                std::cout << "Prediction: " << predictedDigit << " (Confidence: "
                          << (confidence * 100) << "%)" << std::endl;
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error: " << e.what() << std::endl;
            }
        }
        // CLEAR button
        else if (x >= clearButtonRect.x && x < clearButtonRect.x + clearButtonRect.width &&
                 y >= clearButtonRect.y && y < clearButtonRect.y + clearButtonRect.height)
        {
            drawCanvas = cv::Mat::zeros(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1);
            hasPrediction = false;
            updateMainWindow();
            cv::imshow("MNIST Digit Recognizer", mainWindow);
            std::cout << "Canvas cleared." << std::endl;
        }
    }
}

//============================================================================
// Main Function
//============================================================================

void forwardDraw(const std::string &modelPath = "models/model_0.01_100_60000_128_64")
{
    // Initialize
    drawCanvas = cv::Mat::zeros(CANVAS_SIZE, CANVAS_SIZE, CV_8UC1);

    // Create single window
    cv::namedWindow("MNIST Digit Recognizer", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("MNIST Digit Recognizer", onMouse, nullptr);

    // Initial window update
    updateMainWindow();

    std::cout << "=== MNIST Digit Recognition ===" << std::endl;
    std::cout << "Application started. Use the GUI to draw and recognize digits." << std::endl;
    std::cout << "Press ESC to exit." << std::endl;

    // Main loop
    while (true)
    {
        cv::imshow("MNIST Digit Recognizer", mainWindow);

        int key = cv::waitKey(30);
        if (key == 27) // ESC
        {
            break;
        }

        // Check if window was closed (X button)
        if (cv::getWindowProperty("MNIST Digit Recognizer", cv::WND_PROP_VISIBLE) < 1)
        {
            break;
        }
    }

    cv::destroyAllWindows();
}

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
